import json

import numpy as np
import pandas as pd
from numpy import sqrt, square

from config import USER_COL, ITEM_COL, RATING_COL, SGD_HYPER_PARAMS, \
    ALS_HYPER_PARAMS
from utils import get_data, get_test


class MatrixFactorization:

    def __init__(self, k=15, gamma_u=0.01, gamma_i=0.01, gamma_u_b=0.01,
                 gamma_i_b=0.01, lr_u=0.01, lr_i=0.01,
                 lr_u_b=0.01, lr_i_b=0.01):
        self.k = k  # dimension to represent user/item vectors
        self.gamma_u = gamma_u
        self.gamma_i = gamma_i
        self.gamma_u_b = gamma_u_b
        self.gamma_i_b = gamma_i_b
        self.lr_u = lr_u
        self.lr_i = lr_i
        self.lr_u_b = lr_u_b
        self.lr_i_b = lr_i_b

        self.n_users = None
        self.n_items = None
        self.b_u = None
        self.b_i = None
        self.p_u = None
        self.q_i = None
        # self.r_hat = None

        self.current_epoch = None
        self.mu = None

        self.last_epoch_val_loss = np.inf
        self.last_epoch_increase = False
        self.early_stop_epoch = 0
        self.best_rmse = np.inf
        self.r2_valid = np.inf
        self.mae_valid = np.inf

    def mse(self, preds, true_values):
        return np.sum(square(np.subtract(true_values, preds))) / true_values.shape[0]

    def mae(self, preds, true_values):
        return np.sum(np.absolute(np.subtract(true_values, preds))) / true_values.shape[0]

    def record(self, covn_dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"

        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.9}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def set_fit_params(self, train, valid):
        self.n_users = max(train[:, 0].max(), valid[:, 0].max()) + 1
        self.n_items = max(train[:, 1].max(), valid[:, 1].max()) + 1
        self.b_u = np.zeros(self.n_users, dtype='float64')
        self.b_i = np.zeros(self.n_items, dtype='float64')
        self.p_u = np.random.rand(self.n_users, self.k) * 0.1
        self.q_i = np.random.rand(self.n_items, self.k) * 0.1
        self.mu = train[:, 2].mean()
        # self.r_hat = np.dot(self.q_i, self.p_u.T)

        self.current_epoch = 0

    def calc_metrics(self, valid, preds_valid):
        mse_valid = self.mse(preds_valid, valid[:, 2])
        rmse_valid = np.round(sqrt(mse_valid), 4)
        r2_valid = 1 - mse_valid / np.var(valid[:, 2])
        mae_valid = self.mae(preds_valid, valid[:, 2])

        return rmse_valid, r2_valid, mae_valid

    def fit(self, train, valid):
        self.set_fit_params(train.values, valid.values)

        while True:
            self.run_epoch(train)
            # self.r_hat = np.dot(self.q_i, self.p_u.T)
            preds_train = np.array([self.predict(u, i, inference_mode=True)
                                    for u, i in train.values[:, [0, 1]]])
            train_epoch_rmse = np.round(
                sqrt(self.mse(preds_train, train.values[:, 2])), 4)

            preds_valid = np.array([self.predict(u, i, inference_mode=True)
                                    for u, i in valid.values[:, [0, 1]]])
            # check for nan valaues
            if preds_valid[0] != preds_valid[0]:
                print('problem with hyper-params, nan values were found')
                break

            valid_epoch_rmse, valid_epoch_r2, valid_epoch_mae = \
                self.calc_metrics(valid.values, preds_valid)
            epoch_convergence = {"train rmse": train_epoch_rmse,
                                 "valid_rmse": valid_epoch_rmse,
                                 "R^2": valid_epoch_r2,
                                 "mae": valid_epoch_mae}
            self.record(epoch_convergence)
            if (valid_epoch_rmse >= self.last_epoch_val_loss) and \
                    self.last_epoch_increase:
                self.early_stop_epoch = self.current_epoch - 2
                print('early stop! best epochs:', self.early_stop_epoch)
                break

            self.last_epoch_increase = \
                valid_epoch_rmse >= self.last_epoch_val_loss

            if not self.last_epoch_increase:
                self.best_rmse = valid_epoch_rmse
                self.r2_valid = valid_epoch_r2
                self.mae_valid = valid_epoch_mae

            self.current_epoch += 1
            self.last_epoch_val_loss = valid_epoch_rmse

    def fit_early_stop(self, train, valid, epochs):
        full_data = pd.concat([train, validation])

        self.set_fit_params(train.values, valid.values)
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.run_epoch(full_data.values)

    def predict(self, u, i, inference_mode=False):
        r_u_i_pred = self.mu + self.b_u[u] + self.b_i[i] + \
                     self.q_i[i, :].T.dot(self.p_u[u, :])
        if inference_mode:
            r_u_i_pred = round(r_u_i_pred)

        return r_u_i_pred

    def step(self, e_u_i, u, i):
        # implemented in each of son classes
        pass

    def run_epoch(self, train):
        # implemented in each of son classes
        pass


class SGD(MatrixFactorization):
    def step(self, e_u_i, u, i):
        # old_settings = np.seterr(all='raise')
        self.p_u[u, :] += self.lr_u * (
                e_u_i * self.q_i[i, :] - self.gamma_u * self.p_u[u, :])

        self.q_i[i, :] += self.lr_i * (
                e_u_i * self.p_u[u, :] - self.gamma_u * self.q_i[i, :])

        self.b_u[u] += self.lr_u_b * (
            e_u_i - self.gamma_u_b * self.b_u[u])

        self.b_i[i] += self.lr_i_b * (
            e_u_i - self.gamma_i_b * self.b_i[i])

    def run_epoch(self, train):
        for u, i, r_u_i in train.values:
            r_u_i_pred = self.predict(u, i)
            e_u_i = r_u_i - r_u_i_pred
            # if u == 4 and i == 390:
            #     print('predict for user 4 and item 390', r_u_i_pred)
            #     print('error for user 4 and item 390', e_u_i)
            self.step(e_u_i, u, i)
            # if u == 4 and i == 390:
            #     r_u_i_pred = self.predict(u, i)
            #     e_u_i = r_u_i - r_u_i_pred
            #     print('predict for user 4 and item 390', r_u_i_pred)
            #     print('error for user 4 and item 390', e_u_i)

        # exponential decay
        self.lr_i = 0.9 * self.lr_i
        self.lr_i_b = 0.9 * self.lr_i_b
        self.lr_u = 0.9 * self.lr_u
        self.lr_u_b = 0.9 * self.lr_u_b


class ALS(MatrixFactorization):
    def update_b_u(self, train):
        for u in train[USER_COL].unique():
            a = 0
            for i, r_u_i in train[train[USER_COL] == u][
                                  [ITEM_COL, RATING_COL]].values:
                a += r_u_i - self.mu - self.b_i[i] - self.p_u[u].dot(self.q_i[i].T)
            self.b_u[u] = a / (train[train[USER_COL] == u].shape[0] +
                               self.gamma_u_b)

    def update_p_u(self, train):
        for u in train[USER_COL].unique():
            sum_i_mat = np.zeros((self.k, self.k))
            sum_i_vec = np.zeros((self.k, 1))
            for i, r_u_i in train[train[USER_COL] == u][
                    [ITEM_COL, RATING_COL]].values:
                q_i = np.expand_dims(self.q_i[i, :], axis=1)
                i_mat = q_i.dot(q_i.T)
                sum_i_mat = np.add(sum_i_mat, i_mat)
                i_vec = (r_u_i - self.mu - self.b_u[u] - self.b_i[i]) * q_i
                sum_i_vec = np.add(sum_i_vec, i_vec)
            a = np.add(sum_i_mat, self.gamma_u * np.identity(self.k))
            b = sum_i_vec
            self.p_u[u, :] = np.squeeze(np.dot(np.linalg.inv(a), b))

    def update_b_i(self, train):
        for i in train[ITEM_COL].unique():
            a = 0
            for u, r_u_i in train[train[ITEM_COL] == i][
                                  [USER_COL, RATING_COL]].values:
                a += r_u_i - self.mu - self.b_u[u] - self.p_u[u].dot(self.q_i[i].T)
            self.b_i[i] = a / (train[train[ITEM_COL] == i].shape[0] +
                               self.gamma_i_b)

    def update_q_i(self, train):
        for i in train[ITEM_COL].unique():
            sum_u_mat = np.zeros((self.k, self.k))
            sum_u_vec = np.zeros((self.k, 1))
            for u, r_u_i in train[train[ITEM_COL] == i][
                    [USER_COL, RATING_COL]].values:
                p_u = np.expand_dims(self.p_u[u, :], axis=1)
                u_mat = p_u.dot(p_u.T)
                sum_u_mat = np.add(sum_u_mat, u_mat)
                u_vec = (r_u_i - self.mu - self.b_u[u] - self.b_i[i]) * p_u
                sum_u_vec = np.add(sum_u_vec, u_vec)
            a = np.add(sum_u_mat, self.gamma_i * np.identity(self.k))
            b = sum_u_vec
            self.q_i[i, :] = np.squeeze(np.dot(np.linalg.inv(a), b))

    def run_epoch(self, train):
        self.update_p_u(train)
        self.update_b_u(train)
        self.update_q_i(train)
        self.update_b_i(train)


def hyper_param_tuning(method, params):
    trials_num = 10
    best_valid_rmse, best_valid_r_2, best_valid_mae = np.inf, np.inf, np.inf
    best_model_epochs, best_params = None, None

    # run trials
    trials_dict = {}
    for trial in range(trials_num):
        print("------------------------------------------------")
        print("trial number : ", trial)
        trial_params = {k: np.random.choice(params[k]) for k in params.keys()}

        if method is 'SGD':
            model = SGD(**trial_params)
        else:
            model = ALS(**trial_params)
        # 0.912
        # trial_params = {'k': 15, 'gamma_u': 0.08, 'gamma_i': 0.12, 'gamma_u_b': 0.02, 'gamma_i_b': 0.12, 'lr_u': 0.1, 'lr_i': 0.05, 'lr_u_b': 0.05, 'lr_i_b': 0.005}
        # trial_params = {'k': 15, 'gamma_u': 0.001, 'gamma_i': 0.001, 'gamma_u_b': 0.001,
        #  'gamma_i_b': 0.001, 'lr_u': 0.001, 'lr_i': 0.001, 'lr_u_b': 0.001,
        #  'lr_i_b': 0.005}
        print('trial parameters:', trial_params)
        # fit and update num of epochs in early stop
        model.fit(train, validation)
        trials_dict[str(trial)] = (str(model.best_rmse), str(trial_params))
        print('trial best epoch:', model.early_stop_epoch)
        print('trial valid rmse of best epoch:', model.best_rmse)
        print('trial valid r2 of best epoch:', model.r2_valid)
        print('trial valid mae of best epoch:', model.mae_valid)

        # print('trial rmse:', cur_valid_rmse)
        if model.best_rmse < best_valid_rmse:
            best_valid_rmse = model.best_rmse
            best_valid_r_2 = model.r2_valid
            best_valid_mae = model.mae_valid
            best_params = trial_params
            best_model_epochs = model.early_stop_epoch

    with open('params_dict.txt', 'w', encoding="utf8") as outfile:
        json.dump(trials_dict, outfile)

    return best_params, best_model_epochs, best_valid_rmse, \
        best_valid_r_2, best_valid_mae


if __name__ == '__main__':
    train, validation = get_data(True, 1, 1)
    test = get_test()
    # hyper param tuning
    best_params_sgd, best_model_epochs_sgd, best_valid_rmse_sgd, \
        best_valid_r_2_sgd, best_valid_mae_sgd = \
        hyper_param_tuning('SGD', SGD_HYPER_PARAMS)

    print('best SGD model rmse:', best_valid_rmse_sgd)
    print('best SGD model r2:', best_valid_r_2_sgd)
    print('best SGD model mae:', best_valid_mae_sgd)

    final_model = SGD(**best_params_sgd)
    final_model.fit_early_stop(train, validation, best_model_epochs_sgd)
    test['pred'] = pd.apply(lambda row:
                            final_model.predict(row[USER_COL],
                                                row[ITEM_COL],
                                                inference_mode=True))

    best_params_als, best_model_epochs_als, best_valid_rmse_als, \
        best_valid_r_2_als, best_valid_mae_als = \
        hyper_param_tuning('ALS', ALS_HYPER_PARAMS)

    print('best ALS model rmse:', best_valid_rmse_als)
    print('best ALS model r2:', best_valid_r_2_als)
    print('best ALS model mae:', best_valid_mae_als)

    final_model = ALS(**best_params_als)
    final_model.fit_early_stop(train, validation, best_model_epochs_als)
    test['pred'] = pd.apply(lambda row:
                            final_model.predict(row[USER_COL],
                                                row[ITEM_COL],
                                                inference_mode=True))
