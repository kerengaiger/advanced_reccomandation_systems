import numpy as np
from numpy import sqrt, square

from config import USER_COL, ITEM_COL, RATING_COL
from utils import get_data


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

        self.current_epoch = None
        self.mu = None

        self.last_epoch_val_loss = np.inf
        self.last_epoch_increase = False
        self.early_stop_epoch = 0

    def mse(self, preds, true_values):
        return np.sum(np.square(np.subtract(true_values, preds))) / true_values.shape[0]

    def mae(self, preds, true_values):
        return np.sum(np.subtract(true_values, preds)) / true_values.shape[0]

    def rmse(self, data):
        e = 0
        for row in data:
            user, item, rating = row
            e += square(rating - self.predict(user, item))
        # return np.round(sqrt(e / data.shape[0]), 4)
        return sqrt(e / data.shape[0])

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
        self.b_u = np.zeros(self.n_users)
        # self.b_u = np.random.normal(0, 1, self.n_users)
        self.b_i = np.zeros(self.n_items)
        # self.b_i = np.random.normal(0, 1, self.n_items)
        # self.p_u = np.zeros((self.n_users, self.k))
        self.p_u = np.random.normal(0, 1, (self.n_users, self.k))
        # self.q_i = np.zeros((self.n_items, self.k))
        self.q_i = np.random.normal(0, 1, (self.n_items, self.k))
        self.mu = train[:, 2].mean()

        self.current_epoch = 0

    def fit(self, train, valid):
        self.set_fit_params(train.values, valid.values)

        while True:
            self.run_epoch(train)
            preds_train = np.array([self.predict(u, i) for u, i in train.values[:, [0, 1]]])
            preds_valid = np.array([self.predict(u, i) for u, i in valid.values[:, [0, 1]]])
            train_epoch_rmse = np.round(sqrt(self.mse(preds_train, train.values[:, 2])), 5)
            valid_epoch_rmse = np.round(sqrt(self.mse(preds_valid, valid.values[:, 2])), 5)
            epoch_convergence = {"train rmse": train_epoch_rmse,
                                 "valid_rmse": valid_epoch_rmse}
            self.record(epoch_convergence)
            if (valid_epoch_rmse >= self.last_epoch_val_loss) and self.last_epoch_increase:
                self.early_stop_epoch = self.current_epoch - 2
                print('early stop! best epochs:', self.early_stop_epoch)
                break
            self.last_epoch_increase = \
                valid_epoch_rmse >= self.last_epoch_val_loss
            self.current_epoch += 1
            self.last_epoch_val_loss = valid_epoch_rmse

    def save_model_params(self, name_out):
        with open('p_u_' + name_out, 'wb') as f:
            np.save(f, self.p_u)
        with open('q_i_' + name_out, 'wb') as f:
            np.save(f, self.q_i)
        with open('b_u_' + name_out, 'wb') as f:
            np.save(f, self.b_u)
        with open('b_i_' + name_out, 'wb') as f:
            np.save(f, self.b_i)

    def fit_early_stop(self, train, valid):
        self.set_fit_params(train.values, valid.values)

        for epoch in range(self.early_stop_epoch):
            self.current_epoch = epoch
            self.run_epoch(train)
        # self.save_model_params('sgd.npy')

    def predict(self, u, i):
        r_u_i_pred = self.mu + self.b_u[u] + self.b_i[i] + \
                     self.q_i[i, :].T.dot(self.p_u[u, :])
        return r_u_i_pred

    def step(self, e_u_i, u, i):
        # implemented in each of son classes
        pass

    def run_epoch(self, train):
        # implemented in each of son classes
        pass


class SGD(MatrixFactorization):
    def step(self, e_u_i, u, i):
        d_loss_d_qi = -e_u_i * self.p_u[u, :] + self.gamma_i * self.q_i[i, :]
        d_loss_d_pu = -e_u_i * self.q_i[i, :] + self.gamma_u * self.p_u[u, :]
        d_loss_d_bu = -e_u_i + self.gamma_u_b * self.b_u[u]
        d_loss_d_bi = -e_u_i + self.gamma_i_b * self.b_i[i]

        self.b_u[u] = self.b_u[u] - (self.lr_u_b * d_loss_d_bu)
        self.b_i[i] = self.b_i[i] - (self.lr_i_b * d_loss_d_bi)
        self.q_i[i] = self.q_i[i] - (self.lr_i * d_loss_d_qi)
        self.p_u[u] = self.p_u[u] - (self.lr_u * d_loss_d_pu)

    def run_epoch(self, train):
        for u, i, r_u_i in train.values:
            r_u_i_pred = self.predict(u, i)
            e_u_i = r_u_i - r_u_i_pred
            self.step(e_u_i, u, i)

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
            # if u == 1:
            #     print('update user 1 bu:', a / (train[train[USER_COL] == u].shape[0] +
            #                    self.gamma_u_b))

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
            # if u==10:
            #     print('user 1 pu', np.squeeze(np.dot(np.linalg.inv(a), b)))

    def update_b_i(self, train):
        for i in train[ITEM_COL].unique():
            a = 0
            for u, r_u_i in train[train[ITEM_COL] == i][
                                  [USER_COL, RATING_COL]].values:
                a += r_u_i - self.mu - self.b_u[u] - self.p_u[u].dot(self.q_i[i].T)
            self.b_i[i] = a / (train[train[ITEM_COL] == i].shape[0] +
                               self.gamma_i_b)
            # if i == 1055:
            #     print('item 1055 bi', a / (train[train[ITEM_COL] == i].shape[0] +
            #                    self.gamma_i_b))

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
            # if i == 1055:
            #     print('item 1055 qi', np.squeeze(np.dot(np.linalg.inv(a), b)))

    def run_epoch(self, train):
        self.update_p_u(train)
        self.update_b_u(train)
        self.update_q_i(train)
        self.update_b_i(train)


def save_model(model, out_file_name):
    with open('p_u_' + out_file_name, 'wb') as f:
        np.save(f, model.p_u)
    with open('q_i_' + out_file_name, 'wb') as f:
        np.save(f, model.q_i)
    with open('b_u_' + out_file_name, 'wb') as f:
        np.save(f, model.b_u)
    with open('b_i_' + out_file_name, 'wb') as f:
        np.save(f, model.b_i)

if __name__ == '__main__':
    train, validation = get_data()

    # hyper param tuning
    params = {
        'k': [30],
        'gamma_u':[0.02],
        'gamma_i': [0.02],
        'gamma_u_b': [0.02],
        'gamma_i_b': [0.02],
        'lr_u': [0.05],
        'lr_i': [0.05],
        'lr_u_b': [0.05],
        'lr_i_b': [0.05]}

    trials_num = 20
    best_valid_rmse = np.inf
    best_model, best_params = None, None
    
    #run trials
    for trial in range(trials_num):
        trial_params = {k: np.random.choice(params[k]) for k in params.keys()}
        print('trial parameters:', trial_params)
        cur_model = SGD(**trial_params)
        # fit and update num of epochs in early stop
        cur_model.fit(train, validation)
        # refit according to num of epochs
        cur_model.fit_early_stop(train, validation)
        cur_preds = np.array([cur_model.predict(u, i) for u, i in validation.values[:, [0, 1]]])
        cur_valid_mse = cur_model.mse(cur_preds, validation[:, 2])
        cur_valid_rmse = sqrt(cur_valid_mse)
        cur_valid_r_2 = 1 - cur_valid_mse / np.var(validation[:, 2])
        cur_valid_mae = cur_model.mae(cur_preds, validation[:, 2])
        # cur_model_mpr = cur_model.mpr(validation.values)

        print('trial rmse:', cur_valid_rmse)
        if cur_valid_rmse < best_valid_rmse:
            best_valid_rmse = cur_valid_rmse
            best_valid_r_2 = cur_valid_r_2
            best_valid_mae = cur_valid_mae
            # best_valid_mpr = cur_model_mpr
            best_params = trial_params

    print(best_valid_rmse)
    print(best_valid_r_2)
    print(best_valid_mae)
    # print(best_valid_mpr)
    print(best_params)


