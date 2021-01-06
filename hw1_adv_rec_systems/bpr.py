import numpy as np
import pandas as pd
from tqdm import tqdm
import functools
from joblib import Parallel, delayed
from config import TRAIN_BPR_PATH, BPR_HYPER_PARAMS, U_BEST_MODEL_FIT, \
    U_BEST_MODEL_TRIAL, I_BEST_MODEL_FIT, I_BEST_MODEL_TRIAL, \
    RANDOM_TEST_PATH, POPULARITY_TEST_PATH
from operator import itemgetter


class BPR:
    def __init__(self, k=15, lr_u=0.01, lr_i=0.01,
                 lr_j=0.01, n_users=0, n_items=0):
        self.k = k  # dimension to represent user/item vectors
        self.lr_u = lr_u
        self.lr_i = lr_i
        self.lr_j = lr_j
        self.users = np.random.rand(n_users + 1, self.k) * 0.1
        self.items = np.random.rand(n_items + 1, self.k) * 0.1
        self.early_stop_epoch = None
        self.current_epoch = 0
        self.item_popularity = []
        self.sample_method = None

    def step(self, u, i, j, e_u_i_j):
        self.users[u] += self.lr_u * (
                    e_u_i_j * (self.items[i] - self.items[j]))
        self.items[i] += self.lr_i * (e_u_i_j * self.users[u])
        self.items[j] += -self.lr_j * (e_u_i_j * self.users[u])

    def predict(self, u, i, j):
        pred = self.users[u].dot(self.items[i].T) - self.users[u].dot(
            self.items[j].T)
        return pred

    def random_j(self, u, train):
        if self.sample_method == 'Uniform':
            return np.random.choice(
                list(set(train['ItemID'].unique()) -
                     set(train[train['UserID'] == u]['ItemID'].unique())), 1)[0]
        else:
            return np.random.choice(
                list(set(train['ItemID'].unique()) -
                     set(train[train['UserID'] == u]['ItemID'].unique())), 1,
                p=self.item_popularity)[0]

    # def run_epoch(self, train):
    #     for u, i in tqdm(train.values):
    #         j = self.random_j(u, train)
    #         pred = self.predict(u, i, j)
    #         # apply sigmoid
    #         pred = 1 / (1 + np.exp(-pred))
    #         e_u_i_j = 1 - pred
    #         # print('error before', e_u_i_j)
    #         self.step(u, i, j, e_u_i_j)
    #         # pred = self.predict(u, i, j)
    #         # apply sigmoid
    #         # pred = 1 / (1 + np.exp(-pred))
    #         # e_u_i_j = 1 - pred
    #         # print('error after', e_u_i_j)

    def run_epoch(self, train):
        def calc_likelihood_u(u, i, j, users, items):
            res = users[u].dot(items[[i, j]].T)
            diff = res[0] - res[1]
            pred = 1 / (1 + np.exp(-diff))
            return pred

        trained = []
        for u in tqdm(train['UserID'].unique()):
            i = np.random.choice(
                train[train['UserID'] == u]['ItemID'].unique(), 1)[0]

            j = self.random_j(u, train)
            trained.append((u, i, j))
            pred = self.predict(u, i, j)
            # apply sigmoid
            pred = 1 / (1 + np.exp(-pred))
            e_u_i_j = 1 - pred

            self.step(u, i, j, e_u_i_j)
        print('calc likelihood of train:')
        f_partial = functools.partial(calc_likelihood_u,
                                      users=self.users,
                                      items=self.items)

        likelihood_u_lst = Parallel(n_jobs=-1)(
            delayed(f_partial)(u, i, j) for u, i, j in trained)

        return likelihood_u_lst

    def record(self, covn_dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.9}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_mean_loss(self, triples):
        errors = []
        for u, i, j in tqdm(triples):
            pred = self.predict(u, i, j)
            pred = 1 / (1 + np.exp(-pred))
            e_u_i_j = 1 - pred
            errors.append(e_u_i_j)
        return sum(errors) / len(errors)

    def save_params(self, path_out_u, path_out_i):
        with open(path_out_u, 'wb') as f:
            np.save(f, self.users)
        with open(path_out_i, 'wb') as f:
            np.save(f, self.items)

    def load_params(self, path_out_u, path_out_i):
        with open(path_out_u, 'rb') as f:
            self.users = np.load(f, self.users)
        with open(path_out_i, 'rb') as f:
            self.items = np.load(f, self.users)

    def fit(self, S_train, S_valid):
        best_auc_valid = 0
        last_epoch_auc_valid = 0
        last_epoch_decrease = False

        while True:
            # print('epoch:', self.current_epoch)
            S_train = S_train.sample(frac=1)
            train_likelihood = self.run_epoch(S_train)
            # print('calc mean trained loss')
            # cur_epoch_loss_train = self.calc_mean_loss(trained)
            print('calc evaluation AUC')
            cur_epoch_auc_valid = auc(self, S_train, S_valid)
            # epoch_convergence = {"train loss": cur_epoch_loss_train,
            #                      "valid AUC": cur_epoch_auc_valid}
            print('train log likelihood:', np.sum(np.log(np.array(train_likelihood))))
            print('valid AUC:', cur_epoch_auc_valid)

            if (cur_epoch_auc_valid <= last_epoch_auc_valid) and \
                    last_epoch_decrease:
                #TODO: configure epsilon for early stop
                self.early_stop_epoch = self.current_epoch - 2
                print('early stop! best epochs:', self.early_stop_epoch)
                break

            if not last_epoch_decrease:
                best_auc_valid = cur_epoch_auc_valid
                self.save_params(U_BEST_MODEL_FIT, I_BEST_MODEL_FIT)

            self.current_epoch += 1
            last_epoch_decrease = cur_epoch_auc_valid <= last_epoch_auc_valid
            last_epoch_auc_valid = cur_epoch_auc_valid

        return best_auc_valid


def hyper_param_tuning(params, S_train, S_valid, sample_method):
    trials_num = 5
    best_auc = 0
    # model_best = None
    params_best = None

    item_popularity = S_train.groupby(by='ItemID').UserID.size() / \
        len(S_train['ItemID'].unique())
    item_popularity = item_popularity.tolist()

    # run trials
    for trial in range(trials_num):
        print("------------------------------------------------")
        print("trial number : ", trial)
        trial_params = {k: np.random.choice(params[k]) for k in params.keys()}
        trial_params['n_users'] = len(pd.concat([S_train, S_valid])['UserID'].unique())
        trial_params['n_items'] = len(pd.concat([S_train, S_valid])['ItemID'].unique())

        model = BPR(**trial_params)
        model.sample_method = sample_method
        model.item_popularity = item_popularity

        # fit and update num of epochs in early stop
        trial_auc = model.fit(S_train, S_valid)

        if trial_auc > best_auc:
            best_auc = trial_auc
            model.save_params(U_BEST_MODEL_TRIAL, I_BEST_MODEL_TRIAL)
            # model_best = model
            params_best = trial_params

    # print('best model AUC:', best_auc)
    # return params_best, model_best.early_stop_epoch

    return params_best


def calc_delta_u(u, i, j_lst, users_arr, items_arr):
    x_u_i_pred = users_arr[u].T.dot(items_arr[i])
    x_u_j_pred_vec = users_arr[u].dot(items_arr[j_lst].T)
    # good preds are the cases where the dot product of j items with u
    # user are smaller than dot product of i item with u user
    delta_ui_uj = x_u_j_pred_vec[x_u_j_pred_vec < x_u_i_pred].shape[0]
    return delta_ui_uj


def auc_u(u, i, S_full, users_arr, items_arr):
    j_lst = list(set(S_full['ItemID'].unique()) -
                 set(S_full[S_full['UserID'] == u]['ItemID'].unique()))

    delta_ui_uj = calc_delta_u(u, i, j_lst, users_arr, items_arr)
    auc_u = delta_ui_uj / len(j_lst)
    return auc_u


def auc(model, S_train, S_valid):
    S_full = pd.concat([S_train, S_valid])

    f_partial = functools.partial(auc_u, S_full=S_full, users_arr=model.users,
                                  items_arr=model.items)

    auc_u_lst = Parallel(n_jobs=-1)(delayed(f_partial)(u, i) for u, i in S_valid.values)
    auc_tot = sum(auc_u_lst)

    return np.round(auc_tot / S_valid.shape[0], 4)


def rank_items(u, i_lst, S_full, users_arr, items_arr):
    j_lst = list(set(S_full['ItemID'].unique()) -
                 set(S_full[S_full['UserID'] == u]['ItemID'].unique()))
    j_preds = users_arr[u].dot(items_arr[j_lst].T)
    j_dict = dict(zip(j_lst, j_preds))
    i_preds = users_arr[u].dot(items_arr[i_lst].T)
    i_dict = dict(zip(i_lst, i_preds))
    j_dict.update(i_dict)

    rank_k_items = sorted(j_dict.items(),
                          key=itemgetter(1),
                          reverse=True)
    return rank_k_items


def mpr_u(u, i_lst, S_full, users_arr, items_arr):
    rank_k_items = rank_items(u, i_lst, S_full, users_arr, items_arr)
    items_scores_df = pd.DataFrame(rank_k_items, columns=['itemID', 'score'])
    mpr_u_score = items_scores_df[items_scores_df['ItemID'].isin(i_lst)].index[0]
    return 1 / mpr_u_score


def mpr(model, S_train, S_test):
    S_full = pd.concat([S_train, S_test])
    S_test = S_test.groupby('UserID').ItemID.apply(list).reset_index()

    f_partial = functools.partial(mpr_u, S_full=S_full,
                                  users_arr=model.users,
                                  items_arr=model.items)
    precision_k_u_lst = Parallel(n_jobs=-1)(delayed(f_partial)(u, i_lst)
                                            for u, i_lst in S_test.values)

    precision_k_tot = sum(precision_k_u_lst)

    return np.round(precision_k_tot / S_test.shape[0], 4)


def precision_k_u(u, i_lst, S_full, users_arr, items_arr, k):
    rank_k_items = rank_items(u, i_lst, S_full, users_arr, items_arr)
    rank_k_items_set = set(dict(rank_k_items[:k]).keys())

    precision_k_u_score = len(rank_k_items_set.intersection(set(i_lst))) / k
    return precision_k_u_score


def precision_k(model, k, S_train, S_test):
    S_full = pd.concat([S_train, S_test])

    S_test = S_test.groupby('UserID').ItemID.apply(list).reset_index()
    f_partial = functools.partial(precision_k_u, S_full=S_full,
                                  users_arr=model.users,
                                  items_arr=model.items,
                                  k=k)
    precision_k_u_lst = Parallel(n_jobs=-1)(delayed(f_partial)(u, i_lst)
                                            for u, i_lst in S_test.values)

    precision_k_tot = sum(precision_k_u_lst)

    return np.round(precision_k_tot / S_test.shape[0], 4)


def split_data(data):
    S_unobserved = data.groupby('UserID').ItemID.apply(
        lambda x: x.sample(n=1)).reset_index()[['UserID', 'ItemID']]
    S_unobserved['unobserved'] = True
    merged = pd.merge(data, S_unobserved, left_on=['UserID', 'ItemID'],
                      right_on=['UserID', 'ItemID'], how='left')
    S_observed = merged[merged['unobserved'].isnull()][['UserID', 'ItemID']]
    S_unobserved = S_unobserved[['UserID', 'ItemID']]
    return S_observed, S_unobserved


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_BPR_PATH)
    train['UserID'] = train['UserID'] - 1
    train['ItemID'] = train['ItemID'] - 1

    S_train, S_test = split_data(train)
    S_train_in, S_valid = split_data(S_train)

    best_params_uni = hyper_param_tuning(BPR_HYPER_PARAMS, S_train_in, S_valid,
                                         'Uniform')

    model = BPR(**best_params_uni)
    model.load_params(U_BEST_MODEL_TRIAL, I_BEST_MODEL_TRIAL)

    # calc precision_k
    print('model precision@1:', precision_k(model, 1, S_train, S_test))
    print('model precision@10:', precision_k(model, 10, S_train, S_test))
    print('model precision@20:', precision_k(model, 20, S_train, S_test))

    # calc mpr
    print('model MPR:', mpr(model, S_train, S_test))

    test_uniform = pd.read_csv(RANDOM_TEST_PATH)
    test_uniform['UserID'] = test_uniform['UserID'] - 1
    test_uniform['ItemID'] = test_uniform['ItemID'] - 1
    test_uniform['pred'] = \
        test_uniform.apply(lambda row: model.predict(row['UserID'],
                                                     row['Item1'],
                                                     row['Item2']), axis=1)
    test_uniform['result'] = np.where(test_uniform['pred'] > 0, 0, 1)

    ########################## TO DO #################################
    ##Do the same for popularity model##
