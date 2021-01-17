import numpy as np
import pandas as pd
from tqdm import tqdm
import functools
from joblib import Parallel, delayed
# from config import TRAIN_BPR_PATH, BPR_PARAMS, BPR_CANDIDATE_PARAMS,U_BEST_MODEL_FIT, \
#     U_BEST_MODEL_TRIAL, I_BEST_MODEL_FIT, I_BEST_MODEL_TRIAL, \
#     RANDOM_TEST_PATH, POPULARITY_TEST_PATH
from operator import itemgetter
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score


class BPR:
    def __init__(self,
                 k=15,
                 lr_u=0.01, lr_i=0.01,lr_j=0.01,
                 regularizers=dict(au=1e-1,av=1e-1),
                 n_users=0, n_items=0,
                 sample_method='Uniform',
                 max_epochs=10,early_stop_threshold=0.001,early_stopping_lag=10):
        self.k = k  # dimension to represent user/item vectors
        self.lr_u = lr_u
        self.lr_i = lr_i
        self.lr_j = lr_j
        self.users = np.random.rand(n_users + 1, self.k) * 1 #TODO speak with Karen, the vectors dimentation are opposite to the formulas
        self.items = np.random.rand(n_items + 1, self.k) * 1
        self.early_stop_epoch = None
        self.max_epochs=max_epochs
        self.current_epoch = 0
        self.item_popularity = []
        self.sample_method = sample_method
        self.early_stop_threshold=early_stop_threshold
        self.early_stopping_lag = early_stopping_lag
        self.regularizers=regularizers
        self.positive_array= np.array([])

    def step(self, u, i, j, e_u_i_j):
        self.users[u] += self.lr_u * (
                    e_u_i_j * (self.items[i] - self.items[j])) -self.lr_u*self.regularizers['au']*self.users[u]
        self.items[i] += self.lr_i * (e_u_i_j * self.users[u]) -self.lr_i*self.regularizers['av']*self.items[i]
        self.items[j] += -self.lr_j * (e_u_i_j * self.users[u]) -self.lr_j*self.regularizers['av']*self.items[j]

    def predict(self, u, i, j):
        pred = self.users[u].dot(self.items[i].T) - self.users[u].dot(
            self.items[j].T)
        return pred

    def run_epoch(self,mspu=1,train_list=[]):
        trained = []
        train_list_step=train_list.copy()
        random.shuffle(train_list_step) # we shuffle the order of the update
        for u,pos,neg in tqdm(train_list): #we iterate on the users
            # some edge cases:
            # 1. we can have a case where len(neg) < len(pos), here we extend the list
            if len(neg)<len(pos):
                neg=self._extend_neg(neg,len(pos))
            # 2. I defined the mspu to 1 to have a fair run time check -NOT Needed anymore
            # we align to the list size if the number of postive session is lower than the max
            random.shuffle(neg) #should help in convergence
            rmspu=min(mspu,len(pos)) # we shuffle the order of triples

            for idx,i in enumerate(pos[:rmspu]): # we iterate on the positive samples
                # print(i)
                j=neg[idx]
                trained.append((u, i, j))
                pred = self.sigmoid(self.predict(u, i, j))
                e_u_i_j=1-pred
                self.step(u, i, j, e_u_i_j)

        likelihood_u_lst=0

        return likelihood_u_lst

    def sigmoid(self,x):
        return  1 / (1 + np.exp(-x))

    def _extend_neg(self,neg,n):
        """if the neg list is shorter than the positive we just extend it"""
        k=int(n/len(neg))+1
        return neg*k

    def auc_val(self,val_list):
        """
        we average the score for all users in the val set.
        the score for each goes as follows
        the number of times the prediction chance for the positive is higher than rest of negatives:
            sigmoid(xuT*vi) > sigmoid(XuTvj) for every j element in val set
            note, we dont apply sigmoid since when x1>x2 then sigmoid(x1)>sigmoid(x2)
        """
        total_auc=0
        for u,pos,neg in val_list:
            vi=self.items[pos,:]
            vjs=self.items[neg,:]
            # we dont use sigmoid here since it is monotonic increasing function
            pos_score=np.dot(vi,self.users[[u],:].T)
            negs_scores=np.dot(vjs,self.users[[u],:].T)
            if len(pos)>1:
                p=self.sigmoid(np.concatenate((pos_score,negs_scores)))
                t=np.concatenate((np.ones((len(pos_score),1)),np.zeros((len(negs_scores),1))))
                user_auc=roc_auc_score(t,p)
            else:
                user_auc = (negs_scores <= pos_score).sum() / len(neg)
            total_auc += user_auc
        return total_auc / len(val_list)

    def loss_log_likelihood(self,train_list):
        total_loss=0
        count_items=0
        for u,pos,neg in train_list:
            # print (u)
            vis=self.items[pos,:]
            if len(neg)<len(pos):
                neg=self._extend_neg(neg,len(pos))
            end_j=len(pos)
            count_items+=len(pos)
            vjs=self.items[neg[:end_j],:]
            total_loss+=np.log(self.sigmoid(np.dot(vis,self.users[[u],:].T)-np.dot(vjs,self.users[[u],:].T))).sum()
        return total_loss/count_items

    def precision_at_n(self,n,val_list,train_list):
        """
        1. we rank the validation set best on the best scores
        2. Removing positive samples from the training set
        3. we check how many validated true session are in top n
        :parameter
        """
        #Todo: it is not fair to compute precision on validation set unless we zero the training set positives
        if self.positive_array.size == 0:
            self.positive_array = self.lookup_positive(train_list)
        precision=0
        for u,pos,neg in val_list:
            #we predict all items
            pred_u=self.users[[u]].dot(self.items.T).flatten()
            #zero'ing the pos from the train (or in this case minimizing)
            u_idx_t=self.positive_array[u]
            _,pos_item_t,_=train_list[u_idx_t]
            pred_u[pos_item_t]=pred_u.min()-1
            topn=np.argsort(pred_u)[-n:][::-1]
            precision+=len(set(topn).intersection(set(pos))) / n
        return precision/len(val_list)

    def mpr(self,sess_list):
        """"
        what is the rank of one positive prediction among all other negative rank
        if there are more than 1 negative, we choose only the first 1"""
        total_mpr=0
        for u, pos, neg in sess_list:
            if len(pos)>1:
                pos=pos[0]
            vi=self.items[pos,:]
            vjs=self.items[neg,:]
            # we dont use sigmoid here since it is monotonic increasing function
            pos_score=np.dot(vi,self.users[[u],:].T)
            negs_scores=np.dot(vjs,self.users[[u],:].T)
            #if we had the best prediction, the pos score would be the highest
            #best mpr is equal to 0
            total_mpr+=( (negs_scores>pos_score).sum() ) /(len(negs_scores))
        return total_mpr/len(sess_list)

    def classification_accuracy(self,val_list):
        accuracy = 0
        for u, pos, neg in val_list:
            vis = self.items[pos, :]
            if len(neg) < len(pos):
                neg = self._extend_neg(neg, len(pos))
            end_j = len(pos)
            vjs = self.items[neg[:end_j], :]
            correct=np.dot(vis, self.users[[u], :].T) > np.dot(vjs, self.users[[u], :].T)
            accuracy+=correct.flatten().sum()/len(pos)
        return accuracy/len(val_list)

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

    def lookup_positive(self,train_list):
        lookup_arr=np.arange(0,len(train_list))
        for i,(u,p,n) in enumerate(train_list):
            lookup_arr[i]=u
        return lookup_arr

    def fit(self,train_list,val_list):
        self.positive_array=self.lookup_positive(train_list)
        best_auc_valid = 0
        self.loss_curve = dict(training_loglike=[],
                               validation_loglike=[],
                               validation_auc=[],
                               val_accuracy=[],
                               precision_at_5=[])
        while True and self.current_epoch<=self.max_epochs:
            print('epoch:', self.current_epoch)
            # ----  suffling the users ---- #
            train_likelihood  = self.run_epoch(mspu=4000,train_list=train_list)
            # ----  updating losses and scores ---- #
            #TODO: consider create a prediction matrix and have all losses use those scores instead of having each one calculating it
            self.loss_curve['training_loglike'].append(self.loss_log_likelihood(train_list))
            self.loss_curve['validation_loglike'].append(self.loss_log_likelihood(val_list))
            self.loss_curve['validation_auc'].append(self.auc_val(val_list))
            self.loss_curve['val_accuracy'].append(self.classification_accuracy(val_list))
            self.loss_curve['precision_at_5'].append(self.precision_at_n(n=5, val_list=val_list,train_list=train_list))
            print(f"calc evaluation AUC: {self.loss_curve['validation_auc'][self.current_epoch]:.3f}")
            print(f"total train log likelihood: {self.loss_curve['training_loglike'][self.current_epoch]:.3f}")
            # ----  early stopping ---- #
            # Early stopping
            if self.current_epoch > self.early_stopping_lag:
                if self.loss_curve['validation_loglike'][self.current_epoch] - self.early_stop_threshold > \
                        self.loss_curve['validation_loglike'][self.current_epoch - self.early_stopping_lag]:
                    print(f"Reached early stopping in epoch {self.current_epoch}")
                    break

            self.current_epoch += 1

        return best_auc_valid

    def plot_learning_curve(self):
        # ---- plotting the validation and training ---- #
        fig, ax = plt.subplots(2,2,figsize=(10, 5))

        epochs = epochs = range(1, len(self.loss_curve['training_loglike'])+ 1)

        #left side
        tr_mse = ax[0,0].plot(epochs, self.loss_curve['training_loglike'], 'b', label='Training Loss (Normalized log-likelihood)')
        val_mse = ax[0,0].plot(epochs, self.loss_curve['validation_loglike'], 'g', label='Validation Loss (Normalized log-likelihood)')
        ax[0,0].legend()
        #left side
        tr_mse = ax[0,1].plot(epochs, self.loss_curve['validation_auc'], 'g', label='Validation AUC')
        ax[0,1].legend()
        #bottom
        tr_mse = ax[1, 0].plot(epochs, self.loss_curve['val_accuracy'], 'b', label='Validation accuracy')
        ax[1, 0].legend()
        tr_mse = ax[1, 1].plot(epochs, self.loss_curve['precision_at_5'], 'g', label='val precision_at_5')
        ax[1, 1].legend()
        return fig.axes

def hyper_param_tuning(params, S_train, S_valid, sample_method):
    trials_num = 5
    best_auc = 0
    # model_best = None
    params_best = None

    #TODO: why we divide by number of items ? if we divide by numbers of sessions, we might be able to use it to use the distribution
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


def rank_items(u, i_lst, S_full, users_arr, items_arr):
    j_lst = list(set(S_full['ItemID'].unique()) -
                 set(S_full[S_full['UserID'] == u]['ItemID'].unique()))
    j_preds = users_arr[u].dot(items_arr[j_lst].T)
    j_dict = dict(zip(j_lst, j_preds))
    i_preds = users_arr[u].dot(items_arr[i_lst].T)
    i_dict = dict(zip(i_lst, i_preds))
    j_dict = j_dict.update(i_dict)

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

#
# if __name__ == '__main__':
#
#     # calc precision_k
#     print('model precision@1:', precision_k(model, 1, S_train, S_test))
#     print('model precision@10:', precision_k(model, 10, S_train, S_test))
#     print('model precision@20:', precision_k(model, 20, S_train, S_test))
#
#     # calc mpr
#     print('model MPR:', mpr(model, S_train, S_test))
#
#     test_uniform = pd.read_csv(RANDOM_TEST_PATH)
#     test_uniform['UserID'] = test_uniform['UserID'] - 1
#     test_uniform['ItemID'] = test_uniform['ItemID'] - 1
#     test_uniform['pred'] = \
#         test_uniform.apply(lambda row: model.predict(row['UserID'],
#                                                      row['Item1'],
#                                                      row['Item2']), axis=1)
#     test_uniform['result'] = np.where(test_uniform['pred'] > 0, 0, 1)
#
#     ########################## TO DO #################################
#     ##Do the same for popularity model##
