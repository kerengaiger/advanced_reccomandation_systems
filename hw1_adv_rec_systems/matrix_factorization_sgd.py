import pickle
import numpy as np
import pandas as pd
from numpy import sqrt, square
import operator
from config import USER_COL, ITEM_COL, RATING_COL, SGD_HYPER_PARAMS,PARAMS_PATH
from utils import get_data


class MF:

    def __init__(self, k, gamma_u, gamma_i, gamma_u_b,
                 gamma_i_b, lr_u, lr_i,
                 lr_u_b, lr_i_b):
        self.k = k  
        self.gamma_u = gamma_u
        self.gamma_i = gamma_i
        self.gamma_u_b = gamma_u_b
        self.gamma_i_b = gamma_i_b
        self.lr_u = lr_u
        self.lr_i = lr_i
        self.lr_u_b = lr_u_b
        self.lr_i_b = lr_i_b

        self.current_epoch = 0
        self.last_epoch_val_loss = np.inf
        self.last_epoch_increase = False
        self.early_stop_epoch = 0
        self.best_rmse = np.inf
        self.r2_valid = np.inf
        self.mae_valid = np.inf

    
    def record(self, covn_dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"

        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.9}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_metrics(self, data, preds):
        mse = MF.mse(preds, data[:, 2])
        rmse = np.round(sqrt(mse), 4)
        r2 = 1 - mse / np.var(data[:, 2])
        mae = MF.mae(preds, data[:, 2])

        return rmse, r2, mae


    def run_epoch(self,train):
        for u,i,r_u_i in train.values:
            r_u_i_pred = self.mu + self.b_u[u] + self.b_i[i] + self.R[i,u]
            r_u_i_pred = min(r_u_i_pred,5)
            error_u_i = r_u_i-r_u_i_pred
            self.step(error_u_i,u,i)
        # exponential decay
        self.lr_i = 0.9 * self.lr_i
        self.lr_i_b = 0.9 * self.lr_i_b
        self.lr_u = 0.9 * self.lr_u
        self.lr_u_b = 0.9 * self.lr_u_b

    def step(self, e_u_i, u, i):
        self.p_u[u, :] += self.lr_u * (
                e_u_i * self.q_i[i, :] - self.gamma_u * self.p_u[u, :])

        self.q_i[i, :] += self.lr_i * (
                e_u_i * self.p_u[u, :] - self.gamma_u * self.q_i[i, :])

        self.b_u[u] += self.lr_u_b * (
            e_u_i - self.gamma_u_b * self.b_u[u])

        self.b_i[i] += self.lr_i_b * (
            e_u_i - self.gamma_i_b * self.b_i[i])


    def predict(self,u,i):
        return self.mu + self.b_u[u] + self.b_i[i] + self.R[i,u]

    def fit(self,train,valid):
        while True:
            self.run_epoch(train)
            if(np.isnan(self.q_i).any() | np.isnan(self.p_u).any()):
                print('gradiant exploded')
                break
            self.R = np.dot(self.q_i,self.p_u.T)
            #calculate ranks with updated parameters
            preds_train = np.array([self.predict(u, i)
                                        for u, i in train.values[:, [0, 1]]])
            preds_valid = np.array([self.predict(u, i)
                                        for u, i in valid.values[:, [0, 1]]])
            # calculate metrics on train and validation data
            train_epoch_rmse, _, _ = self.calc_metrics(train.values, preds_train)
            valid_epoch_rmse, valid_epoch_r2, valid_epoch_mae = self.calc_metrics(valid.values, preds_valid)
            # print metrics
            epoch_convergence = {"train rmse": train_epoch_rmse,
                                     "valid_rmse": valid_epoch_rmse,
                                     "R^2": valid_epoch_r2,
                                     "mae": valid_epoch_mae}
            self.record(epoch_convergence)
            # stopping rule 
            if (valid_epoch_rmse >= self.last_epoch_val_loss) and self.last_epoch_increase:
                self.early_stop_epoch = self.current_epoch - 2
                print('early stop! best epochs:', self.early_stop_epoch)
                break

            self.last_epoch_increase = valid_epoch_rmse >= self.last_epoch_val_loss
            # update best metrics
            if not self.last_epoch_increase:
                self.best_rmse = valid_epoch_rmse
                self.r2_valid = valid_epoch_r2
                self.mae_valid = valid_epoch_mae

            self.current_epoch += 1
            self.last_epoch_val_loss = valid_epoch_rmse


    def set_params(self,train,valid):
        self.n_users = max(train.values[:, 0].max(), valid.values[:, 0].max()) + 1
        self.n_items = max(train.values[:, 1].max(), valid.values[:, 1].max()) + 1
        self.b_u = np.zeros(self.n_users,dtype='float64')
        self.b_i = np.zeros(self.n_items,dtype='float64')
        self.p_u = np.random.normal(0,0.1,size= (self.n_users, self.k))
        self.q_i = np.random.normal(0,0.1,size= (self.n_items, self.k))
        self.R = np.dot(self.q_i,self.p_u.T)
        self.mu = train.values[:, 2].mean()
    
    def fit_all_data(self,train,validation):
        all_data = pd.concat([train,validation])
        # read best results- best hyperparams and epoch to stop
        with open('params_dict.pickle', 'rb') as handle:
            best_iter_results = pickle.load(handle)
        epochs_to_run = best_iter_results['stop_epoch']
        best_hyper_params = best_iter_results['params']
        model = MF(**best_hyper_params)
        model.set_params(train,validation)
        for epoch in range(epochs_to_run+1):
            self.run_epoch(all_data)
            self.R = np.dot(self.q_i,self.p_u.T)
            preds = np.array([self.predict(u, i)
                                        for u, i in all_data.values[:, [0, 1]]])
            # calculate metrics on the training data (all data)
            epoch_rmse, _, epoch_mae = self.calc_metrics(all_data.values, preds)
            epoch_convergence = {"train rmse": epoch_rmse,
                                "mae": epoch_mae}
            self.record(epoch_convergence)
        # save parameters
        params_dict = {"mu":self.mu, 'b_u':self.b_u, 'b_i': self.b_i, 'q_i':self.q_i, 'p_u': self.p_u}
        pickle.dump(params_dict,open(PARAMS_PATH,'wb'))
    
    def predict_new_data():
        params = pickle.loads(open(PARAMS_PATH,'rb'))
        mu = params['mu']
        b_u = params['bu']
        b_i = params['bi']
        q_i = params['q_i']
        p_u = params['p_u']
        # calculate R_hat
        r_hat = np.dot(q_i,p_u.T)
        # preds_train = np.array([model.predict(u, i)for u, i in train.values[:, [0, 1]]])
    
    
    @staticmethod
    def hyper_param_search(num_of_trials, params):
        trials_dict = {}
        best_rmse_dict = {}
        for trial in range(num_of_trials):
            trial_params = {k: np.random.choice(params[k]) for k in params.keys()}
            trial_params = trial_params = {'k': 15, 'gamma_u': 0.08, 'gamma_i': 0.12, 'gamma_u_b': 0.02, 'gamma_i_b': 0.12, 'lr_u': 0.1, 'lr_i': 0.05, 'lr_u_b': 0.05, 'lr_i_b': 0.005}
            print('----------------------------------------------')
            print(f"trial number {trial} \n Hyper-Params: {trial_params}")
            model = MF(**trial_params)
            model.set_params(train,validation)
            model.fit(train,validation)
            trials_dict[trial] = {"stop_epoch":model.early_stop_epoch,"params":trial_params}
            best_rmse_dict[trial] = model.best_rmse
        
        best_rmse_key = min(best_rmse_dict, key=best_rmse_dict.get)
        best_iteration = trials_dict[best_rmse_key]        
        with open('params_dict.pickle', 'wb') as outfile:
            pickle.dump(best_iteration, outfile)
        return best_iteration

    @staticmethod
    def mse(preds, true_values):
        return np.sum(square(true_values - preds)) / true_values.shape[0]
    @staticmethod
    def mae(preds, true_values):
        return np.sum(np.absolute(true_values-preds)) / true_values.shape[0]


if __name__ == "__main__":
    train, validation = get_data(False)
    MF.hyper_param_search(20,SGD_HYPER_PARAMS)
    # with open('params_dict.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    # print('g')
    





   