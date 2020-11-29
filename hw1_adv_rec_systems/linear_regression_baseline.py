from typing import Dict

import numpy as np

from interface import Regressor
from utils import Config, get_data
from config import BASELINE_PARAMS_FILE_PATH
import pickle


class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        return self.gamma * (np.sum(self.item_biases ** 2) + np.sum(self.user_biases ** 2))

    def fit(self, X):
        self.n_users = X[:, 0].max() + 1
        self.n_items = X[:, 1].max() + 1
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = X[:, 2].mean()

        while self.current_epoch < self.train_epochs:
            self.run_epoch(X)
            train_mse = np.square(self.calculate_rmse(X))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                 "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1

        self.save_params()

    def run_epoch(self, data):
        for u, i, r_u_i in data:
            r_u_i_pred = self.global_bias + self.user_biases[u] + self.item_biases[i]
            e_u_i = r_u_i - r_u_i_pred
            d_loss_d_bu = -e_u_i + self.gamma * self.user_biases[u]
            d_loss_d_bi = -e_u_i + self.gamma * self.item_biases[i]
            self.user_biases[u] = self.user_biases[u] - (self.lr * d_loss_d_bu)
            self.item_biases[i] = self.item_biases[i] - (self.lr * d_loss_d_bi)

    def predict_on_pair(self, user: int, item: int):
        return self.global_bias + self.user_biases[user] + self.item_biases[item]

    def save_params(self):
        bu_bi_params = np.tile(self.user_biases, (self.item_biases.shape[0], 1)).T + np.tile(self.item_biases,
                                                                                      (self.user_biases.shape[0], 1))
        global_params = np.full((self.user_biases.shape[0], self.item_biases.shape[0]), self.global_bias)
        params = bu_bi_params + global_params
        pickle.dump(params, open(BASELINE_PARAMS_FILE_PATH, 'wb'))


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
