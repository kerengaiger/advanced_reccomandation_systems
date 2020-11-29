import numpy as np
import pandas as pd
import pickle

from interface import Regressor
from utils import get_data, Config, find_neighbours, turn_to_sparse

from config import CORRELATION_PARAMS_FILE_PATH, BASELINE_PARAMS_FILE_PATH


class KnnBaseline(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.train = None
        self.sim_mtx = None
        self.biases = None

    def fit(self, X: np.array):
        train_sparse = turn_to_sparse(X[:, 0].max() + 1, X[:, 1].max() + 1, X[:, 0], X[:, 1], X[:, 2])
        self.train = train_sparse
        self.upload_params()

    def predict_on_pair(self, user: int, item: int):
        neibs = find_neighbours(user, item, self.sim_mtx, self.train, self.k)
        if neibs.empty:
            # in case all similarities between the current item and current users's items are not valid, return the mean
            # of the user
            return self.train[self.train.nonzero()].mean()
        ratings = np.squeeze(self.train[user, neibs.index].toarray())
        sims = neibs.values
        predict = \
            self.biases[user, item] - (sims.dot(ratings-np.squeeze(self.biases[user, neibs.index])) / np.sum(sims))
        return predict

    def upload_params(self):
        self.sim_mtx = pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        self.biases = pickle.load(open(BASELINE_PARAMS_FILE_PATH, 'rb'))


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
