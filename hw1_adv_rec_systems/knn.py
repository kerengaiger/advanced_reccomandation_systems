import numpy as np
import pandas as pd

from interface import Regressor
from utils import get_data, Config, find_neighbours, turn_to_sparse
import pickle
from config import CORRELATION_PARAMS_FILE_PATH

class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.train = None
        self.sim_mtx = None

    def fit(self, X: np.array):
        train_sparse = turn_to_sparse(X[:, 0].max() + 1, X[:, 1].max() + 1, X[:, 0], X[:, 1], X[:, 2])
        self.train = train_sparse.copy()
        train_sparse[train_sparse == 0] = np.nan
        df_sparse = pd.DataFrame.sparse.from_spmatrix(train_sparse)
        corrs = df_sparse.corr(method='pearson', min_periods=2)
        self.sim_mtx = corrs
        self.save_params(corrs)

    def predict_on_pair(self, user, item):
        neibs = find_neighbours(user, item, self.sim_mtx, self.train, self.k)
        if neibs.empty:
            # in case all similarities between the current item and current users's items are not valid, return the mean
            # of the user
            return self.train[self.train.nonzero()].mean()
        ratings = np.squeeze(self.train[user, neibs.index].toarray())
        sims = neibs.values
        predict = ratings.dot(sims) / np.sum(sims)
        return predict

    def save_params(self, corrs):
        corrs.to_csv(CORRELATION_PARAMS_FILE_PATH, index=False)


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print('start predict')
    print(knn.calculate_rmse(validation))
