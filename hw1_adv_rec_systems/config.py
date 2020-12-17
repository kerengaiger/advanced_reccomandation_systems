import numpy as np
TRAIN_PATH = "data/Train.csv"
VALIDATION_PATH = "data/Validation.csv"
TEST_PATH = 'data/Test.csv'

USER_COL_NAME_IN_DATAEST = 'User_ID_Alias'
ITEM_COL_NAME_IN_DATASET = 'Movie_ID_Alias'
RATING_COL_NAME_IN_DATASET = 'Ratings_Rating'

# for internal use
USER_COL = 'user'
ITEM_COL = 'item'
RATING_COL = 'rating'

SGD_PARAMS_OUT = 'sgd'

SGD_HYPER_PARAMS = {
        'k': [50, 100],
        'gamma_u': np.logspace(-2, 0, num=1000),
        'gamma_i': np.logspace(-2, 0, num=1000),
        'gamma_u_b': np.logspace(-2, 0, num=1000),
        'gamma_i_b': np.logspace(-2, 0, num=1000),
        'lr_u': np.logspace(-2, 0, num=1000),
        'lr_i': np.logspace(-2, 0, num=1000),
        'lr_u_b': np.logspace(-2, 0, num=1000),
        'lr_i_b': np.logspace(-2, 0, num=1000)}

ALS_HYPER_PARAMS = {'k': [10, 15, 20],
                    'gamma_u': np.logspace(-1.5, 0, num=1000),
                    'gamma_i': np.logspace(-1.5, 0, num=1000),
                    'gamma_u_b': np.logspace(-1.5, 0, num=1000),
                    'gamma_i_b': np.logspace(-1.5, 0, num=1000)}


TEST_OUT_SGD = 'test_preds_sgd.csv'
TEST_OUT_ALS = 'test_preds_als.csv'
