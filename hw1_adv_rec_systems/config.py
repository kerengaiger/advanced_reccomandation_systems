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

# SGD_HYPER_PARAMS = {
#         'k': [12, 15, 17, 40, 100],
#         'gamma_u': [0.001, 0.01, 0.05],
#         'gamma_i': [0.01, 0.08, 0.1, 0.12],
#         'gamma_u_b': [0.001, 0.01, 0.1],
#         'gamma_i_b': [0.1, 0.12, 0.09, 0.001],
#         'lr_u': [0.1, 0.07, 0.13],
#         'lr_i': [0.05, 0.01, 0.005],
#         'lr_u_b': [0.005, 0.05, 0.01, 0.1],
#         'lr_i_b': [0.01, 0.005, 0.1]}

SGD_HYPER_PARAMS = {
        'k': [12,15,17,40],
        'gamma_u':[0.08,0.1,0.12],
        'gamma_i': [0.08,0.1,0.12],
        'gamma_u_b': [0.02,0.06,0.09],
        'gamma_i_b': [0.1,0.12,0.09],
        'lr_u': [0.1,0.07,0.13],
        'lr_i': [0.05,0.01,0.005],
        'lr_u_b': [0.05,0.01,0.1],
        'lr_i_b': [0.01,0.005]}

ALS_HYPER_PARAMS = {'k': [13, 15, 17, 20],
                    'gamma_u': [0.2, 0.1, 0.3],
                    'gamma_i': [0.3, 0.2, 0.1, 0.4],
                    'gamma_u_b': [0.02, 0.01, 0.1],
                    'gamma_i_b': [0.02, 0.01, 0.1],
                    'lr_u': [0.05, 0.01, 0.005, 0.1],
                    'lr_i': [0.05, 0.01, 0.005, 0.1],
                    'lr_u_b': [0.05, 0.01, 0.005, 0.1],
                    'lr_i_b': [0.05, 0.01, 0.005, 0.1]}

TEST_OUT_SGD = 'test_preds_sgd.csv'
TEST_OUT_ALS = 'test_preds_als.csv'
