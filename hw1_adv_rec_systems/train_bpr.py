import numpy as np
import pandas as pd
from tqdm import tqdm
import functools
from joblib import Parallel, delayed
from config import TRAIN_BPR_PATH, BPR_HYPER_PARAMS, U_BEST_MODEL_FIT, \
    U_BEST_MODEL_TRIAL, I_BEST_MODEL_FIT, I_BEST_MODEL_TRIAL, \
    RANDOM_TEST_PATH, POPULARITY_TEST_PATH
from operator import itemgetter
from bpr import *
import pickle
import random


if __name__ == '__main__':
    load_training_list= True
    pickle_path = 'training_list.pkl'

    trial_params = {'k': 20,
                    'lr_u': 0.01,
                    'lr_i': 0.01,
                    'lr_j': 0.01,
                    'n_users': 6040,
                    'n_items': 3705,
                    'sample_method': 'Uniform',
                    'max_epochs': 5}

    train = pd.read_csv(TRAIN_BPR_PATH)

    train['UserID'] = train['UserID'] - 1
    train['ItemID'] = train['ItemID'] - 1

    data = train.copy()  #TODO: will change to include 2 tests files

    S_train, S_test = split_data(data=train,train=train)
    S_train_in, S_valid = split_data(data=S_train,train=train)

    #TODO; this will be a function / class that we do once ever. basically reformating the training data
    if load_training_list:
        S_train_list = pickle.load(open(pickle_path, 'rb'))
    else:
        #-------------this is how I prepare the data-------#
        """
        it is consisted of tuples of sessions [(user_id,[positive items],[random negative items]),...,]
        """
        # S_train : our cv training ds
        # train : our entire training
        items=data['ItemID'].unique()
        users=data['UserID'].unique()
        set
        u=1962
        S_train_list=[]
        for u in users:
            pos=S_train[S_train['UserID']==u].ItemID.to_list()
            neg=list(set(items)-set(pos))
            random.shuffle(neg)#TODO this can be ordered by priority or we can think on a way to sample by priority
            S_train_list.append((u,pos, neg))
        a,b,c=S_train_list[0]

        #save to local file
        outfile = open(pickle_path, 'wb')
        pickle.dump(S_train_list, outfile)
        outfile.close()

    # a short viz of the training data
    for ses in S_train_list:
        u, p, n = ses
        print(u, p[0:2], n[0:2])

    model = BPR(**trial_params)
    print(model.loss_log_likelihood(S_train_list))
    trial_auc = model.fit(S_train, S_valid,S_train_list)
