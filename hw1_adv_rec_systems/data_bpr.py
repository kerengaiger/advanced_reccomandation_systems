import numpy as np
import pandas as pd
from config import *
import random
from tqdm import tqdm
class prep_data(object):
    def __init__(self):
        self.items=None
        self.users = None
        self.get_items_users()
    def get_items_users(self):
        file_list=[RANDOM_TEST_PATH,POPULARITY_TEST_PATH]
        df_tests=pd.DataFrame()
        for file in file_list:
            # print(self.load_sessions_file(file).head())
            df_tests=df_tests.append(self.load_sessions_file(file))
        df_train=self.load_sessions_file(TRAIN_BPR_PATH)
        self.users=pd.concat([df_train['UserID'],df_train['UserID']]).unique()
        self.items=pd.concat([df_train['ItemID'],df_tests['Item1'],df_tests['Item2']]).unique()
        print(f"Loaded {len(self.users):,} users and {len(self.items):,} items")
    def load_sessions_file(self,path):
        df=pd.read_csv(path)
        for col in df.columns:
            df[col] = df[col] - 1
        return df
    def leave_one_out(self,sess_df):
        # np.random.seed(9)
        sess_df_1 = sess_df.groupby('UserID').ItemID.apply(
            lambda x: x.sample(n=1)).reset_index()[['UserID', 'ItemID']]
        idx_to_remove=list(zip(sess_df_1['UserID'],sess_df_1['ItemID']))
        sess_df_all=sess_df.set_index(['UserID', 'ItemID']).drop(index=idx_to_remove).reset_index()
        sess_df_all.head()

        #sanity check:
        assert(len(sess_df_1)+len(sess_df_all)==len(sess_df))
        return sess_df_1, sess_df_all

    def get_training_list(self,sess_df_v,neg_method):
        """
        returns as training list consisted of tuples of sessions [(user_id,[positive items],[random negative items]),...,]
        """

        session_list = []
        sess_df=sess_df_v.set_index('UserID')
        sess_df.sort_index(inplace=True)

        if neg_method == 'distribution':
            item_dist=sess_df.groupby('ItemID').size()/len(sess_df)
            item_dist=item_dist.sort_values(ascending=False).reset_index().rename(columns={0:'weight'})
            # item_dist.dtypes
            # t=item_dist.sample(10,weights='weight',replace=True)
            # t.groupby('ItemID').size().sort_values(ascending=False)

        print(f"constructing users info in list, negative selections are by: {neg_method} \n")
        for u in tqdm(self.users):
            pos = sess_df.loc[u].ItemID
            if type(pos)==pd.core.series.Series:
                pos=pos.to_list()
            else:
                pos=[pos]
            if len(pos)==0:
                continue
            if neg_method=='uniform':
                neg = list(set(self.items) - set(pos))
                random.shuffle(neg)
            elif neg_method=='distribution':# this is sampled by priority
                neg=item_dist.sample(len(pos),weights='weight',replace=True).index.values
                #TODO we need to remove the positive items from the item_dist before we sample
            elif neg_method == 'all_others':
                neg=self.items #TODO: I need to remove the positive item, since it only one, we can use pop or something like that
            session_list.append((u, pos, neg))
        return session_list
    def get_train_val_lists(self,neg_method='uniform'):
        sess_df_all=self.load_sessions_file(TRAIN_BPR_PATH)
        sess_df_val, sess_df_train =self.leave_one_out(sess_df_all)
        train_list=self.get_training_list(sess_df_train,neg_method=neg_method)
        val_list = self.get_training_list(sess_df_val, neg_method='all_others')
        return train_list,val_list


if __name__ == '__main__':
    rd=prep_data()
    train_list,val_list=rd.get_train_val_lists(neg_method='distribution')


    #short viz of the data model
    for ses in train_list:
        u, p, n = ses
        print(u, p[0:5], n[0:10])
