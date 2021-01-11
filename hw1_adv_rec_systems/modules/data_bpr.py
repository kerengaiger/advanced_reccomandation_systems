import numpy as np
import pandas as pd
from config.config import *
import random
from tqdm import tqdm
import pickle


class prep_data(object):
    def __init__(self,sample_users=None,sample_items=None):
        self.items=None
        self.users = None
        self.sample_set = (sample_users, sample_items)
        self.get_items_users()
        self.train_list=None
        self.val_list=None

    def get_items_users(self):
        file_list=[RANDOM_TEST_PATH,POPULARITY_TEST_PATH]
        df_tests=pd.DataFrame()
        for file in file_list:
            # print(self.load_sessions_file(file).head())
            df_tests=df_tests.append(self.load_sessions_file(file))
        df_train=self.load_sessions_file(TRAIN_BPR_PATH)
        if self.sample_set!=(None,None):
            #minizing the training data to iterate faster when coding
            #this is not random selecting since I did not want to reindex the items and user aliases
            sample_users, sample_items=self.sample_set
            self.users=np.arange(0,sample_users)
            self.items=np.arange(0,sample_items)
        else:
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
            item_dist.name = 'weight'
            item_dist=item_dist.sort_values(ascending=False)#.reset_index().rename(columns={0:'weight'})
            # t=item_dist.sample(10,weights='weight',replace=True)
            # t.groupby('ItemID').size().sort_values(ascending=False)

        print(f"constructing users info in list, negative selections are by: {neg_method} \n")
        for u in tqdm(self.users):
            # print(u)
            if self.sample_set != (None, None):
                if pd.Series(u).isin(sess_df.index)[0]==False:
                    continue
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
                #we need to remove the positive items from the item_dist before we sample
                neg = list(item_dist.drop(pos).sample(len(pos), weights=item_dist, replace=True).index.values)
                #trick to remove pos element from the documentation:  index values in sampled object not in weights will be assigned weights of zero
                """
                #test example 
                #pos1=[813,404]
                #t=item_dist.drop(pos1).sample(1000000, weights=item_dist, replace=True)
                #t.groupby(t.index).size().sort_values(ascending=False)
                """
            elif neg_method == 'all_others':
                #we need to remove the positive item, since it only one, we can find it and remove from neg vector
                pos_index=np.where(self.items==pos)
                neg=list(np.delete(self.items,pos_index))
            session_list.append((u, pos, neg))
        return session_list
    def get_train_val_lists(self,neg_method='uniform'):
        if self.sample_set!=(None,None):
            sess_df_all = self.load_sessions_file(TRAIN_BPR_PATH)
            sess_df_all=sess_df_all[sess_df_all['UserID'].isin(self.users) & sess_df_all['ItemID'].isin(self.items) ]
        else:
            sess_df_all=self.load_sessions_file(TRAIN_BPR_PATH)
        sess_df_val, sess_df_train =self.leave_one_out(sess_df_all)
        self.train_list=self.get_training_list(sess_df_train,neg_method=neg_method)
        self.val_list = self.get_training_list(sess_df_val, neg_method='all_others')
        return self.train_list,self.val_list
    def save_local_train_val_list(self,pkl_path):
        # save to local file
        outfile = open(pkl_path, 'wb')
        print(f"saving lists to disk {pkl_path}")
        pickle.dump( (self.train_list,self.val_list), outfile)
        outfile.close()
    def load_local_train_val_list(self,pkl_path):
        print(f"loading lists from disk {pkl_path}")
        self.train_list, self.val_list = pickle.load(open(pkl_path, 'rb'))
        return self.train_list, self.val_list

if __name__ == '__main__':
    rd=prep_data(sample_users=100,sample_items=50)
    train_list,val_list=rd.get_train_val_lists(neg_method='uniform')
    # rd.save_local_train_val_list(pkl_path="saved_models/train_val_uniform.pkl")
    # train_list, val_list = rd.load_local_train_val_list("saved_models/train_val.pkl")

    #short viz of the data model
    for ses in train_list:
        u, p, n = ses
        print(u, p[0:5], n[0:10])
