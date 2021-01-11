
from bpr import *
from data_bpr import prep_data



if __name__ == '__main__':
    trial_params = {'k': 20,
                    'lr_u': 0.01,
                    'lr_i': 0.01,
                    'lr_j': 0.01,
                    'n_users': 6040,
                    'n_items': 3705,
                    'sample_method': 'Uniform',
                    'max_epochs': 100}

    # rd=prep_data(sample_users=100,sample_items=50)
    rd=prep_data()
    train_list, val_list = rd.get_train_val_lists(neg_method='uniform')
    # train_list, val_list = rd.load_local_train_val_list("saved_models/train_val_uniform.pkl")

    # # short viz of the data model and a random sanity check
    # user_list=[]
    # for ses in train_list:
    #     u, p, n = ses
    #     user_list.append(u)
    #     print(u, p[0:5], n[0:10])
    # ut_idx=user_list.index(4)
    # user_list=[]
    # for ses in val_list:
    #     u, p, n = ses
    #     user_list.append(u)
    #     print(u, p[0:5], n[0:10])
    # uv_idx=user_list.index(4)
    # u_t,p_t,n_t=train_list[ut_idx]
    # u,p_v,n=val_list[uv_idx]

    ###
    model = BPR(**trial_params)
    print(model.auc_val(val_list))
    print(model.loss_log_likelihood(train_list))
    trial_auc = model.fit(train_list, val_list)
    fig=model.plot_learning_curve()
    plt.show()
