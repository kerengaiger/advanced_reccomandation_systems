
from modules.bpr import *
from modules.data_bpr import prep_data
from config.config import BPR_PARAMS


if __name__ == '__main__':
    sampling= False #run on less data just to test the code
    sample_users = 100
    sample_items = 50
    if sampling:
        rd = prep_data(sample_users=sample_users, sample_items=sample_items)
        BPR_PARAMS['n_users']=sample_users
        BPR_PARAMS['n_items']=sample_items
    else:
        rd = prep_data()
    train_list, val_list = rd.get_train_val_lists(neg_method='uniform')

    ###
    model = BPR(**BPR_PARAMS)

    print('Starting point: ')
    print('---------------------')
    print(model.auc_val(val_list))
    print(model.loss_log_likelihood(val_list))
    print(model.loss_log_likelihood(train_list))
    print(model.precision_at_n(n=5,train_list=train_list))

    print('Training phase: ')
    trial_auc = model.fit(train_list, val_list)
    print(model.precision_at_n(n=5, train_list=train_list))
    fig=model.plot_learning_curve()
    plt.show()

