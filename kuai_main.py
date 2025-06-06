# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
# np.random.seed(2024)
# torch.manual_seed(2024)
import pdb

from dataset import load_data
from new_model import MF_BV_Multi_bin_JL_G

from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU,recall_func, precision_func, setup_seed

import optuna
from parse import parse_args
import pandas as pd

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

dataset_name = "kuai"

rdf_train = np.array(pd.read_table("./data/kuai/user.txt", header = None, sep = ','))     
rdf_test = np.array(pd.read_table("./data/kuai/random.txt", header = None, sep = ','))
rdf_train_new = np.c_[rdf_train, np.ones(rdf_train.shape[0])]
rdf_test_new = np.c_[rdf_test, np.zeros(rdf_test.shape[0])]
rdf = np.r_[rdf_train_new, rdf_test_new]
    
rdf = rdf[np.argsort(rdf[:, 0])]
c = rdf.copy()
for i in range(rdf.shape[0]):
    if i == 0:
        c[:, 0][i] = i
        temp = rdf[:, 0][0]
    else:
        if c[:, 0][i] == temp:
            c[:, 0][i] = c[:, 0][i-1]
        else:
            c[:, 0][i] = c[:, 0][i-1] + 1
        temp = rdf[:, 0][i]
    
c = c[np.argsort(c[:, 1])]
d = c.copy()
for i in range(rdf.shape[0]):
    if i == 0:
        d[:, 1][i] = i
        temp = c[:, 1][0]
    else:
        if d[:, 1][i] == temp:
            d[:, 1][i] = d[:, 1][i-1]
        else:
            d[:, 1][i] = d[:, 1][i-1] + 1
        temp = c[:, 1][i]

y_train = d[:, 2][d[:, 3] == 1]
y_test = d[:, 2][d[:, 3] == 0]
x_train = d[:, :2][d[:, 3] == 1]
x_test = d[:, :2][d[:, 3] == 0]
    
num_user = x_train[:,0].max() + 1
num_item = x_train[:,1].max() + 1

    
y_train = binarize(y_train, 2)
y_test = binarize(y_test, 2)
print(y_train.sum())
num_user = int(num_user)
num_item = int(num_item)
print("# user: {}, # item: {}".format(num_user, num_item))


# #%%
# "MF naive"
# mf = MF(num_user, num_item, batch_size=128)
# mf.cuda()
# mf.fit(x_train, y_train, 
#     lr=0.01,
#     lamb=5e-4,
#     tol=1e-5,
#     verbose=False)
# test_pred = mf.predict(x_test)
# mse_mf = mse_func(y_test, test_pred)
# auc_mf = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(mf, x_test, y_test)

# print("***"*5 + "[MF]" + "***"*5)
# print("[MF] test mse:", mse_func(y_test, test_pred))
# print("[MF] test auc:", auc_mf)
# print("[MF] ndcg@20:{:.6f}, ndcg@50:{:.6f}".format(
#         np.mean(ndcg_res["ndcg_20"]), np.mean(ndcg_res["ndcg_50"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF]" + "***"*5)



#%%
def exp(args):
    "MF BV"
    setup_seed(args.seed)
    print(args)
    # mf_ips = MF_BV(num_user, num_item, batch_size=args.training_args['batch_size'], monte_sample_num=args.monte_sample_num, embedding_k=args.base_model_args['emb_dim'])
    mf_ips = MF_BV_Multi_bin_JL_G(num_user, num_item, batch_size=args.training_args['batch_size'], monte_sample_num=args.monte_sample_num, embedding_k=args.base_model_args['emb_dim'], batch_size_prop=args.training_args['batch_size_prop'])
    mf_ips = mf_ips.cuda()

    # mf_ips._compute_IPS(x_train)

    # mf_ips.fit(x_train, y_train, 
    #     lr=args.base_model_args['learning_rate'],
    #     lamb=args.base_model_args['weight_decay'],
    #     tol=1e-5,
    #     verbose=False,
    #     sigma_y=args.sigma_y,
    #     sigma_o=args.sigma_o)

    mf_ips.fit(x_train, y_train, 
        lr=args.base_model_args['learning_rate'],
        lamb=args.base_model_args['weight_decay'],
        tol=1e-5,
        verbose=False,
        alpha=args.training_args['alpha'],
        G=args.training_args['G'])
    test_pred = mf_ips.predict(x_test)
    mse_mfips = mse_func(y_test, test_pred)
    auc_mfips = roc_auc_score(y_test, test_pred)
    ndcg_res = ndcg_func(mf_ips, x_test, y_test,top_k_list = [20, 50])
    recall_res = recall_func(mf_ips, x_test, y_test,top_k_list = [20, 50])
    precision_res = precision_func(mf_ips, x_test, y_test,top_k_list = [20, 50])

    print("***"*5 + "[MF_BV]" + "***"*5)
    print("[MF_BV] test mse:", mse_func(y_test, test_pred))
    print("[MF_BV] test auc:", auc_mfips)
    print("[MF_BV] ndcg@20:{:.6f}, ndcg@50:{:.6f}".format(
            np.mean(ndcg_res["ndcg_20"]), np.mean(ndcg_res["ndcg_50"])))
    print("[MF_BV] recall@20:{:.6f}, recall@50:{:.6f}".format(
            np.mean(recall_res["recall_20"]), np.mean(recall_res["recall_50"])))
    print("[MF_BV] precision@20:{:.6f}, precision@50:{:.6f}".format(
            np.mean(precision_res["precision_20"]), np.mean(precision_res["precision_50"])))
    print("[MF_BV] f1@20:{:.6f}, f1@50:{:.6f}".format(
            2 * (np.mean(precision_res["precision_20"]) * np.mean(recall_res["recall_20"])) / (np.mean(precision_res["precision_20"]) + np.mean(recall_res["recall_20"])),
            2 * (np.mean(precision_res["precision_50"]) * np.mean(recall_res["recall_50"])) / (np.mean(precision_res["precision_50"]) + np.mean(recall_res["recall_50"]))))

    # user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    # gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[MF_BV]" + "***"*5)
    return auc_mfips, np.mean(ndcg_res["ndcg_50"]), np.mean(recall_res["recall_50"])
# %%
if __name__ == "__main__": 
    args = parse_args()
    setup_seed(args.seed)
    def objective(trial):
        # emb_dim = trial.suggest_categorical('emb_dim',[8,16,32,64])
        # learning_rate = trial.suggest_categorical('learning_rate',[0.001, 0.005,0.01,0.05])
        # weight_decay=trial.suggest_categorical('weight_decay',[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        # batch_size=trial.suggest_categorical('batch_size',[2048,4096,8192])
        # monte_sample_num=trial.suggest_int('monte_sample_num',10,60)
        # # sigma_y=trial.suggest_float('sigma_y',1,10)
        # # sigma_o=trial.suggest_float('sigma_o',1,10)
        # alpha =trial.suggest_float('alpha',0.05,0.95) 
        # G=trial.suggest_int('G',1,5)

        emb_dim = 16
        learning_rate = 0.005
        weight_decay=0.0001
        batch_size=2048
        monte_sample_num=40
        # alpha =0.7360043622186028
        alpha=args.sigma_y
        G=4
        # sigma_y=4.69570946478236
        # sigma_o=4.8374586194153375
        learning_rate_pretrain = 0.05
        weight_decay_pretrain = 0.001
        learning_rate_prop = 0.01
        weight_decay_prop = 5e-05



        args.base_model_args['emb_dim']=emb_dim
        args.base_model_args['learning_rate']=learning_rate
        args.base_model_args['weight_decay']=weight_decay
        args.training_args['batch_size']=batch_size
        args.monte_sample_num=monte_sample_num
        # args.sigma_y=sigma_y
        # args.sigma_o=sigma_o
        args.training_args['learning_rate_pretrain']=learning_rate_pretrain
        args.training_args['weight_decay_pretrain']=weight_decay_pretrain
        args.training_args['learning_rate_prop']=learning_rate_prop
        args.training_args['weight_decay_prop']=weight_decay_prop
        args.training_args['alpha']=alpha
        args.training_args['G']=G

        auc, ndcg, recall = exp(args)
        return auc+ndcg+recall
        # try:
        #     auc, ndcg, recall = exp(args)
        #     return auc+ndcg+recall
        # except Exception as e:
        #     return 0
    
    if args.optuna_continue:
        study = optuna.create_study(direction='maximize', study_name='{}_{}'.format(args.dataset,args.optuna_name), storage='sqlite:///optuna_db/{}.db'.format(args.dataset),load_if_exists=args.optuna_continue)
    else:
        study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}') 
