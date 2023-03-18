# code to run cycles of MPRA predictions on reduced dataset sizes

import torch
from torch import nn

import altair as alt
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import yaml


import models as m
import torch_utils as tu
from torch_utils import DatasetSpec
import utils as u


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    #os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# def shuffle_seq(dna):
#     to_shuffle = list(dna)
#     random.shuffle(to_shuffle)
#     return  ''.join(to_shuffle)

# def load_cuperus_data():
#     df = pd.read_csv("data/cuperus_random_utrs.csv",index_col=0).reset_index()
#     df['shuffled_seq'] = df['UTR'].apply(lambda x: shuffle_seq(x))
#     return df

def top_n_split(df, n, sort_col):
    '''
    Given a value n, take the top n% of reads and use those as the test split.
    '''
    sorted_df = df.sort_values(sort_col,ascending=False)
    
    total_samples = df.shape[0]
    num_test_samples = int(n*total_samples)
    test_df = sorted_df.head(num_test_samples)
    train_df = sorted_df.tail(total_samples - num_test_samples)
    
    return train_df, test_df


# def collect_model_stats(model_name,seq_len,
#                         train_dl,val_dl,
#                         lr=0.001,ep=1000,pat=100,
#                         opt=None,model=None,load_best=True):
#     '''
#     Execute run of a model and return stats and objects related
#     to its results
#     '''
#     # default model if none specified
#     if not model:
#         model = m.DNA_2CNN_2FC_Multi(
#             seq_len,
#             1, # num tasks
#         )
#     model.to(DEVICE)

#     # currently hardcoded for regression
#     loss_func = torch.nn.MSELoss() 
    
#     if opt:
#         opt = opt(model.parameters(), lr=lr)
    
#     train_losses, \
#     val_losses, \
#     epoch_stop, \
#     best_val_score = tu.run_model(
#         train_dl,
#         val_dl, 
#         model, 
#         loss_func, 
#         DEVICE,
#         lr=lr, 
#         epochs=ep, 
#         opt=opt,
#         patience=pat,
#         load_best=load_best
#     )

#     # to plot loss
#     data_label = [((train_losses,val_losses),model_name,epoch_stop,best_val_score)]
#     #tu.quick_loss_plot(data_label)
    
#     return {
#         'model_name':model_name,
#         'model':model,
#         'train_losses':train_losses,
#         'val_losses':val_losses,
#         'epoch_stop':epoch_stop,
#         'best_val_score':best_val_score,
#         'data_label':data_label
#     }

# def parity_pred_by_split(model,
#                          model_name,
#                          device,
#                          split_dfs,
#                          locus_col='locus_tag',
#                          seq_col='seq',
#                          target_col="score",
#                          splits=['train','val'],
#                          alpha=0.2,
#                          save_file=None
#                         ):
#     '''
#     Given a trained model, get the model's predictions on each split
#     of the data and create parity plots of the y predictions vs actual ys
#     '''
#     # init subplots
#     fig, axs = plt.subplots(1,len(splits), sharex=True, sharey=True,figsize=(10,4))
#     #pred_dfs = {}
#     pred_res = [] # collect prediction results for dataFrame
    
#     def parity_plot(title,ytrue,ypred,rigid=True):
#         '''
#         Individual parity plot for a specific split
#         '''
#         axs[i].scatter(ytrue, ypred, alpha=alpha)

#         r2 = r2_score(ytrue,ypred)
#         pr = pearsonr(ytrue,ypred)[0]
#         sp = spearmanr(ytrue,ypred).correlation

#         # y=x line
#         xpoints = ypoints = plt.xlim()
#         if rigid:
#             axs[i].set_ylim(min(xpoints),max(xpoints)) 
#         axs[i].plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)
#         axs[i].set_title(f"{title} (r2:{r2:.2f}|p:{pr:.2f}|sp:{sp:.2f})",fontsize=14)
#         axs[i].set_xlabel("Actual Score",fontsize=14)
#         axs[i].set_ylabel("Predicted Score",fontsize=14)

#         return r2, pr, sp
    
#     for i,split in enumerate(splits):
#         print(f"{split} split")
#         df = split_dfs[split]
#         loci = df[locus_col].values
#         seqs = list(df[seq_col].values)        
#         ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs]).to(device)
#         labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
    
#     #dfs = {} # key: model name, value: parity_df
    
#         # initialize prediction df with just locus col
#         pred_df = df[[locus_col]]
#         pred_df['truth'] = df[target_col]
#         print(f"Predicting for {model_name}")
        
        
#         # ask model to predict on seqs
#         preds = model(ohe_seqs.float()).tolist()
#         # preds is a tensor converted to a list, 
#         # single elements returned as a list, hence x[0]
#         pred_df['pred'] = [x[0] for x in preds]
        
#         # do I want the result dfs? revise if so
#         #dfs[model_name] = pred_df
        
#         # plot stuff
#         ytrue = pred_df['truth'].values
#         ypred = pred_df['pred'].values
        
#         #plt.subplot(len(splits),i+1,1)
#         split_title = split
#         r2,pr,sp = parity_plot(split_title,ytrue,ypred,rigid=True)
        
#         # save predictions
#         #pred_dfs[split] = pred_df
#         row = [model_name,split,r2,pr,sp]
#         pred_res.append(row)
    
#     # show combined plot
#     plt.suptitle(model_name,fontsize=14)
#     plt.tight_layout()
#     plt.show()
#     if save_file:
#         plt.savefig(save_file)
    
#     return pd.DataFrame(pred_res, columns=['model_name','split','r2','pearson','spearman'])

def load_train_test_splits(train_df,test_df,seq_col,target_col):
    '''
    Given a train and test df and specific seq(feature) and 
    target(label) columns, load into PyTorch dataloaders 
    '''
    # specify type of data encoding to use (just one-hot for now)
    # Maybe add to config.yaml later
    dataset_types = [
        DatasetSpec('ohe'),
    ]

    # load train/test into dataloaders
    dls = tu.build_dataloaders(
        train_df, 
        test_df, 
        dataset_types,
        seq_col=seq_col,
        target_col=target_col,
    )
    train_dl,test_dl = dls['ohe']

    return train_dl, test_dl

def quick_model_setup(model_type,input_size):
    '''
    Some quick model types - make customizable later
    '''
    if model_type == 'CNN':
        model = m.DNA_2CNN_2FC(
            input_size,
            1, # num tasks
        )
    elif model_type == 'biLSTM':
        model = m.DNA_biLSTM(
            input_size,
            DEVICE,
            num_classes=1
        )
    elif model_type == 'CNNLSTM':
        model = m.DNA_CNNLSTM(
            input_size,
            DEVICE,
            num_classes=1
        )

    else:
        raise ValueError(f"Unknown model type {model_type}. (Current: CNN, biLSTM, CNNLSTM)")

    return model


def main():
    set_seed()

    # previous data loading before kfold
    # df = load_cuperus_data()
    # # split data (originals)
    # full_train_df_og,test_df_og = top_n_split(df, 0.05, 't0')
    # train_df_og, val_df_og = tu.quick_split(full_train_df_og)
    # assert(train_df_og.shape[0] + val_df_og.shape[0] == full_train_df_og.shape[0])

    # load pre-split data
    train_file = 'data/mpra_splits/full_train_df_top5test.tsv'
    test_file = 'data/mpra_splits/test_df_top5test.tsv'
    full_train_df_og = pd.read_csv(train_file,sep='\t',index_col=0)
    test_df_og = pd.read_csv(test_file,sep='\t',index_col=0)

    # +----------------------------------------------+
    # TODO load info from config file
    # cvs = [0,1,2,3,4] 
    # reductions = [0.005,0.025,0.25,1.0]
    # models_to_try = ['CNN','biLSTM','CNNLSTM']
    # out_dir = 'pred_out_5fold' #'pred_out'

    config_file = 'config/mpra_reduction_config.yaml'
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("*****")
    print(config)
    print("*****")
    # +----------------------------------------------+
    # make out and checkpoint dirs
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Making out_dir:",out_dir)

    # init result collection objects
    train_res = {}  # training results
    pred_res = {}   # prediction results

    # load the pre-made KFold CV splits
    for fold in config['cvs']:
        train_idxs = np.load(f'data/mpra_splits/{fold}_train.npy')
        val_idxs = np.load(f'data/mpra_splits/{fold}_val.npy')
        
        train_df_og = full_train_df_og.iloc[train_idxs]
        val_df_og = full_train_df_og.iloc[val_idxs]

        # for each round of reductions
        for r in config['reductions']:
            print(f"r = {r}")
            # reduce the dataset size
            train_df = train_df_og.sample(frac=r)
            val_df = val_df_og.sample(frac=r)
            test_df = test_df_og.sample(frac=r)
            train_size = train_df.shape[0]

            split_dfs = {
                'train':train_df,
                'val':val_df,
                'test':test_df,   
            }

            dataset_types = [
                DatasetSpec('ohe'),
            ]

            
            target_col = config['target_col']
            locus_col = config['locus_col']

            for mode in config['seq_mode']:
                if mode == 'regular':
                    seq_col = 'UTR'
                elif mode == 'shuffle':
                    seq_col = 'shuffled_seq'
                else:
                    raise ValueError(f"Unknown seq_mode {mode}. Currently: [regular, shuffle]")

                seq_len = len(train_df[seq_col].values[0])

                train_dl, val_dl = load_train_test_splits(train_df,val_df,seq_col,target_col)

                # for each type of model
                for model_type in config['models_to_try']:
                    model_name = f"{model_type}_cv{fold}_r{r}_{mode}Seqs"
                    print(f"running model {model_type} for r={r} (CVfold {fold}, {mode})")
                    # model + reduction combo
                
                    # initialize a model
                    model = quick_model_setup(model_type,seq_len)
                    
                    # run model and collect stats from training
                    t_res = tu.collect_model_stats(
                        model_name,
                        seq_len,
                        train_dl,
                        val_dl,
                        DEVICE,
                        lr=config['lr'],
                        ep=config['epochs'],
                        pat=config['patience'],
                        opt=torch.optim.Adam,
                        model=model
                    )
                    # save this in training res
                    train_res[model_name] = t_res # does this go anywhere? get saved?
                    # plot loss 
                    tu.quick_loss_plot(t_res['data_label'],save_file=f"{out_dir}/{model_name}_loss_plot.png")

                    #splits_to_plot = ['val','test'] if train_size > 10000 else ['train','val','test']
                    splits_to_plot = ['train','val','test']
                    # collect prediction stats
                    p_res_df = tu.parity_pred_by_split(
                        model,
                        model_name,
                        DEVICE,
                        split_dfs,
                        locus_col=locus_col,
                        seq_col=seq_col,
                        target_col=target_col,
                        splits=splits_to_plot,
                        save_file=f"{out_dir}/{model_name}_parity_plot.png"
                    )
                    p_res_df['reduction'] = r
                    p_res_df['train_size'] = train_size
                    p_res_df['cv_fold'] = fold # which cv split
                    p_res_df['seq_mode'] = mode # regular or shuffled seq
                    p_res_df['model_type'] = model_type
                    pred_file_name = f"{out_dir}/{model_name}_pred_res.tsv"
                    # save a temp copy of the results?
                    p_res_df.to_csv(pred_file_name,sep='\t',index=False)
                    
                    pred_res[model_name] = p_res_df
                    print(f"Finished {model_name}\n")
                    # # add to master collection df
                    # all_pred_res = pd.concat([all_pred_res,p_res_df]) # add whole data row to final collection


    # # after all this save the whole df
    # all_pred_res.to_csv(f"{out_dir}/all_pred_res.tsv",sep='\t',index=False)
    print("\nDone training/predicting for:")    
    for x in pred_res:
        print(x)
    print("DONE!")

if __name__ == '__main__':
    main()
