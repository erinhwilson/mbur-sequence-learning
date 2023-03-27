# code to run cycles of MPRA predictions on reduced dataset sizes
# targetted for Evolution Vaishnav paper

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

    # load pre-split data
    filename = 'data/defined_media_traning_data_0.05sample.txt'
    df = pd.read_csv('data/defined_media_traning_data_0.05sample.txt',sep='\t')

    # +----------------------------------------------+
    config_file = 'config/evo_reduction_config.yaml'
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

    # quick split randomly for different folds
    for fold in config['cvs']:
        full_train_df_og,test_df_og = tu.quick_split(df)
        train_df_og,val_df_og = tu.quick_split(full_train_df_og)

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
            seq_col = config['seq_col']

            for mode in config['seq_mode']:
                if mode == 'regular':
                    seq_col = seq_col
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
                    #train_res[model_name] = t_res # does this go anywhere? get saved?
                    # plot loss 
                    tu.quick_loss_plot(t_res['data_label'],save_file=f"{out_dir}/{model_name}_loss_plot.png")

                    splits_to_plot = ['val','test'] if train_size > 10000 else ['train','val','test']
                    #splits_to_plot = ['train','val','test']
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
                    
                    #pred_res[model_name] = p_res_df
                    print(f"Finished {model_name}\n")


    # # after all this save the whole df
    print("DONE!")

if __name__ == '__main__':
    main()
