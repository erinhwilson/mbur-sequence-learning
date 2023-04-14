# code to run cycles of synthetic scoring prediction from randomized 300bp seqs
# with data reduction and 5-fold CV

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
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import StratifiedShuffleSplit 
import time
from torch.utils.data.sampler import WeightedRandomSampler
import yaml

import models as m
import utils as u
import torch_utils as tu
from torch_utils import DatasetSpec
import viz as v

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

UP_MOTIF = 'TATATA'
DOWN_MOTIF = 'GCGCGC'
def synthetic_score(seq):
    '''
    Given a DNA sequence, return a simple synthetic score based on
    it's sequence content and presence of a specific 6mer motif
    '''
    score_dict = {
        'A':5,
        'C':2,
        'G':-2,
        'T':-5
    }

    score = np.mean([score_dict[base] for base in seq])
    if UP_MOTIF in seq:
        score += 10
    if DOWN_MOTIF in seq:
        score -= 10
    return score


def quick_model_setup(model_type,input_size):
    '''
    Some quick model types - make customizable later
    '''
    if model_type == 'CNN_simple':
        model = m.DNA_CNN(
            input_size,
            num_classes=3,
            num_filters=8,
            kernel_size=6,
            fc_node_num=10
        )

    elif model_type == '2CNN':

        # start smaller model for synth task
        model = m.DNA_2CNN_2FC(
            input_size,
            num_classes=3,
            num_filters1=8,
            num_filters2=8,
            kernel_size1=6,
            kernel_size2=6,
            fc_node_num1=10,
            fc_node_num2=10,
        )

    elif model_type == '2CNN_pool':

        # start smaller model for synth task
        model = m.DNA_2CNN_2FC(
            input_size,
            num_classes=3,
            num_filters1=8,
            num_filters2=8,
            kernel_size1=6,
            kernel_size2=6,
            conv_pool_size1=2,
            fc_node_num1=10,
            fc_node_num2=10,
        )

    else:
        raise ValueError(f"Unknown model type {model_type}. (Current: CNN_simple, 2CNN, 2CNN_pool)")

    return model

def make_random_seq_dataset(num_seqs, seq_len):
    '''
    Given a number of sequences to make and a sequence length,
    return a dataframe of randomly generated sequences
    '''
    print(f"Making {num_seqs} seq dataset of {seq_len}bp")
    syn_seqs = []
    # make num_seqs random sequences of length seq_len
    for i in range(num_seqs):
        my_seq = ''.join(np.random.choice(('C','G','T','A'), seq_len))
        syn_seqs.append((i,my_seq))

    syn_df = pd.DataFrame(syn_seqs,columns=['id','seq'])
    return syn_df

def make_random_seq_dataset_with_2_motifs(num_seqs, seq_len, m1, m2, proportion):
    '''
    Given a number of sequences to make and a sequence length,
    return a dataframe of randomly generated sequences. 
    Also, make sure two specific motifs are represented at at least 
    a certain proportion of the samples.
    '''
    print(f"Making {num_seqs} seq dataset of {seq_len}bp with balanced motifs")
    syn_seqs = []
    m1_seqs = []
    m2_seqs = []
    
    m1_enough = False
    m2_enough = False
    m1_prop = 0 # proportion of samples with motif 1
    m2_prop = 0 # proportion of samples with motif 2

    i = 0

    while not (m1_enough and m2_enough):
        my_seq = ''.join(np.random.choice(('C','G','T','A'), seq_len))
        if m1 in my_seq and m2 in my_seq:
            # skip doubles for now
            pass

        elif m1 in my_seq:
            m1_seqs.append((i,my_seq))
            i+=1

            m1_prop = len(m1_seqs)/num_seqs
            if m1_prop > proportion:
                m1_enough = True

        elif m2 in my_seq:
            m2_seqs.append((i,my_seq))
            i+=1

            m2_prop = len(m2_seqs)/num_seqs
            if m2_prop > proportion:
                m2_enough = True

        else:
            syn_seqs.append((i,my_seq))
            i+=1

    # Once we're out of this loop, check if we have enough total sequences
    total_seqs = len(syn_seqs) + len(m1_seqs) + len(m2_seqs)

    # If too many, downsample the syn_seqs without motifs
    if total_seqs > num_seqs:
        print(f"downsampling {total_seqs} randomseqs to {num_seqs}")
        overlow = total_seqs - num_seqs
        num_to_keep = len(syn_seqs) - overlow
        syn_seqs = u.downselect_list(syn_seqs,num_to_keep)

    # If not enough, make some more syn_seqs to fill the gap
    elif total_seqs < num_seqs:
        num_seqs_needed = num_seqs - total_seqs
        print(f"backfilling {num_seqs_needed} up to {num_seqs}")
        #for j in range(i,i+num_seqs_needed):
        while i < num_seqs_needed:
            my_seq = ''.join(np.random.choice(('C','G','T','A'), seq_len))
            # to preserve the intended motif balance, don't backfill sequences with 
            # M1 or M2
            if (m1 in my_seq) or (m2 in my_seq):
                pass
            else:
                syn_seqs.append((i,my_seq))
                i += 1

    else:
        print("Wow. What a magical coincidence that we have exactly the number of seqs needed!")

    # shuffle it all up for good measure
    all_seqs = m1_seqs + m2_seqs + syn_seqs
    random.shuffle(all_seqs)

    syn_df = pd.DataFrame(all_seqs,columns=['id','seq'])
    return syn_df

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

    # create weight sampler for this train split
    sampler = tu.make_weighted_sampler(train_df,target_col)

    # load train/test into dataloaders
    dls = tu.build_dataloaders(
        train_df, 
        test_df, 
        dataset_types,
        seq_col=seq_col,
        target_col=target_col,
        sampler=sampler,
        shuffle=False # set to False since sampler is used
    )
    train_dl,test_dl = dls['ohe']

    return train_dl, test_dl

def main():
    set_seed()


    # +----------------------------------------------+
    # TODO load info from config file
    # cvs = [0,1,2,3,4]
    # #augs = [0]
    # reductions = [0.1,0.25,0.5]
    # models_to_try = ['CNN','CNN_simple']
    # out_dir = 'out_synth_cls_randomseq_rw_5fold' #'pred_out'

    # seq_col_name = 'upstream_region' # TODO: put in config
    # target_col_name = 'score_reg_UD' # TODO: put in config
    # locus_col_name = 'locus_tag'

    # reweight_samples = True 
    config_file = 'config/synth_cls_config.yaml'
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
    # chkpt_dir = os.path.join(out_dir, "chkpts")
    # if not os.path.exists(chkpt_dir):
    #     os.makedirs(chkpt_dir)
    #     print("Making chkpt_dir:",chkpt_dir)

    # init result collection objects
    train_res = {}  # training results
    pred_res = {}   # prediction results

    

    # init train/test splitter
    sss = StratifiedShuffleSplit(n_splits=config['folds'], test_size=0.2, random_state=0)
    
    # for each num_seqs/seq_len combo requested
    for num_seqs in config['num_seqs']:
        for seq_len in config['seq_len']:
            # create synthetic dataset and add custom labels
            #syn_df = make_random_seq_dataset(num_seqs, seq_len)
            minority_balance = config['minority_balance']
            syn_df = make_random_seq_dataset_with_2_motifs(num_seqs, seq_len, UP_MOTIF,DOWN_MOTIF,minority_balance)
            syn_df['score'] = syn_df['seq'].apply(lambda x: synthetic_score(x))
            tu.set_reg_class_up_down(syn_df,'score',thresh=5)
            print("__ SynDF Value Counts__")
            counts = syn_df.value_counts('score_reg_UD')
            print(counts)
            print("0:", counts[0]/num_seqs)
            print("2:", counts[2]/num_seqs)
            print("____________________")

            target_col = 'score_reg_UD'
            seq_col = 'seq'

            # start looping through different folds of the synthetic data
            for fold, (train_idxs, test_idxs) in enumerate(sss.split(syn_df, syn_df[target_col])):
                train_df = syn_df.iloc[train_idxs]
                test_df = syn_df.iloc[test_idxs]
                seq_len = len(train_df[seq_col].values[0])

                train_dl,test_dl = load_train_test_splits(train_df,test_df,seq_col,target_col)

                # sequences to predict on after training
                seq_list = [
                    (train_df[seq_col].values,train_df[target_col],"train"),
                    (test_df[seq_col].values,test_df[target_col],"test")
                ]

                # for each type of model requested
                for model_type in config['models_to_try']:
                    model_name = f"{model_type}_fold{fold}_{seq_len}bp_{num_seqs}seqs"
                    model = quick_model_setup(model_type,seq_len)
                    print(f"\n___Starting training for {model_name}___")

                    # +-------------+
                    # | TRAIN MODEL |
                    # +-------------+
                    t_res = tu.collect_model_stats(
                        model_name,
                        seq_len,
                        train_dl,
                        test_dl,
                        DEVICE,
                        lr=config['lr'],
                        ep=config['epochs'],
                        pat=config['patience'],
                        opt=torch.optim.Adam,
                        model=model,
                        loss_type='classification',
                        chkpt_path=f"checkpoint_{config['minority_balance']}.pt"

                    )
                    tu.quick_loss_plot(
                        t_res['data_label'],
                        title=f"{model_name} Loss Curve",
                        save_file=f"{out_dir}/{model_name}_loss_plot.png"
                    )
                    # save training results for later
                    train_res[model_name] = t_res

                    # +------------+
                    # | EVAL MODEL |
                    # +------------+
                    p_res_df = tu.get_confusion_stats(
                        t_res['model'],
                        t_res['model_name'],
                        seq_list,
                        DEVICE,
                        title=f"{t_res['model_name']}",
                        save_file=f"{out_dir}/{model_name}_confmat.png"
                    )
                    p_res_df['model_type'] = model_type
                    p_res_df['ds_size'] = num_seqs
                    p_res_df['seq_len'] = seq_len
                    p_res_df['fold'] = fold # which cv split
                    p_res_df['best_val_score'] = t_res['best_val_score']
                    p_res_df['epoch_stop'] = t_res['epoch_stop']
                    p_res_df['total_time'] = t_res['total_time']
                    
                    pred_file_name = f"{out_dir}/{model_name}_pred_res.tsv"
                    # save a temp copy of the results?
                    p_res_df.to_csv(pred_file_name,sep='\t',index=False)

                    pred_res[model_name] = p_res_df
                    print(f"Finished {model_name}\n")

    print("\nDone training/predicting for:")    
    for x in pred_res:
        print(x)
    print("\n ALL DONE!")

if __name__ == '__main__':
    main()
