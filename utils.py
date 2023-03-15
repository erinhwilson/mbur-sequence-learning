import altair as alt
import copy
#import logomaker
#import math
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
random.seed(7)

from Bio.Seq import reverse_complement


# import torch
# from torch.utils.data import TensorDataset,DataLoader
# from torch import nn

from override import GENE_NAME_OVERRIDE, GENE_PRODUCT_OVERRIDE, SYS_LOOKUP

def load_promoter_seqs(filename):
    '''
    Load fasta file of promoters into ID, desc, and seq. It expects
    each fasta header to be divided by "|" with in the format:
    LOCUS_TAG|GENE_SYMBOL|PRODUCT
    '''
    proms = []
    with open(filename,'r') as f:
        for line in f:
            if line.startswith(">"):
                full_header = line.strip()[1:].strip()
                locus_tag = full_header.split('|')[0]
            else:
                seq = line.strip().upper()
                proms.append((locus_tag,full_header,seq))
                
    return proms

def alter_upstream_seq_len(d,verbose=True):
    seq_lens = [len(x) for x in d.values()]
    max_seq_len = max(seq_lens)
    for loc in d:
        seq_len = len(d[loc])
        if seq_len < max_seq_len:
            ns = ''.join(["N" for x in range(max_seq_len-seq_len)])
            d[loc] = ns+d[loc]
            if verbose: 
                print(f"WARNING: sequence for locus {loc} is shorter ({seq_len}) than max of {max_seq_len}. Padding with Ns")
                print(d[loc])

def load_data(
        upstream_region_file,
        op_file,
        data_mat_file, 
        sample2cond_file, 
        sample_file, 
        condition_file,
        coded_meta_file=None
        ):
    '''
    Wrapper function to load data from files into relavent objects
    '''
    # load upstream seq regions
    seqs = load_promoter_seqs(upstream_region_file)
    loc2seq = dict([(x,z) for (x,y,z) in seqs])
    alter_upstream_seq_len(loc2seq)

    
    # load operon estimate file
    op_df = pd.read_csv(op_file,sep='\t')
    op_leads = set(op_df[op_df['op?']==False]['locus_tag'].values)
    
    # load TPM data
    tpm_df = pd.read_csv(data_mat_file,sep='\t').fillna('')

    
    # load mapping from sample to condition
    with open(sample2cond_file,'r') as f:
        sample2condition = dict(x.strip().split() for x in f.readlines())

    
    # load sample to include file
    if sample_file:
        with open(sample_file,'r') as f:
            samples = list(x.strip() for x in f.readlines())
    # if none provided, just use all the samples from the sample2condition dict
    else: 
        samples = list(sample2condition.keys())

        
    # load the conditions to include file
    if condition_file:
        with open(condition_file,'r') as f:
            conditions = list(x.strip() for x in f.readlines())
    # if none provided, just use all the conditions
    else:
        conditions = list(set([sample2condition[x] for x in sample2condition]))

    # load coded metadata file
    if coded_meta_file:
        print("Warning: using a 'coded metadata file' is a feature currently hard coded for adjusting 5G data file and does not generalize to other organisms!")
        meta_df = pd.read_csv(coded_meta_file,sep='\t')
        meta_df['sample'] = meta_df['#sample']+'_tpm'
    else:
        meta_df = None

    return loc2seq, op_leads, tpm_df, sample2condition, samples, conditions, meta_df

def make_info_dict(df):
    '''
    Given a TPM df, make a dict with some metadata keyed by
    locus tag
    '''
    
    def get_info_for_row(row):
        d = {
        'gene':row[1],
        'product':row[2],
        'type':row[3],
        }
        return d

        
    info = df[['locus_tag','gene_symbol','product','type']].values
    locus2info = dict(
        [(row[0],get_info_for_row(row)) for row in info]
    )

    # update with overrides
    for loc in GENE_NAME_OVERRIDE:
        locus2info[loc]['gene'] = GENE_NAME_OVERRIDE[loc]
    for loc in GENE_PRODUCT_OVERRIDE:
        locus2info[loc]['product'] = GENE_PRODUCT_OVERRIDE[loc]
        
    return locus2info
    
    
def one_hot_encode(seq):
    #print("one hot encoding...")
    
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'A':[1.0,0.0,0.0,0.0],
             'C':[0.0,1.0,0.0,0.0],
             'G':[0.0,0.0,1.0,0.0],
             'T':[0.0,0.0,0.0,1.0],
             'N':[0.0,0.0,0.0,0.0]}
    
    # Create empty matrix.
    #vec=torch.tensor([nuc_d[x] for x in seq])
    vec=np.array([nuc_d[x] for x in seq])#.flatten()
        
    return vec
    

def downselect_list(l,n):
    '''
    Given a list l and a target number n, shuffle l and reduce to n items. 
    '''
    # copy the list so we don't alter the original
    l2 = copy.deepcopy(l)
    # shuffle the list
    random.shuffle(l2)
    # return n of the shuffled list
    return l2[:n]

def kmers(k):
    return [''.join(x) for x in product(['A','C','G','T'], repeat=k)]

def count_kmers_in_seq(seq, mers):
    '''
    For a given sequence and kmer set, return the count
    '''
    return [seq.count(mer) for mer in mers]


# +------------------------------+
# | Data transformation functions |
# +------------------------------+

def get_gene_means_by_condition(df,samples,sample2condition):
    '''
    Given a df of genes with samples in the columns, return a df with
    conditions as the rows and columns that are genes with values averaged
    by condition
    '''
    # transpose dataframe to make experiments the rows and genes the columns
    df_T = df.set_index('locus_tag')[samples].T.reset_index().rename(columns={'index':'sample'})
    # ^^ kinda gross pandas syntax... try to understand what each part does
    df_T['exp_condition'] = df_T['sample'].apply(lambda x: sample2condition[x])
    
    # list of all locus tags
    loci = list(df['locus_tag'].values)

    # get average value of each locus_tag in each condition
    # (unique same ids go away)
    df_means = df_T[['exp_condition']+loci]\
                        .groupby('exp_condition',as_index=False)\
                        .mean()

    return df_means

def make_lr_df(raw_df, denom_col):
    '''
    Given a df of raw tpm data, convert each gene's TPM value
    to a log ratio df relative to a specific denominator column
    '''
    # initialize log ratio df from locus tags in df_meansT
    lr_df = pd.DataFrame()
    lr_df['locus_tag'] = raw_df.reset_index()['locus_tag']
    lr_df = lr_df.set_index('locus_tag')
    
    
    if denom_col not in raw_df.columns:
        raise ValueError(f"Column {denom_col} not in dataframe. Choose from {raw_df.columns}")

    # calculate LOG RATIO of each condition divided by denom column
    for cond in raw_df.columns:
        lr_df[cond] = np.log2(raw_df[cond]/raw_df[denom_col])
        
    return lr_df


def check_row(row,conds,thresh=0.5):
    '''
    Quick check to see if a given gene changed much (had a big log
    ratio value) across any of the conditions specified. Used in a
    lambad function.
    '''
    change = False
    for c in conds:
        val = np.abs(row[c])
        if val > thresh:
            change=True
        
    return change

def make_lr_melt_df(lr_df,conds,neutral_filt=False):
    '''
    Given a dataframe of log ratio'd tpms, focus on a few specific conditions and make
    a melted dataframe for plotting. 

    Optionally, filter out genes that don't really change much in any condition

    TODO: change cu var names
    '''

    lr_cu = lr_df[conds]
    lr_cu_melt = pd.melt(lr_cu.reset_index(), id_vars=['locus_tag'],var_name="cond",value_name="tpm_lr")
    for i in ['gene','product','type']:
        lr_cu_melt[i] = lr_cu_melt['locus_tag'].apply(lambda x: locus2info[x][i])

    lr_cu_melt['desc_string'] = lr_cu_melt.apply(
        lambda row: f"{row['locus_tag']}|{row['gene']}|{row['product']}",axis=1
    )

    # filter out genes that don't really change
    if neutral_filt:
        cu_scores =lr_df[conds]
        cu_scores['change?'] = cu_scores.apply(lambda row: check_row(row,conds),axis=1)
        changing_genes = set(cu_scores[cu_scores['change?']==True].index)
        
        return lr_cu_melt[lr_cu_melt['locus_tag'].isin(changing_genes)]
        
    return lr_cu_melt


def make_XY_df(df,conds,loc2seq, op_leads, op_filt=True):
    '''
    For a particular df and a set of conditions, add the upstream
    sequence to each gene and filter to only the set of conditions.
    
    By default, filter out genes possibly inside operons (not in op_leads)
    '''
    XYdf = df.reset_index()
    XYdf['upstream_region'] = XYdf['locus_tag'].apply(lambda x: loc2seq[x])
    XYdf = XYdf[['locus_tag','upstream_region']+conds]
    
    # filter out genes likely to be inside operons
    if op_filt:
        XYdf = XYdf[XYdf['locus_tag'].isin(op_leads)]

    return XYdf.reset_index().rename(columns={'index':'og_index'})

# +-----------------------------+
# | Data Augmentation functions |
# +-----------------------------+

def augment_revcomp(df,seq_col='upstream_region'):
    '''
    Given a dataframe of training data, augment it by adding the
    reverse complemented version of the sequence
    '''
    
    new_rows = []
    # for each row in the original df
    for i,row in df.iterrows():
        seq = row[seq_col]
        new_row = copy.deepcopy(row)
        # get the revcomp and set it as the seq_col value
        new_row[seq_col] = reverse_complement(seq)         
        # add row to new rows
        new_rows.append(new_row.values)
    
    # put new rows into a df
    new_rows = pd.DataFrame(new_rows,columns=df.columns)

    # add direction to og df and new rows
    df['dir'] = 'fwd'
    new_rows['dir'] = 'rev'
    
    return pd.concat([df,new_rows])
        
def augment_slide(df,w,flankseq_dict,s=50,id_col='locus_tag',seq_col='upstream_region'):
    '''
    Given a dataframe of training data, augment it by sliding a window
    of size w down the length of seq_col by stride s
    '''
    seq_len = max([len(x) for x in flankseq_dict.values()])
    # make sure the sliding window and stride are equally divisible
    assert (seq_len-w) % s == 0
    slides = int((seq_len-w)/s) + 1 # number of full slides down seq
    
    new_rows = []
    # for each row in the original df
    for i,row in df.iterrows():
        # get seq from the flank dict
        lt = row[id_col]
        seq = flankseq_dict[lt]

        # slide across seq
        left=0
        for i in range(slides):
            # take sequence slice
            right = left+w # window size
            new_seq = seq[left:right]
            
            # add to new row
            new_row = copy.deepcopy(row)
            new_row[seq_col] = new_seq
            new_row['slide'] = i
            new_rows.append(new_row.values)
            
            # take next stride
            left += s
            
    # use index order of final new_row
    aug_df = pd.DataFrame(new_rows,columns=new_row.index)
    return aug_df

def augment_mutate(df,n,seq_col='upstream_region',mutation_rate=0.03):
    '''
    Given a dataframe of training data, augment it by adding 
    mutated versions back into the data frame
    '''
    mutation_dict = {
        'A':['C','G','T'],
        'C':['G','T','A'],
        'G':['T','A','C'],
        'T':['A','C','G']
    }
    new_rows = []
    # for each row in the original df
    for i,row in df.iterrows():
        seq = row[seq_col]
                
        # generate n mutants
        for j in range(n):
            new_row = copy.deepcopy(row)
            new_seq = list(seq)
            mutate_vec = [random.random() for x in range(len(seq))]
            
            # loop through mutation values along length of the seq
            for k in range(len(seq)):
                # if random value is below mutation rate, then make a change
                if mutate_vec[k] < mutation_rate:
                    cur_base = seq[k]
                    # select new base randomly
                    new_base = random.choice(mutation_dict[cur_base])
                    new_seq[k] = new_base
            
            new_row[seq_col] = ''.join(new_seq)
            new_row['seq_version'] = j+1
            new_rows.append(new_row.values)

    # put new rows into a df
    new_rows = pd.DataFrame(new_rows,columns=new_row.index)
    # add version to og df 
    df['seq_version'] = 0
    
    return pd.concat([df,new_rows])




# def quick_split(df, split_frac=0.8, verbose=False):
#     '''
#     Given a df, randomly split between
#     train and test. Not a formal train/test split, just a quick n dirty version.
    
#     '''

#     # train test split
#     idxs = list(range(df.shape[0]))
#     random.shuffle(idxs)

#     split = int(len(idxs)*split_frac)
#     train_idxs = idxs[:split]
#     test_idxs = idxs[split:]
    
#     # split df and convert to tensors
#     train_df = df[df.index.isin(train_idxs)]
#     test_df = df[df.index.isin(test_idxs)]
        
#     return train_df, test_df

# +-------------------+
# | PYTORCH FUNCTIONS |
# +-------------------+

# MOVED TO torch_utils.py
# REPLACED WITH UDPATED VERSION - includes decision for OHE vs Kmer
# def build_dataloaders_single(df, seq_encoding="oh",target_col="score",batch_size=32):
#     '''
#     Given a df, encode the sequence for modeling, split into train and test 
#     and put into pytorch loaders
#     '''
#     # if one-hot encoding, add oh column
#     if seq_encoding == 'oh':
#         df[seq_encoding] = df['seq'].apply(lambda x: one_hot_encode(x))
#     elif seq_encoding == 'kmer':
#         raise ValueError("Not implemented yet")
#     else: 
#         raise ValueError(f"Unknown seq encoding {seq_encoding}")
    
#     # split
#     train_df, test_df = quick_split(df)
    
#     # make train test tensors and load into DataLoaders
#     x_train = torch.tensor(list(train_df[seq_encoding].values))
#     y_train = torch.tensor(list(train_df[target_col].values)).unsqueeze(1)
#     x_test  = torch.tensor(list(test_df[seq_encoding].values))
#     y_test  = torch.tensor(list(test_df[target_col].values)).unsqueeze(1)
    
#     train_ds = TensorDataset(x_train, y_train)
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

#     test_ds = TensorDataset(x_test, y_test)
#     test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
    
#     return train_dl, test_dl, train_df, test_df

# MOVED TO torch_utils.py
# def loss_batch(model, loss_func, xb, yb, opt=None,verbose=False):
#     '''
#     Apply loss function to a batch of inputs. If no optimizer
#     is provided, skip the back prop step.
#     '''
#     if verbose:
#         print('loss batch ****')
#         print("xb shape:",xb.shape)
#         print("yb shape:",yb.shape)

#     xb_out = model(xb.float())
#     if verbose:
#         print("model out pre loss", xb_out.shape)
#     loss = loss_func(xb_out, yb.float())

#     if opt is not None:
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#     #print("lb returning:",loss.item(), len(xb))
#     return loss.item(), len(xb)


# def fit(epochs, model, loss_func, opt, train_dl, test_dl):
#     '''
#     Fit the model params to the training data, eval on unseen data.
#     Loop for a number of epochs and keep train of train and val losses 
#     along the way
#     '''
#     # keep track of losses
#     train_losses = []    
#     test_losses = []
    
#     # loops through epochs
#     for epoch in range(epochs):
#         #print("TRAIN")
#         model.train()
#         ts = []
#         ns = []
#         # collect train loss; provide opt so backpropo happens
#         for xb, yb in train_dl:
#             t, n = loss_batch(model, loss_func, xb, yb, opt=opt)
#             ts.append(t)
#             ns.append(n)
#         train_loss = np.sum(np.multiply(ts, ns)) / np.sum(ns)
#         train_losses.append(train_loss)
        
#         #print("EVAL")
#         model.eval()
#         with torch.no_grad():
#             losses, nums = zip(
#                 # loop through test batches
#                 # returns loss calc for test set batch size
#                 # unzips into two lists
#                 *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
#                 # Note: no opt provided, backprop won't happen
#             )
#         # Gets average MSE loss across all batches (may be of diff sizes, hence the multiply)
#         #print("losses", losses)
#         #print("nums", nums)
#         test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

#         print(epoch, test_loss)
#         test_losses.append(test_loss)

#     return train_losses, test_losses

# def run_model(train_dl,test_dl, model, lr=0.01, epochs=20):
#     '''
#     Given data and a model type, run dataloaders with MSE loss and SGD opt
#     '''
#     # define loss func and optimizer
#     loss_func = torch.nn.MSELoss() 
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
#     # run the training loop
#     train_losses, test_losses = fit(epochs, model, loss_func, optimizer, train_dl, test_dl)
    
#     #return model, train_losses, test_losses
#     return train_losses, test_losses

# def quick_loss_plot(data_label):
#     '''Plot loss by epoch'''
#     for data, label in data_label:
#         plt.plot(data, label=label)
#     plt.legend()
#     plt.ylabel("MSE loss")
#     plt.xlabel("Epoch")
#     plt.show()
    
# def quick_seq_pred(model, seqs, oracle):
#     '''Given some sequences, get the model's predictions '''
#     for dna in seqs:
#         s = torch.tensor(one_hot_encode(dna)).unsqueeze(0)
#         pred = model(s.float())
#         actual = oracle[dna]
#         diff = actual - pred.item()
#         print(f"{dna}: pred:{pred.item():.3f} actual:{actual:.3f} ({diff:.3f})")



# def get_conv_layers_from_model(model):
#     '''
#     Given a trained model, extract its convolutional layers
#     '''
#     model_children = list(model.children())
    
#     # counter to keep count of the conv layers
#     model_weights = [] # we will save the conv layer weights in this list
#     conv_layers = [] # we will save the actual conv layers in this list
#     bias_weights = []
#     counter = 0 

#     # append all the conv layers and their respective weights to the list
#     for i in range(len(model_children)):
#         # get model type of Conv1d
#         if type(model_children[i]) == nn.Conv1d:
#             counter += 1
#             model_weights.append(model_children[i].weight)
#             conv_layers.append(model_children[i])
#             bias_weights.append(model_children[i].bias)

#         # also check sequential objects' children for conv1d
#         elif type(model_children[i]) == nn.Sequential:
#             for child in model_children[i]:
#                 if type(child) == nn.Conv1d:
#                     counter += 1
#                     model_weights.append(child.weight)
#                     conv_layers.append(child)
#                     bias_weights.append(child.bias)

#     print(f"Total convolutional layers: {counter}")
#     return conv_layers, model_weights, bias_weights

# def view_filters(model_weights, num_cols=8):
#     num_filt = model_weights[0].shape[0]
#     filt_width = model_weights[0][0].shape[1]
#     num_rows = int(np.ceil(num_filt/num_cols))
    
#     # visualize the first conv layer filters
#     plt.figure(figsize=(20, 17))

#     for i, filter in enumerate(model_weights[0]):
#         ax = plt.subplot(num_rows, num_cols, i+1)
#         ax.imshow(filter.detach(), cmap='gray')
#         ax.set_yticks(np.arange(4))
#         ax.set_yticklabels(['A', 'C', 'G','T'])
#         ax.set_xticks(np.arange(filt_width))
#         ax.set_title(f"Filter {i}")

#     plt.tight_layout()
#     plt.show()
    
# def get_conv_output_for_seq(seq, conv_layer):
#     '''
#     Given an input sequeunce, get the output tensor containing the filter activations
#     '''
#     print(f"Running seq {seq}")
#     # format seq for input to conv layer (OHE, reshape)
#     seq = torch.tensor(one_hot_encode(seq))#.view(-1,len(seq),4).permute(0,2,1)
#     # OHE FIX??
    
#     # run through conv layer
#     with torch.no_grad(): # don't want as part of gradient graph?
#         res = conv_layer(seq.float())
#         return res[0]
    

# def get_filter_activations(seqs, conv_layer):
#     '''
#     Given a set of input sequences and a trained convolutional layer, 
#     determine the subsequences for which each filter in the conv layer 
#     activate most strongly. 
    
#     1.) Run inputs through conv layer. 
#     2.) Loop through filter activations of the resulting tensor, saving the
#             position where filter activations were >0. 
#     3.) Compile a count matrix for each filter by accumulating subsequences which
#             activate the filter
#     '''
#     # initialize dict of pwms for each filter in the conv layer
#     num_filters = conv_layer.out_channels
#     filt_width = conv_layer.kernel_size[0]
#     filter_pwms = dict((i,torch.zeros(4,filt_width)) for i in range(num_filters))
    
#     # loop through a set of sequences and collect subseqs where each filter activated
#     for seq in seqs:
#         res = get_conv_output_for_seq(seq, conv_layer)
#         # for each filter and it's activation vector
#         for filt_id,act_vec in enumerate(res):
#             activated_positions = [x.item() for x in torch.where(act_vec>0)[0]]
            
#             # get subsequences that caused filter to activate
#             for pos in activated_positions:
#                 subseq = seq[pos:pos+filt_width]
#                 #subseq_tensor = torch.tensor(one_hot_encode(subseq)).view(-1,filt_width,4).permute(0,2,1).squeeze(0)
#                 subseq_tensor = torch.tensor(one_hot_encode(subseq)).permute(0,2,1).squeeze(0)
#                 # OHE FIX??
                
#                 # add this subseq to the pwm count for this filter
#                 filter_pwms[filt_id] += subseq_tensor
            
            
            
#     return filter_pwms


# def view_filters_and_logos(model_weights,filter_activations, num_cols=8):
    
#     assert(model_weights[0].shape[0] == len(filter_activations))
#     # make sure the model weights agree with the number of filters
#     num_filts = len(filter_activations)
#     num_rows = int(np.ceil(num_filts/num_cols))*2+1 # not sure why +1 is needed... complained otherwise
    
#     plt.figure(figsize=(20, 17))

#     j=0 # use to make sure a filter and it's logo end up vertically paired
#     for i, filter in enumerate(model_weights[0]):
#         if (i)%num_cols == 0:
#             j += num_cols
#     #     print('i:', i)
#     #     print('j:', j)
#     #     print('i%8 == 0', i%8 == 0)
#     # #     print('i+1%9 =?', (i+1)%9)
#     #     print("i+j+1=", i+j+1)
#     #     print("i+j+1+4=", i+j+1+8)
#     #     print("*******")

#         # display raw filter
#         ax1 = plt.subplot(num_rows, num_cols, i+j+1)
#         ax1.imshow(filter.detach(), cmap='gray')
#         ax1.set_yticks(np.arange(4))
#         ax1.set_yticklabels(['A', 'C', 'G','T'])
#         ax1.set_xticks(np.arange(3))
#         ax1.set_title(f"Filter {i}")

#         # display sequence logo
#         ax2 = plt.subplot(num_rows, num_cols, i+j+1+num_cols)
#         filt_df = pd.DataFrame(filter_activations[i].T.numpy(),columns=['A','C','G','T'])
#         filt_df_info = logomaker.transform_matrix(filt_df,from_type='counts',to_type='information')
#         logo = logomaker.Logo(filt_df_info,ax=ax2)
#         ax2.set_ylim(0,2)
#         ax2.set_title(f"Filter {i}")

#     plt.tight_layout()



# def parity_plot(model,df, pearson):
#     plt.scatter(df['pred'].values, df['truth'].values, alpha=0.2)
    
#     # y=x line
#     xpoints = ypoints = plt.xlim()
#     plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)

#     plt.xlabel("Predicted Score",fontsize=14)
#     plt.ylabel("Actual Score",fontsize=14)
#     plt.title(f"{model} (pearson:{pearson:.3f})",fontsize=20)
#     plt.show()
    
# def alt_parity_plot(model,df, pearson,task):
#     chart = alt.Chart(df).mark_circle(opacity=0.2).encode(
#         alt.X('pred:Q'),
#         alt.Y('truth:Q'),
#         tooltip=['seq:N']
#     ).properties(
#         title=f'Model (pearson:{pearson})'
#     ).interactive()
    
#     chart.save(f'alt_out/parity_plot_{task}_{model}.html')
    

# def parity_pred(models, seqs, oracle,task,alt=True):
#     '''Given some sequences, get the model's predictions '''
#     dfs = {} # key: model name, value: parity_df
    
    
#     for model_name,model in models:
#         print(f"Running {model_name}")
#         data = []
#         for dna in seqs:
#             s = torch.tensor(one_hot_encode(dna))#.unsqueeze(0)
#             actual = oracle[dna]
#             pred = model(s.float())
#             data.append([dna,actual,pred.item()])
#         df = pd.DataFrame(data, columns=['seq','truth','pred'])
#         pearson = df['truth'].corr(df['pred'])
#         dfs[model_name] = (pearson,df)
        
#         #plot parity plot
#         if alt: # make an altair plot
#             alt_parity_plot(model_name, df, pearson,task)
#         parity_plot(model_name, df, pearson)

#     return dfs
