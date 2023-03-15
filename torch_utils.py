# torch_utils.py
# functions for doing things in Pytorch
import altair as alt
import logomaker
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import random
import time
import tqdm

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,precision_recall_curve,matthews_corrcoef,r2_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report


import torch
from torch.utils.data import Dataset,DataLoader #,TensorDataset
from torch import nn
from torch.utils.data.sampler import WeightedRandomSampler


import utils as u
# import EarlyStopping
from pytorchtools import EarlyStopping

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

# +------------------------+
# | Custom Dataset classes |
# +------------------------+
class SeqDatasetOHE(Dataset):
    '''
    Multi-task for one-hot-encoded sequences
    '''
    def __init__(self,df,seq_col='seq',target_col='score'):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        
        self.ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in self.seqs])
    
        self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
        
        #print("which seq?",self.seqs)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]
        
        return seq, label
    
class SeqDatasetKmer(Dataset):
    '''
    Multi-task for k-mer vector sequences
    '''
    def __init__(self,df,k=3,seq_col='seq',target_col='score'):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        self.kmers = u.kmers(k)
        
        self.kmer_vecs = torch.stack([torch.tensor(u.count_kmers_in_seq(x,self.kmers)) for x in self.seqs])
    
        self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        kmer_vec = self.kmer_vecs[idx]
        label = self.labels[idx]
        
        return kmer_vec, label


class DatasetSpec():
    '''
    Quick access class for speciying the type of dataset to build
    '''
    def __init__(self,ds_type,k=None):
        self.name = ds_type
        self.k = k
        
        self.id = self.name if not k else f"{self.name}_{k}"

# +-----------------------+
# | Dataloaders functions |
# +-----------------------+

def quick_split(df, split_frac=0.8, verbose=False):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    '''
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs)*split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]
    
    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]
        
    return train_df, test_df

def stratified_partition(df, cpd,class_col='reg'):
    '''
    Given a specification for how to split specific classes into 
    train-test and train-val, implement those splits independently 
    and return final dfs for train/test/val splits
    '''
    
    # make sure classes and CPD specs match
    assert set(cpd.keys()) == set(df[class_col].unique())
    
    final_full_train = pd.DataFrame(columns=df.columns)
    final_test = pd.DataFrame(columns=df.columns)
    final_train = pd.DataFrame(columns=df.columns)
    final_val = pd.DataFrame(columns=df.columns)

    # for class in class_partition_dict
    for c in cpd:
        temp_df = df[df[class_col]==c]
        print(f"class {c}: {temp_df.shape[0]} examples")
        full_train,test = quick_split(temp_df, split_frac=cpd[c]['train_test'])
        train,val = quick_split(full_train, split_frac=cpd[c]['train_val'])
        
        final_full_train = pd.concat([final_full_train, full_train])
        final_test = pd.concat([final_test, test])
        final_train = pd.concat([final_train, train])
        final_val = pd.concat([final_val, val])
        
    return final_full_train, final_test, final_train, final_val



def build_dataloaders_single(train_df,
                             test_df, 
                             ds_specs,
                             seq_col='seq',
                             target_col="score",
                             batch_size=128,
                             split_frac=0.8,
                             sampler=None,
                             shuffle=True
                            ):
    '''
    Given a df, split into train and test, and encode the sequence for modeling 
    based on the requested dataset types (eg OHE or Kmer counts). Load each 
    Dataset into a pytorch loaders. 
    '''
    
    # split
    #train_df, test_df = u.quick_split(df,split_frac=split_frac)
    
    dls = {} # collect data loaders
    
    for ds in ds_specs:
        # Kmer data set
        if ds.name == 'kmer':
            if not ds.k:
                raise ValueError(f"To use SeqDatasetKmer, you must specify an integer value for k in DatasetSpec")
            assert(type(ds.k) == int)
            
            train_ds = SeqDatasetKmer(train_df, ds.k,seq_col=seq_col,target_col=target_col)
            test_ds = SeqDatasetKmer(test_df, ds.k,seq_col=seq_col,target_col=target_col)
            
        # One-hot encoding
        elif ds.name == 'ohe':
            train_ds = SeqDatasetOHE(train_df,seq_col=seq_col,target_col=target_col)
            test_ds = SeqDatasetOHE(test_df,seq_col=seq_col,target_col=target_col)
            
        # unknown datatype?
        else:
            raise ValueError(f"Unknown Dataset Type {ds.name}.")

        # Put DataSets into DataLoaders
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,sampler=sampler)
        #test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
        test_dl = DataLoader(test_ds, batch_size=batch_size) # why was *2 ever used?
        dls[ds.id] = (train_dl,test_dl)
    
    return dls


def make_st_skorch_dfs(df,seq_col='seq',target_col='score'):
    '''
    Make basic X,y matrix,vec for skorch fit() loop.
    '''
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs])

    labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
    # had to unsqueeze here or else errors later
    
    return ohe_seqs.float(), labels.float()


def make_mt_skorch_dfs(df,seq_col='seq',target_cols=['highCu','noCu']):
    '''
    Make multi-task X,y matrix,vec for skorch fit() loop.
    '''
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs])

    # number of labels = len(target_cols)
    labels = torch.tensor(list(df[target_cols].values))
    # bad dimension? fixed in model.forward for now
    
    return ohe_seqs.float(), labels.float()

# +-------------------------------------------+
# | Classification group assignment functions |
# +-------------------------------------------+

def set_reg_class_up_down(df, col,thresh=1.0):
    '''
    Given a dataframe of log ratio TPMS, add a column splitting genes into categories
    * Below -thresh: class 0
    * Between -thresh:thresh: class 1
    * Above thresh: class 2
    '''
    def get_class(val):
        if val < -thresh:
            return 0
        elif val > thresh:
            return 2
        else:
            return 1
    
    reg_col = f"{col}_reg_UD"
    df[reg_col] = df[col].apply(lambda x: get_class(x))

    return reg_col
    
def set_reg_class_yes_no(df, col,thresh=1.0):
    '''
    Given a dataframe of log ratio TPMS, add a column splitting genes into categories
    * Below -thresh: class 0
    * Between -thresh:thresh: class 1
    * Above thresh: class 0
    '''
    def get_class(val):
        if val < -thresh:
            return 0
        elif val > thresh:
            return 0
        else:
            return 1
    
    reg_col = f"{col}_reg_YN"
    df[reg_col] = df[col].apply(lambda x: get_class(x))


# +--------------------------------+
# | Training and fitting functions |
# +--------------------------------+

def loss_batch(model, loss_func, xb, yb, opt=None,verbose=False):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    if verbose:
        print('loss batch ****')
        print("xb shape:",xb.shape)
        print("yb shape:",yb.shape)
        print("yb shape:",yb.squeeze(1).shape)
        #print("yb",yb)

    xb_out = model(xb.float())
    if verbose:
        print("model out pre loss", xb_out.shape)
        #print('xb_out', xb_out)
        print("xb_out:",xb_out.shape)
        print("yb:",yb.shape)
        print("yb.long:",yb.long().shape)
    
    # determine formatting of yb for loss function
    # determined by Erin's trial and error >.<
    loss_str = str(loss_func)
    if loss_str in ['MSELoss()']:
        loss = loss_func(xb_out, yb.float()) # for MSE/regression
    elif loss_str in ['CrossEntropyLoss()']:
        loss = loss_func(xb_out, yb.long().squeeze(1))
        # ^^ changes for CrossEntropyLoss...
    elif loss_str in ['BCEWithLogitsLoss()']:
        #print("shape yb into loss:",yb.float().squeeze(1).shape)
        loss = loss_func(xb_out, yb.float().squeeze(1))
            # ^^ changes for BCEWithLogitsLoss...?
    else:
        print(f"Warning: I don't know if loss function {loss_str} needs special formatting. Using MSE format for now!")
        loss = loss_func(xb_out, yb.float()) # for MSE/regression
        
    if verbose:
        print("loss",loss)

    if opt is not None: # if opt
        #print('opt:',opt)
        loss.backward()
        opt.step()
        opt.zero_grad()

    #print("lb returning:",loss.item(), len(xb))
    return loss.item(), len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # Set model to Training mode
    model.train()
    tl = [] # train losses
    ns = [] # batch sizes, n
    # collect train loss; provide opt so backpropo happens
    for xb, yb in train_dl:
        # put on GPU
        xb, yb = xb.to(device),yb.to(device)

        t, n = loss_batch(model, loss_func, xb, yb, opt=opt)
        tl.append(t)
        ns.append(n)
    
    # average the losses over all batches    
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)
    
    return train_loss

def val_step(model, val_dl, loss_func, device):
    '''
    Execute 1 set of batched validation within an epoch
    '''
    # Set model to Evaluation mode
    model.eval()
    with torch.no_grad():
        vl = [] # val losses
        ns = [] # batch sizes
        for xb, yb in val_dl:
            # put on GPU
            xb, yb = xb.to(device),yb.to(device)

            v, n = loss_batch(model, loss_func, xb, yb)
            vl.append(v)
            ns.append(n)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(vl, ns)) / np.sum(ns)
    
    return val_loss


# def fit(epochs, model, loss_func, opt, train_dl, test_dl, device):
#     '''
#     Fit the model params to the training data, eval on unseen data.
#     Loop for a number of epochs and keep train of train and test losses 
#     along the way
#     '''
#     # keep track of losses
#     train_losses = []    
#     test_losses = []
    
#     # loops through epochs
#     for epoch in range(epochs):
#         # train step
#         train_loss = train_step(model, train_dl, loss_func, device, opt)
#         train_losses.append(train_loss)
        
#         # test step
#         test_loss = test_step(model, test_dl, loss_func, device)
#         print(epoch, test_loss)
#         test_losses.append(test_loss)

#     return train_losses, test_losses

def fit(epochs, model, loss_func, opt, train_dl, val_dl,device,
        patience=1000,load_best=False,chkpt_path='checkpoint.pt'):
    '''
    Fit the model params to the training data, eval on unseen data.
    Loop for a number of epochs and keep train of train and val losses 
    along the way
    '''
    # keep track of losses
    train_losses = []    
    val_losses = []
    
    # create early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False,path=chkpt_path)
    
    # loops through epochs
    #for epoch in range(epochs): #tqdm?
    with tqdm.trange(epochs) as pbar:
        for i in pbar:
            train_loss = train_step(model, train_dl, loss_func, device,opt)
            train_losses.append(train_loss)


            val_loss = val_step(model, val_dl, loss_func, device)
            #print(epoch, val_loss)
            val_losses.append(val_loss)
            
            pbar.set_description(f"E:{i} | train loss:{train_loss:.3f} | val loss: {val_loss:.3f}")
            
            # copied from https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, model,i)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    # Epoch and value of best model checkpoint
    estop = early_stopping.best_model_epoch
    best_val_score = early_stopping.val_loss_min 

    print("fit checkpoint path:",chkpt_path)

    # load the last checkpoint with the best model
    if load_best:
        model.load_state_dict(torch.load(chkpt_path))
        # ^^ Does this need to be returned? I dont' think so... loads in place

    return train_losses, val_losses,estop,best_val_score


def run_model(train_dl,val_dl, model, loss_func, device,lr=0.01, 
              epochs=20, opt=None,patience=1000,load_best=False,
              chkpt_path='checkpoint.pt'):
    '''
    Given data and a model type, run dataloaders with MSE loss and SGD opt
    '''
    # define optimizer
    if opt:
        optimizer = opt
    else: # if no opt provided, just use SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    print("runmodel checkpoint path:",chkpt_path)
    
    # run the training loop
    #train_losses, test_losses = fit(epochs, model, loss_func, optimizer, train_dl, test_dl, device)
    train_losses, \
    val_losses,\
    epoch_stop,\
    best_val_score = fit(epochs, model, loss_func, optimizer, train_dl, val_dl,
                         device,patience=patience, load_best=load_best,chkpt_path=chkpt_path)

    #return model, train_losses, test_losses
    return train_losses, val_losses, epoch_stop, best_val_score


def collect_model_stats(model_name,seq_len,
                        train_dl,val_dl,device,
                        lr=0.001,ep=1000,pat=100,
                        model=None, loss_type='regression',
                        opt=None,opt_warm=False,
                        load_best=True,chkpt_path='checkpoint.pt'):
    '''
    Execute run of a model and return stats and objects related
    to its results
    '''
    # default model if none specified
    if not model:
        model = m.DNA_2CNN_2FC_Multi(
            seq_len,
            3, # num tasks
        )
    model.to(device)

    if loss_type=='regression':
        loss_func = torch.nn.MSELoss() 
        loss_str = "MSE Loss"
    elif loss_type=='classification':
        loss_func = torch.nn.CrossEntropyLoss()
        loss_str = "Cross Entropy Loss"
    else:
        raise ValueError(f"Unimplmented loss for type {loss_type}")
    
    if opt:
        opt = opt(model.parameters(), lr=lr)
        if opt_warm:
            opt.load_state_dict(torch.load(opt_warm))
           
            

    # collect run time
    start_time = time.time()
    print("collect checkpoint path:",chkpt_path)
    
    train_losses, \
    val_losses, \
    epoch_stop, \
    best_val_score = run_model(
        train_dl,
        val_dl, 
        model, 
        loss_func, 
        device,
        lr=lr, 
        epochs=ep, 
        opt=opt,
        patience=pat,
        load_best=load_best,
        chkpt_path=chkpt_path
    )
    total_time = time.time() - start_time

    # to plot loss
    data_label = [((train_losses,val_losses),model_name,epoch_stop,best_val_score)]
    quick_loss_plot(data_label, loss_str = loss_str)
    
    return {
        'model_name':model_name,
        'model':model,
        'opt':opt,
        'train_losses':train_losses,
        'val_losses':val_losses,
        'epoch_stop':epoch_stop,
        'best_val_score':best_val_score,
        'data_label':data_label,
        'total_time':total_time
    }



### DO THIS
def make_weighted_sampler(df, reg):
    '''
    Given a training dataframe, create a balanced sampler for the class
    indicated
    '''
    # make weighted sampler for data loader
    class_sample_count = df[reg].value_counts()
    # get 1/count as weight for each class
    weight = dict([(x,(1. / class_sample_count[x])) for x in class_sample_count.keys()])
    # apply new weight to each sample
    samples_weight = np.array([weight[t] for t in df[reg].values])
    samples_weight = torch.from_numpy(samples_weight).double()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler



# +--------------------------------+
# | Prediction checking/validation |
# +--------------------------------+

def quick_seq_pred(model, seqs, oracle):
    '''
    Given a model and some sequences, get the model's predictions
    for those sequences and compare to the oracle (true) output
    '''
    for dna in seqs:
        s = torch.tensor(u.one_hot_encode(dna)).unsqueeze(0)
        pred = model(s.float())
        actual = oracle[dna]
        diff = actual - pred.item()
        print(f"{dna}: pred:{pred.item():.3f} actual:{actual:.3f} ({diff:.3f})")

# trying to replace with function below
# def parity_plot(model_name,pred_df):
#     '''
#     Given a dataframe of samples with their true and predicted values,
#     make a scatterplot.
#     '''
#     plt.scatter(pred_df['truth'].values, pred_df['pred'].values, alpha=0.2)
    
#     # y=x line
#     xpoints = ypoints = plt.xlim()
#     plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)

#     plt.ylim(xpoints)
#     plt.ylabel("Predicted Score",fontsize=14)
#     plt.xlabel("Actual Score",fontsize=14)
#     plt.title(f"{model_name} (pearson:{pearson:.3f})",fontsize=20)
#     plt.show()

def parity_plot(model_title,ytrue,ypred,rigid=True,title=None):
    plt.scatter(ytrue, ypred, alpha=0.2)
    
    r2 = r2_score(ytrue,ypred)
    
    if not title:
        title = model_title
    
    # y=x line
    xpoints = ypoints = plt.xlim()
    if rigid:
        plt.ylim(min(xpoints),max(xpoints)) 
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)

    plt.xlabel("Actual Score",fontsize=14)
    plt.ylabel("Predicted Score",fontsize=14)
    plt.title(f"{title} (r2:{r2:.3f})",fontsize=20)
    plt.show()
    
    
def alt_parity_plot(model_name,pred_df,task):
    '''
    Make an interactive parity plot with altair
    '''
    ytrue = pred_df['truth'].values
    ypred = pred_df['pred'].values
    r2 = r2_score(ytrue,ypred)
    
    chart = alt.Chart(pred_df).mark_circle(opacity=0.2).encode(
        alt.X('pred:Q'),
        alt.Y('truth:Q'),
        tooltip=['seq:N']
    ).properties(
        title=f'{task} {model_name} (r2:{r2})'
    ).interactive()
    
    chart.save(f'alt_out/parity_plot_{task}_{model_name}.html')
    

def parity_pred_seqs(models, seqs, oracle,task,alt=True):
    '''
    Given some short sequences (e.g. a synthetic practice
    task, get the model's predictions and plot parity
    '''
    dfs = {} # key: model name, value: parity_df
    
    # TODO: fix this so it doesn't loop through each seq?
    for model_name,model in models:
        print(f"Running {model_name}")
        data = []
        for dna in seqs:
            s = torch.tensor(u.one_hot_encode(dna))#.unsqueeze(0)
            actual = oracle[dna]
            pred = model(s.float())
            data.append([dna,actual,pred.item()])
        pred_df = pd.DataFrame(data, columns=['seq','truth','pred'])
        #pearson = df['truth'].corr(df['pred'])
        dfs[model_name] = (pred_df)
        
        #plot parity plot
        if alt: # make an altair plot
            alt_parity_plot(model_name, pred_df,task)
        parity_plot(model_name, pred_df['truth'],pred_df['pred'])

    return dfs

def parity_pred_loci(models,df,device,locus_col='locus_tag',seq_col='seq',target_col="score",alt=True):
    '''
    Given some X examples of gene loci and y labels, get the model's predictions
    for X and plot vs the actual y
    '''
    loci = df[locus_col].values
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs]).to(device)
    labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
    
    dfs = {} # key: model name, value: parity_df
    
    for model_name,model in models:
        pred_df = df[[locus_col]]
        pred_df['truth'] = df[target_col]
        print(f"Running {model_name}")
        
        preds = model(ohe_seqs.float()).tolist()
        pred_df['pred'] = [x[0] for x in preds]
        
        dfs[model_name] = pred_df
        
        # plot stuff
        ytrue = pred_df['truth'].values
        ypred = pred_df['pred'].values
        
        if alt: # make an altair plot
            alt_parity_plot(model_name, pred_df,target_col)
        parity_plot(model_name, ytrue,ypred)
        
    return dfs



# def quick_loss_plot(data_label_list,loss_str="MSE Loss",title="Train/test Loss",sparse_n=0,figsize=(10,5),save_file=None):
#     '''
#     For each train/test loss trajectory, plot loss by epoch
#     '''
#     plt.figure(figsize=figsize)
#     for i,((train_data,test_data),label,epoch_stop,best_val) in enumerate(data_label_list):
#         # plot only 1 in every sparse_n points
#         if sparse_n:
#             train_data = [x for i,x in enumerate(train_data) if (i%sparse_n==0)]
#             test_data = [x for i,x in enumerate(test_data) if (i%sparse_n==0)]
#             epoch_stop = epoch_stop/sparse_n
            
#         plt.plot(train_data,linestyle='dashed',color=f"C{i}",linewidth=2.0, label=f"{label} Train")
#         plt.plot(test_data,color=f"C{i}", label=f"{label} Test",linewidth=3.5)
#         plt.axvline(x=epoch_stop,c=f"C{i}",linewidth=0.75,linestyle='dashed',label=f"{label} best test score")
#         plt.axhline(y=best_val,c=f"C{i}",linewidth=0.75,linestyle='dashed')

#     plt.legend()
#     plt.ylabel(loss_str)
#     plt.xlabel("Epoch")
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1,1),loc='upper left')
#     plt.show()

#     if save_file:
#         plt.savefig(save_file,bbox_inches='tight')

def quick_loss_plot(data_label_list,loss_str="MSE Loss",title="Train/test Loss",sparse_n=0,figsize=(10,5),save_file=None):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    colors =  matplotlib.cm.get_cmap("tab20").colors
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i,((train_data,test_data),label,epoch_stop,best_val) in enumerate(data_label_list):
        # plot only 1 in every sparse_n points
        if sparse_n:
            train_data = [(i,x) for i,x in enumerate(train_data) if (i%sparse_n==0)]
            test_data = [(i,x) for i,x in enumerate(test_data) if (i%sparse_n==0)]
            
            plt.plot(*zip(*train_data),linestyle='dashed',color=colors[2*i+1],linewidth=2.0, label=f"{label} Train")
            plt.plot(*zip(*test_data),color=colors[2*i], label=f"{label} Test",linewidth=3.5)
            
        else:
            plt.plot(train_data,linestyle='dashed',color=colors[2*i+1],linewidth=2.0, label=f"{label} Train")
            plt.plot(test_data,color=colors[2*i], label=f"{label} Test",linewidth=3.5)
        
        plt.axvline(x=epoch_stop,c=colors[2*i],linewidth=0.75,linestyle='dashed',label=f"{label} best test score")
        plt.axhline(y=best_val,c=colors[2*i],linewidth=0.75,linestyle='dashed')

    plt.legend()
    plt.ylabel(loss_str)
    plt.xlabel("Epoch")
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.show()

    if save_file:
        plt.savefig(save_file,bbox_inches='tight')

def get_confusion_data(model, model_name, ds, genes, oracle,loc2seq,device):
    '''
    Given a trained model and set of genes, evaluate the model's
    ability to predict these genes' reg class
    '''
    model.eval()
    data = []
    for gene in genes:
        dna = loc2seq[gene]
        if ds.name == 'ohe':
            s = torch.tensor(u.one_hot_encode(dna)).unsqueeze(0).to(device)
        elif ds.name == 'kmer':
            #s = torch.tensor(u.count_kmers_in_seq(dna,u.kmers(ds.k))).to(device)
            s = torch.tensor(u.count_kmers_in_seq(dna,u.kmers(ds.k))).unsqueeze(0).to(device)
            # need unsqueeze?
        else:
            raise ValueError(f"Unknown DataSetSpec Type {ds.name}. Currently just [ohe, kmer]")

        actual = oracle[gene]
        preds = [x.topk(1) for x in model(s.float())]
        
        for i in range(len(preds)):
            prob,clss = [x.item() for x in preds[i]]
            data.append((gene,actual[i], clss,prob,dna))
            
    df = pd.DataFrame(data, columns=['locus_tag','truth','pred','prob','seq'])
    return df


def cls_report(df,labels=[0,1,2]):
    '''
    Basic print out of precicion/recall/f1 scores for a classification problem
    '''
    
    acc = accuracy_score(df['truth'].values,df['pred'].values)
    mcc = matthews_corrcoef(df['truth'].values,df['pred'].values)
    
    # micro
    mi_p = precision_score(df['truth'].values,df['pred'].values,labels=labels,average='micro')
    mi_r = recall_score(df['truth'].values,df['pred'].values,labels=labels,average='micro')
    mi_f1 = f1_score(df['truth'].values,df['pred'].values,labels=labels,average='micro')
    
    # macro
    ma_p = precision_score(df['truth'].values,df['pred'].values,labels=labels,average='macro')
    ma_r = recall_score(df['truth'].values,df['pred'].values,labels=labels,average='macro')
    ma_f1 = f1_score(df['truth'].values,df['pred'].values,labels=labels,average='macro')

    report = {
        'acc':acc,'mcc':mcc,
        'mi_p':mi_p,'mi_r':mi_r,'mi_f1':mi_f1,
        'ma_p':ma_p,'ma_r':ma_r,'ma_f1':ma_f1,
    }
    return report

def get_confusion_stats(model,model_name,seq_list,device,save_file=False,title=None):#seqs,labels,seq_name):
    '''Get class predictions and plot confusion matrix'''

    def plot_confusion_raw_norm(mats):
        f, axes = plt.subplots(len(seq_list), 2, figsize=(9.8, 4.2*len(seq_list)))#, sharey='row')
        #axes = list(axes)
        axes_list = [item for sublist in axes for item in sublist]

        for i,(mat,subtitle) in enumerate(mats):
            disp = ConfusionMatrixDisplay(confusion_matrix=mat)
            disp.plot(ax=axes_list.pop(0))
            #disp.plot(ax=axes.pop(0))
            disp.ax_.set_title(f"{subtitle}")
            disp.im_.colorbar.remove()

        title_str=title if title else model_name
        f.suptitle(f"{title_str}",fontsize=20)
        plt.tight_layout()
        if save_file:
            plt.savefig(save_file)

    model.eval()
    print(f"Running {model_name}")
    
    mats = [] # conf matrices
    res_data = [] # classification results

    for seqs, labels, split_name in seq_list:
        ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs]).to(device)
        preds = [x.topk(1).indices.item() for x in model(ohe_seqs.float())]#.tolist()        
        
        cls_rep = classification_report(labels, preds,output_dict=True)
        pr = cls_rep['macro avg']['precision']
        re = cls_rep['macro avg']['recall']
        f1 = cls_rep['macro avg']['f1-score']
        sp = cls_rep['macro avg']['support']
        res_data.append([model_name,split_name,pr,re,f1,sp])
        
        c = confusion_matrix(labels, preds)
        mats.append((c,f"raw counts ({split_name})"))
        # get the normalized confusino matrix
        cp = np.zeros(c.shape)
        for i,row in enumerate(c):
            rowsum = sum(row)
            for j,item in enumerate(row):
                val = item/rowsum
                cp[i][j] = val

        mats.append((cp,f"normed counts ({split_name})"))

    plot_confusion_raw_norm(mats)
    
    res_df = pd.DataFrame(res_data,columns=['model_name','split','mac_precision','mac_recall','mac_f1','support'])
    
    return res_df

