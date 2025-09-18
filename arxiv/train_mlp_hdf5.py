import os,sys,time
from math import pow

print("TRAIN LIGHT-MODEL MLP with HDF5 data")

tstart = time.time()

import numpy as np
import torch
import torch.nn as nn
import wandb
import geomloss

import flashmatchnet
from flashmatch_hdf5_reader import create_flashmatch_dataloader, FlashMatchTransform
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor
from flashmatchnet.utils.trackingmetrics import validation_calculations
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings
from flashmatchnet.losses.loss_poisson_emd import PoissonNLLwithEMDLoss

print("modules loaded: ", time.time()-tstart," sec")

# Configuration
USE_WANDB=True
TRAIN_HDF5_FILES=['path/to/train_data.h5']  # Update with actual paths
VALID_HDF5_FILES=['path/to/valid_data.h5']  # Update with actual paths
NUM_EPOCHS=None
WORKERS_COUNT=4
BATCHSIZE=32
NPMTS=32
SHUFFLE_ROWS=True
FREEZE_BATCH=False # True, for small batch testing
mag_loss_on_sum=False
VERBOSITY=0
NVALID_ITERS=10
CHECKPOINT_NITERS=1000
checkpoint_folder = "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/checkpoints/"
LOAD_FROM_CHECKPOINT=True
checkpoint_file=checkpoint_folder+"/lightmodel_mlp_enditer_12500.pth"

start_iteration = 12501
num_iterations = 125000
iterations_per_validation_step = 100
learning_rate = 1.0e-5
learning_rate_ly = 1.0e-7
end_iteration = start_iteration + num_iterations
starting_iepoch = 1

def hdf5_to_petastorm_format(batch):
    """
    Convert HDF5 batch format to match the expected format from Petastorm reader.
    """
    # The HDF5 reader returns different format, so we need to convert
    # Original Petastorm format expected:
    # - coord: MinkowskiEngine sparse tensor coordinates
    # - feat: Features with shape matching coord
    # - flashpe: Flash PE data
    # - batchentries: Number of entries per batch item
    # - batchstart/batchend: Start/end indices for each batch item
    
    batchsize = len(batch['run'])
    
    # Convert to the format expected by the training loop
    result = {
        'coord': batch['coords'],  # Already concatenated coordinates
        'feat': batch['feats'],    # Already concatenated features
        'flashpe': batch['flashpe'].squeeze(1),  # Remove extra dimension if present
        'event': batch['event'],
        'matchindex': batch['matchindex'],
        'batchentries': [],
        'batchstart': [],
        'batchend': []
    }
    
    # Calculate batch start/end indices
    start_idx = 0
    unique_batch_indices = torch.unique(batch['batch_indices'])
    
    for batch_idx in unique_batch_indices:
        mask = (batch['batch_indices'] == batch_idx)
        n_entries = mask.sum().item()
        
        result['batchentries'].append(n_entries)
        result['batchstart'].append(start_idx)
        result['batchend'].append(start_idx + n_entries)
        start_idx += n_entries
    
    # Convert to numpy arrays as expected
    result['batchentries'] = np.array(result['batchentries'], dtype=np.int64)
    result['batchstart'] = np.array(result['batchstart'], dtype=np.int64)
    result['batchend'] = np.array(result['batchend'], dtype=np.int64)
    
    return result

def hdf5_transform_row(data):
    """
    Transform function to match the Petastorm preprocessing.
    """
    coord = data['coord']
    feat = data['feat']
    
    # TPC bounds filtering (same as Petastorm version)
    goodmask_i = coord[:,0] < 53
    goodmask_j = coord[:,1] < 49
    goodmask_k = coord[:,2] < 210
    goodmask = goodmask_i * goodmask_j * goodmask_k
    
    coord = coord[goodmask, :]
    feat = feat[goodmask, :]
    
    # Normalize charge features
    feat = feat / 10000.0
    
    # Normalize PE features
    pe = data['flashpe'] / 1000.0 + 1.0e-4
    
    # Update data
    data['coord'] = coord
    data['feat'] = feat
    data['flashpe'] = pe
    
    return data

if USE_WANDB:
    print("LOGIN TO WANDB")
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlashMatchMLP(input_nfeatures=112,
                    hidden_layer_nfeatures=[512,512,512,512,512]).to(device)

net.train()

loss_fn_train = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                      mag_loss_on_sum=False,
                                      full_poisson_calc=False).to(device)
loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                      mag_loss_on_sum=False,
                                      full_poisson_calc=True).to(device)
loss_tot = None

param_list = []
for name,param in net.named_parameters():
    if name=="light_yield":
        param_list.append( {'params':param,"lr":learning_rate_ly,"weight_decay":1.0e-5} )
    else:
        param_list.append( {'params':param,"lr":learning_rate} )        
optimizer = torch.optim.AdamW( param_list, lr=learning_rate)

if LOAD_FROM_CHECKPOINT:
    print("LOADING MODEL/OPTIMIZER/LOSS STATE FROM CHECKPOINT")
    print("Loading from: ",checkpoint_file)
    checkpoint_data = torch.load( checkpoint_file )
    net.load_state_dict( checkpoint_data['model_state_dict'] )
    optimizer.load_state_dict( checkpoint_data['optimizer_state_dict'] )
    loss_tot = checkpoint_data['loss']

if USE_WANDB:
    run = wandb.init(
        project="lightmodel-tmw-test-hdf5",
        config={
            "learning_rate": learning_rate,
            "mag_loss_on_sum":mag_loss_on_sum,
            "start_iteration":start_iteration,
            "batchsize":BATCHSIZE,
            "nvalid_iters":NVALID_ITERS,
            "end_iteration":end_iteration,
            "data_format": "hdf5"
        })
    wandb.watch(net, log="all", log_freq=1000) 

# Create data transforms
transform = FlashMatchTransform(voxel_size=5.0, augment=False)
transform._old_call = transform.__call__
transform.__call__ = lambda data: hdf5_transform_row(transform._old_call(data))

# Create data loaders
print("Creating HDF5 data loaders...")
train_dataloader = create_flashmatch_dataloader(
    TRAIN_HDF5_FILES,
    batch_size=BATCHSIZE,
    shuffle=SHUFFLE_ROWS,
    num_workers=WORKERS_COUNT,
    use_sparse=False,
    transform=transform
)

valid_dataloader = create_flashmatch_dataloader(
    VALID_HDF5_FILES,
    batch_size=BATCHSIZE,
    shuffle=SHUFFLE_ROWS,
    num_workers=WORKERS_COUNT,
    use_sparse=False,
    transform=transform
)

num_train_examples = len(train_dataloader.dataset)
num_valid_examples = len(valid_dataloader.dataset)
print("Number Training Examples: ",num_train_examples)
print("Number Validation Examples: ",num_valid_examples)

train_iter = iter(train_dataloader)
valid_iter = iter(valid_dataloader)

print(net)
epoch = 0.0
last_iepoch = -1
lr_updated = learning_rate

for iteration in range(start_iteration,end_iteration): 

    net.train()
    
    tstart_dataprep = time.time()

    if FREEZE_BATCH:
        if iteration==start_iteration:
            raw_batch = next(train_iter)
            row = hdf5_to_petastorm_format(raw_batch)
        else:
            pass # intentionally not changing data
    else:
        try:
            raw_batch = next(train_iter)
        except StopIteration:
            # Restart iterator when we reach the end
            train_iter = iter(train_dataloader)
            raw_batch = next(train_iter)
        row = hdf5_to_petastorm_format(raw_batch)

    coord = row['coord'].to(device)
    q_feat = row['feat'][:,:3].to(device)
    entries_per_batch = row['batchentries']
    start_per_batch = torch.from_numpy(row['batchstart']).to(device)
    end_per_batch   = torch.from_numpy(row['batchend']).to(device)

    # for each coord, we produce the other features
    vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, net )

    dt_dataprep = time.time()-tstart_dataprep

    if VERBOSITY>=2:
        print("[ITERATION ",iteration,"] ======================")
        print("  coord.shape=",coord.shape)
        print("  pe[target].shape=",row['flashpe'].shape)
        print("  entries per batch: ",entries_per_batch)
        print("dt_dataprep=",dt_dataprep," seconds")

    # Training step
    tstart_fwd = time.time()
    optimizer.zero_grad()

    # make prediction
    pe_pred = net(vox_feat, start_per_batch, end_per_batch)
    pe_target = row['flashpe'].to(device)

    # losses
    loss_tot = loss_fn_train(pe_pred, pe_target)

    dt_fwd = time.time()-tstart_fwd

    # backward
    tstart_bwd = time.time()    
    loss_tot.backward()
    optimizer.step()
    dt_bwd = time.time()-tstart_bwd

    if iteration%10==0:
        print("[ITERATION ",iteration,"] loss: ",loss_tot.item(), 
              " | dt_dataprep=",dt_dataprep," dt_fwd=",dt_fwd," dt_bwd=",dt_bwd," secs")

    if USE_WANDB:
        wandb.log({"train_loss":loss_tot.item(),
                   "iteration":iteration,
                   "learning_rate":optimizer.param_groups[0]['lr'],
                   "dt_dataprep":dt_dataprep,
                   "dt_fwd":dt_fwd,
                   "dt_bwd":dt_bwd})

    # Validation
    if iteration%iterations_per_validation_step==0 and iteration>start_iteration:
        print("==========================")
        print("Run Validation @ iter ",iteration)

        net.eval()
        valid_loss_total = 0.0
        
        for vv in range(NVALID_ITERS):
            try:
                raw_valid_batch = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_dataloader)
                raw_valid_batch = next(valid_iter)
            
            valid_row = hdf5_to_petastorm_format(raw_valid_batch)
            
            with torch.no_grad():
                valid_coord = valid_row['coord'].to(device)
                valid_q_feat = valid_row['feat'][:,:3].to(device)
                valid_entries_per_batch = valid_row['batchentries']
                valid_start_per_batch = torch.from_numpy(valid_row['batchstart']).to(device)
                valid_end_per_batch = torch.from_numpy(valid_row['batchend']).to(device)

                valid_vox_feat, valid_q = prepare_mlp_input_embeddings(valid_coord, valid_q_feat, net)
                valid_pe_pred = net(valid_vox_feat, valid_start_per_batch, valid_end_per_batch)
                valid_pe_target = valid_row['flashpe'].to(device)
                valid_loss = loss_fn_valid(valid_pe_pred, valid_pe_target)
                valid_loss_total += valid_loss.item()

        valid_loss_ave = valid_loss_total/float(NVALID_ITERS)
        print("Validation Loss: ",valid_loss_ave)

        if USE_WANDB:
            wandb.log({"valid_loss":valid_loss_ave,
                       "iteration":iteration})

    # Save checkpoint
    if iteration%CHECKPOINT_NITERS==0 and iteration>start_iteration:
        print("saving checkpoint")
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tot,
        }
        outname = checkpoint_folder+"/lightmodel_mlp_hdf5_enditer_%d.pth"%(iteration)
        torch.save(checkpoint_data, outname)
        print("saved ",outname)

print("Training completed!")

if USE_WANDB:
    wandb.finish()