import os,sys,time
from math import pow,cos

print("TRAIN LIGHT-MODEL SIREN")

tstart = time.time()

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import wandb
import geomloss

import flashmatchnet
from flashmatchnet.data.reader import make_dataloader
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor
from flashmatchnet.utils.trackingmetrics import validation_calculations
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
from flashmatchnet.losses.loss_poisson_emd import PoissonNLLwithEMDLoss
from flashmatchnet.model.lightmodel_siren import LightModelSiren
from flashmatchnet.data.augment import scale_small_charge, mixup

print("modules loaded: ", time.time()-tstart," sec")


USE_WANDB=True
TRAIN_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_training/'
VALID_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_validation/'
NUM_EPOCHS=None
WORKERS_COUNT=4
BATCHSIZE=32
NPMTS=32
SHUFFLE_ROWS=True
FREEZE_BATCH=False # True, for small batch testing
FREEZE_LY_PARAM=True
mag_loss_on_sum=False
USE_COS_INPUT_EMBEDDING_VECTORS=False
VERBOSITY=1
NVALID_ITERS=10
CHECKPOINT_NITERS=10000
checkpoint_folder = "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/checkpoints/siren/curious-universe-230/"
LOAD_FROM_CHECKPOINT=True
checkpoint_file=checkpoint_folder+"/lightmodel_mlp_enditer_312500.pth"
start_iteration = 312501

# below number of examples are estimates. need to write something to get actual amount
if not FREEZE_BATCH:
    num_train_examples = 400e3
    num_valid_examples = 100e3
    # number of validation entries is: 128,832
    print("Number Training Examples: ",num_train_examples)
    print("Number Validation Examples: ",num_valid_examples)
    train_iters_per_epoch = int(num_train_examples/float(BATCHSIZE))
else:
    num_train_examples = BATCHSIZE
    num_valid_examples = BATCHSIZE
    train_iters_per_epoch = 1

num_iterations = train_iters_per_epoch*25
#num_iterations = 10000
iterations_per_validation_step = 100
if FREEZE_BATCH:
    iterations_per_validation_step = 1

# Learning rate config params
learning_rate_warmup_lr = 1.0e-5
learning_rate_warmup_nepochs = 20
learning_rate_warmup_niters = int(learning_rate_warmup_nepochs*train_iters_per_epoch)
learning_rate_max = 1.0e-3
learning_rate_min = 1.0e-6
learning_rate_ly_max = 1.0e-3
learning_rate_ly_min = 1.0e-7
learning_rate_cosine_nepoch = 50
learning_rate_cosine_niters = int(learning_rate_cosine_nepoch*train_iters_per_epoch)
end_iteration = start_iteration + num_iterations
starting_iepoch = 0
w0_initial = 30.0

#torch.autograd.set_detect_anomaly(True)

if USE_WANDB:
    print("LOGIN TO WANDB")
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# we create an instance of the mlp in order to access its embedding functions
mlp = FlashMatchMLP(input_nfeatures=112,
                    hidden_layer_nfeatures=[512,512,512,512,512]).to(device)

# we create a siren network
net = LightModelSiren(
    #dim_in = 112,                    # input dimension, ex. 2d coor
    dim_in = 7,                       # input dimension, ex. 2d coor (x,y,z,dx,dy,dz,dist)
    dim_hidden = 512,                 # hidden dimension
    dim_out = 1,                      # output dimension, ex. rgb value
    num_layers = 5,                   # number of layers
    final_activation = nn.Identity(), # activation of final layer (nn.Identity() for direct output)
    w0_initial = w0_initial           # different signals may require different omega_0 in the first layer - this is a hyperparameter
).to(device)
net.train()

loss_fn_train = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                      mag_loss_on_sum=False,
                                      full_poisson_calc=False).to(device)
loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                      mag_loss_on_sum=False,
                                      full_poisson_calc=True).to(device)
loss_tot = None


param_group_main = []
param_group_ly   = []

for name,param in net.named_parameters():
    if name=="light_yield":
        if FREEZE_LY_PARAM:
            param.requires_grad = False
        param_group_ly.append(param)
    else:
        param_group_main.append(param)
param_group_list = [{"params":param_group_ly,"lr":learning_rate_warmup_lr*0.1,"weight_decay":1.0e-5},
                    {"params":param_group_main,"lr":learning_rate_warmup_lr}]
optimizer = torch.optim.AdamW( param_group_list )

#lr_scheduler_warmup = ConstantLR( optimizer, factor=learning_rate_warmup_lr, total_iters=learning_rate_warmup_niters )
#lr_scheduler_start  = ConstantLR( optimizer, factor=learning_rate_max, total_iters=1 )
# warmup: constant LR
#warmup_lambda0_ly = lambda epoch: learning_rate_warmup_lr*0.1
#warmup_lambda1 = lambda epoch: learning_rate_warmup_lr
#lr_scheduler_warmup = LambdaLR( optimizer, lr_lambda=[warmup_lambda0_ly,warmup_lambda1] )
## start: one step to set LR max for cosine annealing
#start_lambda0_ly = lambda epoch: learning_rate_ly_max
#start_lambda1    = lambda epoch: learning_rate_max
#lr_scheduler_start  = LambdaLR( optimizer, lr_lambda=[start_lambda0_ly,start_lambda1] )
# cosine annealing
#lr_scheduler_cosine = CosineAnnealingLR( optimizer, learning_rate_cosine_niters, learning_rate_min, last_epoch=learning_rate_warmup_niters+1 )
#scheduler_v  = [lr_scheduler_warmup, lr_scheduler_start, lr_scheduler_cosine ]
#milestones_v = [learning_rate_warmup_niters,learning_rate_warmup_niters+1]
#print("Setup LR schedulers")

def get_learning_rate( epoch, warmup_epoch, cosine_epoch_period, warmup_lr, cosine_max_lr, cosine_min_lr ):
    if epoch < warmup_epoch:
        return warmup_lr
    elif epoch>=warmup_epoch and epoch-warmup_epoch<cosine_epoch_period:
        lr = cosine_min_lr + 0.5*(cosine_max_lr-cosine_min_lr)*(1+cos( (epoch-warmup_epoch)/float(cosine_epoch_period)*3.14159 ) )
        return lr
    else:
        return cosine_min_lr


if LOAD_FROM_CHECKPOINT:
    print("LOADING MODEL/OPTIMIZER/LOSS STATE FROM CHECKPOINT")
    print("Loading from: ",checkpoint_file)
    checkpoint_data = torch.load( checkpoint_file )
    net.load_state_dict( checkpoint_data['model_state_dict'] )
    optimizer.load_state_dict( checkpoint_data['optimizer_state_dict'] )
    loss_tot = checkpoint_data['loss']
else:
    print("TRAINING FROM INITIALIZATION")
    #net.init_custom()
    pass

if USE_WANDB:
    run = wandb.init(
        # Set the project where this run will be logged
        project="lightmodel-tmw-test",
        # Track hyperparameters and run metadata
        config={
            "learning_rate_max":learning_rate_max,
            "learning_rate_warmup":learning_rate_warmup_lr,
            "learning_rate_warmup_nepochs":learning_rate_warmup_nepochs,
            "learning_rate_cosine_max":learning_rate_max,
            "learning_rate_cosine_max_ly":learning_rate_ly_max,
            "learning_rate_cosine_min":learning_rate_min,
            "learning_rate_cosine_nepoch":learning_rate_cosine_nepoch,
            "mag_loss_on_sum":mag_loss_on_sum,
            "start_iteration":start_iteration,
            "batchsize":BATCHSIZE,
            "nvalid_iters":NVALID_ITERS,
            "end_iteration":end_iteration,
        })
    wandb.watch(net, log="all", log_freq=1000) 
    

####### 
# load SA tensor in here
# put on device too
###############
#print("Loading SA Table")
#sa_coords, sa_values = load_satable_fromnpz()

train_dataloader = make_dataloader( TRAIN_DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, BATCHSIZE,
                                    workers_count=WORKERS_COUNT )
valid_dataloader = make_dataloader( VALID_DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, BATCHSIZE,
                                    workers_count=WORKERS_COUNT )


train_iter = iter(train_dataloader)
valid_iter = iter(valid_dataloader)

# put net in training mode (vs. validation)
print(net)
epoch = 0.0
last_iepoch = -1

for iteration in range(start_iteration,end_iteration): 

    net.train()
    
    tstart_dataprep = time.time()

    if FREEZE_BATCH:
        if iteration==0:
            row = next(train_iter)
        else:
            pass # intentionally not changing data
    else:
        row = next(train_iter)

    if FREEZE_BATCH:
        epoch += 1.0
    else:
        epoch = float(iteration)/float(train_iters_per_epoch)

    # scale small charge clusters
    row = scale_small_charge( row )
    # mixup
    row = mixup( row, device, factor_range=[0.5,1.5] )
    # make batched sparse tensor
    #row['flashpe'] = torch.from_numpy(row['flashpe']).to(device)
    #coords, feats = me.utils.sparse_collate( coords=collated["coord"], feats=collated["feat"] )    
        
    coord = row['coord'].to(device)
    q_feat = row['feat'][:,:3].to(device)
    entries_per_batch = row['batchentries']
    start_per_batch = torch.from_numpy(row['batchstart']).to(device)
    end_per_batch   = torch.from_numpy(row['batchend']).to(device)

    # for each coord, we produce the other features
    with torch.no_grad():
        if USE_COS_INPUT_EMBEDDING_VECTORS:
            vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, mlp )
        else:
            vox_feat, q = prepare_mlp_input_variables( coord, q_feat, mlp )

    dt_dataprep = time.time()-tstart_dataprep

    if VERBOSITY>=2:
        print("[ITERATION ",iteration,"] ======================")
        print("  coord.shape=",coord.shape)
        print("  pe[target].shape=",row['flashpe'].shape)
        print("  entries per batch: ",entries_per_batch)
        print("dt_dataprep=",dt_dataprep," seconds")

        #print("VOX FEATURES ====================== ")
        #print(vox_feat)
        #print("=================================== ")

    # set learning rate for lightyield
    iepoch = max( int(epoch)-starting_iepoch, 0 )
    #if iepoch>last_iepoch and not FREEZE_BATCH:
    #    lr_updated    = learning_rate*pow(0.5,iepoch)
    #    lr_updated_LY = learning_rate_ly*pow(0.5,iepoch)
    #    print("new epoch @ iepoch=",iepoch," (with iepoch offset=",starting_iepoch,": set learning rate to: ",lr_updated)
    #    print("new epoch @ iepoch=",iepoch,": set learning rate (LY) to: ",lr_updated_LY)
    #    last_iepoch = iepoch
    #    optimizer.param_groups[0]["lr"] = lr_updated_LY
    #    optimizer.param_groups[1]["lr"] = lr_updated
    optimizer.zero_grad()

    # we run the MLP on every voxel,pmt pair
    # do this by looping over batch and pmt
    # start with dumb loop, then figure out how to make fast with proper syntax
    tstart_forward = time.time()
    # unroll everything to ( (NB)*PMTS, nfeatures)
    N,C,K = vox_feat.shape
    #print(N," ",C," ",K)
    vox_feat = vox_feat.reshape( (N*C,K) )
    q = q.reshape( (N*C,1) )

    if VERBOSITY>=2:
        with torch.no_grad():
            print("INPUTS ==================")
            print("vox_feat.shape=",vox_feat.shape," from (N,C,K)=",(N,C,K))
            print("q.shape=",q.shape)

    pe_per_voxel = net(vox_feat, q)
    pe_per_voxel = pe_per_voxel.reshape( (N,C) )

    if VERBOSITY>=2:
        print("PE_PER_VOXEL ================")
        print("pe_per_voxel.shape=",pe_per_voxel.shape)
        print("pe_per_voxel stats: ")
        with torch.no_grad():
            print(" mean: ",pe_per_voxel.mean())
            print(" var: ",pe_per_voxel.var())
            print(" max: ",pe_per_voxel.max())
            print(" min: ",pe_per_voxel.min())
            print(pe_per_voxel[rand_entries[:],:2])


    # loss
    #pe_per_pmt_target = torch.from_numpy(row['flashpe']).to(device)
    pe_per_pmt_target = row['flashpe']
    loss_tot,(floss_tot,floss_emd,floss_mag,pred_pesum,pred_pemax) = loss_fn_train( pe_per_voxel,
                                                                                    pe_per_pmt_target,
                                                                                    start_per_batch,
                                                                                    end_per_batch )
                
    dt_forward = time.time()-tstart_forward
    #print("dt_forward: ",dt_forward," secs")
    
    # ----------------------------------------
    # Backprop
    dt_backward = time.time()
    
    loss_tot.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    # get the lr we just used
    iter_lr    = optimizer.param_groups[1]["lr"]
    iter_lr_ly = optimizer.param_groups[0]["lr"]

    # update the next learning rate
    #nstep = 0
    #for im in range(len(milestones_v)):
    #    if iteration<milestones_v[im]:
    #        if im==0 or (im>0 and iteration>=milestones_v[im-1]):
    #            # step the scheduler
    #            print("stepping with LR scheduler index=",im," @iteration=",iteration)
    #            scheduler_v[im].step()
    #            nstep += 1
    #            break
    #print("number of scheduler steps=",nstep)
    next_lr = get_learning_rate( epoch, learning_rate_warmup_nepochs, learning_rate_cosine_nepoch,
                                 learning_rate_warmup_lr, learning_rate_max, learning_rate_min )
    optimizer.param_groups[1]["lr"] = next_lr
    optimizer.param_groups[0]["lr"] = next_lr*0.01 # LY learning rate
    
    dt_backward = time.time()-dt_backward
    #print("dt_backward: ",dt_backward," sec")
    # ----------------------------------------    
    

    if iteration%CHECKPOINT_NITERS==0 and iteration>start_iteration:
        print("save checkpoint @ iteration = ",iteration)
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_tot}, 
	           checkpoint_folder+'/lightmodel_mlp_iter_%d.pth'%(int(iteration)))
            
    if iteration%iterations_per_validation_step==0:

        with torch.no_grad():
            print("=========================")
            print("[Validation at iter=",iteration,"]")            
            print("Epoch: ",epoch)
            print("LY=",net.get_light_yield().cpu().item())
            print("(next) learning_rate=",next_lr," :  lr_LY=",next_lr*0.01)
            print("=========================")
        
        with torch.no_grad():
            pe_target_sum, pe_target_sum_idx = pe_per_pmt_target.max(1)
            #bmax = torch.argmax( pe_target_sum )
            #print("pe_target[bmax]: ",pe_per_pmt_target[bmax])
            #print("pe_batch[bmax]: ",pe_batch[bmax])
            #print("pdf_target[bmax]: ",pdf_target[bmax])
            #print("pdf_batch[bmax]: ",pdf_batch[bmax])
            #print("pe sum (target)=",pe_target_sum[bmax])
            #print("pe sum (predict)=",pe_sum[bmax])
            #print(" ------------------ ")
            print("pe sum target: [prev train iter]",pe_target_sum)
            print("pe sum predict [prev train iter]: ",pred_pesum)
            pred_pesum = pred_pesum.to(device)
            frac_diff = (pe_target_sum-pred_pesum)/pe_target_sum
            print("fractional diff (true-pred)/true: ",frac_diff)
            print("ave err^2: ",(frac_diff*frac_diff).mean())
        #input()
        
        with torch.no_grad():
            print("Run validation calculations on valid data") 
            valid_info_dict = validation_calculations( valid_iter, net, loss_fn_valid, BATCHSIZE, device, NVALID_ITERS, mlp=mlp,
                                                       use_embed_inputs=USE_COS_INPUT_EMBEDDING_VECTORS )
            print("  [valid total loss]: ",valid_info_dict["loss_tot_ave"])
            print("  [valid emd loss]: ",valid_info_dict["loss_emd_ave"])
            print("  [valid mag loss]: ",valid_info_dict["loss_mag_ave"])

            print("Run validation calculations on train data")
            train_info_dict = validation_calculations( train_iter, net, loss_fn_valid, BATCHSIZE, device, NVALID_ITERS,
                                                       mlp=mlp,
                                                       use_embed_inputs=USE_COS_INPUT_EMBEDDING_VECTORS)
            print("  [train total loss]: ",train_info_dict["loss_tot_ave"])
            print("  [train emd loss]: ",train_info_dict["loss_emd_ave"])
            print("  [train mag loss]: ",train_info_dict["loss_mag_ave"])            

            if USE_WANDB:
                #tabledata = valid_info_dict["table_data"]
                for_wandb = {"loss_tot_ave_train":train_info_dict["loss_tot_ave"],
                             "loss_emd_ave_train":train_info_dict["loss_emd_ave"],
                             "loss_mag_ave_train":train_info_dict["loss_mag_ave"],
                             "loss_tot_ave_valid":valid_info_dict["loss_tot_ave"],
                             "loss_emd_ave_valid":valid_info_dict["loss_emd_ave"],
                             "loss_mag_ave_valid":valid_info_dict["loss_mag_ave"],
                             #"pesum_v_x_pred":wandb.plot.scatter(tabledata, "x", "pe_sum_pred",
                             #                                    title="PE sum vs. x (cm)"),
                             #"pesum_v_q_pred":wandb.plot.scatter(tabledata, "qmean","pe_sum_pred",
                             #                                    title="PE sum vs. q sum"),
                             "lr":iter_lr,
                             "lr_lightyield":iter_lr_ly,
                             "lightyield":net.get_light_yield().cpu().item(),
                             "epoch":epoch}
            
                wandb.log(for_wandb, step=iteration)

print("FINISHED ITERATION LOOP!")

torch.save({'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tot}, 
	   checkpoint_folder+'/lightmodel_mlp_enditer_%d.pth'%(int(end_iteration)))
    
