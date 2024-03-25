import os,sys,time

print("TRAIN LIGHT-MODEL MLP")

tstart = time.time()

import numpy as np
import torch
import torch.nn as nn
import wandb
import geomloss

import flashmatchnet
from flashmatchnet.data.reader import make_dataloader
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor
from flashmatchnet.utils.trackingmetrics import validation_calculations
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings
from flashmatchnet.losses.loss_poisson_emd import PoissonNLLwithEMDLoss


print("modules loaded: ", time.time()-tstart," sec")


USE_WANDB=True
TRAIN_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2'
VALID_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2_validation'
NUM_EPOCHS=None
WORKERS_COUNT=4
BATCHSIZE=32
NPMTS=32
SHUFFLE_ROWS=True
FREEZE_BATCH=False # True, for small batch testing
VERBOSITY=0
NVALID_ITERS=10
CHECKPOINT_NITERS=1000
checkpoint_folder = "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/checkpoints/"

start_iteration = 0
num_iterations = 10000
iterations_per_validation_step = 100
learning_rate = 1.0e-4
end_iteration = start_iteration + num_iterations

#torch.autograd.set_detect_anomaly(True)

if USE_WANDB:
    print("LOGIN TO WANDB")
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlashMatchMLP(input_nfeatures=112,
                    hidden_layer_nfeatures=[512,512,512,512,512]).to(device)
#print("network par list: ")
#for x in net.parameters():#
#    print(x.data.dtype)
# def custom_init(model):
#     for name, param in model.named_parameters():
#         #print(name)
#         if name=="output.weight":
#             print("pre-custom action param values: ",param)
#             param.data *= 0.001
#         elif name=="output.bias":
#             print("pre-custom action param values: ",param)
#             param.data.fill_(0.0)
#         elif name=="light_yield":
#             print("pre-custom action param values: ",param)
#             param.data.fill_(0.0)

# net.apply(custom_init)

net.train()

loss_fn_train = PoissonNLLwithEMDLoss(magloss_weight=1.0,full_poisson_calc=False).to(device)
loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,full_poisson_calc=True).to(device)

#loss_sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
#loss_mse      = nn.MSELoss(reduction='mean')
#loss_poisson  = nn.PoissonNLLLoss(log_input=False,reduction='mean')


# we make the x and y tensors
#print("========================")
#x_pred   = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
#y_target = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
#x_pred   = x_pred.repeat(BATCHSIZE,1).reshape( (BATCHSIZE,NPMTS,2) ).to(device)
#y_target = y_target.repeat(BATCHSIZE,1).reshape( (BATCHSIZE,NPMTS,2) ).to(device)
#print("x_pred.shape=",x_pred.shape)
#print("x_pred [these are the 2D positions of the PMTs] =================")
#print(x_pred)
#print("========================")

param_list = []
for name,param in net.named_parameters():
    if name=="light_yield":
        param_list.append( {'params':param,"lr":1.0e-3,"weight_decay":1.0e-5} )
    else:
        param_list.append( {'params':param} )
optimizer = torch.optim.AdamW( param_list, lr=learning_rate)

if USE_WANDB:
    run = wandb.init(
        # Set the project where this run will be logged
        project="lightmodel-tmw-test",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "start_iteration":start_iteration,
            "batchsize":BATCHSIZE,
            "nvalid_iters":NVALID_ITERS,
            "end_iteration":end_iteration,
        })
    wandb.watch(net, log="all", log_freq=1000, log_graph=True) 

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

num_train_examples = 400e3
num_valid_examples = 100e3
print("Number Training Examples: ",num_train_examples)
print("Number Validation Examples: ",num_valid_examples)

train_iter = iter(train_dataloader)
valid_iter = iter(valid_dataloader)

# put net in training mode (vs. validation)
print(net)
epoch = 0.0

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

        #print("VOX FEATURES ====================== ")
        #print(vox_feat)
        #print("=================================== ")

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
        print("INPUTS ==================")
        print("vox_feat.shape=",vox_feat.shape)
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
    pe_per_pmt_target = torch.from_numpy(row['flashpe']).to(device)
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
    
    dt_backward = time.time()-dt_backward
    #print("dt_backward: ",dt_backward," sec")
    # ----------------------------------------    

    if FREEZE_BATCH:
        epoch += 1.0
    else:
        epoch = float(iteration*BATCHSIZE)/float(num_train_examples)
    

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
            print("Epoch: ",epoch)
            print("Loss[ total ]: ", floss_tot)
            print("emdloss: ",floss_emd)
            print("sum mse loss: ",floss_mag)
            print("LY=",net.get_light_yield())        
            print("=========================")
        
        print("[ITERATION ",iteration,"]")
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
            print("pe sum target: ",pe_target_sum)
            print("pe sum predict: ",pred_pesum)
        #input()
        
        with torch.no_grad():
            print("Run validation calculations on valid data")            
            valid_info_dict = validation_calculations( valid_iter, net, loss_fn_valid, BATCHSIZE, device, NVALID_ITERS )
            print("Run validation calculations on train data")                        
            train_info_dict = validation_calculations( train_iter, net, loss_fn_valid, BATCHSIZE, device, NVALID_ITERS )

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
                         "epoch":epoch}
            
            wandb.log(for_wandb, step=iteration)

print("FINISHED ITERATION LOOP!")

torch.save({'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tot}, 
	   checkpoint_folder+'/lightmodel_mlp_enditer_%d.pth'%(int(end_iteration)))
    
