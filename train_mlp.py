import os,sys,time
import numpy as np
import torch
import torch.nn as nn

import wandb

import flashmatchnet
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP

from flashmatchdata import make_dataloader, get_rows_from_data_iterator
from flashmatchnet.utils.pmtutils import make_weights,get_2d_zy_pmtpos_tensor

import geomloss

USE_WANDB=False
TRAIN_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data'
NUM_EPOCHS=1
WORKERS_COUNT=4
BATCHSIZE=16
NPMTS=32
SHUFFLE_ROWS=False
FREEZE_BATCH=False
VERBOSITY=0

if USE_WANDB:
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlashMatchMLP(input_nfeatures=112,
                    hidden_layer_nfeatures=[512,512,512,512,512]).to(device)
#print("network par list: ")
#for x in net.parameters():
#    print(x.data.dtype)

def custom_init(model):
    for name, param in model.named_parameters():
        #print(name)
        if name=="output.weight":
            print("pre-custom action param values: ",param)
            param.data *= 0.001
        elif name=="output.bias":
            print("pre-custom action param values: ",param)
            param.data.fill_(0.0)
        elif name=="light_yield":
            print("pre-custom action param values: ",param)
            param.data.fill_(0.0)

net.apply(custom_init)
net.train()
#net.apply(custom_init) # hack to verify the value

loss_sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
loss_mse      = nn.MSELoss(reduction='mean')

# we make the x and y tensors
x_pred   = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
y_target = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)

x_pred   = x_pred.repeat(BATCHSIZE,1).reshape( (BATCHSIZE,NPMTS,2) ).to(device)
y_target = y_target.repeat(BATCHSIZE,1).reshape( (BATCHSIZE,NPMTS,2) ).to(device)
print("x_pred.shape=",x_pred.shape)
print("x_pred [these are the 2D positions of the PMTs] =================")
print(x_pred)
print("========================")

num_iterations = 1000

learning_rate = 1.0e-5
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

#EPOCH = 5
#PATH = "model.pt"
#LOSS = 0.4

if USE_WANDB:
    run = wandb.init(
        # Set the project where this run will be logged
        project="lightmodel-tmw-test",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs":1,
        })
    wandb.watch(net, log="all", log_freq=1)

####### 
# load SA tensor in here
# put on device too
###############
print("Loading SA Table")
#sa_coords, sa_values = load_satable_fromnpz()

train_dataloader = make_dataloader( TRAIN_DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, BATCHSIZE,
                                    workers_count=WORKERS_COUNT )

train_iter = iter(train_dataloader)

# put net in training mode (vs. validation)
net.train()
print(net)


for iteration in range(num_iterations): 

    tstart_dataprep = time.time()

    if FREEZE_BATCH:
        if iteration==0:
            row = next(train_iter)
        else:
            pass # intentionally not changing data
    else:
        row = next(train_iter)

    coord = row["coord"]
    entries_per_batch = row["batchentries"]
    start_per_batch = row["batchstart"]
    end_per_batch = start_per_batch+entries_per_batch
    
    # for each coord, we produce the other features
    print("[ITERATION ",iteration,"] ======================")
    print("  coord.shape=",coord.shape)
    print("  pe[target].shape=",row['flashpe'].shape)
    print("  entries per batch: ",entries_per_batch)

    # make det pos. don't need gradients for this.
    with torch.no_grad():

        nvoxels = coord.shape[0]
        npmt = 32

        if VERBOSITY>=2:
            print("coord tensor ===================")
            print(coord[:10])
            print("================================")
            # if our verbosity is so high,
            # we will track the values of random entries in order to
            # verify that all the manipulations are correct
            rand_entries = torch.randint(0,nvoxels,(5,),device=device,requires_grad=False)
        
        # convert tensor index to det positions: # (N,4) -> (N,3), we lose the batch index
        detpos_cm = net.index2pos( coord, dtype=torch.float32 )
        detpos_cm.requires_grad = False
        if VERBOSITY>=2:
            print("    detpos_cm.shape=",detpos_cm.shape," type=",detpos_cm.dtype)
            print("DETPOS_CM ================== ")
            print(detpos_cm[rand_entries[:],:])
            print("================================")

        print("nvoxels=",nvoxels," npmt=",npmt)
    
        # convert det positions into (sinusoidal) embedding vectors: # (N,3) -> (N,16x3)
        detpos_embed = net.get_detposition_embedding( detpos_cm, [16,16,16] )
        detpos_embed.requires_grad = False
        if VERBOSITY>=2:
            print("    detpos_embed.shape=",detpos_embed.shape)
            print("DETPOS_EMBED =============================")
            print(detpos_embed[rand_entries[:],:])
            
        detpos_embed_perpmt = torch.repeat_interleave( detpos_embed, npmt, dim=0).reshape( (nvoxels,npmt,48) ).to(torch.float32)
        # above is (N,32,48)
        if VERBOSITY>=2:
            print("DETPOS_EMBED_PERPMT (repeated)  =============================")
            print("detpos_embed_perpmt.shape=",detpos_embed_perpmt.shape)            
            print(detpos_embed_perpmt[rand_entries[:],:2,:]) # look at the first two pmts of the random entries
            

        # we calculate the distance to each pmt for each charge voxel position
        # we get two outputs, dist and the vector diff
        # this maps (N,3) -> (N,32,1) for distance
        # this maps (N,3) -> (N,32,3) for dvec2pmt
        dist2pmts_cm, dvec2pmts_cm = net.calc_dist_to_pmts( detpos_cm )
        if VERBOSITY>=2:
            print("DIST2PMTS_CM ========================")            
            print("dist2pmts_cm.shape=",dist2pmts_cm.shape," dtype=",dist2pmts_cm.dtype)
            print(dist2pmts_cm[rand_entries[:],:2,:])
            print("DVEC2PMT ==========================")
            print("dev2pmts_cm.shape=",dvec2pmts_cm.shape," dtype=",dvec2pmts_cm.dtype)            
            print(dvec2pmts_cm[rand_entries[:],:2,:])

        # we want an embedding vector for each distance we calculated
        # first unroll the distances: (32,N,1) -> (32*N,1)
        # then pass the unroll into the embedding function: (32*N,1) -> (32*N,16)
        # reshape this to include the PMT channel dim
        dist_embed_dims = 16
        dist_embed = net.get_detposition_embedding( dist2pmts_cm.reshape( (npmt*nvoxels,1) ),
                                                    [dist_embed_dims],
                                                    maxwavelength_per_dim=[net._dim_len[2]] )
        dist_embed = dist_embed.reshape( (nvoxels,npmt,dist_embed_dims) )
        if VERBOSITY>=2:
            print("DIST EMBED ==============================")
            print("dist_embed.shape=",dist_embed.shape)
            print(dist_embed[rand_entries[:],:2])

        # same thing for the dvec embeddings
        dvec2pmts_embed = net.get_detposition_embedding( dvec2pmts_cm.reshape( (npmt*nvoxels,3) ),
                                                         [16,16,16] )
        dvec2pmts_embed = dvec2pmts_embed.reshape( (nvoxels,npmt,3*16) ) # (N,32,48)
        if VERBOSITY>=2:
            print("DVEC2PMTS EMBED ==============================")
            print("dvec2pmts_embed.shape=",dvec2pmts_embed.shape)
            print(dvec2pmts_embed[rand_entries[:],:2])

        # transpose, as we need the shape to be: (N,32,48)
        #dvec2pmts_embed = torch.transpose( dvec2pmts_embed, 1, 0)
        #dist_embed      = torch.transpose( dist_embed, 1, 0)
        #print("    dist_embed.shape=",dist_embed.shape)        
        #print("    dvec2pmts_embed.shape=",dvec2pmts_embed.shape)
        #print("dvec2pmts_embed =====================")
        #print(dvec2pmts_embed)

        # repeat the charge information 32 times for the pmts
        q = row['feat'][:,0:3]  # (NB,3+32) feature tensor has 3 values for charge plus 32 value for the SA to each PMT        
        q = torch.mean(q,dim=1) # take mean charge over plane
        q = torch.repeat_interleave( q, npmt, dim=0).reshape( (nvoxels,npmt,1) ).to(torch.float32).to(device)
        if VERBOSITY>=2:
            print("Q SHAPE ==============================")
            print("pre-repeats")
            print(torch.mean(row['feat'][rand_entries[:],0:3],dim=1))
            print("q.shape=",q.shape)        
            print(q[rand_entries[:],:2])
              
        #print("concat")
        vox_feat = torch.cat( [detpos_embed_perpmt, dvec2pmts_embed, dist_embed], dim=2 ).to(device) # 48+48+16=97
        if VERBOSITY>=2:
            print(" VOXEL FEAT TENSOR ========================")
            print("  voxel feature tensor: ",vox_feat.shape," ",vox_feat.dtype)
            print(vox_feat[rand_entries[:],:2])
                
    #     ntries += 1
    #     if ntries>10*BATCHSIZE:
    #         print("infinite loop trying to fill nonzero charge tensors")
    #         sys.exit(1)
        

    dt_dataprep = time.time()-tstart_dataprep
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
            
    pe_batch = torch.zeros((BATCHSIZE,32),dtype=torch.float32,device=device)

    
    for ibatch in range(BATCHSIZE):

        out_event = pe_per_voxel[start_per_batch[ibatch]:end_per_batch[ibatch],:]
        out_ch = torch.sum(out_event,dim=0)
        pe_batch[ibatch,:] += out_ch[:]        
        if VERBOSITY>=2:
            with torch.no_grad():
                print("  ----------------------------------------")
                print("  batch[",ibatch,"] out_ch.shape=",out_ch.shape," out_event.shape=",out_event.shape)
                print("  event start=",start_per_batch[ibatch]," end=",end_per_batch[ibatch])
                print("  out_event")
                print(out_event[rand_entries[:],:2])
                print("  out_event stats:")
                print("    min: ",out_event.min())
                print("    max: ",out_event.max())
                print("    mean: ",out_event.mean())
                print("    var: ",out_event.var())
                print("  out_ch")
                print(out_ch)
                print("  ----------------------------------------")
                
        

    if VERBOSITY>=2:
        print("PE_BATCH ============================")        
        print("  pe_batch.shape=",pe_batch.shape)
        print("  ",pe_batch)
          
    
    dt_forward = time.time()-tstart_forward
    print("dt_forward: ",dt_forward," secs")

    # LOSS
    pe_target = torch.from_numpy( row['flashpe'] ).to(device)
    if VERBOSITY>=2:
        print("PE_TARGET ===================================")
        print("pe_target.shape=",pe_target.shape)
        print(pe_target)

    #print("")
    #print("sum ===============")
    # stop gradient on sum
    pe_sum = torch.sum(pe_batch,dim=1) # (B)
    print("pe_sum: ",pe_sum)    
    with torch.no_grad():
        pe_sum_perpmt = torch.repeat_interleave( pe_sum.detach(), NPMTS, dim=0).reshape( (BATCHSIZE,NPMTS) )
        print("pe_sum_perpmt: ")
        print(pe_sum_perpmt)


    with torch.no_grad():
        pe_target_sum = torch.sum(pe_target,dim=1) # (B)        
        pdf_target = nn.functional.normalize( pe_target, dim=1, p=1 )
        print("pe_target_sum: ",pe_target_sum)
        
    pdf_batch  = pe_batch / pe_sum_perpmt

    print("pdf_batch: ",pdf_batch)
    print("pdf_target: ",pdf_target)

    floss_emd = loss_sinkhorn( pdf_batch, x_pred, pdf_target, y_target ).mean()
    print("emdloss: ",floss_emd)

    floss_magnitude = loss_mse( pe_batch, pe_target ).mean()
    print("sum mse loss: ",floss_magnitude)

    floss = floss_emd + floss_magnitude
    #floss = floss_magnitude
    #floss = floss_emd
    print("LY=",net.light_yield)
    print("=========================")
    print("Loss[ total ]: ", floss)
    print("=========================")
    
    floss.backward()
    optimizer.step()
    if iteration%100==0:


        
        
        input()

    if USE_WANDB:
        wandb.log({"loss": floss})
    

#net.save(os.path.join(wandb.run.dir, "model.h5"))
#wandb.save('model.h5')
#wandb.save('../logs/*ckpt*')
#wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

