import os,sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

import wandb

import flashmatchnet
import flashmatchnet.model
from flashmatchnet.model.flashmatchnet import FlashMatchNet
from sa_table import load_satable_fromnpz

from flashmatchdata import make_dataloader


USE_WANDB=False
TRAIN_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data'
NUM_EPOCHS=1
WORKERS_COUNT=4
BATCHSIZE=32

if USE_WANDB:
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlashMatchNet().to(device)

num_iterations = 1000

error = nn.MSELoss(reduction='mean')

learning_rate = 1.0e-4
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
sa_coords, sa_values = load_satable_fromnpz()

train_dataloader = make_dataloader( TRAIN_DATAFOLDER, NUM_EPOCHS, True, 1,
                                    workers_count=WORKERS_COUNT )

train_iter = iter(train_dataloader)

# put net in training mode (vs. validation)
net.train()

for iteration in range(num_iterations): 

    coordList = []
    featList  = []
    saList    = []
    nList = []
    peList = []

    optimizer.zero_grad()            
    ntries = 0
    
    while len(coordList)<BATCHSIZE:
        try:
            row = next(train_iter)
        except:
            print("iterator exhausted. reset")
            train_iter = iter(train_dataloader)
            row = next(train_iter)
        
        #print(" [ncall ",ntries,"] ==================")
        #print(" event: ",row['event']," matchindex: ",row['matchindex'])
        #print(" coord: ",row['coord'].shape)
        #print(" feat: ",row['feat'].shape)
        #print(" flashpe: ",row['flashpe'].shape)

        n_worker_return = row['coord'].shape[0]
        coord = row['coord']
        feat  = row['feat']
        sa    = row['sa']
        pe    = row['flashpe']

        for i in range(n_worker_return):
            if coord.shape[1]>0:
                #print("[",ntries,",",i,"]: ",coord.shape)
                coordList.append( coord[i,:,:] )
                featList.append( feat[i,:,:] )
                saList.append( sa[i,:,:] )
                peList.append( pe[i,:].unsqueeze(0) )
                nList.append( int(coord.shape[1]) )
                
        ntries += 1
        if ntries>100:
            print("infinite loop trying to fill nonzero charge tensors")
            sys.exit(1)
        

    print("num coord tensors ready for training iteration: ",len(coordList))
    assert(len(coordList)==BATCHSIZE) # check that we have the right number of tensors

    # Make the sparse tensor: charge tensor
    coords, feats = ME.utils.sparse_collate( coords=coordList, feats=featList )

    # Make solid angle tensor
    #coords_sa, feats_sa = ME.utils.sparse_collate( coords=coordList, feats=saList )
    solidAngle = torch.from_numpy( np.concatenate( saList ) ).to(device) # (N,32)
    target_pe = torch.from_numpy( np.concatenate( peList, axis=0 ) ).to(device) # (N,32)    
    
    #print("coords (after sparse_collate) --------------- ")
    #print(coords.shape)
    #print("feats (after sparse_collate) ---------------- ")
    #print(feats.shape)
    #print("sa_feats (after sparse_collate) ---------------- ")
    #print(solidAngle.shape)
    #print("true pe (after sparse_collate) ---------------- ")
    #print(target_pe.shape)

    ##coords, feats = ME.utils.sparse_collate( coords=[coord0], feats=[feat0] )
    input_charge = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    #uni_coords = np.unique(coords, axis=0)
    #print("uni_coords", uni_coords)
    #print("uni_coords.shape", uni_coords.shape)

    # Forward
    output = net(input_charge) # expect shape like: (N,32), N covers all batches

    #print("Printing output. Is it all between 0 and 1?")
    #print("Output: ", output.shape)
    
    output *= solidAngle
    output = output.squeeze()
    #print("Printing output (after SA multiplication). Is it all between 0 and 1?")
    #print("Output: ", output.shape)

    # Compress into 32 PMT predictions for each batch
    out_pe = []
    ntot = 0
    for ib in range(BATCHSIZE):
        pe_sum = torch.sum( output[ ntot:ntot+nList[ib], : ], dim=0 ).unsqueeze(0)
        #print("[",ib,"] pe_sum: ",pe_sum.shape)
        out_pe.append( pe_sum )
        ntot += nList[ib]

    out_pe = torch.cat(out_pe, dim=0)
    #print("final pe prediction: ",out_pe.shape)

    loss = error( out_pe, target_pe )

    floss = loss.detach().item()    
    print("Loss: ", floss)
    if USE_WANDB:
        wandb.log({"loss": floss})
    
    loss.backward()
    optimizer.step()
    

#net.save(os.path.join(wandb.run.dir, "model.h5"))
#wandb.save('model.h5')
#wandb.save('../logs/*ckpt*')
#wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

