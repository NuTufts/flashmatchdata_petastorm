import os,sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

import wandb

from lightmodelnet import LightModelNet

import flashmatchnet
import flashmatchnet.model
from flashmatchnet.model.flashmatchnet import FlashMatchNet
from sa_table import load_satable_fromnpz

from flashmatchdata import make_dataloader


USE_WANDB=True
TRAIN_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data'
NUM_EPOCHS=1
WORKERS_COUNT=4
BATCHSIZE=2 #increase to 32

if USE_WANDB:
    wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = LightModelNet(3, 32, D=3).to(device)
#net = FlashMatchNet().to(device)

num_iterations = 3

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
    input = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    #uni_coords = np.unique(coords, axis=0)
    #print("uni_coords", uni_coords)
    #print("uni_coords.shape", uni_coords.shape)

    # Forward
    output = net(input) # expect shape like: (N,32), N covers all batches

    #print("Printing output. Is it all between 0 and 1?")
    print("Output: ", output.shape)

    print("solidAngle: ", solidAngle)
    print("solidAngle.shape: ", solidAngle.shape)
    
    eltmult = output*solidAngle
    print("it worked. eltmult.shape: ", eltmult.shape)

    C = eltmult.C
    F = eltmult.F

    firstBatch = torch.empty(32)
    
    for ib in range(0,BATCHSIZE):

        mask = C[:,0]==ib
        #print("mask: ", mask)
        maskedF = F[mask]
        #print("masked out F (eltmult): ", maskedF)
        #print("Size of masked out F: ", maskedF.shape)
        sum_t = torch.sum(maskedF, 0)
        print("This is summed over 1 batch: ", sum_t)
        #maxPE = torch.max(labelList[ib], 0)

        print("peList[ib]: ", peList[ib])
        print("peList[ib] shape: ", peList[ib].shape)

        peList[ib] = torch.reshape(peList[ib], (32,))

        maxThreePE = torch.topk(peList[ib], 3, dim=0)
        maxFirstPE = maxThreePE[0][0]
        maxSecondPE = maxThreePE[0][1]
        maxThirdPE = maxThreePE[0][2]
        indexFirstPE = maxThreePE[1][0].item()
        indexSecondPE = maxThreePE[1][1].item()
        indexThirdPE = maxThreePE[1][2].item()
        print("This is the maxPE (highest value in maxThreePE): ", maxFirstPE)

        if (ib==0): 
            batchedLosses = sum_t
            batchedLosses_truth = peList[ib]

            maxPEBatch_truth = maxFirstPE
            maxPEBatch_output = sum_t[indexFirstPE]

            secondMaxPEBatch_truth = maxSecondPE
            secondMaxPEBatch_output = sum_t[indexSecondPE]

            thirdMaxPEBatch_truth = maxThirdPE
            thirdMaxPEBatch_output = sum_t[indexThirdPE]

        else:
            batchedLosses = torch.stack([batchedLosses,sum_t]) 
            batchedLosses_truth = torch.stack([batchedLosses_truth, peList[ib]])

            maxPEBatch_truth = torch.stack([maxPEBatch_truth, maxFirstPE])
            maxPEBatch_output = torch.stack([maxPEBatch_output, sum_t[indexFirstPE]])

            secondMaxPEBatch_truth = torch.stack([secondMaxPEBatch_truth, maxSecondPE])
            secondMaxPEBatch_output = torch.stack([secondMaxPEBatch_output, sum_t[indexSecondPE]])

            thirdMaxPEBatch_truth = torch.stack([thirdMaxPEBatch_truth, maxThirdPE])
            thirdMaxPEBatch_output = torch.stack([thirdMaxPEBatch_output, sum_t[indexThirdPE]])
            
                          
    if (BATCHSIZE > 1): # need to reshape tensor to be (BATCHSIZE, 1)
        maxPEBatch_truth = torch.reshape(maxPEBatch_truth, (BATCHSIZE, 1))
        maxPEBatch_output = torch.reshape(maxPEBatch_output, (BATCHSIZE, 1))

        secondMaxPEBatch_truth = torch.reshape(secondMaxPEBatch_truth, (BATCHSIZE, 1))
        secondMaxPEBatch_output = torch.reshape(secondMaxPEBatch_output, (BATCHSIZE, 1))

        thirdMaxPEBatch_truth = torch.reshape(thirdMaxPEBatch_truth, (BATCHSIZE, 1))
        thirdMaxPEBatch_output = torch.reshape(thirdMaxPEBatch_output, (BATCHSIZE, 1))
    
    '''
    #output *= solidAngle
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
    '''

    #loss = error( out_pe, target_pe )

    loss = error(batchedLosses, batchedLosses_truth)
    lossMaxPE = error(maxPEBatch_output, maxPEBatch_truth)
    lossSecondMaxPE = error(secondMaxPEBatch_output, secondMaxPEBatch_truth)
    lossThirdMaxPE = error(thirdMaxPEBatch_output, thirdMaxPEBatch_truth)
    #print("This is the loss: ", loss)
    #wandb.log({"loss": loss.detach().item()})

    floss = loss.detach().item()    
    print("Loss: ", floss)

    flossMaxPE = lossMaxPE.detach().item()
    flossSecondMaxPE = lossSecondMaxPE.detach().item()
    flossThirdMaxPE = lossThirdMaxPE.detach().item()
    if USE_WANDB:
        wandb.log({"loss": floss})
        wandb.log({"lossMaxPE": flossMaxPE})
        wandb.log({"lossSecondMaxPE ": flossSecondMaxPE })
        wandb.log({"lossThirdMaxPE": flossThirdMaxPE})
    
    loss.backward()
    optimizer.step()
    

#net.save(os.path.join(wandb.run.dir, "model.h5"))
#wandb.save('model.h5')
#wandb.save('../logs/*ckpt*')
#wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

