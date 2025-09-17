import os,sys,time
import ROOT as rt
import torch
import torch.nn as nn
from array import array
import yaml
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# HDF5 datasets
from flashmatchnet.data.read_flashmatch_hdf5 import FlashMatchVoxelDataset
from flashmatchnet.data.flashmatch_mixup import create_mixup_dataloader, MixUpFlashMatchDataset

# Model
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
try:
    from flashmatchnet.model.lightmodel_siren import LightModelSiren
except e:
    print("Trouble loading Siren Model")
    print("Did you active the siren-pytorch submodule? :: git submodule init; git submodule update")
    print("Did you set the environment varibles? :: source setenv_flashmatchdata.sh")
    print(e.what())
    sys.exit(1)

"""
run_siren_inference.py

We use this script to study the performance of the model. 
Our goal is to run on the validation data set rather than on provide a module one can run in the ntuple-maker or in the Reco.

We need to:
1. configure the model via a yaml config
2. also via the yaml, we need to setup a data reader for the validation data
3. we run the inference, saving prediction
4. we have the light model prediction in the validation data set, 
   so we can compare via pe total and the sinkhorn divergence loss.
   so we should save the difference in terms of a change in MSE for the totl
   and a change in the sinkhorn divergence (positive or negative)
"""

import flashmatchnet

def create_data_loaders(config: Dict[str, Any]) -> (DataLoader, Dataset):
    """Create training and validation data loaders using the new HDF5 dataset"""
    
    dataloader_config = config['dataloader']
    
    print(f"Loading validation data from: {dataloader_config['valid_filelist']}")
    valid_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['valid_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    # Apply MixUp augmentation if specified
    use_mixup  = dataloader_config.get('use_mixup',True)
    mixup_prob = dataloader_config.get('mixup_prob')
    
    if use_mixup and mixup_prob > 0:
        print(f"Applying MixUp augmentation with probability {mixup_prob}")
        
        # Create MixUp datasets
        valid_dataset = MixUpFlashMatchDataset(
            base_dataset=valid_base_dataset,
            mixup_prob=mixup_prob,
            alpha=dataloader_config.get('mixup_alpha', 1.0),
            max_total_voxels=dataloader_config.get('max_voxels', 500) * 2
        )
        
        # Use custom collate function for MixUp
        from flashmatchnet.data.flashmatch_mixup import mixup_collate_fn
        collate_fn = mixup_collate_fn
    else:
        valid_dataset = valid_base_dataset
        collate_fn = None
    
    # Create data loaders 
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=dataloader_config['batchsize'],
        shuffle=False,  # Don't shuffle validation
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', False),
        drop_last=True
    )
    
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Batch size: {dataloader_config['batchsize']}")
    
    return valid_dataloader, valid_dataset

def load_model(config: Dict[str, Any]) -> (FlashMatchMLP, LightModelSiren):

    model_config = config['model']
    device = torch.device(config['inference'].get('device','cpu'))

    # Create MLP for embeddings
    flashmlp_config = model_config['flashmlp']
    mlp = FlashMatchMLP(
        input_nfeatures=flashmlp_config['input_nfeatures'],
        hidden_layer_nfeatures=flashmlp_config['hidden_layer_nfeatures']
    ).to(device)
    
    # Create SIREN network
    siren_config = model_config['lightmodelsiren']
    
    # Handle final activation
    if siren_config.get('final_activation') == 'identity':
        final_activation = nn.Identity()
    else:
        raise ValueError(f"Invalid final_activation: {siren_config.get('final_activation')}")
    
    # Create SIREN model
    siren = LightModelSiren(
        dim_in=siren_config['dim_in'],
        dim_hidden=siren_config['dim_hidden'],
        dim_out=siren_config['dim_out'],
        num_layers=siren_config['num_layers'],
        w0_initial=siren_config['w0_initial'],
        final_activation=final_activation,
        use_logpe=config.get('use_logpe')
    ).to(torch.device('cpu'))

    # Load checkpoint
    checkpoint_file = model_config.get('checkpoint',None)
    if checkpoint_file is not None:
        state_dict = torch.load( checkpoint_file, map_location=torch.device('cpu') )
        print("stat_dict keys: ",state_dict.keys())
        print("model_states keys: ",state_dict['model_states'].keys())
        siren.load_state_dict( state_dict['model_states']['siren'] )

    siren = siren.to(device)
    
    return mlp, siren

def main(config_path):

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataloader, dataset = create_data_loaders(config)

    print("Loaded DataLoader. Num entries: ",len(dataset))

    mlp,siren = load_model(config)
    print("Loaded SIREN Model")
    print("="*80)
    print(siren)
    print("="*80)

if __name__ == "__main__":

    main(sys.argv[1])
    

# from flashmatchnet.data.reader import make_dataloader, _default_transform_row, flashmatchdata_collate_fn
# from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
# from flashmatchnet.model.lightmodel_siren import LightModelSiren
# from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
# from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor
# from flashmatchnet.utils.root_vis import single_event_visualization
# from flashmatchnet.data.augment import mixup, scale_small_charge
# from data_studies import get_vars_q_x_targetpe

# rt.gStyle.SetOptStat(0)

# # we want more from the data loader
# def my_transform_row( row ):
#     #print(row.keys())
#     out = _default_transform_row(row)
#     for k in ['run', 'subrun', 'sourcefile']:
#         out[k] = row[k]
#     return out

# def my_collate_fn( datalist ):
#     collated = flashmatchdata_collate_fn(datalist)
#     for k in ['run', 'subrun', 'sourcefile']:
#         x = []
#         for data in datalist:
#             x.append( data[k] )
#         collated[k] = x
#     return collated

# if __name__=="__main__":

#     VALID_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_validation'

#     NUM_EPOCHS=1
#     WORKERS_COUNT=1
#     batchsize=32
#     npmts=32
#     SHUFFLE_ROWS=False
#     FREEZE_BATCH=False # True, for small batch testing
#     VERBOSITY=0
#     NVALID_ITERS=10
#     CHECKPOINT_NITERS=1000
#     checkpoint_folder = "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/checkpoints/"
#     LOAD_FROM_CHECKPOINT=True
#     #checkpoint_file=checkpoint_folder+"/rosy-music-197/lightmodel_mlp_enditer_137501.pth"
#     #checkpoint_file=checkpoint_folder+"/siren/revived-water-213-icy-eon-214/lightmodel_mlp_enditer_332501.pth"
#     #checkpoint_file=checkpoint_folder+"/siren/captain-maquis-216/lightmodel_mlp_enditer_312500.pth"
#     #checkpoint_file=checkpoint_folder+"/siren/curious-universe-230/lightmodel_mlp_iter_90000.pth"
#     checkpoint_file=checkpoint_folder+"/siren/curious-universe-230/lightmodel_mlp_enditer_625001.pth"
#     num_entries = -1
#     RUN_VIS=True

#     num_entries = 5000

#     if RUN_VIS:
#         num_entries=2

#     # LOAD the Model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("USING DEVICE: ",device)

#     mlp = FlashMatchMLP(input_nfeatures=112,
#                         hidden_layer_nfeatures=[512,512,512,512,512]).to(device)

#     # we create a siren network
#     w0_initial = 30.0
#     net = LightModelSiren(
#         #dim_in = 112,                     # input dimension, ex. 2d coor
#         dim_in = 7,                     # input dimension, ex. 2d coor
#         dim_hidden = 512,                 # hidden dimension
#         dim_out = 1,                      # output dimension, ex. rgb value
#         num_layers = 5,                   # number of layers
#         final_activation = nn.Identity(), # activation of final layer (nn.Identity() for direct output)
#         w0_initial = w0_initial           # different signals may require different omega_0 in the first layer - this is a hyperparameter
#     ).to(device)
#     net.eval()
    
    
#     #loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,full_poisson_calc=True).to(device)

#     print("LOADING MODEL STATE FROM CHECKPOINT")
#     print("Loading from: ",checkpoint_file)
#     checkpoint_data = torch.load( checkpoint_file )
#     net.load_state_dict( checkpoint_data['model_state_dict'] )


#     valid_dataloader = make_dataloader( VALID_DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, batchsize,
#                                         row_transformer=my_transform_row,
#                                         custom_collate_fn=my_collate_fn,
#                                         workers_count=WORKERS_COUNT,
#                                         removed_fields=['ancestorid'] )    
#     valid_iter = iter(valid_dataloader)

#     pmt_xy_cm = get_2d_zy_pmtpos_tensor(scaled=False)

#     # measures:
#     #  - error in pe per pmt vs. x
#     #  - error in pe sum vs. x
#     #  - 3D histogram to compare to true value plots made before
#     #  - single event visualizations for many events

#     net.eval()

#     rootout = rt.TFile("out_model_anaysis.root","recreate")

#     h3d_pesum = rt.TH3F("hpesum",";x (cm); q; pe (sum)", 256,0,256, 100,0,2.0, 100,0,10.0)
#     h3d_pemax = rt.TH3F("hpemax",";x (cm); q; pe (max)", 256,0,256, 100,0,2.0, 100,0,1.0)
#     h3d_pesum_true = rt.TH3F("hpesum_true",";x (cm); q; pe (sum)", 256,0,256, 100,0,2.0, 100,0,10.0)
#     h3d_pemax_true = rt.TH3F("hpemax_true",";x (cm); q; pe (max)", 256,0,256, 100,0,2.0, 100,0,1.0)
#     h3d_pesum_z = rt.TH3F("hpesum_z",";z (cm); q; pe (sum)", 259,0,1036, 100,0,2.0, 100,0,10.0)
#     h3d_pemax_z = rt.TH3F("hpemax_z",";z (cm); q; pe (max)", 259,0,1036, 100,0,2.0, 100,0,1.0)    
#     h3d_pesum_z_true = rt.TH3F("hpesum_z_true",";z (cm); q; pe (sum)", 259,0,1036, 100,0,2.0, 100,0,10.0)
#     h3d_pemax_z_true = rt.TH3F("hpemax_z_true",";z (cm); q; pe (max)", 259,0,1036, 100,0,2.0, 100,0,1.0)
#     h2d_pefracerr_v_pe   = rt.TH2D("hpefracerr_v_pe",";pe(true);(pe(pred)-pe(true))/pe(true);",200,0,5.0,201,-10.0,10.0)

#     # should just make a tree ...
#     lmana = rt.TTree("lmanalysis","light model analysis tree")
#     px_mean = array('f',[0.0])
#     pz_mean = array('f',[0.0])
#     pqsum = array('f',[0.0])
#     ppesum_pred = array('f',[0.0])
#     ppesum_true = array('f',[0.0])
#     ppemax_pred = array('f',[0.0])
#     ppemax_true = array('f',[0.0])
#     ppe_pred = array('f',[0.0]*32)
#     ppe_true = array('f',[0.0]*32)
#     lmana.Branch("x",px_mean,"x/F")
#     lmana.Branch("z",pz_mean,"z/F")
#     lmana.Branch("qsum",pqsum,"qsum/F")
#     lmana.Branch("pesum_pred",ppesum_pred,"pesum_pred/F")
#     lmana.Branch("pesum_true",ppesum_true,"pesum_true/F")    
#     lmana.Branch("pemax_pred",ppemax_pred,"pemax_pred/F")
#     lmana.Branch("pemax_true",ppemax_true,"pemax_true/F")
#     lmana.Branch("pe_pred",ppe_pred,"pe_pred[32]/F")
#     lmana.Branch("pe_true",ppe_true,"pe_true[32]/F")    
    
#     print("Start analysis loop")

#     nentries = 0
#     t_start = time.time()
    
#     while (num_entries>=0 and nentries<num_entries) or num_entries<0:

#         if nentries%100==0:
#             dt_elapsed = time.time()-t_start
#             sec_per_iter = 0.0
#             if nentries>0:
#                 sec_per_iter = dt_elapsed/float(nentries)
#             print("Running iteration ",nentries,"; elapsed=%.02f"%(dt_elapsed)," secs; sec/iter=%0.2f secs"%(sec_per_iter))

#         try:
#             row = next(valid_iter)
#         except:
#             print("end of epoch: num charge clusters processed=",nentries*batchsize)
#             break
#         #info = (row['sourcefile'],row['run'],row['subrun'],row['event'],row['matchindex'])
#         #print("[",i,"]: ",info)

#         # scale small charge clusters
#         #row = scale_small_charge( row )
#         # mixup
#         #row = mixup( row, device, factor_range=[0.5,1.5] )        

#         coord = row['coord'].to(device)
#         q_feat = row['feat'][:,:3].to(device)
#         entries_per_batch = row['batchentries']
#         batchstart = torch.from_numpy(row['batchstart']).to(device)
#         batchend   = torch.from_numpy(row['batchend']).to(device)

#         iter_batchsize = batchstart.shape[0]

#         # for each coord, we produce the other features
#         with torch.no_grad():
            
#             #vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, mlp )
#             vox_feat, q = prepare_mlp_input_variables( coord, q_feat, mlp )

#             N,C,K = vox_feat.shape
#             vox_feat = vox_feat.reshape( (N*C,K) )
#             q = q.reshape( (N*C,1) )

#             pe_per_voxel = net(vox_feat, q)
#             pe_per_voxel = pe_per_voxel.reshape( (N,C) )

#             # need to first calculate the total predicted pe per pmt for each batch index
#             pe_batch = torch.zeros((iter_batchsize,npmts),dtype=torch.float32,device=device)

#             for ibatch in range(iter_batchsize):

#                 out_event = pe_per_voxel[batchstart[ibatch]:batchend[ibatch],:] # (N_ibatch,npmts)
#                 out_ch = torch.sum(out_event,dim=0) # (npmts,)
#                 pe_batch[ibatch,:] += out_ch[:]

#             # pred sum
#             pe_sum = torch.sum(pe_batch,dim=1) # (B,)

#             # pred pe max
#             pe_max,pe_max_idx = torch.max(pe_batch,1)

#             # truth
#             #if type(row['flashpe'])=
#             #pe_per_pmt_target = torch.from_numpy(row['flashpe']).to(device)
#             pe_per_pmt_target = row['flashpe'].to(device)
#             pe_sum_target = torch.sum(pe_per_pmt_target,dim=1)

#             # pe frac error
#             pe_fracerr = (pe_batch-pe_per_pmt_target)/pe_per_pmt_target


#             truthvars = get_vars_q_x_targetpe( row, iter_batchsize )
#             for ii in range(iter_batchsize):
#                 if truthvars["q"][ii] is None:
#                     continue
#                 h3d_pesum.Fill( truthvars["x"][ii], truthvars["q"][ii], float(pe_sum[ii].item()) )
#                 h3d_pesum_true.Fill( truthvars["x"][ii], truthvars["q"][ii], truthvars["pesum"][ii] )
                
#                 h3d_pemax.Fill( truthvars["x"][ii], truthvars["q"][ii], float(pe_max[ii].item()) )
#                 h3d_pemax_true.Fill( truthvars["x"][ii], truthvars["q"][ii], truthvars["pemax"][ii] )
                
#                 h3d_pesum_z.Fill( truthvars["z"][ii], truthvars["q"][ii], float(pe_sum[ii].item()) )
#                 h3d_pesum_z_true.Fill( truthvars["z"][ii], truthvars["q"][ii], truthvars["pesum"][ii] )
#                 h3d_pemax_z.Fill( truthvars["z"][ii], truthvars["q"][ii], float(pe_max[ii].item()) )
#                 h3d_pemax_z_true.Fill( truthvars["z"][ii], truthvars["q"][ii], truthvars["pemax"][ii] )

#                 px_mean[0] = truthvars["x"][ii]
#                 pz_mean[0] = truthvars["z"][ii]
#                 pqsum[0] = truthvars["q"][ii]
#                 ppesum_true[0] = truthvars["pesum"][ii]
#                 ppemax_true[0] = truthvars["pemax"][ii]
#                 ppesum_pred[0] = pe_sum[ii]
#                 ppemax_pred[0] = pe_max[ii]
                
#                 for p in range(32):
#                     h2d_pefracerr_v_pe.Fill( pe_batch[ii,p], pe_fracerr[ii,p] )
#                     ppe_pred[p] = pe_batch[ii,p]
#                     ppe_true[p] = pe_per_pmt_target[ii,p]
#                 lmana.Fill()

#             # visualize
#             if RUN_VIS:
#                 for bmax in range( len(row['batchstart']) ):
#                     #bmax = torch.argmax( pe_sum_target )
#                     print("bmax=",bmax)
        
#                     vox_pos_cm = coord[batchstart[bmax]:batchend[bmax],1:4]*5.0+2.5 # should  be (N,3)
#                     vox_pos_cm[:,1] -= 116.5 # remove y-axis offset
#                     qmean_vis = q.reshape( (N,C) )[batchstart[bmax]:batchend[bmax],0]
#                     canvas, vis_prods = single_event_visualization( pe_batch[bmax,:].cpu(), pe_per_pmt_target[bmax,:].cpu(),
#                                                                     vox_pos_cm.cpu(), qmean_vis.cpu(), 
#                                                                     pmt_xy_cm[:,0], pmt_xy_cm[:,1] )
#                     indexing_info = ( row['run'][bmax], row['subrun'][bmax], row['event'][bmax], row['matchindex'][bmax] )
#                     canvas.SaveAs( "flash_visualization_run%03d_subrun%05d_event%05d_index%03d.png"%(indexing_info) )
#                     del vis_prods
#                     del canvas
#                 # end of if RUN_VIS

#         # end of event loop
#         nentries += 1
#         if num_entries>0 and nentries>=num_entries:
#             break

#     print("Number of times we've filled the hists: ",h3d_pesum.GetEntries())
#     rootout.Write()
#     print("Done with loop")


