import os,sys
import ROOT as rt
import torch

import flashmatchnet
from flashmatchnet.data.reader import make_dataloader, _default_transform_row, flashmatchdata_collate_fn
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor

rt.gStyle.SetOptStat(0)

VALID_DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2_validation'

NUM_EPOCHS=1
WORKERS_COUNT=1
batchsize=16
npmts=32
SHUFFLE_ROWS=False
FREEZE_BATCH=False # True, for small batch testing
VERBOSITY=0
NVALID_ITERS=10
CHECKPOINT_NITERS=1000
checkpoint_folder = "/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/checkpoints/"
LOAD_FROM_CHECKPOINT=True
checkpoint_file=checkpoint_folder+"/rosy-music-197/lightmodel_mlp_enditer_137501.pth"
num_events = 2

# LOAD the Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlashMatchMLP(input_nfeatures=112,
                    hidden_layer_nfeatures=[512,512,512,512,512]).to(device)
#loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,full_poisson_calc=True).to(device)

print("LOADING MODEL STATE FROM CHECKPOINT")
print("Loading from: ",checkpoint_file)
checkpoint_data = torch.load( checkpoint_file )
net.load_state_dict( checkpoint_data['model_state_dict'] )

# we want more from the data loader
def my_transform_row( row ):
    #print(row.keys())
    out = _default_transform_row(row)
    for k in ['run', 'subrun', 'sourcefile']:
        out[k] = row[k]
    return out

def my_collate_fn( datalist ):
    collated = flashmatchdata_collate_fn(datalist)
    for k in ['run', 'subrun', 'sourcefile']:
        x = []
        for data in datalist:
            x.append( data[k] )
        collated[k] = x
    return collated
    

valid_dataloader = make_dataloader( VALID_DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, batchsize,
                                    row_transformer=my_transform_row,
                                    custom_collate_fn=my_collate_fn,
                                    workers_count=WORKERS_COUNT,
                                    removed_fields=['ancestorid'] )
valid_iter = iter(valid_dataloader)

pmt_xy_cm = get_2d_zy_pmtpos_tensor(scaled=False)


# measures:
#  - error in pe per pmt vs. x
#  - error in pe sum vs. x
#  - 3D histogram to compare to true value plots made before
#  - single event visualizations for many events

net.eval()


def single_event_visualization( pe_per_pmt, pe_per_pmt_truth, vox_pos_cm, qmean_per_voxel, pmt_x, pmt_y ):
    # make canvas
    c = rt.TCanvas("c", "",1500,600)
    c.Divide(2,2)

    # define draw tpc box
    yzbox = rt.TBox( 0.0, -116.5, 1036.0, +116.5 )
    yzbox.SetLineColor( rt.kBlack )
    yzbox.SetFillStyle(0)
    xybox = rt.TBox( 0.0, -116.5,  256.0, +116.5 )
    xybox.SetLineColor( rt.kBlack )
    xybox.SetFillStyle(0)

    # histograms to compare pe predictions
    hpe_pred    = rt.TH1F("hpe_pred","",32,0,32)
    hpe_target  = rt.TH1F("hpe_target","",32,0,32)

    
    # canvas 1: y-z projection
    c.cd(1)
    # make hist to set coordinates
    hyz = rt.TH2D("hyz",";z (cm); y(cm)", 100, -20.0, 1056.0, 100, -140, +140.0)
    hyz.Draw()
    yzbox.Draw()

    pred_circ_v = []
    for i in range(32):
        pe = pe_per_pmt[i]
        #print("pred pe[",i,"]: ",pe)
        hpe_pred.SetBinContent(i+1,pe)
        r = min(10.0*pe,50.0)
        pe_circle = rt.TEllipse( pmt_x[i], pmt_y[i], r, r )
        pe_circle.SetLineColor(rt.kRed)
        pe_circle.SetFillStyle(0)
        pe_circle.SetLineWidth(3)
        pe_circle.Draw()        
        pred_circ_v.append( pe_circle )
        
    # draw target
    target_circ_v = []
    for i in range(32):
        pe = pe_per_pmt_truth[i]
        #print("target pe[",i,"]: ",pe)
        hpe_target.SetBinContent(i+1,pe)
        r = min(10.0*pe,50.0)
        pe_circle = rt.TEllipse( pmt_x[i], pmt_y[i], r, r )
        pe_circle.SetFillStyle(0)                
        pe_circle.Draw()
        target_circ_v.append( pe_circle )

    # draw voxels: maybe a set of boxes?
    nvoxels = vox_pos_cm.shape[0]
    vox_box_xy_v = []
    vox_box_zy_v = []
    vox_graph = rt.TGraph( vox_pos_cm.shape[0] )
    for i in range(nvoxels):
        # average is maybe a sum of 0.4 of scaled value
        # an there is about 48 pixels if you move directly vertically.
        # calibrate box size to a sum of 0.4 assuming on average 65 voxels ~ 48*sqrt(2)
        l = min(qmean_per_voxel[i]*5.0*(0.4/65.0),10.0) #
        box_zy = rt.TBox( vox_pos_cm[i,2]-l/2.0, vox_pos_cm[i,1]-l/2.0,
                          vox_pos_cm[i,2]+l/2.0, vox_pos_cm[i,1]+l/2.0 )
        box_xy = rt.TBox( vox_pos_cm[i,0]-l/2.0, vox_pos_cm[i,1]-l/2.0,
                          vox_pos_cm[i,0]+l/2.0, vox_pos_cm[i,1]+l/2.0 )
        box_xy.SetFillStyle(0)
        box_zy.SetFillStyle(0)
        box_zy.Draw() # draw in zy
        vox_box_zy_v.append( box_zy )
        vox_box_xy_v.append( box_xy )
        vox_graph.SetPoint(i, vox_pos_cm[i,2], vox_pos_cm[i,1]  )
    vox_graph.SetMarkerStyle(21)
    vox_graph.SetMarkerSize(2)
    #vox_graph.Draw("Psame")

    # change to xy plot (only charge is shown)
    c.cd(2)
    hxy = rt.TH2D("hxy",";x (cm); y(cm)", 100, -10.0, 266.0, 100, -140.0, +140.0)
    hxy.Draw()
    xybox.Draw()
    # draw in the charge boxes
    for b in vox_box_xy_v:
        b.Draw()
    

    # histogram: pe
    c.cd(3)
    hpe_pred.SetMinimum(0.0)
    hpe_target.SetMinimum(0.0)
    hpe_pred.SetLineColor(rt.kRed)    
    if hpe_pred.GetMaximum()>hpe_target.GetMaximum():
        hpe_pred.Draw("hist")
        hpe_target.Draw("histsame")
    else:
        hpe_target.Draw("hist")        
        hpe_pred.Draw("histsame")

    # histogram pdf
    c.cd(4)
    hpdf_pred   = hpe_pred.Clone("hpdf_pred")
    hpdf_target = hpe_target.Clone("hpdf_target")    
    if hpdf_pred.Integral()>0.0:
        hpdf_pred.Scale( 1.0/hpdf_pred.Integral() )
        hpdf_pred.SetLineColor(rt.kRed)
    if hpdf_target.Integral()>0.0:
        hpdf_target.Scale( 1.0/hpdf_target.Integral() )
    if hpdf_pred.GetMaximum()>hpdf_target.GetMaximum():
        hpdf_pred.Draw("hist")
        hpdf_target.Draw("histsame")
    else:
        hpdf_target.Draw("hist")
        hpdf_pred.Draw("histsame")        


    c.Update()

    # return all products
    return c, (hyz, hxy, yzbox, xybox, pred_circ_v, target_circ_v,
               hpdf_pred, hpdf_target,
               vox_box_xy_v, vox_box_zy_v,
               hpe_pred, hpe_target)

print("Start analysis loop")    
    
for i in range(num_events):
    
    row = next(valid_iter)
    #info = (row['sourcefile'],row['run'],row['subrun'],row['event'],row['matchindex'])
    #print("[",i,"]: ",info)

    coord = row['coord'].to(device)
    q_feat = row['feat'][:,:3].to(device)
    entries_per_batch = row['batchentries']
    batchstart = torch.from_numpy(row['batchstart']).to(device)
    batchend   = torch.from_numpy(row['batchend']).to(device)

    # for each coord, we produce the other features
    with torch.no_grad():
        vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, net )

        N,C,K = vox_feat.shape
        vox_feat = vox_feat.reshape( (N*C,K) )
        q = q.reshape( (N*C,1) )

        pe_per_voxel = net(vox_feat, q)
        pe_per_voxel = pe_per_voxel.reshape( (N,C) )

        # need to first calculate the total predicted pe per pmt for each batch index
        pe_batch = torch.zeros((batchsize,npmts),dtype=torch.float32,device=device)

        for ibatch in range(batchsize):

            out_event = pe_per_voxel[batchstart[ibatch]:batchend[ibatch],:] # (N_ibatch,npmts)
            out_ch = torch.sum(out_event,dim=0) # (npmts,)
            pe_batch[ibatch,:] += out_ch[:]

        # pred sum
        pe_sum = torch.sum(pe_batch,dim=1) # (B,)            

        # truth
        pe_per_pmt_target = torch.from_numpy(row['flashpe']).to(device)
        pe_sum_target = torch.sum(pe_per_pmt_target,dim=1)

        # visualize
        for bmax in range( len(row['batchstart']) ):
            #bmax = torch.argmax( pe_sum_target )
            print("bmax=",bmax)
        
            vox_pos_cm = coord[batchstart[bmax]:batchend[bmax],1:4]*5.0+2.5 # should  be (N,3)
            vox_pos_cm[:,1] -= 116.5 # remove y-axis offset

            qmean_vis = q.reshape( (N,C) )[batchstart[bmax]:batchend[bmax],0]

            canvas, vis_prods = single_event_visualization( pe_batch[bmax,:].cpu(), pe_per_pmt_target[bmax,:].cpu(),
                                                    vox_pos_cm.cpu(), qmean_vis.cpu(), 
                                                    pmt_xy_cm[:,0], pmt_xy_cm[:,1] )
            indexing_info = ( row['run'][bmax], row['subrun'][bmax], row['event'][bmax], row['matchindex'][bmax] )
            canvas.SaveAs( "flash_visualization_run%03d_subrun%05d_event%05d_index%03d.png"%(indexing_info) )
            del vis_prods
            del canvas
        

        # end of loop
        
print("Done with loop")


