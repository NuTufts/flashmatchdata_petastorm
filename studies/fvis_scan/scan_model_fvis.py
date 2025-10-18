import sys, os

sys.path.append( os.environ['FLASHMATCH_BASEDIR'] )

import yaml
import numpy as np
import flashmatchnet
from flashmatchnet.utils.load_model import load_model
from flashmatchnet.utils.prepare_model_input import prepare_batch_input
from flashmatchnet.utils.pmtpos import create_pmtpos_tensor
import ROOT as rt
import torch


def main(args):

    config_path = args[1]
    outfile = args[2]
    #if os.path.exists(outfile):
    #    raise ValueError("output file already exists: ",outfile)

    rout = rt.TFile(outfile,'recreate')


    nybins = 48
    nzbins = 208
    

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    siren = load_model( config, False, 0 )
    siren.eval()
    device = torch.device( config['model'].get('device'))
    print(siren)
    print("[enter] to start.")
    input()

    xpts = np.linspace(10.0,240.0,5,endpoint=True)
    #xpts = np.linspace(100.0,100.0,1,endpoint=True)
    ypts = np.linspace(-120,120,nybins, endpoint=False)
    zpts = np.linspace(0,1036.0,nzbins,endpoint=False)

    pmtpos = create_pmtpos_tensor()

    print("xpts: ",xpts)
    print("ypts: ",ypts)
    print("zpts: ",zpts)

    for ix,xpt in enumerate(xpts):

        hyz_v = {}
        hdist_vs_yz = {}
        # declare pmts
        for ipmt in range(32):
            z_pmt = pmtpos[ipmt,2]
            y_pmt = pmtpos[ipmt,1]
            hname  = f"hyz_pmt{ipmt}_x{ix}"
            htitle = f"PMT {ipmt} ({z_pmt:.1f},{y_pmt:.1f}): fvis(y,z) @ x[{ix}]={xpt:.1f};z (cm); y (cm)"
            hyz_pmt  = rt.TH2D(hname,htitle, nzbins,0,1036.0,nybins,-120,120)
            hnamedist  = f"hyz_dist_pmt{ipmt}_x{ix}"
            htitledist = f"PMT {ipmt} ({z_pmt:.1f},{y_pmt:.1f}): dist to pmt vs. (z,y) @ x[{ix}]={xpt:.1f};z (cm); y (cm)"            
            hyz_dist = rt.TH2D(hnamedist,htitledist,nzbins,0,1036.0,nybins,-120,120)
            hyz_v[ipmt] = hyz_pmt
            hdist_vs_yz[ipmt] = hyz_dist
        
        for iy,ypt in enumerate(ypts):
            print("ybin[",iy,"/",nybins,"]")
            for iz,zpt in enumerate(zpts):

                avepos = torch.tensor( (xpt,ypt,zpt), dtype=torch.float32 ).to(device)
                q      = torch.tensor((50000.0), dtype=torch.float32).to(device)
                mask   = torch.ones((1), dtype=torch.float32).to(device)
                n_voxels = torch.ones((1),dtype=torch.int32).to(device)

                # add batch dimension
                avepos = avepos.reshape((1,1,-1)) # (1,1,3)
                q      = q.reshape((1,1,-1))      # (1,1,1)
                mask   = mask.reshape((1,1,-1))   # (1,1,1)
                n_voxels = n_voxels.reshape((1,1)) # (1,1)


                #print("avepos: ",avepos.shape,": ",avepos)
                #print("q: ",q.shape,": ",q)
                #print("mask: ",mask)

                batch = {'avepos':avepos,'planecharge':q,'mask':mask,'n_voxels':n_voxels}

                input_feats, input_charge, input_mask = prepare_batch_input( batch, config, device, pmtpos=pmtpos )
                #print('input_charge: ',input_charge.shape)
                #print('input_feats: ',input_feats.shape)
                #print('  input_charge[ibatch=0,Nvoxels,ipmt=0,0]',input_charge.reshape((1,1,32,1))[0,:,0,0])

                pe_per_voxel_per_pmt = siren.forward( input_feats, input_charge ).reshape((1,1,32,1))
                #print("pe_per_voxel_per_pmt: ",pe_per_voxel_per_pmt)
                

                for ipmt in range(32):
                    hyz_v[ipmt].SetBinContent(iz+1,iy+1,pe_per_voxel_per_pmt[0,0,ipmt,0])
                    hdist_vs_yz[ipmt].SetBinContent(iz+1,iy+1, input_feats.reshape((1,1,32,7))[0,0,ipmt,-1]) 

        for ipmt in range(32):
            hyz_v[ipmt].Write()
            hdist_vs_yz[ipmt].Write()

        if False:
            break

    return

if __name__=="__main__":

    main(sys.argv)



