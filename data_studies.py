import os,sys
import ROOT as rt
import numpy as np
import torch as torch



# -----------
# this script is meant to loop over the dataset and make plots to understand the data
# -----------

# need a loader of the dataset
#from flashmatchdata import make_dataloader, get_rows_from_data_iterator
import flashmatchnet
from flashmatchnet.data.reader import make_dataloader
from flashmatchnet.utils.pmtutils import make_weights,get_2d_zy_pmtpos_tensor


# things we can study
    
# let's understand how some objective functions might behave for simple solutions
# 1) pe=0 for all pmts
# 2) guess the mean pe of the dataset for each pmt

# metrics
# 1) L2 loss or the mean-squared error
# 2) KL divergence
# 3) Balanced Sinkhorn with epsilon=0.1 (this will remove the normalization)

def fill_Q_x_pemax_rowdata( hist_pesum, hist_pemax, hist_pesum_z, hist_pemax_z, rowdata ):
    # calculate 3 items
    # (1) q_mean for each voxel
    # (2) q_mean-weighted x-position
    # (3) q_mean sum over all voxels
    # (4) pe_max
    # (5) pe_sum

    coord = rowdata['coord']
    feat  = rowdata['feat']
    ibatch_start = rowdata['batchstart']
    ibatch_nvox  = rowdata['batchentries']
    ibatch_end   = ibatch_start + ibatch_nvox
    batch_size = len(ibatch_start)
    pe_v = rowdata['flashpe']

    nvoxels, dims = coord.shape

    for ib in range( batch_size ):

        nvox = ibatch_nvox[ib]
        fq_sum  = 0.0
        fx_mean = 0.0
        fpe_sum = 0.0
        fpe_max = 0.0

        with torch.no_grad():
        
            if nvox>0:
                q_plane = feat[ibatch_start[ib]:ibatch_end[ib],0:3]
                coord_i = coord[ibatch_start[ib]:ibatch_end[ib],1:]
                x_pos = coord_i[:,0].float()*5.0 # convert index number to position
                z_pos = coord_i[:,2].float()*5.0
                q_mean = torch.mean(q_plane,1) # returns (N,)

                q_sum = q_mean.sum()                

                fx_mean = float( ((x_pos*q_mean).sum()/q_sum).item() )
                fz_mean = float( ((z_pos*q_mean).sum()/q_sum).item() )                

                fpe_sum = float( pe_v[ib,:].sum().item() )
                fpe_max = float( pe_v[ib,:].max().item() )
                fq_sum  = float( q_sum.item() )
                
                #print(type(fx_mean), type(fq_sum), type(fpe_max), type(fpe_sum) )
                #print(fx_mean, fq_sum, fpe_max, fpe_sum)
            
        hist_pesum.Fill( fx_mean, fq_sum, fpe_sum, 1.0 )
        hist_pemax.Fill( fx_mean, fq_sum, fpe_max, 1.0 )
        hist_pesum_z.Fill( fz_mean, fq_sum, fpe_sum, 1.0 )
        hist_pemax_z.Fill( fz_mean, fq_sum, fpe_max, 1.0 )

    return

def hist3d_Q_x_pemax_fulldataset( data_iter ):

    fout = rt.TFile("out_datastudies_qxpe_hist3d_v2.root","recreate")
    fout.cd()
    
    h3d_pesum = rt.TH3F("hpesum",";x (cm); q; pe (sum)", 256,0,256, 100,0,2.0, 100,0,10.0)
    h3d_pemax = rt.TH3F("hpemax",";x (cm); q; pe (max)", 256,0,256, 100,0,2.0, 100,0,1.0)
    h3d_pesum_z = rt.TH3F("hpesum_z",";z (cm); q; pe (sum)", 259,0,1036, 100,0,2.0, 100,0,10.0)
    h3d_pemax_z = rt.TH3F("hpemax_z",";z (cm); q; pe (max)", 259,0,1036, 100,0,2.0, 100,0,1.0)    
    
    moredata = True
    nrows = 0.0
    while moredata:

        if int(nrows)%1000==0:
            print("processing iteration=",int(nrows))

        try:
            row = next(data_iter)
        except:
            print("iterator out of entries")
            break

        fill_Q_x_pemax_rowdata( h3d_pesum, h3d_pemax, h3d_pesum_z, h3d_pemax_z, row )
        nrows += 1.0
    
    print("nrows processed: ",nrows)
    h3d_pesum.Write()
    h3d_pemax.Write()
    h3d_pesum_z.Write()
    h3d_pemax_z.Write()
    fout.Close()


    

def calculate_mean_variance( data_iter ):
    # calculate mean, variance for absolute, non-zero values, and the pdf value
    pe_x = torch.zeros( 32 )
    pe_x2 = torch.zeros( 32 )

    pdf_x = torch.zeros( 32 )
    pdf_x2 = torch.zeros( 32 )

    nrows = 0.0
    nrows_nonzero = 0.0

    ave_pe_sum = 0.0
    ave_pe_sum_nonzero = 0.0

    moredata = True
    
    while moredata:

        row = get_rows_from_data_iterator(data_iter)
        if row is None:
            break

        x = row['flashpe'][0]
        
        pe_x += x
        pe_x2 += x*x
        pe_sum = x.sum()
        ave_pe_sum += pe_sum
        
        if pe_sum>0.0:
            nrows_nonzero += 1.0
            x_pdf = x/pe_sum
            pdf_x += x_pdf
            pdf_x2 += x_pdf*x_pdf
            ave_pe_sum_nonzero += pe_sum
        
        nrows += 1.0

    ave_pe = pe_x/float(nrows)
    ave_pe2 = pe_x2/float(nrows)
    var_pe = torch.sqrt( ave_pe2-ave_pe*ave_pe )

    ave_pdf  = pdf_x/float(nrows_nonzero)
    ave_pdf2 = pdf_x2/float(nrows_nonzero)
    var_pdf  = torch.sqrt( ave_pdf2-ave_pdf*ave_pdf )

    print("mean pe: ",ave_pe)
    print("var pe: ",var_pe)

    print("mean p(x_i): ",ave_pdf)
    print("var p(x_i): ",var_pdf)
    print("num non-zero: ",nrows_nonzero)

    print("ave total: ",(ave_pe_sum/nrows),")")
    print("ave total (non-zero): ",(ave_pe_sum_nonzero/nrows_nonzero),")")    
    
    return


def simple_count( data_iter, batchsize ):
    nentries = 0
    ncalls = 0
    while True:
        if ncalls%100==0:
            print("ncalls: ",ncalls)
        try:
            rows = next(data_iter)
        except:
            print("end of epoch")
            break
        nentries += batchsize
        ncalls += 1

    print("number of entries: ",nentries)
    print("number of calls: ",ncalls)

if  __name__ == "__main__":

    DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2'
    DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2_validation'    

    NUM_EPOCHS=1
    WORKERS_COUNT=4
    BATCH_SIZE=64
    SHUFFLE_ROWS=False
    
    dataloader = make_dataloader( DATAFOLDER, NUM_EPOCHS, SHUFFLE_ROWS, BATCH_SIZE,
                                  row_transformer=None )
    data_iter = iter(dataloader)


    simple_count(data_iter, BATCH_SIZE)
    sys.exit(0)

    #calculate_mean_variance( data_iter )    
    hist3d_Q_x_pemax_fulldataset( data_iter )

    if True:
        sys.exit(0)
    
    totpe = 0.0
    while totpe==0.0:
        row = next(data_iter)
        pe_v = row['flashpe'][0]
        totpe = pe_v.sum()

    meanpdf = torch.tensor([0.0307, 0.0319, 0.0292, 0.0348, 0.0272, 0.0271, 0.0284, 0.0322, 0.0327,
        0.0344, 0.0362, 0.0291, 0.0295, 0.0331, 0.0331, 0.0360, 0.0361, 0.0301,
        0.0300, 0.0332, 0.0329, 0.0363, 0.0356, 0.0301, 0.0300, 0.0330, 0.0314,
        0.0302, 0.0197, 0.0283, 0.0296, 0.0282])
    
    x = get_2d_zy_pmtpos_tensor()
    y = get_2d_zy_pmtpos_tensor()
    
    a = make_weights(meanpdf)
    b = make_weights(pe_v)

    mse = torch.pow(a-b,2).mean()
    
    print("tot pe: ",totpe)
    
    print("pe_v: ",pe_v)
    print("a [ pdf of mean p(x_i) ]: ",a)
    print("b [ pdf of pe_v ]: ",b)
    print("")
    print("x.shape: ",x.shape)
    print("a.shape: ",a.shape," ",a.sum())
    print("b.shape: ",b.shape," ",b.sum())
    
    import geomloss
    print("test geomloss")

    loss = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)
    emd = loss( a, x, b, y)
    print("emd loss: ",emd)
    print("mse loss: ",mse)
