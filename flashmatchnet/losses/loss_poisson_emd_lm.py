import torch
import torch.nn as nn
from ..utils.pmtutils import get_2d_zy_pmtpos_tensor

import geomloss

class PoissonNLLwithEMDLoss(nn.Module):
    def __init__(self,magloss_weight=1.0,
                 full_poisson_calc=False,
                 mag_loss_on_sum=False ):
        super(PoissonNLLwithEMDLoss,self).__init__()

        self.poisson_fn  = nn.PoissonNLLLoss(log_input=False,reduction='mean',full=full_poisson_calc)
        self.sinkhorn_fn = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05)

        # we make the x and y tensors
        self.x_pred   = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
        self.y_target = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)

        self.x_pred_batch   = None 
        self.y_target_batch = None 

        self.magloss_weight  = magloss_weight
        self.mag_loss_on_sum = mag_loss_on_sum

    def forward( self, pred_pmtpe_per_voxel, charge, target_pe, batchstart, batchend, npmts=32 ):
        """
        pred_pmtpe_per_voxel: (N,32) where N is over all voxels in the batch, with a prediction for all 32 pmts
        charge: (N,3) Charge from 3 ADC planes for the batch
        target_ps: (B,32)
        batchstart: (B,) indices in pred_pe that mark the start of a batch
        batchend: (B,) indices in pred_pmtpe_per_voxel that mark the end+1 of a batch
        """

        print("This is what we started with (net output): ", pred_pmtpe_per_voxel.shape)
        print("This is what we started with (truth): ", target_pe.shape)

        batchsize = batchstart.shape[0]
        device = pred_pmtpe_per_voxel.device
        
        '''
        # need to first calculate the total predicted pe per pmt for each batch index
        pe_batch = torch.zeros((batchsize,npmts),dtype=torch.float32,device=device)

        for ibatch in range(batchsize):

            out_event = pred_pmtpe_per_voxel[batchstart[ibatch]:batchend[ibatch],:] # (N_ibatch,npmts)
            out_ch = torch.sum(out_event,dim=0) # (npmts,)
            pe_batch[ibatch,:] += out_ch[:]

        print("LOSS FUNCTION CALL")
        '''


        pe_batch = torch.zeros( (batchsize,npmts), dtype=torch.float32,device=device)
        for ib in range(batchsize):

            #print("this is ib:", ib)
            #print("batchstart[ib]: ", batchstart[ib])
            #print("batchend[ib]: ", batchend[ib])

            q_ib = charge[ batchstart[ib]:batchend[ib],:]

            #print("q_ib = q_feat[ batchstart[ib]:batchend[ib],:]: ", q_ib)
            #print("q_ib.shape (before mean): ", q_ib.shape)

            q_ib = torch.mean( q_ib, axis=1 )

            #print("q_ib.shape (after mean): ", q_ib.shape)

            pe_ib = torch.transpose( pred_pmtpe_per_voxel[ batchstart[ib]:batchend[ib],:], 1,0)

            #print("Actual net output for this respective N: ", pred_pmtpe_per_voxel[ batchstart[ib]:batchend[ib],:])
            #print("Actual net output shape: ", pred_pmtpe_per_voxel.shape)
            #print("Actual net output shape for this N: ", pred_pmtpe_per_voxel[ batchstart[ib]:batchend[ib],:].shape)

            #print("transpose'd : ", pe_ib)
            #print("transpose'd shape: ", pe_ib.shape)

            output = pe_ib*q_ib
            #print("output: ", output)
            #print("output.shape: ", output.shape)

            pe_batch[ib,:] = torch.sum(output, 1)

            #print("summed output along N: ", output)
            #print("shape of summed output along N: ", output.shape)

        print("pe_batch.shape: ", pe_batch.shape)
        print("pe_batch: ", pe_batch)


        pe_sum = torch.sum(pe_batch,dim=1) # (B,)


        with torch.no_grad():
            # scale the normalize with detached tensor to stop gradient flow through normalization
            # and the repeating
            pe_sum_perpmt = torch.repeat_interleave( pe_sum.detach(), npmts, dim=0).reshape( (batchsize,npmts) )
            print("type(target_pe): ", type(target_pe))
            pdf_target = nn.functional.normalize( target_pe, dim=1, p=1 )

        pdf_batch  = pe_batch / pe_sum_perpmt # (B,npmts)

        if self.x_pred_batch is None or self.y_target_batch is None:
            with torch.no_grad():
                self.x_pred_batch   = self.x_pred.repeat(batchsize,1).reshape( (batchsize,npmts,2) ).to(device)
                self.y_target_batch = self.y_target.repeat(batchsize,1).reshape( (batchsize,npmts,2) ).to(device)
                self.x_pred_batch.requires_grad = False
                self.y_target_batch.requires_grad = False

        # Earth mover's distance calculated using unbiased sinkhorn divergence
        floss_emd = self.sinkhorn_fn( pdf_batch, self.x_pred_batch, pdf_target, self.y_target_batch ).mean()

        if self.mag_loss_on_sum:
            # Poisson loss on the sum
            # (would it be better to calculate on individual pmts? (would double count the emd loss)
            pe_target_sum = target_pe.sum(dim=1)
            pe_target_sum.requires_grad = False
            floss_magnitude = self.poisson_fn( pe_sum, pe_target_sum )
        else:
            floss_magnitude = self.poisson_fn( pe_batch, target_pe )

        floss = floss_emd + self.magloss_weight*floss_magnitude

        # reporting variables
        pred_pemax, pemax_indices = pe_batch.detach().max(1)
        reporting = (floss.detach().cpu().item(),
                     floss_emd.detach().cpu().item(),
                     floss_magnitude.detach().cpu().item(),
                     pe_sum.detach().cpu(),
                     pred_pemax.cpu())
        
        return floss, reporting
