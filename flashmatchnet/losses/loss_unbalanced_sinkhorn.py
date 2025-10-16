import torch
import torch.nn as nn
from ..utils.pmtutils import get_2d_zy_pmtpos_tensor

import geomloss

class UnbalancedSinkhornLoss(nn.Module):
    def __init__(self,batchsize,npmts=32):
        super(UnbalancedSinkhornLoss,self).__init__()

        self.sinkhorn_fn = geomloss.SamplesLoss(loss='sinkhorn', p=2, reach=1.0, blur=0.10)
        self.mse = nn.MSELoss()

        # we make the x and y tensors
        self.x_pred   = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
        self.y_target = get_2d_zy_pmtpos_tensor(scaled=True) # (32,2)
        self.x_pred_batch   = self.x_pred.repeat(batchsize,1).reshape( (batchsize,npmts,2) )
        self.y_target_batch = self.y_target.repeat(batchsize,1).reshape( (batchsize,npmts,2) )
        self.x_pred_batch.requires_grad = False
        self.y_target_batch.requires_grad = False


    def forward( self, pred_pmtpe, target_pe, batchstart, batchend, npmts=32, mask=None ):
        """
        pred_pmtpe_per_voxel: (B,32) where B is the batch index, and 32 is the number of PMTs
        target_pe: (B,32)
        batchstart: (B,) indices in pred_pe that mark the start of a batch
        batchend: (B,) indices in pred_pmtpe_per_voxel that mark the end+1 of a batch
        """

        batchsize,npmts = pred_pmtpe.shape
        device = pred_pmtpe.device
        if self.x_pred_batch.device!=device:
            self.x_pred_batch   = self.x_pred_batch.to(device)
            self.y_target_batch = self.y_target_batch.to(device)
        

        # Earth mover's distance calculated using unbiased sinkhorn divergence
        loss = self.sinkhorn_fn( pred_pmtpe, self.x_pred_batch, target_pe, self.y_target_batch ).mean()


        with torch.no_grad():
            pe_sum = torch.sum(pred_pmtpe,dim=1).reshape( (batchsize, 1) ).detach().cpu()
            target_sum = torch.sum(target_pe,dim=1).reshape( (batchsize,1) ).detach().cpu()
            pred_pemax, pemax_indices = pred_pmtpe.detach().max(1)
            floss = loss.detach().cpu().item()
            floss_emd = loss.detach().cpu().item()
            floss_mag = self.mse(pe_sum,target_sum).detach().cpu().item()            
            reporting = (floss,floss_emd, floss_mag, pe_sum, pred_pemax)
        
        return loss, reporting
