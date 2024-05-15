import os,sys

from .pmtpos import getPMTPosByOpChannel
import torch

def get_2d_zy_pmtpos_tensor(scaled=True):
    x = torch.zeros( (32,2) ) # index by op-channels
    zscale = 1036.0
    yscale = (2*116.5)
    if not scaled:
        zscale = 1.0
        yscale = 1.0
    for i in range(32):
        opchpos = getPMTPosByOpChannel(i, use_v4_geom=True)
        x[i,0] = opchpos[2]/zscale # pmt geant4 z-position
        x[i,1] = opchpos[1]/yscale # pmt geant4 y-position
    return x

def get_3d_pmtpos_tensor():
    x = torch.zeros( (32,3) ) # index by op-channels
    for i in range(32):
        opchpos = getPMTPosByOpChannel(i, use_v4_geom=True)
        for j in range(3):
            x[i,j] = opchpos[j]/zscale    # pmt geant4 z-position
    return x

def make_weights( pe_value ):

    pe_sum = pe_value.sum()
    w = pe_value/pe_sum
    return w
