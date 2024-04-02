import os,sys

from .pmtpos import pmtposmap
import torch

def get_2d_zy_pmtpos_tensor(scaled=True):
    x = torch.zeros( (32,2) )
    zscale = 1036.0
    yscale = (2*116.5)
    if not scaled:
        zscale = 1.0
        yscale = 1.0
    for i in range(32):
        x[i,0] = pmtposmap[i][2]/zscale    # pmt geant4 z-position
        x[i,1] = pmtposmap[i][1]/yscale # pmt geant4 y-position
    return x

def make_weights( pe_value ):

    pe_sum = pe_value.sum()
    w = pe_value/pe_sum
    return w
