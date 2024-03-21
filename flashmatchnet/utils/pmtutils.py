import os,sys

from .pmtpos import pmtposmap
import torch

def get_2d_zy_pmtpos_tensor(scaled=True):
    x = torch.zeros( (32,2) )
    for i in range(32):
        x[i,0] = pmtposmap[i][2]/1036.0    # pmt geant4 z-position
        x[i,1] = pmtposmap[i][1]/(2*116.5) # pmt geant4 y-position
    return x

def make_weights( pe_value ):

    pe_sum = pe_value.sum()
    w = pe_value/pe_sum
    return w
