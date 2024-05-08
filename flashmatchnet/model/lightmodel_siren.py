import os,sys
import torch
import torch.nn as nn
from siren_pytorch import SirenNet

class LightModelSiren(SirenNet):
    def __init__(self,*args,**kwargs):
        super(LightModelSiren,self).__init__(*args,**kwargs)

        # need overall scale factor for light yield
        self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
        #self.light_yield = nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) )
        self.tanh_fn = nn.Tanh()
        self.softplus_fn = nn.Softplus(beta=1.0, threshold=20.0)

    def get_light_yield(self):
        return 0.5*self.tanh_fn( self.light_yield )+0.5

    def forward(self,x,q):
        # run the siren MLP
        out = super(LightModelSiren,self).forward(x)
        # provide a scalce factor for overall light-yield
        out = self.softplus_fn(out)*self.get_light_yield()
        # multiply by the charge and add an epsilon to prevent zero problems
        out = out*q + 1.0e-8
        return out


        
