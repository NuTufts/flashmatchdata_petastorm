import os,sys
import torch
import torch.nn as nn
from siren_pytorch import SirenNet

class LightModelSiren(SirenNet):
    def __init__(self,*args,**kwargs):
        super(LightModelSiren,self).__init__(*args,**kwargs)

        # need overall scale factor for light yield
        #self.light_yield = nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) )
        self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
        self.sigmoid_fn = nn.Sigmoid()

        #if not self.use_logpe:
        self.softplus_fn = nn.Softplus(beta=1.0, threshold=20.0)
        print("LightModelSiren: use linear LY")
        #else:use_logpe
        #    self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
        #    print("LightModelSiren: use log(LY)")

    def get_light_yield(self):
        return self.light_yield

    def forward(self,x,q,return_fvis=False):
        # run the siren MLP
        out = super(LightModelSiren,self).forward(x)

        # bound visibiity function between 0 and 1 and then scale by charge
        out = self.sigmoid_fn(out)

        # multiply by positive voxel charge and LY nunmber        
        pe = out*q*self.get_light_yield() + 1.0e-6

        if return_fvis:
            return pe,out
        else:
            return pe


        
