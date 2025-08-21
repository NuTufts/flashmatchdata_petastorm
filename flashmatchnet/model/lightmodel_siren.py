import os,sys
import torch
import torch.nn as nn
from siren_pytorch import SirenNet

class LightModelSiren(SirenNet):
    def __init__(self,use_logpe=True,*args,**kwargs):
        super(LightModelSiren,self).__init__(*args,**kwargs)

        # need overall scale factor for light yield
        #self.light_yield = nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) )
        
        self.use_logpe = use_logpe
        self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
        self.sigmoid_fn = nn.Sigmoid()

        if not self.use_logpe:
            self.softplus_fn = nn.Softplus(beta=1.0, threshold=20.0)
            print("LightModelSiren: use liner LY")
        else:
            self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
            print("LightModelSiren: use log(LY)")

    def get_light_yield(self):
        if not self.use_logpe:
            return self.softplus_fn( self.light_yield )
        else:
            return torch.exp(self.light_yield)

    def forward(self,x,q):
        # run the siren MLP
        out = super(LightModelSiren,self).forward(x)

        # bound visibiity function between 0 and 1 and then scale by charge
        out = self.sigmoid_fn(out)*q 

        if not self.use_logpe:
            # keep pe linear, multiply by positive LY nunmber
            out = out*self.get_light_yield() + 1.0e-6
        else:
            # output is log(pe) = log(visibility*LY_scale) = log(visibility) + log(LY_scale)
            out = torch.log(out+1.0e-6) + self.light_yield

        return out


        
