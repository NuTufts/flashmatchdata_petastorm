from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .backbone_resunetme import MinkEncode4Layer, MinkDecode4Layer
from .resnetinstance_block import BasicBlockInstanceNorm

class FlashMatchNet(nn.Module):

    def __init__(self,ndimensions=3,
                 inputshape=(128,128,512),
                 input_nfeatures=3):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        input_nfeatures [int] number of features in the input tensor, default=1 (the image charge)
        """
        super(FlashMatchNet,self).__init__()
        
        # STEM
        stem_nfeatures = 16
        stem_nlayers = 3
        stem_layers = OrderedDict()
        if stem_nlayers==1:
            respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
            block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
                else:
                    block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        self.encoder = MinkEncode4Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=3 )
        self.decoder = MinkDecode4Layer( in_channels=stem_nfeatures, out_channels=32, D=3 )

        # sparse to dense operation
        #self.sparse_to_dense = [ ME.MinkowskiToFeature() for p in range(input_nplanes) ]
        self.out_feature = ME.MinkowskiToFeature()

        self.register_parameter( name="light_yield",param=nn.parameter.Parameter( torch.zeros(1,dtype=torch.float32) ) )
        self.tanh_fn = nn.Tanh()

        # softplus to clamp values
        self.softplus_fn = nn.Softplus(beta=1.0, threshold=20.0)

        # Regression MLP
        reg_nfeatures = [32,64]
        npmts = 32
        reg_layers = OrderedDict()
        reg_layers["reg0conv"] = torch.nn.Conv1d(MinkDecode4Layer.PLANES[-1],
                                                 reg_nfeatures[0],1)
        reg_layers["reg0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(reg_nfeatures[1:]):
            reg_layers["reg%dconv"%(ilayer+1)] = torch.nn.Conv1d(reg_nfeatures[ilayer],nfeats,1)
            reg_layers["reg%drelu"%(ilayer+1)] = torch.nn.ReLU()
        reg_layers["regout"] = torch.nn.Conv1d(reg_nfeatures[-1],npmts,1)
        self.reg = torch.nn.Sequential( reg_layers )



    def get_light_yield(self):
        return 0.5*self.tanh_fn( self.light_yield )+0.5
        
        

    def forward( self, input_sparsetensors ):

        # check input
        
        # we push through each sparse image through the stem and backbone (e.g. unet)
        x = self.stem(input_sparsetensors)
        x_encode = self.encoder(x)
        #print("post encoder")
        #for xx in x_encode:
        #    print(xx)
        
        x_decode = self.decoder(x_encode)
        print("x_decode.F: ",x_decode.F.shape)
        #print("------------------------------------------------------------")
        #print("output features plane[",p,"] ",x_decode.shape)
        #print(x_decode)
        #print("------------------------------------------------------------")
        
        # then we have to extract a feature tensor
        #batch_spacepoint_feat = self.extract_features(x_feat_v, matchtriplets, batch_size )
        #for b,spacepoint_feat in enumerate(batch_spacepoint_feat):
        #    print("--------------------------------------------------------")
        #    print("extracted features batch[",b,"]_spacepoint_feat")            
        #    print(spacepoint_feat)
        #print("--------------------------------------------------------")

        out = self.out_feature( x_decode ) # (N,F)
        # need to get to (None,C,H,W) so (F,N,1)
        #out = torch.transpose( out, 1, 0 )
        out = self.softplus_fn(out)
        print("out_feature: ",out.shape)        
        #out = self.reg( out ).squeeze()
        #out = torch.transpose( out, 1, 0 )

        return out
                                        
