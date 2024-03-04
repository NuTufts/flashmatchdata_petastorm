import os,sys
import torch
import torch.nn as nn

from pmtpos import pmtposmap # uboone pmt pos. in cm in uboone coordinates


class FlashMatchMLP(nn.Module):

    def __init__(self,ndimensions=3,
                 inputshape=(64,64,256),
                 input_nfeatures=3):

        # set the voxel lengths
        self._voxel_len = torch.zeros(1)
        self._voxel_len[0] = 5.0

        # set the max num indices per dimenson
        self._nvoxels_dim = torch.zeros(3)
        self._nvoxels_dim[0] = 54.0
        self._nvoxels_dim[1] = 50.0
        self._nvoxels_dim[2] = 210.0

        # above defines the length of the dims
        self._dim_len = self._nvoxels_dim*self._voxel_len

        self.prepare_pmtpos()

        # network
        
        pass
    
    def prepare_pmtpos(self):
        # copy position data into numpy array format
        pmtpos = torch.zeros( (32, 3) )
        for i in range(32):
            for j in range(3):
                pmtpos[i,j] = pmtposmap[i][j]
        # change coordinate system to 'tensor' system
        # main difference is y=0 is at bottom of TPC        
        pmtpos[:,1] -= -117.0
        # the pmt x-positions are wrong. they would be in the TPC
        pmtpos[:,0] = -20.0

        self._pmtpos = pmtpos

    def index2pos(self,coord_index):
        """
        convert (N,3) tensor of indices to positions in the detector.
        we are hardcoding 5.0 cm voxels ... not great
        """
        coord_pos = coord_index*self._voxel_len

    def norm_det_pos(self,pos_cm):
        for i in range(3):
            pos_cm[:,i] *= 1.0/self._dim_len[i]
        return pos_cm

    def index2scaledpos(self,coord_index):
        pos_cm = self.index2pos(coord_index)
        return self.norm_det_pos(pos_cm)

    def calc_dist_to_pmts(self,pos_cm, dim_w_n=0):
        """
        for tensor (N,3) containing the positions of charge deposits,
        calculate the distance to each of the PMTs, whose positions
        are in self._pmtpos
        """
        
        # we copy the positions by the number of pmts
        pos = pos_cm
        n = pos.shape[0]
        n_feat_dims = pos.shape[1]
        n_repeats = 32

        pos_per_pmt = torch.repeat_interleave( pos, n_repeats,dim=0).reshape( (n,n_repeats,n_feat_dims) )
        #print(pos)
        #print(pos_per_pmt)
        # output is now: (N,32,3)

        # calc per 32 PMTs
        dist_v = [] # list of distance per pos tensors
        dpos_v = [] # list of relative vector tensors
        for ipmt in range(32):
            # get relative position: translation from pmt to charge position
            dx = pos_per_pmt[:,ipmt,0]-self._pmtpos[ipmt,0]
            dy = pos_per_pmt[:,ipmt,1]-self._pmtpos[ipmt,1]
            dz = pos_per_pmt[:,ipmt,2]-self._pmtpos[ipmt,2]
            print(dx.shape)
            # each appended tensor is (N,1,3)            
            dpos = torch.cat( [dx.unsqueeze(-1),dy.unsqueeze(-1),dz.unsqueeze(-1)], dim=1 ).unsqueeze(0)
            print(dpos.shape)
            dpos_v.append( dpos )
            
            # each appended tensor is (1,N)
            dist_v.append( torch.sqrt(dx*dx+dy*dy+dz*dz).unsqueeze(0) )
            
        # concatenate the results for each pmt
        dist   = torch.cat( dist_v, dim=0 )
        dpos_v = torch.cat( dpos_v, dim=0 )
        
        return dist,dpos_v


    ###########################################################################
    # Embedding functions taken from
    #  https://github.com/yang-song/score_sde/blob/main/models/layers.py
    # Functions below are ported over from the DDPM codebase:
    #  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    ###########################################################################

    def get_detposition_embedding(self,det_pos_cm,nembedfeats_per_dim):
        """
        det_pos_cm: (N,3) torch tensor with detector positions of voxels in cms
        nembed_per_dim: list of 3 integers indicating the number of embedding features
          per spatial dimension. this is what we will use to encode the position
          or add to features.
        """
        assert(len(nembedfeats_per_dim)==det_pos_cm.shape[1])
        
        embed_v = []

        nfeats_tot = 0
        for nfeats in nembedfeats_per_dim:
            nfeats_tot += nfeats
        
        for idim in range(len(nembedfeats_per_dim)):
            # loop over spatial dimensions

            # nfeats in this spatial dimension
            nembed = nembedfeats_per_dim[idim]
            
            # we split the number of embedding feats per dim
            # in half, one for sin, one for cos
            half_dim = float( nembed // 2 )

            # get the length of this axis
            axis_len = self._dim_len[idim]

            # nvoxels along this axis
            nvoxels_dim = self._nvoxels_dim[idim] # note its a float

            max_positions = 10000
            # this magic number 10000 is from transformers
            # define the longest wavelength and take log
            # goes into the sinusoids as
            #  sin( x/(max_wavelength)^(2i/d) )
            # where i is the feature index and d is the number of embedding feats
            # we will use a max positions twice the length of tpc in order to explore distances
            # longer than the tpc (to correlate with positions ourside the tpc where light can be made)
            
            # to reduce numerical problems, the denominator of the argument is calculated using
            # the exp of a log
            # wavelength = exp[ - i*(2/d)*log(max_wavelength) ]
            
            emb = torch.log( nvoxels_dim ) / (half_dim - 1)
            # emb = math.log(2.) / (half_dim - 1)
            # we create a tensor with sequence [0,1,2, ..., half_dim-1]
            # multiply by
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=det_pos_cm.device) * -emb)
            # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
            # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
            #emb = timesteps[:, None] * emb[None, :]
            emb = (det_pos_cm[:,idim])[:,None]*emb[None,:]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if nembed % 2 == 1:  # zero pad
                emb = F.pad(emb, (0, 1), mode='constant')
            assert emb.shape == (det_pos_cm[:,idim].shape[0], nembed)
            embed_v.append( emb )

        emb = torch.cat(embed_v,dim=1)
        # returns tensor of (N,D)
        # where D=(nembedding features x-dim) + (nembedding features y-dim) + (nembedding fearures z-dim)
        assert( emb.shape==(det_pos_cm.shape[0],nfeats_tot) )
        return emb

    # make dist2pmt embeddings
    def make_dist2pmt_embed(self,dist_cm):
        pass

def test_calc_dist_to_pmts(Ntest=100):

    model = FlashMatchMLP()

    # make some positions
    pos_cm = torch.rand( (Ntest,3), dtype=torch.float32 )
    for i in range(3):
        pos_cm[:,i] *= model._dim_len[i]

    # do it with proper tensor syntax?
    dist1,dpos1 = model.calc_dist_to_pmts( pos_cm )
    print("dist1.shape: ",dist1.shape)
    print("dpos1.shape: ",dpos1.shape)

    # do it the loop way (which should be right)
    dist2 = torch.zeros( (32,Ntest) )
    for ipmt in range(32):
        dx = pos_cm[:,0]-model._pmtpos[ipmt,0]
        dy = pos_cm[:,1]-model._pmtpos[ipmt,1]
        dz = pos_cm[:,2]-model._pmtpos[ipmt,2]
        dist2[ipmt,:] = torch.sqrt( dx*dx + dy*dy + dz*dz )[:]
    print("dist2.shape: ",dist2.shape)        
        
    diff = (dist1-dist2).sum()
    print("diff.sum()=",diff)
    print("first row comparison")
    print(dist1[0,:])
    print(dist2[0,:])

def test_get_detposition_embedding(Ntest=100,verbose=False):
    model = FlashMatchMLP()

    # make some positions
    pos_cm = torch.rand( (Ntest,3), dtype=torch.float32 )
    for i in range(3):
        pos_cm[:,i] *= model._dim_len[i]

    # make embedding features
    embed1 = model.get_detposition_embedding(pos_cm,nembedfeats_per_dim=[16,16,16])
    print("embed1.shape=",embed1.shape)
    
    if verbose:
        print("pos---------------")
        print(pos_cm)
        print("embed------------")
        print(embed1)
        
if __name__ == "__main__":
    
    # some unit tests
    #test_calc_dist_to_pmts(Ntest=2)
    #test_get_detposition_embedding(Ntest=3,verbose=True)
        
        
