import torch
import torch.nn as nn

from .pmtutils import get_2d_zy_pmtpos_tensor

###########################################################################
# Embedding functions taken from
#  https://github.com/yang-song/score_sde/blob/main/models/layers.py
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################
def make_embedding(det_pos_cm, nembedfeats_per_dim, maxwavelength_per_dim):
    """
    det_pos_cm: (N,3) torch tensor with detector positions of voxels in cms
    nembed_per_dim: list of 3 integers indicating the number of embedding features
       per spatial dimension. this is what we will use to encode the position
       or add to features.
    """
    ndims = det_pos_cm.shape[1]    
    assert(len(nembedfeats_per_dim)==ndims)
    assert(len(maxwavelength_per_dim)==ndims)

    nfeats_tot = 0
    for nfeats in nembedfeats_per_dim:
        nfeats_tot += nfeats

    embed_v = []

    # loop over spatial dimensions    
    for idim in range(len(nembedfeats_per_dim)):

        # nfeats in this spatial dimension
        nembed = nembedfeats_per_dim[idim]
            
        # we split the number of embedding feats per dim
        # in half, one for sin, one for cos
        half_dim = float( nembed // 2 )

        # get the length of this axis
        max_embed_wavelength = maxwavelength_per_dim[idim] # cm
            
        max_positions = torch.ones(1,dtype=torch.float32,device=det_pos_cm.device)*10.0 
        # this magic number was 10000 from transformers,
        # each batch had 25k source and 25k target tokens.
        # 100.0 is one order of magnitude larger than the number of tokens
        # In the transfomer paper, the embedding is defined as
        #  sin( x/(max_wavelength)^((i)/(d/2)) )
        # where i-th is the feature index and d is the number of embedding feats
        # the number of feats is split in 2, hence (d/2), to include sin and cos embedding components
        # we will use a max positions twice the length of tpc in order to explore distances
        # longer than the tpc (to correlate with positions ourside the tpc where light can be made)
        # note for this to work, x needs to already by unit-less and pre-scaled.
    
        # to reduce numerical problems, the denominator of the argument is calculated using
        # the exp of a log
        # sin argument (in radians)
        # arg = exp( log(x)-(i/(d/2))log(maxL) )*rad
            
        # wavelength = exp[ - i*(2/d)*log(max_wavelength) ]
            
        emb = torch.log( max_positions ) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1) # this was used for time-embedding of the diffusion model
        # we create a tensor with sequence [0,1,2, ..., half_dim-1]
        # multiply by
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=det_pos_cm.device) * -emb)
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        #emb = timesteps[:, None] * emb[None, :]
        emb = (det_pos_cm[:,idim]/max_embed_wavelength)[:,None]*emb[None,:]*3.14159 # this is in radians
        #print("det_pos_cm/max_embed_wavelength, dim=",idim," maxL=",max_embed_wavelength," cm --------------")
        #print((det_pos_cm[:,idim]/max_embed_wavelength)[:10])
        #print("----------------------------")            
        #print("embed argument, dim=",idim," maxL=",max_embed_wavelength," cm --------------")
        #print(emb[:10]/3.14159)
        #print("----------------------------")
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


def prepare_mlp_input_embeddings( coord_batch, q_perplane_batch, net,
                                  vox_len_cm=5.0, npmt=32 ):

    nvoxels = coord_batch.shape[0]
    device = coord_batch.device

    detpos_cm = coord_batch.to(torch.float32)[:,1:4]*vox_len_cm
    detpos_cm.requires_grad = False

    #print(detpos_cm[:10,:])

    detpos_embed = make_embedding( detpos_cm, [16, 16, 16], [54.0*5.0,50.0*5.0,210.0*5.0] )
    
    detpos_embed_perpmt = torch.repeat_interleave( detpos_embed, npmt, dim=0).reshape( (nvoxels,npmt,48) )

    dist2pmts_cm, dvec2pmts_cm = net.calc_dist_to_pmts( detpos_cm )

    dist_embed_dims = 16
    dist_embed = make_embedding( dist2pmts_cm.reshape( (npmt*nvoxels,1) ),
                                 [dist_embed_dims],
                                 maxwavelength_per_dim=[210.0*5.0] )
    dist_embed = dist_embed.reshape( (nvoxels,npmt,dist_embed_dims) )

    dvec2pmts_embed = make_embedding( dvec2pmts_cm.reshape( (npmt*nvoxels,3) ), [16,16,16],
                                      [210.0*5, 210.0*5, 210.0*5] )
                                      
    dvec2pmts_embed = dvec2pmts_embed.reshape( (nvoxels,npmt,48) ) # (N,32,48)

    vox_feat = torch.cat( [detpos_embed_perpmt, dvec2pmts_embed, dist_embed], dim=2 ).to(device) # 48+48+16=97

    q_per_pmt = torch.mean(q_perplane_batch,dim=1) # take mean charge over plane
    q_per_pmt = torch.repeat_interleave( q_per_pmt, npmt, dim=0).reshape( (nvoxels,npmt,1) ).to(torch.float32).to(device)
    
    return vox_feat, q_per_pmt

def prepare_mlp_input_variables( coord_batch, q_perplane_batch, net,
                                 vox_len_cm=5.0, npmt=32 ):
    """
    we provide for each (voxel,pmt) pair the following 9-d input vector:
    (x,y,z,dx,dy,dz,dist,azimuth,zenith)
    """

    nvoxels = coord_batch.shape[0]
    device = coord_batch.device

    # detector positions in cm
    detpos = coord_batch.to(torch.float32)[:,1:4]*vox_len_cm # shape=(N,3)
    detpos.requires_grad = False

    # calculate distance to each pmt and dx from the voxel to the pmts
    dist2pmts, dvec2pmts = net.calc_dist_to_pmts( detpos )

    # the positions and the distances need to be normalized
    detlens = torch.zeros((1,3),dtype=torch.float32).to(device) # shape=(1,3)
    detlens[0,0] = 54.0*5.0
    detlens[0,1] = 50.0*5.0
    detlens[0,2] = 210.0*5.0
    #print("detlens.shape=",detlens.shape)

    # matches 
    # should trigger broadcast of division to all instances of N
    # transpose returns just a view of the same data, so changes should occur to original detpos_cm
    #print("detpos.shape=",detpos.shape)    
    detpos /= detlens
    
    # we make copies of the coordinates
    detpos_perpmt = torch.repeat_interleave( detpos, npmt, dim=0).reshape( (nvoxels,npmt,3) )

    # we scaled the distances
    dist2pmts /= (210.0*5.0)

    # we also scale the relative vector accordingly
    # reshape dvec2pmts from (N,32,3) to (N*32,3)
    # then apply transpose to (3,N*32)
    # and apply the same broadcast: (N,N*32) / 3
    dvec2pmts.reshape( (npmt*nvoxels,3) )
    #dvec2pmts_T = torch.transpose( dvec2pmts, 0, 1 ) # now should be (3,N)
    #dvec2pmts_T /= detlens # will activate broadcast
    dvec2pmts /= detlens # will activate broadcast
    dvec2pmts.reshape( (nvoxels, npmt, 3) )
    vox_feat = torch.cat( [detpos_perpmt, dvec2pmts, dist2pmts], dim=2 ).to(device) # (N,C,7)

    q_per_pmt = torch.mean(q_perplane_batch,dim=1) # take mean charge over plane
    q_per_pmt = torch.repeat_interleave( q_per_pmt, npmt, dim=0).reshape( (nvoxels,npmt,1) ).to(torch.float32).to(device)
    
    return vox_feat, q_per_pmt
