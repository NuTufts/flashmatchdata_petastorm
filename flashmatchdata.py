import os,sys
from larlite import larlite
from ublarcvapp import ublarcvapp
from larflow import larflow
import numpy as np

def make_flashmatch_data( fm, triplet ):
    """
    Makes flash and charge pairings from merging info from
      1) ublarcvapp::mctools::FlashMatcherV2
      2) larflow::TripletMaker
    """
    pass

def get_reco_flash_vectors( ioll ):

    # reco flash vectors
    producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
    flash_np_v = {}
    for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets() )
            
            for iopdet in range( flash.nOpDets() ):
                flash_np[iopdet] = flash.PE(iopdet)

            # uboone has 4 pmt groupings
            score_group = {}
            for igroup in range(4):
                score_group[igroup] = flash_np[ 100*igroup: 100*igroup+32 ].sum()
            print(" [",producer,"] iflash[",iflash,"]: ",score_group)
            
            if producer=="simpleFlashBeam":
                flash_np_v[(iproducer,iflash)] = flash_np[0:32]
            elif producer=="simpleFlashCosmic":
                flash_np_v[(iproducer,iflash)] = flash_np[200:232]
    return flash_np_v
    

def process_one_entry( filename, ientry, iolcv, ioll, fmbuilder, voxelizer ):

    adc_name = "wiremc"

    run = int(ioll.run_id())
    subrun = int(ioll.subrun_id())
    event = int(ioll.event_id())

    # match reco flashes to true track and shower information
    fmbuilder.process( ioll )

    # make vectors of reco opflashes
    flash_np_v = get_reco_flash_vectors( ioll )

    # build candidate spacepoints and true labels
    voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name )

    voxdata = voxelizer.make_voxeldata_dict()
    coord   = voxdata["voxcoord"]
    feat    = voxdata["voxfeat"]
    truth   = voxdata["voxlabel"]
    instancedict = voxelizer.make_instance_dict_labels( voxelizer._triplet_maker )
    voxseqid = instancedict["voxinstance"]    
    trackid_to_seqid = instancedict["voxinstance2id"]

    print("coord: ",coord.shape)
    print("feat: ",feat.shape)
    print("truth: ",truth.shape)
    print("voxel sequential id tensor: ",voxseqid.shape)

    # mask out the ghost voxels
    mask = truth[:]==0
    coord_real   = coord[ mask[:], : ]
    feat_real    = feat[ mask[:], : ]
    trackid_real = voxseqid[ mask[:] ]

    print("Prepare Match Rows")
    row_data = []    

    # loop over RecoFlash_t objects.
    # get the reco flash, if exists
    # get the coordinate pixels    
    for imatch in range(fmbuilder.recoflash_v.size()):
        print(" MATCH[",imatch,"]: ")
        matchdata = fmbuilder.recoflash_v.at(imatch)

        # get the flash vector
        pe_v = np.zeros( 32 )
        if matchdata.producerid>=0:
            # if producer is not -1 (null flash)
            # set the pe values to the reco pe
            flashkey = (matchdata.producerid,matchdata.index)
            pe_v = flash_np_v[flashkey]

        # get the charge vector        
        coord_list = []
        feat_list  = []
        trackid_v = matchdata.trackid_list()
        if trackid_v.size()==0:
            continue
        
        for itrackid in range(trackid_v.size()):
            trackid = trackid_v.at(itrackid)
            
            if trackid in trackid_to_seqid:
                seqid = trackid_to_seqid[trackid]
                mask_id = trackid_real[:]==seqid
                
                coord_tid = coord_real[ mask_id[:], : ]
                feat_tid  = feat_real[ mask_id[:], : ]

                coord_list.append( coord_tid )
                feat_list.append( feat_tid )
                
        # concat the coord and feat tensors
        if len(coord_list)==0:
            continue
        
        coord_full = np.concatenate( coord_list )
        feat_full  = np.concatenate( feat_list )

        # make the row
        row = {"coord":coord_full,
               "feat":feat_full,
               "flashpe":pe_v,
               "run":run,
               "subrun":subrun,
               "event":event,
               "index":int(imatch),
               "ancestorid":int(matchdata.ancestorid),
               "file":filename}
        row_data.append(row)
        
    return row_data
    

