import os,sys

class FlashmatchTrainingDataProducer:
    
    def __init__(self,config):
        self.config = config

        # this needs to be part of a configuration
        self.max_indices_per_dim = [54,50,210]

        # where is the TPC origin assumed to be for the voxelization
        self.tpc_origin = std.vector("float")(3)
        self.tpc_origin[0] = 0.3
        self.tpc_origin[1] = -117.0
        self.tpc_origin[2] = 0.3
    
        self.tpc_end = std.vector("float")(3)
        self.tpc_end[0] = 256.0
        self.tpc_end[1] = 117.0
        self.tpc_end[2] = 1035.7
        
        return
    
    def process_one_entry( self, filename, ientry, iolcv, ioll, fmbuilder, voxelizer,
                           min_charge_voxels=1,
                           max_frac_out_of_tpc=0.3 ):
        """
        process one ROOT file entry.
        In the ROOT file, one entry is one event trigger, which contains many 
        pairs of ionization clusters and flashes.
        For each event, we return with a list of rows for our
        flashmatch training data database.
        One row is one charge and flash pair, to be used for training the light model.
    
        We assume that the larcv and larlite IO interfaces (iolcv, ioll)
        have already been loaded to the same entry/event.
        """
        
        # get info that lets us modify the coordinate index tensors originally
        #  defined by the Voxelizer coordinates to the coordinates used by
        #  Polina's solid angle calculations.

        sa_maxdims = self.max_indices_per_dim


        index_tpc_origin = [ int(voxelizer.get_axis_voxel(i,self.tpc_origin[i])) for i in range(3) ]
        index_tpc_end    = [ int(voxelizer.get_axis_voxel(i,self.tpc_end[i])) for i in range(3) ]
        print("[process one entry] index_tpc_origin: ",index_tpc_origin)
    
        adc_name = "wiremc"

        run = int(ioll.run_id())
        subrun = int(ioll.subrun_id())
        event = int(ioll.event_id())

        # match reco flashes to true track and shower information
        fmbuilder.process( ioll )

        # make vectors of reco opflashes
        flash_np_v = get_reco_flash_vectors( ioll )

        # build candidate spacepoints and true labels
        truth_correct_tdrift = True
        voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name, truth_correct_tdrift )

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

        # keep the non-ghost coordinates
        mask = truth[:]==1
        coord_real   = coord[ mask[:], : ]
        feat_real    = feat[ mask[:], : ]
        trackid_real = voxseqid[ mask[:] ]
        print("unique IDs: ",np.unique(trackid_real))

        print("Prepare Match Rows")
        row_data = []    
        nrejected = 0
    
        # loop over RecoFlash_t objects.
        # get the reco flash, if exists
        # get the coordinate pixels    
        for imatch in range(fmbuilder.recoflash_v.size()):
            print(" MATCH[",imatch,"]: ")
            matchdata = fmbuilder.recoflash_v.at(imatch)

            # get the flash vector
            pe_v = np.zeros( 32, dtype=np.float32 )
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
                print("  empty track ID list. skip.")
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
                print("  empty coord list? skip.")            
                continue
        
            coord_full = np.concatenate( coord_list )
            feat_full  = np.concatenate( feat_list )

            # we need to remove index offsets
            for i in range(3):
                coord_full[:,i] -= index_tpc_origin[i]

            # let's make sure we do not have any charge voxels outside the  TPC/theSA lookup table
            if coord_full.shape[0]==0:
                print("match failing because there are no charge voxels")
                continue
        
            n_f = float(coord_full.shape[0])        
            below_mask = [ coord_full[:,i]>=0 for i in range(3) ]
            above_mask = [ coord_full[:,i]<int(sa_maxdims[i]) for i in range(3) ]

            good_mask = above_mask[0]*below_mask[0]
            good_mask *= above_mask[1]*below_mask[1]
            good_mask *= above_mask[2]*below_mask[2]

            coord_good = coord_full[ good_mask[:], : ]
            feat_good  = feat_full[ good_mask[:], : ]

            frac_below = [ 1.0-float(below_mask[i].sum())/n_f for i in range(3) ]
            frac_above = [ 1.0-float(above_mask[i].sum())/n_f for i in range(3) ]
        
            print("below index fraction by coords: ",frac_below)
            print("above max-index fraction by coords: ",frac_above)

            too_many_outofbounds = False
            for i in range(3):
                if frac_below[i]>max_frac_out_of_tpc or frac_above[i]>max_frac_out_of_tpc:
                    too_many_outofbounds = True
                    print("match failing because too many out of bounds along axis: ",i)
                    print("coord_full.shape: ",coord_full.shape)
                    print("dump of bad indices (post-offset): ")
                    print(coord_full[ good_mask[:]==False, :])

            if coord_good.shape[0]<min_charge_voxels:
                print("match failing because too little voxels inside tpc:  ",coord_good.shape)

            if coord_good.shape[0]<min_charge_voxels or too_many_outofbounds:
                nrejected += 1
                print("match rejected ------")
                print("  ancestorid: ",matchdata.ancestorid)
                print("  time_us: ",matchdata.time_us)
                print("  tick: ",matchdata.tick)            
                continue
        
        
            # make the row of data
            row = {"coord":coord_good,
                   "feat":feat_good,
                   "flashpe":pe_v,
                   "run":run,
                   "subrun":subrun,
                   "event":event,
                   "matchindex":int(imatch),
                   "ancestorid":int(matchdata.ancestorid),
                   "sourcefile":filename}
            row_data.append( dict_to_spark_row(FlashMatchSchema,row) )

        print("[prcess-one-entry] nrejected=",nrejected," nproduced=",len(row_data))

        # now filter out events
        filtered_data = self.filter_matches( row_data )
        
        return row_data


    def get_reco_flash_vectors( self, ioll ):
        """
        Gets the reconstructed flashes we will match charge clusters to.
        We get the data from the larlite data structures.
        We assume that an event has already been loaded before
        this is called.
        """
    
        # reco flash vectors
        producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
        flash_np_v = {}
        for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets(), dtype=np.float32 )
            
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

    def write_event_data_to_spark_session( self, spark_session, output_url, row_data,
                                           rowgroup_size_mb=256, write_mode='append' ):
        
        with materialize_dataset(spark_session, output_url, FlashMatchSchema, rowgroup_size_mb):
            print("store rows to parquet file")
            spark_session.createDataFrame(row_data, FlashMatchSchema.as_spark_schema() ) \
                         .coalesce( 1 ) \
                         .write \
                         .partitionBy('sourcefile') \
                        .mode(write_mode) \
                        .parquet( output_url )
            print("spark write operation")
            
        return True

    def filter_matches(self,row_data):
        
        for irow, row in enumerate(row_data):
            pass
    
