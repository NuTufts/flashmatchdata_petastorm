/**
 * @file make_mc_training_data
 * @brief Program to produce light model training data using Corsika Simulated data
 * 
 * This program implements a pipeline to prepare data
 * - voxelize the image data
 * - use the true trajectories and truth information to cluster charge voxels into cosmic tracks.
 *   this replaces actual cosmic reconstruction.
 * - Match optical flashes with cosmic ray tracks
 * - Integrate CRT information when available
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

// ROOT includes
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>

// ubdl includes (conditional compilation)
// TODO: Add proper UBDL includes when implementing ROOT I/O
// The skeleton currently uses dummy data for compilation testing
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larflow/Reco/NuVertexFlashPrediction.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"
#include "larflow/PrepFlowMatchData/FlowTriples.h"
#include "larflow/OpticalModel/OpModelMCDataPrep.h"

#include "larlite/DataFormat/storage_manager.h"

#include "ublarcvapp/MCTools/FlashMatcherV2.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"

// Local includes
#include "DataStructures.h"
#include "FlashTrackMatcher.h"
#include "CRTMatcher.h"
#include "FlashMatchOutputData.h"
#include "FlashMatchHDF5Output.h"
#include "LarliteDataInterface.h"
#include "CosmicRecoInput.h"
#include "PrepareVoxelOutput.h"
#include "TruthFlashTrackMatcher.h"

using namespace flashmatch::dataprep;

/**
 * @brief Program configuration structure
 */
struct ProgramConfig {
    std::string input_file; // cosmic reco information
    std::string input_mcinfo; // root file containing mcinfo larlite products
    std::string output_hdf5_file;
    std::string quality_cuts_config;
    std::string flash_matching_config;
    std::string debug_output_file;
    std::string larcv_input_file;
    
    int max_events = -1;
    int start_event = 0;
    int verbosity = 1;
    bool debug_mode = false;
    bool enable_crt = true;
    bool have_larcv = false;
    bool have_mcinfo = false;
    bool output_hdf5 = false;
    bool exclude_anode = false;
    
    ProgramConfig() = default;
};

/**
 * @brief Print program usage
 */
void PrintUsage(std::string& program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Flash-Track Matching Data Preparation\n"
              << "Applies quality cuts and performs flash-track matching on cosmic ray data\n\n"
              << "Required Arguments:\n"
              << "  --input FILE              Input ROOT file from cosmic reconstruction\n"
              << "  --input-mcinfo FILE       Input ROOT file containing simulation truth (larlite mcinfo)\n"
              << "Must specify HDF5 output file\n"
              << "  --output-hdf5 FILE             Output HDF5 file with matched data\n\n"
              << "Optional Arguments:\n"
              << "  --config FILE             Quality cuts configuration (YAML)\n"
              << "  --flash-config FILE       Flash matching configuration (YAML)\n"
              << "  --debug-output FILE       Debug output file\n"
              << "  --max-events N            Maximum number of events to process\n"
              << "  --start-event N           Starting event number (default: 0)\n"
              << "  --verbosity N             Verbosity level 0-3 (default: 1)\n"
              << "  --debug                   Enable debug mode\n"
              << "  --no-crt                  Disable CRT matching\n"
              << "  --help                    Display this help message\n"
              << "  --exclude-anode           Exclude Anode-crossing matches\n"
              << "  --larcv FILE              LArCV file containing images. Used to make flash prediction.\n\n"
              << "Examples:\n"
              << "  " << program_name << " --input cosmic_tracks.root --output matched_data.root\n"
              << "  " << program_name << " --input cosmic_tracks.root --output matched_data.root \\\n"
              << "    --config quality_cuts.yaml --flash-config flash_matching.yaml --debug\n\n";
}

/**
 * @brief Parse command line arguments
 */
bool ParseArguments(int argc, char* argv[], ProgramConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--input" && i + 1 < argc) {
            config.input_file = argv[++i];
        } else if (arg == "--input-mcinfo" && i + 1 < argc) {
            config.input_mcinfo = argv[++i];
            config.have_mcinfo = true;
        } else if (arg == "--output-hdf5" && i + 1 < argc) {
            config.output_hdf5_file = argv[++i];
            config.output_hdf5 = true;
        } else if (arg == "--config" && i + 1 < argc) {
            config.quality_cuts_config = argv[++i];
        } else if (arg == "--flash-config" && i + 1 < argc) {
            config.flash_matching_config = argv[++i];
        } else if (arg == "--debug-output" && i + 1 < argc) {
            config.debug_output_file = argv[++i];
        } else if (arg == "--max-events" && i + 1 < argc) {
            config.max_events = std::atoi(argv[++i]);
        } else if (arg == "--start-event" && i + 1 < argc) {
            config.start_event = std::atoi(argv[++i]);
        } else if (arg == "--verbosity" && i + 1 < argc) {
            config.verbosity = std::atoi(argv[++i]);
        } else if (arg == "--debug") {
            config.debug_mode = true;
        } else if (arg == "--no-crt") {
            config.enable_crt = false;
        } else if (arg == "--exclude-anode" ) {
            config.exclude_anode = true;
        } else if (arg == "--larcv" ) {
            config.have_larcv = true;
            config.larcv_input_file = std::string( argv[++i] );
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            return false;
        }
    }
    
    // Check required arguments
    if (config.input_file.empty()) {
        std::cerr << "Error: Input file is required" << std::endl;
        return false;
    }
    
    if (!config.output_hdf5 ) {
        std::cerr << "Error: Must specify hdf5 output path" << std::endl;
        return false;
    }

    if (config.output_hdf5 && config.output_hdf5_file.empty()) {
        std::cerr << "Error: Output HDF5 file path is empty" << std::endl;
        return false;
    }



    return true;
}

bool check_output_data( EventData& output_data, ProgramConfig& config )
{

     // check everything is OK

    if (output_data.num_matches() != output_data.optical_flashes.size()) {
        throw std::runtime_error("Number of tracks and flashes saved in matched output EventData disagree!");
    }
    if (output_data.num_matches() != output_data.crt_hits.size()) {
        throw std::runtime_error("Number of tracks and CRT Hits saved in matched output EventData disagree!");
    }
    if (output_data.num_matches() != output_data.crt_tracks.size()) {
        throw std::runtime_error("Number of tracks and CRT Tracks saved in matched output EventData disagree!");
    }
    if (output_data.num_matches() != output_data.match_type.size()) {
        throw std::runtime_error("Number of tracks and match_type labels saved in matched output EventData disagree!");
    }

    if (config.have_larcv) {
        if (output_data.num_matches()!=output_data.predicted_flashes.size()) {
            throw std::runtime_error("Number of tracks and predicted flashes saved in matched output EventData disagree!");
        }
        if (output_data.num_matches()!=output_data.voxel_planecharge_vvv.size()) {
            throw std::runtime_error("Number of tracks and voxel plane charges in the EventData disagree!");
        }
        if (output_data.num_matches()!=output_data.voxel_indices_vvv.size()) {
            throw std::runtime_error("Number of tracks and voxel indices in the EventData disagree!");
        }
        if (output_data.num_matches()!=output_data.voxel_avepos_vvv.size()) {
            throw std::runtime_error("Number of tracks and voxel ave pos in the EventData disagree!");
        }
        if (output_data.num_matches()!=output_data.voxel_centers_vvv.size()) {
            throw std::runtime_error("Number of tracks and voxel ave pos in the EventData disagree!");
        }
    }

    return true;
}

/**
 * @brief Main program entry point
 */
int main(int argc, char* argv[]) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Parse command line arguments
    ProgramConfig config;
    if (!ParseArguments(argc, argv, config)) {
        std::string programName = argv[0];
        PrintUsage(programName);
        return 1;
    }

    // Set debug environment if requested
    if (config.debug_mode) {
        setenv("FLASHMATCH_DEBUG", "1", 1);
        setenv("FLASHMATCH_LOG_LEVEL", "DEBUG", 1);
    }

    std::cout << "Flash-Track Matching Data Preparation" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Input file: " << config.input_file << std::endl;
    if ( config.output_hdf5 )
        std::cout << "Output HDF5 file: " << config.output_hdf5_file << std::endl;

    if (!config.quality_cuts_config.empty()) {
        std::cout << "Quality cuts config: " << config.quality_cuts_config << std::endl;
    }
    if (!config.flash_matching_config.empty()) {
        std::cout << "Flash matching config: " << config.flash_matching_config << std::endl;
    }
    std::cout << "CRT matching: " << (config.enable_crt ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;


    // we use a truth-based flash-track matcher
    larflow::opticalmodel::OpModelMCDataPrep fmutil;
    ublarcvapp::mctools::FlashMatcherV2 truth_fm;
    fmutil.setVerboseLevel(config.verbosity);

    // Create our truth-based flash-track matcher
    TruthFlashTrackMatcher truth_track_matcher;
    truth_track_matcher.SetVerbosity(config.verbosity);
    if ( config.exclude_anode )
        truth_track_matcher.SetExcludeAnode( true );

    // Initialize processing components
    FlashMatchConfig flash_config;

    // Load configurations if provided

    // FlashTrackMatcher flash_matcher(flash_config);
    // if (!config.flash_matching_config.empty()) {
    //     if (!flash_matcher.LoadConfigFromFile(config.flash_matching_config)) {
    //         std::cerr << "Warning: Could not load flash matching config, using defaults" << std::endl;
    //     }
    // }

    // CRTMatcher crt_matcher;
    // crt_matcher.set_verbosity_level(2); // kINFO for now. TODO: make a configuration or agument line parameter

    // We use this make the flash prediction
    larflow::reco::NuVertexFlashPrediction flashpredicter;

    // Utility class to bin spacepoints into voxels
    //bool HAS_MC = false;
    //float sparseimg_adc_threshold = 10.0;
    larflow::voxelizer::VoxelizeTriplets voxelizer;
    float voxel_len = 5.0;
    voxelizer.set_voxel_size_cm( voxel_len ); // re-define voxels to 5 cm spaces
    auto const ndims_v = voxelizer.get_dim_len(); // number of voxels per dimension
    auto const voxel_origin_v = voxelizer.get_origin(); // (x,y,z) of origin voxel (0,0,0)
    std::vector<float> tpc_origin = { 0.0, -117.0, 0.0 };
    std::vector<float> tpc_end = { 256.0, 117.0, 1036.0 }; 
    std::vector<int> index_tpc_origin(3,0);
    std::vector<int> index_tpc_end(3,0);
    for (int i=0; i<3; i++) {
        index_tpc_origin[i] = voxelizer.get_axis_voxel(i,tpc_origin[i]);
        index_tpc_end[i]    = voxelizer.get_axis_voxel(i,tpc_end[i]);
    }

    std::cout << "VOXELIZER SETUP =====================" << std::endl;
    std::cout << "origin: (" << tpc_origin[0] << "," << tpc_origin[1] << "," << tpc_origin[2] << ")" << std::endl;
    std::cout << "ndims: (" << ndims_v[0] << "," << ndims_v[1] << "," << ndims_v[2] << ")" << std::endl;
    std::cout << "index-tpc-origin: ("
              << index_tpc_origin[0] << ","
              << index_tpc_origin[1] << ","
              << index_tpc_origin[2] << ")" 
              << std::endl;
    std::cout << "index-tpc-end: ("
              << index_tpc_end[0] << ","
              << index_tpc_end[1] << ","
              << index_tpc_end[2] << ")" 
              << std::endl;
    std::cout << "=====================================" << std::endl;

    PrepareVoxelOutput voxelprep;

    // Process events
    int events_processed = 0;
    int events_with_matches = 0;

    // Load the input file
    CosmicRecoInput cosmic_reco_input_file( config.input_file );
    int num_events = cosmic_reco_input_file.get_num_entries();

    std::cout << "Loaded input file. Number of entries: " << cosmic_reco_input_file.get_num_entries() << std::endl;

    larcv::IOManager iolcv( larcv::IOManager::kREAD, "larcv", larcv::IOManager::kTickForward );
    if ( config.have_larcv ) {
        iolcv.add_in_file( config.larcv_input_file );
        iolcv.specify_data_read( "image2d", "wire" );
        iolcv.specify_data_read( "image2d", "instance" );
        iolcv.specify_data_read( "image2d", "ancestor" );
        iolcv.set_verbosity( larcv::msg::kINFO );
        iolcv.initialize();

        if ( num_events != (int)iolcv.get_n_entries() ) {
            std::cout << "WARNING: Number of larcv events does not match cosmic reco events." << std::endl;
            if ( num_events > (int)iolcv.get_n_entries() ) {
                num_events = iolcv.get_n_entries();
            }
        }
    }

    larlite::storage_manager ioll( larlite::storage_manager::kREAD );
    if ( config.have_mcinfo ) {
        ioll.add_in_filename( config.input_mcinfo );
        ioll.set_verbosity( larlite::msg::kINFO );
        ioll.open();
    }

    int total_events = (config.max_events > 0 ) ? config.max_events : num_events;
    int end_entry = config.start_event + total_events;
    if ( end_entry > num_events )
        end_entry = num_events;

    // Define the output file(s)
    FlashMatchHDF5Output* hdf5_output_man = nullptr;
    if ( config.output_hdf5 ) {
        hdf5_output_man = new FlashMatchHDF5Output( config.output_hdf5_file, true );
    }

    std::cout << "Starting Event Loop" << std::endl;
    std::cout << "Start entry: " << config.start_event << std::endl;
    std::cout << "End entry: " << end_entry << std::endl;

    for (int entry = config.start_event; entry < end_entry; ++entry) {
        
        std::cout << "[ENTRY " << entry << "]" << std::endl;
        ioll.go_to( entry );
        iolcv.read_entry( entry );

        // run truth-based flash matcher
        // matches reco optical flashes to MC-trajectories based on timing
        truth_fm.process(ioll);
        truth_fm.printMatches();

        // load cosmic reco
        // has line-segment representations of cosmic tracks
        // along with 3d hits associated to those tracks
        cosmic_reco_input_file.load_entry( entry );

        // convert input data into format we will use to then convert into 
        // training data arrays stored in HDF5 format.

        EventData input_data;
        input_data.run    = cosmic_reco_input_file.get_run();
        input_data.subrun = cosmic_reco_input_file.get_subrun();
        input_data.event  = cosmic_reco_input_file.get_event();

        input_data.optical_flashes = convert_event_opflashes( cosmic_reco_input_file.get_opflash_v() );
        std::cout << "  number of optical flashes: " << input_data.optical_flashes.size() << std::endl;

        input_data.cosmic_tracks = convert_event_trackinfo( 
            cosmic_reco_input_file.get_track_v(),
            cosmic_reco_input_file.get_hitinfo_v()
        );
        std::cout << "  number of cosmic tracks: " << input_data.cosmic_tracks.size() << std::endl;

        input_data.crt_tracks = convert_event_crttracks( cosmic_reco_input_file.get_crttrack_v() );
        std::cout << "  number of CRT tracks: " << input_data.crt_tracks.size() << std::endl;

        input_data.crt_hits   = convert_event_crthits( cosmic_reco_input_file.get_crthit_v() );
        std::cout << "  number of CRT hits: " << input_data.crt_hits.size() << std::endl;

        EventData output_data; // class to store results.

        // Load instance images for truth matching
        larcv::EventImage2D* ev_instance = nullptr;
        std::vector<larcv::Image2D> instance_img_v;

        if (config.have_larcv) {
            ev_instance = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "ancestor");
            if (ev_instance) {
                instance_img_v = ev_instance->as_vector();
                std::cout << "  loaded " << instance_img_v.size() << " instance images" << std::endl;
            } else {
                std::cerr << "Warning: Could not load instance images" << std::endl;
            }
        }

        // Perform truth-based flash-track matching
        int num_matches = 0;
        if (!instance_img_v.empty()) {
            num_matches = truth_track_matcher.MatchTracksToFlashes(input_data, output_data,
                                                                  instance_img_v, truth_fm);
            std::cout << "  truth matcher found " << num_matches << " matches" << std::endl;
        }

        // Process matched data
        if (num_matches > 0) {

	  EventData filtered_data;
	  filtered_data.run    = output_data.run;
	  filtered_data.subrun = output_data.subrun;
	  filtered_data.event  = output_data.event;

            // prepare voxel data for each matches
            larcv::EventImage2D* ev_adc =
                (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
            auto const& adc_v = ev_adc->as_vector();
            
            for (size_t imatch=0; imatch<output_data.cosmic_tracks.size(); imatch++) {

                auto& cflash = output_data.optical_flashes.at(imatch);
                auto& ctrack = output_data.cosmic_tracks.at(imatch);

                // Prepare voxel represention of track
                std::vector< std::vector<float> > voxel_planecharge_vv;
                std::vector< std::vector<float> > voxel_avepos_vv;
                std::vector< std::vector<float> > voxel_centers_vv;
                std::vector< std::vector<int> >   voxel_indices_vv;
                //int nvoxels = voxelprep.makeVoxelChargeTensor( 
                voxelprep.makeVoxelChargeTensor(
                    ctrack,
                    adc_v,
                    voxelizer,
                    voxel_indices_vv,
                    voxel_centers_vv,
                    voxel_avepos_vv,
                    voxel_planecharge_vv
                );

		if ( voxel_indices_vv.size()==0 ) {
		  std::cout << "Number of voxels for this track is zero. Skip." << std::endl;
		  continue;
		}

                // store observed pe and produce predicted pe
                float totpe_observed = 0.0;
                for (size_t i=0; i<32; i++) {
                    totpe_observed += cflash.pe_per_pmt[i];
                }

                // now we want to create a predicted flash for this track
                // the interface to the flash prediction code requires making
                // a temporary nu vertex candidate object.
                larflow::reco::NuVertexCandidate nuvtx;
                nuvtx.track_v.push_back( cosmic_reco_input_file.get_track_v().at(ctrack.index) );
                nuvtx.track_isSecondary_v.push_back(0);

                // now use the flashprediction routine
                larlite::opflash predicted_opflash = flashpredicter.predictFlash( nuvtx, adc_v );

                // pass the result to the flash
                OpticalFlash predflash;
                predflash.readout = 2; // prediction index
                predflash.index   = cflash.index;
                predflash.flash_center  = cflash.flash_center;
                predflash.flash_width_y = cflash.flash_width_y;
                predflash.flash_width_z = cflash.flash_width_z;
                predflash.flash_time    = cflash.flash_time;
                double totpe = 0.0;
                predflash.pe_per_pmt.resize(32,0);
                for (size_t i=0; i<32; i++) {
                    predflash.pe_per_pmt[i] = 1000*predicted_opflash.PE(i);
                    totpe += predflash.pe_per_pmt[i];
                }
                predflash.total_pe = totpe;

		// store everything
                filtered_data.predicted_flashes.push_back( predflash );		
		filtered_data.optical_flashes.emplace_back( std::move(cflash) );
		
		filtered_data.cosmic_tracks.emplace_back( std::move(ctrack) );		
                filtered_data.voxel_indices_vvv.emplace_back(std::move(voxel_indices_vv));
                filtered_data.voxel_centers_vvv.emplace_back(std::move(voxel_centers_vv));
                filtered_data.voxel_avepos_vvv.emplace_back(std::move(voxel_avepos_vv));
                filtered_data.voxel_planecharge_vvv.emplace_back(std::move(voxel_planecharge_vv));

		filtered_data.crt_hits.emplace_back( std::move(output_data.crt_hits.at(imatch)) );
		filtered_data.crt_tracks.emplace_back( std::move(output_data.crt_tracks.at(imatch)) );

		filtered_data.match_type.push_back( output_data.match_type.at(imatch) );

                // calculate logratio for filtering (not used now)
                // double flash_logratio
                //     = TMath::Log(predflash.total_pe)-TMath::Log(totpe_observed);
                // if ( flash_logratio<-2.0 || flash_logratio>2.0 ) {
                //     std::cout << "out-of-range PE agreement: log(predicted)-log(observed)=" << flash_logratio << std::endl;
                //     continue;
                // }

            }//end of match loop

	    std::swap( output_data, filtered_data );

            // Save processed data
            int num_matches_saves = output_data.num_matches();
            
            bool isok = check_output_data(output_data, config);
            if ( !isok ) {
                throw std::runtime_error("Container for matched track-flashes does not pass consistency check!");
            }

            std::cout << "Saving Matches - "
                      << "output_data.run=" << output_data.run 
                      << ", output_data.subrun=" << output_data.subrun 
                      << ", output_data.event=" << output_data.event 
                      << ", num_matches=" << num_matches_saves << std::endl;

            if ( config.output_hdf5 && hdf5_output_man ) {
                hdf5_output_man->storeEventVoxelData( output_data );
            }

            events_processed++;
            if ( num_matches_saves > 0) {
                events_with_matches++;
            }

            if (config.verbosity >= 1 && events_processed % 100 == 0) {
                std::cout << "Processed " << events_processed << " events..." << std::endl;
            }
        }

        

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // Print final statistics
    std::cout << std::endl;
    std::cout << "Processing Complete!" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Events processed: " << events_processed << std::endl;
    std::cout << "Events with matches: " << events_with_matches << std::endl;
    std::cout << "Processing time: " << duration.count() << " seconds" << std::endl;
    std::cout << std::endl;

    // Print component statistics
    if (config.verbosity >= 1) {
        // Print final statistics
        truth_track_matcher.PrintStatistics();
    }

    if ( config.output_hdf5 && hdf5_output_man ) {
        hdf5_output_man->close(); // write whats remaining in the batch and then close file
    }

    return 0;
}
