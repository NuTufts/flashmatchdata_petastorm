/**
 * @file main.cxx
 * @brief Main program for flash-track matching data preparation
 * 
 * This program implements steps 2-3 of the data preparation pipeline:
 * - Apply quality cuts to cosmic ray tracks
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

// Local includes
#include "DataStructures.h"
#include "CosmicTrackSelector.h"
#include "FlashTrackMatcher.h"
#include "CRTMatcher.h"
#include "FlashMatchOutputData.h"
#include "LarliteDataInterface.h"
#include "CosmicRecoInput.h"

using namespace flashmatch::dataprep;

/**
 * @brief Program configuration structure
 */
struct ProgramConfig {
    std::string input_file;
    std::string output_file;
    std::string quality_cuts_config;
    std::string flash_matching_config;
    std::string debug_output_file;
    
    int max_events = -1;
    int start_event = 0;
    int verbosity = 1;
    bool debug_mode = false;
    bool enable_crt = true;
    
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
              << "  --output FILE             Output ROOT file with matched data\n\n"
              << "Optional Arguments:\n"
              << "  --config FILE             Quality cuts configuration (YAML)\n"
              << "  --flash-config FILE       Flash matching configuration (YAML)\n"
              << "  --debug-output FILE       Debug output file\n"
              << "  --max-events N            Maximum number of events to process\n"
              << "  --start-event N           Starting event number (default: 0)\n"
              << "  --verbosity N             Verbosity level 0-3 (default: 1)\n"
              << "  --debug                   Enable debug mode\n"
              << "  --no-crt                  Disable CRT matching\n"
              << "  --help                    Display this help message\n\n"
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
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
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
    
    if (config.output_file.empty()) {
        std::cerr << "Error: Output file is required" << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Load event data from ROOT file
 * @param file_path Path to input ROOT file
 * @param event_data Output event data structure
 * @param entry Entry number to load
 * @return true if successful
 */
bool LoadDummyData(std::string& file_path, EventData& event_data, int entry) {
    // TODO: Implement ROOT file reading
    // This would read the cosmic reconstruction output and populate EventData structure
    // For now, create dummy data for compilation

    event_data.run = 1;
    event_data.subrun = 1;
    event_data.event = entry;

    // Create dummy cosmic track
    CosmicTrack dummy_track;
    dummy_track.track_length = 100.0;
    dummy_track.total_charge = 1000.0;
    dummy_track.hit_density = 10.0;
    dummy_track.boundary_distance = 20.0;
    event_data.cosmic_tracks.push_back(dummy_track);
    
    // Create dummy optical flash
    OpticalFlash dummy_flash;
    dummy_flash.flash_time = 5.0;
    dummy_flash.total_pe = 500.0;
    event_data.optical_flashes.push_back(dummy_flash);
    
    return true;
}


/**
 * @brief Load event data from ROOT file
 * @param file_path Path to input ROOT file
 * @param event_data Output event data structure
 * @param entry Entry number to load
 * @return true if successful
 */
bool LoadEventData(std::string& file_path, EventData& event_data, int entry) {
    // TODO: Implement ROOT file reading
    // This would read the cosmic reconstruction output and populate EventData structure
    // For now, create dummy data for compilation

    event_data.run = 1;
    event_data.subrun = 1;
    event_data.event = entry;

    // Create dummy cosmic track
    // for (int i = 0; i < track_v.size(); i++) {
    //     track = track_v.at(i)
    //     int n_points = track.NumberTrajectoryPoints()
        

    // }

    return true;
}


/**
 * @brief Save processed event data to ROOT file
 * @param file_path Path to output ROOT file
 * @param event_data Event data to save
 * @return true if successful
 */
bool SaveEventData(std::string& file_path, EventData& event_data) {
    // TODO: Implement ROOT file writing
    // This would save the processed event data to output ROOT file
    std::cout << "Saving event data to " << file_path << std::endl;
    return true;
}

/**
 * @brief Process a single event
 */
bool ProcessEvent(EventData& input_data, 
                  EventData& output_data,
                  CosmicTrackSelector& track_selector,
                  FlashTrackMatcher& flash_matcher,
                  CRTMatcher& crt_matcher,
                  ProgramConfig& config) {
    


    if (config.verbosity >= 2) {
        std::cout << "Processing event " << input_data.run << ":" 
                  << input_data.subrun << ":" << input_data.event << std::endl;
        std::cout << "  Input tracks: " << input_data.cosmic_tracks.size() << std::endl;
        std::cout << "  Input flashes: " << input_data.optical_flashes.size() << std::endl;
    }

    // // Step 1: Apply quality cuts to tracks
    // for (auto& track : input_data.cosmic_tracks) {
    //     if (track_selector.PassesQualityCuts(track)) {
    //         output_data.cosmic_tracks.push_back(track);
    //     }
    // }

    // if (config.verbosity >= 2) {
    //     std::cout << "  Quality tracks: " << output_data.cosmic_tracks.size() << std::endl;
    // }
    
    // // Step 2: Perform flash-track matching
    // if (!output_data.cosmic_tracks.empty() && !input_data.optical_flashes.empty()) {
    //     output_data.flash_track_matches = flash_matcher.FindMatches(input_data);

    //     if (config.verbosity >= 2) {
    //         std::cout << "  Flash matches: " << output_data.flash_track_matches.size() << std::endl;
    //     }
    // }
    
    // Step X: CRT Matcher
    for ( int icrt_track=0; icrt_track<(int)input_data.crt_tracks.size(); icrt_track++ ) {
        std::cout << "CRT-TRACK[" << icrt_track << "] ================== " << std::endl;
        auto& crttrack = input_data.crt_tracks.at(icrt_track);

        std::cout << "  time: " << crttrack.startpt_time << " usec" << std::endl;

        crt_matcher.MatchToCRTTrack( crttrack, input_data.cosmic_tracks );
    }

    // // Update event-level statistics
    // output_data.num_quality_tracks = output_data.cosmic_tracks.size();
    // output_data.num_matched_flashes = output_data.flash_track_matches.size();

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
    std::cout << "Output file: " << config.output_file << std::endl;
    if (!config.quality_cuts_config.empty()) {
        std::cout << "Quality cuts config: " << config.quality_cuts_config << std::endl;
    }
    if (!config.flash_matching_config.empty()) {
        std::cout << "Flash matching config: " << config.flash_matching_config << std::endl;
    }
    std::cout << "CRT matching: " << (config.enable_crt ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    // Initialize processing components
    QualityCutConfig quality_config;
    FlashMatchConfig flash_config;

    // Load configurations if provided
    CosmicTrackSelector track_selector(quality_config);
    if (!config.quality_cuts_config.empty()) {
        if (!track_selector.LoadConfigFromFile(config.quality_cuts_config)) {
            std::cerr << "Warning: Could not load quality cuts config, using defaults" << std::endl;
        }
    }

    FlashTrackMatcher flash_matcher(flash_config);
    if (!config.flash_matching_config.empty()) {
        if (!flash_matcher.LoadConfigFromFile(config.flash_matching_config)) {
            std::cerr << "Warning: Could not load flash matching config, using defaults" << std::endl;
        }
    }

    CRTMatcher crt_matcher;

    // Process events
    int events_processed = 0;
    int events_with_matches = 0;

    // Load the input file
    CosmicRecoInput cosmic_reco_input_file( config.input_file );

    std::cout << "Loaded input file. Number of entries: " << cosmic_reco_input_file.get_num_entries() << std::endl;

    // TODO: Implement proper event loop over ROOT file
    // For now, process a dummy set of events
    int total_events = (config.max_events > 0) ? config.max_events : cosmic_reco_input_file.get_num_entries();
    int end_entry = config.start_event + total_events;
    if ( end_entry > cosmic_reco_input_file.get_num_entries() )
        end_entry = cosmic_reco_input_file.get_num_entries();

    // Define the output file
    FlashMatchOutputData output_file( config.output_file, false ); 

    std::cout << "Starting Event Loop" << std::endl;
    std::cout << "Start entry: " << config.start_event << std::endl;
    std::cout << "End entry: " << end_entry << std::endl;

    for (int entry = config.start_event; entry < end_entry; ++entry) {
        
        std::cout << "[ENTRY " << entry << "]" << std::endl;

        cosmic_reco_input_file.load_entry( entry );

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

    //     // Load event data
    //     if (!LoadEventData(config.input_file, input_data, entry)) {
    //         std::cerr << "Error loading event " << entry << std::endl;
    //         continue;
    //     }

        EventData output_data;

        // Process event
        if (ProcessEvent(input_data, output_data, track_selector, flash_matcher, 
                        crt_matcher, config)) {

            // Save processed data
            int num_matches_saves = output_file.saveEventMatches();

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
        std::cout << "Quality Cut Statistics:" << std::endl;
        track_selector.PrintStatistics();
        std::cout << std::endl;

        std::cout << "Flash Matching Statistics:" << std::endl;
        flash_matcher.PrintStatistics();
        std::cout << std::endl;

        if (config.enable_crt) {
            std::cout << "CRT Matching Statistics:" << std::endl;
            crt_matcher.PrintStatistics();
            std::cout << std::endl;
        }
    }

    output_file.writeTree();
    output_file.closeFile();

    std::cout << "Output saved to: " << config.output_file << std::endl;

    return 0;
}