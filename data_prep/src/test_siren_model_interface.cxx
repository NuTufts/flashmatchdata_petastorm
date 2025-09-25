#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// ROOT headers
#include "TFile.h"
#include "TTree.h"

// HDF5 headers
#ifdef HAVE_HDF5
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#endif

// Torch headers
#include <torch/torch.h>

// Project headers
#include "SirenTorchModel.h"
#include "ModelInputInterface.h"

void printUsage() {
    std::cout << "Usage: test_siren_model_interface [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input HDF5_FILE    Input HDF5 file with flashmatch data" << std::endl;
    std::cout << "  -m, --model MODEL_FILE   Siren model file (.pt)" << std::endl;
    std::cout << "  -o, --output ROOT_FILE   Output ROOT file (default: test_siren_output.root)" << std::endl;
    std::cout << "  -n, --max-entries N      Maximum number of entries to process (default: all)" << std::endl;
    std::cout << "  -v, --verbose            Enable verbose output" << std::endl;
    std::cout << "  -h, --help               Show this help message" << std::endl;
}

int main( int nargs, char** argv ) {

    std::cout << "Test Siren Model Interface" << std::endl;

#ifndef HAVE_HDF5
    std::cerr << "Error: HDF5 support not compiled. Rebuild with HDF5." << std::endl;
    return 1;
#endif

    // Parse command line arguments
    std::string hdf5_file;
    std::string model_file;
    std::string output_file = "test_siren_output.root";
    int max_entries = -1;
    bool verbose = false;

    for (int i = 1; i < nargs; i++) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < nargs) {
                hdf5_file = argv[++i];
            }
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < nargs) {
                model_file = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < nargs) {
                output_file = argv[++i];
            }
        } else if (arg == "-n" || arg == "--max-entries") {
            if (i + 1 < nargs) {
                max_entries = std::stoi(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage();
            return 0;
        }
    }

    // Validate arguments
    if (hdf5_file.empty() || model_file.empty()) {
        std::cerr << "Error: Input HDF5 file and model file are required." << std::endl;
        printUsage();
        return 1;
    }

    try {
        // 1. Open HDF5 file
        std::cout << "Opening HDF5 file: " << hdf5_file << std::endl;
        HighFive::File h5file(hdf5_file, HighFive::File::ReadOnly);

        // Get the voxel_data group
        auto voxel_group = h5file.getGroup("voxel_data");

        // Count entries
        int num_entries = 0;
        while (voxel_group.exist("entry_" + std::to_string(num_entries))) {
            num_entries++;
        }
        std::cout << "Found " << num_entries << " entries in HDF5 file" << std::endl;

        if (num_entries == 0) {
            std::cerr << "Error: No entries found in HDF5 file" << std::endl;
            return 1;
        }

        if (max_entries > 0 && max_entries < num_entries) {
            num_entries = max_entries;
            std::cout << "Processing only first " << num_entries << " entries" << std::endl;
        }

        // 2. Create SirenTorchModel and load weights
        std::cout << "Loading Siren model from: " << model_file << std::endl;
        flashmatch::SirenTorchModel siren_model;
        if (verbose) {
            siren_model.set_verbosity(1);
        }
        siren_model.load_model_file(model_file);

        // Create ModelInputInterface for preparing input tensors
        flashmatch::ModelInputInterface input_interface;

        // Set normalization parameters (matching Python code)
        std::vector<float> planecharge_offset = {0.0, 0.0, 0.0};
        std::vector<float> planecharge_scale = {50000.0, 50000.0, 50000.0};
        input_interface.set_planecharge_normalization(planecharge_offset, planecharge_scale);
        input_interface.set_use_log_normalization(false);

        // 3. Create output ROOT file and TTree
        std::cout << "Creating output ROOT file: " << output_file << std::endl;
        TFile* outfile = new TFile(output_file.c_str(), "RECREATE");
        TTree* tree = new TTree("siren_inference", "Siren Model Inference Results");

        // Define tree branches
        int run, subrun, event, match_index, match_type;
        int num_voxels;
        float obs_pe_tot, pred_pe_tot, siren_pe_tot;
        std::vector<float> obs_pe_per_pmt(32);
        std::vector<float> pred_pe_per_pmt(32);
        std::vector<float> siren_pe_per_pmt(32);

        tree->Branch("run", &run);
        tree->Branch("subrun", &subrun);
        tree->Branch("event", &event);
        tree->Branch("match_index", &match_index);
        tree->Branch("match_type", &match_type);
        tree->Branch("num_voxels", &num_voxels);
        tree->Branch("obs_pe_tot", &obs_pe_tot);
        tree->Branch("pred_pe_tot", &pred_pe_tot);
        tree->Branch("siren_pe_tot", &siren_pe_tot);
        tree->Branch("obs_pe_per_pmt", &obs_pe_per_pmt);
        tree->Branch("pred_pe_per_pmt", &pred_pe_per_pmt);
        tree->Branch("siren_pe_per_pmt", &siren_pe_per_pmt);

        // 4. Loop over entries in HDF file
        for (int entry_idx = 0; entry_idx < num_entries; entry_idx++) {
            if (verbose || entry_idx % 100 == 0) {
                std::cout << "Processing entry " << entry_idx << "/" << num_entries << std::endl;
            }

            std::string entry_name = "entry_" + std::to_string(entry_idx);
            auto entry_group = voxel_group.getGroup(entry_name);

            // 5. Load arrays: planecharge, avepos
            std::vector<std::vector<float>> planecharge_vv =
                H5Easy::load<std::vector<std::vector<float>>>(h5file, "voxel_data/" + entry_name + "/planecharge");
            std::vector<std::vector<float>> avepos_vv =
                H5Easy::load<std::vector<std::vector<float>>>(h5file, "voxel_data/" + entry_name + "/avepos");

            // Load observed and predicted PE
            obs_pe_per_pmt = H5Easy::load<std::vector<float>>(h5file, "voxel_data/" + entry_name + "/observed_pe_per_pmt");
            pred_pe_per_pmt = H5Easy::load<std::vector<float>>(h5file, "voxel_data/" + entry_name + "/predicted_pe_per_pmt");

            // Load metadata
            run = H5Easy::loadAttribute<int>(h5file, "voxel_data/" + entry_name, "run");
            subrun = H5Easy::loadAttribute<int>(h5file, "voxel_data/" + entry_name, "subrun");
            event = H5Easy::loadAttribute<int>(h5file, "voxel_data/" + entry_name, "event");
            match_index = H5Easy::loadAttribute<int>(h5file, "voxel_data/" + entry_name, "match_index");

            // Load match_type if available
            match_type = H5Easy::load<int>(h5file, "voxel_data/" + entry_name + "/match_type");

            num_voxels = planecharge_vv.size();

            if (num_voxels == 0) {
                if (verbose) {
                    std::cout << "  Entry " << entry_idx << " has no voxels, skipping" << std::endl;
                }
                // Fill with zeros
                siren_pe_tot = 0;
                obs_pe_tot = 0;
                pred_pe_tot = 0;
                std::fill(siren_pe_per_pmt.begin(), siren_pe_per_pmt.end(), 0.0);
                tree->Fill();
                continue;
            }

            // 6. Prepare input tensors using ModelInputInterface
            // Convert to tensors
            torch::Tensor coord = torch::zeros({num_voxels, 3}, torch::kFloat32);
            torch::Tensor planecharge = torch::zeros({num_voxels, 3}, torch::kFloat32);

            for (int i = 0; i < num_voxels; i++) {
                for (int j = 0; j < 3; j++) {
                    coord[i][j] = avepos_vv[i][j];
                    planecharge[i][j] = planecharge_vv[i][j];
                }
            }

            // Normalize planecharge
            for (int plane = 0; plane < 3; plane++) {
                planecharge.index({torch::indexing::Slice(), plane}) =
                    (planecharge.index({torch::indexing::Slice(), plane}) + planecharge_offset[plane]) / planecharge_scale[plane];
            }

            // Prepare features for Siren model
            // The model expects (N*32, 7) for features and (N*32, 1) for charge
            // We need to manually create the input following prepare_mlp_input_variables logic

            // Convert coordinates from cm to normalized detector coordinates
            float voxel_len_cm = 1.0;
            torch::Tensor detpos = coord * voxel_len_cm;  // avepos already in cm in detector coordinates
            torch::Tensor detlens = torch::tensor({300.0, 300.0, 1500.0}, torch::kFloat32);
            detpos = detpos / detlens;  // Normalize

            // Get PMT positions from ModelInputInterface
            torch::Tensor pmtpos = input_interface.get_pmt_positions();

            // Create features for each voxel-PMT pair
            int npmt = 32;
            torch::Tensor features = torch::zeros({num_voxels * npmt, 7}, torch::kFloat32);
            torch::Tensor charge_tensor = torch::zeros({num_voxels * npmt, 1}, torch::kFloat32);

            // Calculate mean charge per voxel
            torch::Tensor q_mean = planecharge.mean(1, true);  // (N, 1)

            for (int vox_idx = 0; vox_idx < num_voxels; vox_idx++) {
                for (int pmt_idx = 0; pmt_idx < npmt; pmt_idx++) {
                    int idx = vox_idx * npmt + pmt_idx;

                    // Position features (normalized)
                    features[idx][0] = detpos[vox_idx][0].item<float>();
                    features[idx][1] = detpos[vox_idx][1].item<float>();
                    features[idx][2] = detpos[vox_idx][2].item<float>();

                    // Calculate relative position to PMT
                    float dx = coord[vox_idx][0].item<float>() * voxel_len_cm - pmtpos[pmt_idx][0].item<float>();
                    float dy = coord[vox_idx][1].item<float>() * voxel_len_cm - pmtpos[pmt_idx][1].item<float>();
                    float dz = coord[vox_idx][2].item<float>() * voxel_len_cm - pmtpos[pmt_idx][2].item<float>();

                    // Normalize relative positions
                    features[idx][3] = dx / detlens[0].item<float>();
                    features[idx][4] = dy / detlens[1].item<float>();
                    features[idx][5] = dz / detlens[2].item<float>();

                    // Distance (normalized)
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    features[idx][6] = dist / (210.0 * 5.0);

                    // Charge
                    charge_tensor[idx][0] = q_mean[vox_idx][0].item<float>();
                }
            }

            // 7. Run model
            std::vector<float> siren_output = siren_model.predict_pe(features, charge_tensor);

            // 8. & 9. Output is already summed over voxels and rescaled by predict_pe function
            // Copy to output vector
            siren_pe_per_pmt = siren_output;

            // Calculate totals
            siren_pe_tot = 0;
            obs_pe_tot = 0;
            pred_pe_tot = 0;
            for (int i = 0; i < 32; i++) {
                siren_pe_tot += siren_pe_per_pmt[i];
                obs_pe_tot += obs_pe_per_pmt[i];
                pred_pe_tot += pred_pe_per_pmt[i];
            }

            if (verbose) {
                std::cout << "  Entry " << entry_idx
                          << " (R" << run << ":S" << subrun << ":E" << event << ")"
                          << " nvoxels=" << num_voxels
                          << " obs_tot=" << obs_pe_tot
                          << " pred_tot=" << pred_pe_tot
                          << " siren_tot=" << siren_pe_tot << std::endl;
            }

            // 10. Save to TTree
            tree->Fill();
        }

        // Write and close files
        std::cout << "Writing output file..." << std::endl;
        outfile->cd();
        tree->Write();
        outfile->Close();
        delete outfile;

        std::cout << "Successfully processed " << num_entries << " entries" << std::endl;
        std::cout << "Output saved to: " << output_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}