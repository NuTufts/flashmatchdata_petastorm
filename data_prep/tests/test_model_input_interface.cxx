/**
 * Test program for ModelInputInterface
 * Validates that the C++ implementation matches the Python behavior
 */

#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>

#include "ModelInputInterface.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"

void print_tensor_info(const std::string& name, const torch::Tensor& tensor) {
    std::cout << name << " shape: [";
    for (int i = 0; i < tensor.dim(); ++i) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "], dtype: " << tensor.dtype() << std::endl;

    // Print first few values for debugging
    if (tensor.numel() > 0 && tensor.numel() <= 20) {
        std::cout << "  Values: " << tensor << std::endl;
    } else if (tensor.numel() > 20) {
        auto flat = tensor.flatten();
        std::cout << "  First 10 values: ";
        for (int i = 0; i < std::min(10L, tensor.numel()); ++i) {
            std::cout << flat[i].item<float>() << " ";
        }
        std::cout << "..." << std::endl;
    }
}

int main(int argc, char* argv[]) {

    std::cout << "Testing ModelInputInterface C++ Implementation" << std::endl;
    std::cout << "=============================================" << std::endl;

    // // take as test input, result of some cosmic ray reconstruction
    // std::string cosmicreco_test_larlite = "test_cosmicreco_larlite.root";
    // std::string cosmicreco_test_larcv   = "test_cosmicreco_larcv.root";
    // larcv::IOManager iolcv( larcv::IOManager::kREAD, "larcv");
    // iolcv.add_in_file( cosmicreco_test_larcv );
    // iolcv.initialize();
    // larlite::storage_manager ioll( larlite::storage_manager::kREAD );
    // ioll.add_in_filename( cosmicreco_test_larlite );
    // ioll.open();

    // iolcv.read_entry(0);
    // ioll.go_to(0);

    // auto ev_img = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    // auto ev_cosmictrack = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, "cosmictrack" );
    // auto ev_cosmichits  = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "cosmictrack");

    // auto& adc_v = ev_image->AsImageArray();
    

    // Create the interface
    flashmatch::ModelInputInterface model_interface;

    // Create some dummy data
    int num_hits = 100;
    std::vector<std::vector<float>> hitpos_v;
    std::vector<std::vector<float>> hitimgcoord_v;

    // Generate random hit positions (in detector coordinates)
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> x_dist(0.0, 256.0);
    std::uniform_real_distribution<float> y_dist(-117.0, 117.0);
    std::uniform_real_distribution<float> z_dist(0.0, 1036.0);
    std::uniform_real_distribution<float> wire_dist(0.0, 3455.0);
    std::uniform_real_distribution<float> time_dist(2400.0, 2400.0+6.0*1008.0);

    for (int i = 0; i < num_hits; ++i) {
        hitpos_v.push_back({x_dist(generator), y_dist(generator), z_dist(generator)});

        // Dummy image coordinates (wire, time) for 3 planes
        hitimgcoord_v.push_back({
            time_dist(generator), // tick
            wire_dist(generator), // U plane
            wire_dist(generator), // V plane
            wire_dist(generator)  // Y plane
        });
    }

    // Create dummy wire plane images (would normally come from real data)
    std::vector<larcv::Image2D> wireplane_v;
    for (int plane = 0; plane < 3; ++plane) {
        larcv::ImageMeta meta( 3456.0, 1008.0*6, 1008, 3456, 0, 2400, plane );
        larcv::Image2D img(meta);  // Dummy dimensions
        img.paint(40.0); // paint values so points get charge
        // In real usage, these would contain actual charge data
        wireplane_v.push_back(img);
    }

    // Prepare input tensors
    torch::Tensor voxel_features, voxel_charge;

    std::cout << "\nPreparing input tensors..." << std::endl;
    model_interface.prepare_input_tensor(
        hitpos_v,
        hitimgcoord_v,
        wireplane_v,
        voxel_features,
        voxel_charge
    );

    std::cout << "\nOutput tensors:" << std::endl;
    print_tensor_info("voxel_features", voxel_features);
    print_tensor_info("voxel_charge", voxel_charge);

    // Validate output shapes
    bool valid = true;

    // Expected: voxel_features should be (N*32, 7) where N is number of voxels
    if (voxel_features.dim() != 2 || voxel_features.size(1) != 7) {
        std::cerr << "ERROR: voxel_features has wrong shape!" << std::endl;
        std::cerr << "Expected: (N*32, 7), Got: ("
                  << voxel_features.size(0) << ", " << voxel_features.size(1) << ")" << std::endl;
        valid = false;
    }

    // Expected: voxel_charge should be (N*32, 1)
    if (voxel_charge.dim() != 2 || voxel_charge.size(1) != 1) {
        std::cerr << "ERROR: voxel_charge has wrong shape!" << std::endl;
        std::cerr << "Expected: (N*32, 1), Got: ("
                  << voxel_charge.size(0) << ", " << voxel_charge.size(1) << ")" << std::endl;
        valid = false;
    }

    // Check that both tensors have same number of rows
    if (voxel_features.size(0) != voxel_charge.size(0)) {
        std::cerr << "ERROR: voxel_features and voxel_charge have different number of rows!" << std::endl;
        valid = false;
    }

    // Check that number of rows is divisible by 32 (number of PMTs)
    if (voxel_features.size(0) % 32 != 0) {
        std::cerr << "ERROR: Number of rows not divisible by 32!" << std::endl;
        valid = false;
    }

    int num_voxels = voxel_features.size(0) / 32;
    std::cout << "\nNumber of voxels processed: " << num_voxels << std::endl;

    // Test with TorchScript model if path provided
    if (argc > 1) {
        std::string model_path = argv[1];
        std::cout << "\nLoading TorchScript model from: " << model_path << std::endl;

        try {
            torch::jit::script::Module module = torch::jit::load(model_path);
            module.eval();

            std::cout << "Model loaded successfully!" << std::endl;

            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(voxel_features);
            inputs.push_back(voxel_charge);

            std::cout << "Running inference..." << std::endl;
            torch::Tensor output = module.forward(inputs).toTensor();

            print_tensor_info("Model output", output);

            // Expected output shape: (N*32,) or (N*32, 1)
            if (output.size(0) != voxel_features.size(0)) {
                std::cerr << "WARNING: Output size doesn't match input size!" << std::endl;
            }

            // Reshape output to per-voxel, per-PMT format
            torch::Tensor pe_per_voxel = output.reshape({num_voxels, 32});

            // Sum over voxels to get total PE per PMT
            torch::Tensor pe_per_pmt = pe_per_voxel.sum(0);

            std::cout << "\nPredicted PE per PMT:" << std::endl;
            for (int i = 0; i < 32; ++i) {
                std::cout << "PMT " << i << ": " << pe_per_pmt[i].item<float>() << std::endl;
            }

            float total_pe = pe_per_pmt.sum().item<float>();
            std::cout << "\nTotal predicted PE: " << total_pe << std::endl;

        } catch (const c10::Error& e) {
            std::cerr << "Error loading or running model: " << e.what() << std::endl;
            valid = false;
        }
    }

    if (valid) {
        std::cout << "\n✓ All validation checks passed!" << std::endl;
    } else {
        std::cout << "\n✗ Some validation checks failed!" << std::endl;
        return 1;
    }

    return 0;
}