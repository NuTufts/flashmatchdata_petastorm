#include "SirenTorchModel.h"

#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
#include <c10/util/TypeIndex.h>

#include  "PrepareVoxelOutput.h"

namespace flashmatch {
    
SirenTorchModel::SirenTorchModel()
: _verbosity(0)
{}

int SirenTorchModel::load_model_file( std::string model_filepath )
{

    _model_filepath = model_filepath;

    try {
        // Load the model
        _model = torch::jit::load(model_filepath);
        _model.eval();
        
        std::cout << "[SirenTorchModel::load_model_file] "
            << " loaded from " << _model_filepath << std::endl;
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }

    return 0;
}


std::vector<float> SirenTorchModel::predict_pe( torch::Tensor& features, torch::Tensor& charge )
{
    // Check if model is loaded
    if (!is_loaded()) {
        throw std::runtime_error("Model not loaded. Call load_model_file() first.");
    }

    // Ensure tensors are on the correct device and in eval mode
    torch::NoGradGuard no_grad;

    // Validate input shapes
    if (features.dim() != 2 || features.size(1) != 7) {
        throw std::runtime_error("Features tensor must be shape (N*32, 7)");
    }
    if (charge.dim() != 2 || charge.size(1) != 1) {
        throw std::runtime_error("Charge tensor must be shape (N*32, 1)");
    }
    if (features.size(0) != charge.size(0)) {
        throw std::runtime_error("Features and charge must have same number of rows");
    }
    if (features.size(0) % 32 != 0) {
        throw std::runtime_error("Number of rows must be divisible by 32 (number of PMTs)");
    }

    std::cout << "Inputs to SirenTorchModel: " << std::endl;
    std::cout << "  " << features.sizes() << std::endl;
    std::cout << "  " << charge.sizes() << std::endl;

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(features);
    inputs.push_back(charge);

    // number of voxels - use size() instead of shape
    int num_voxels = features.size(0) / 32;  // features is (N*32, 7) where N is num_voxels

    // Fix typo: three colons should be two, and variable name should be consistent
    torch::Tensor out_pe_per_voxel = _model.forward(inputs).toTensor();
    out_pe_per_voxel = out_pe_per_voxel.reshape({num_voxels, 32});
    std::cout << "SirenTorchModel output: " << out_pe_per_voxel.sizes() << std::endl;

    // Sum over voxels to get total PE per PMT
    torch::Tensor pe_per_pmt = out_pe_per_voxel.sum(0);

    std::vector<float> pe_per_pmt_v(32,0);

    for (int ipmt=0; ipmt<32; ipmt++) {
        pe_per_pmt_v[ipmt] = 1000.0*pe_per_pmt[ipmt].item<float>(); // 1000.0 comes from scale factor used in training
    }

    return pe_per_pmt_v;

}





}