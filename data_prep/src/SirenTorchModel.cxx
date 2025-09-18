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




}