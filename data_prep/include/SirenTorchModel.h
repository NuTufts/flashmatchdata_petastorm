#ifndef __FLASHMATCH_SIREN_TORCH_MODEL_H__
#define __FLASHMATCH_SIREN_TORCH_MODEL_H__

// Prevent the ROOT Interpretter from parsing anything in this header
#ifndef __CINT__
#ifndef __CLING__

#include <string>
#include <torch/torch.h>
#include <torch/script.h>

namespace flashmatch {

    class SirenTorchModel {

    public:

        SirenTorchModel();
        ~SirenTorchModel() {};

        void set_verbosity( int v ) { _verbosity=v; };
        int load_model_file( std::string model_filepath );
        std::vector<float> predict_pe( torch::Tensor& features, torch::Tensor& charge );
        bool is_loaded() const { return !_model_filepath.empty(); }


    protected:

        int _verbosity;

        torch::jit::script::Module _model;
        std::string _model_filepath;

    };

}


#endif
#endif

#endif