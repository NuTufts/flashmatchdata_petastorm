#ifndef __FLASHMATCH_SIREN_TORCH_MODEL_H__
#define __FLASHMATCH_SIREN_TORCH_MODEL_H__

// Prevent the ROOT Interpretter from parsing anything in this header
#ifndef __CINT__
#ifndef __CLING__

#include <string>
#include <torch/script.h>

namespace flashmatch {

    class SirenTorchModel {

    public:

        SirenTorchModel();
        ~SirenTorchModel() {};

        void set_verbosity( int v ) { _verbosity=v; };
        int load_model_file( std::string model_filepath );


    protected:

        int _verbosity;

        torch::jit::script::Module _model;
        std::string _model_filepath;

    };

}


#endif
#endif

#endif