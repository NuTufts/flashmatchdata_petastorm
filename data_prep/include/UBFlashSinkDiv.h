#ifndef __FLASHMATCH_UBFLASHSINKDIV_H__
#define __FLASHMATCH_UBFLASHSINKDIV_H__

#include <vector>
#include <torch/torch.h>

namespace flashmatch {

class UBFlashSinkDiv {

public:

    UBFlashSinkDiv() {};
    ~UBFlashSinkDiv() {};

    double calc( const std::vector<float>& pe_a, const std::vector<float>& pe_b, 
                    bool balanced=true, float length_scale_cm=1000.0 );

    torch::Tensor compute_cost_matrix(const torch::Tensor& x, const torch::Tensor& y);

    torch::Tensor generate_pmt_positions(int n_pmts, float dist_scale=1.0 );

};

}

#endif