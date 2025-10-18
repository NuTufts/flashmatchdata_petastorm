#include "UBFlashSinkDiv.h"

#include "UnbalancedSinkhornDivergence.h"
#include "PMTPositions.h"

using namespace flashmatch::sinkhorn;

namespace flashmatch {

/** 
 * @brief get the UB PMT Position tensor
 * 
 */
torch::Tensor UBFlashSinkDiv::generate_pmt_positions(int n_pmts, float dist_scale ) {

    auto positions = torch::zeros({n_pmts, 3});

    for (int i = 0; i < n_pmts; ++i) {
        auto pmtpos = flashmatch::PMTPositions::getOpDetPos( i );
        for (int j=0; j<3; j++)
            positions[i][j] = pmtpos[j]/dist_scale;
    }

    return positions;
}


// Compute squared Euclidean distance matrix
torch::Tensor UBFlashSinkDiv::compute_cost_matrix(const torch::Tensor& x, const torch::Tensor& y, int p) {
    // x: (N, D), y: (M, D)
    // Returns: (N, M) matrix of squared distances

    auto N = x.size(0);
    auto M = y.size(0);

    if (p!=1 && p!=2) {
        throw std::runtime_error("Error [UBFlashSinkDiv::compute_cost_matrix]: p must be 1 or 2");
    }

    torch::Tensor C = torch::zeros( {N,M} );

    for (int i=0; i<N; i++) {
        auto x_pos = x.index( {i,torch::indexing::Slice()} );
        //std::cout << "cost_matrix x_pos [" << i << "]: " << x_pos << std::endl;
        for (int j=0; j<M; j++) {
            auto y_pos = y.index( {j,torch::indexing::Slice()} );
            auto diff = x_pos-y_pos;
            if (p==2) {
                auto d = 0.5*torch::sum(diff*diff);
                C.index( {i,j} ) = d;
            }
            else if ( p==1 ) {
                auto d = torch::sqrt(torch::sum(diff*diff));
                C.index( {i,j} ) = d;
            }
        }
    }

    // std::cout << "cos matrix [p=" << p << "]"  << std::endl;
    // std::cout << C << std::endl;

    return C;
}


double UBFlashSinkDiv::calc( const std::vector<float>& pe_a, const std::vector<float>& pe_b, 
                                bool balanced, float length_scale_cm, int p ) 
{

    const int n_pmts = 32;  // MicroBooNE has 32 PMTs
    if ( pe_a.size()!=n_pmts) {
        throw std::runtime_error("[UBFlashSinkDiv::calc] number of entries in pe_a is not 32, the number of UB PMTs");
    }
    if ( pe_b.size()!=n_pmts) {
        throw std::runtime_error("[UBFlashSinkDiv::calc] number of entries in pe_b is not 32, the number of UB PMTs");
    }

    auto masses_a = torch::zeros({n_pmts});
    auto masses_b = torch::zeros({n_pmts});

    for (size_t ipmt=0; ipmt<n_pmts; ipmt++) {
        masses_a[ipmt] = pe_a[ipmt];
        masses_b[ipmt] = pe_b[ipmt];
    }

    // Normalize masses to make them proper probability measures
    auto pdf_a = masses_a / torch::sum(masses_a);
    auto pdf_b = masses_b / torch::sum(masses_b);

    // Generate PMT positions
    auto pmt_positions = generate_pmt_positions(n_pmts, length_scale_cm);

    // Compute cost matrices (squared distances between PMT positions)
    auto C_xx = compute_cost_matrix(pmt_positions, pmt_positions, p);
    auto C_yy = C_xx;  // Same positions for both measures
    auto C_xy = C_xx;
    auto C_yx = C_xx;

    // std::cout << "Cost matrix shape: " << C_xx.sizes() << "\n";
    // std::cout << "Max distance squared: " << torch::max(C_xx).item<double>() << " cm²\n\n";

    // Compute log weights
    auto a_log     = UnbalancedSinkhornDivergence::log_weights(masses_a);
    auto b_log     = UnbalancedSinkhornDivergence::log_weights(masses_b);
    auto pdf_a_log = UnbalancedSinkhornDivergence::log_weights(pdf_a);
    auto pdf_b_log = UnbalancedSinkhornDivergence::log_weights(pdf_b);

    // std::cout << "log(pdf_a): " << pdf_a_log << std::endl;
    // std::cout << "log(pdf_b): " << pdf_b_log << std::endl;

    // Set up epsilon schedule for annealing
    //double diameter = std::sqrt(UnbalancedSinkhornDivergence::max_diameter(pmt_positions, pmt_positions));
    double diameter = 0.9466; // from python code
    //std::cout << "Point cloud diameter: " << diameter << " cm\n";

    auto eps_schedule = UnbalancedSinkhornDivergence::epsilon_schedule(
        2.0,     // p=2 for squared Euclidean distance
        diameter,
        0.05,     // target blur scale
        0.5      // scaling factor
    );

    // std::cout << "Epsilon annealing schedule: ";
    // for (double eps : eps_schedule) {
    //     std::cout << eps << " ";
    // }
    // std::cout << "\n\n";

    

    if ( balanced ) {

        double* rho = nullptr; // balanced case

        // // Test balanced optimal transport (rho = nullptr)
        // std::cout << "Testing balanced optimal transport...\n";
        // std::cout << "Using official softmin_tensorized from geomloss\n";
        auto balanced_result = UnbalancedSinkhornDivergence::sinkhorn_loop_single_scale(
            UnbalancedSinkhornDivergence::softmin_tensorized,
            pdf_a_log,
            pdf_b_log,
            C_xx,
            C_yy,
            C_xy,
            C_yx,
            eps_schedule,
            rho,
            true  // debias
        );
        auto f_aa_bal = std::get<0>(balanced_result);
        auto g_bb_bal = std::get<1>(balanced_result);
        auto g_ab_bal = std::get<2>(balanced_result);
        auto f_ba_bal = std::get<3>(balanced_result);

        auto balanced_cost = UnbalancedSinkhornDivergence::sinkhorn_cost(
            eps_schedule.back(),  // final epsilon
            rho,             
            pdf_a,
            pdf_b,
            f_aa_bal,
            g_bb_bal,
            g_ab_bal,
            f_ba_bal,
            false,  // not batch
            true,   // debias
            false   // return cost, not potentials
        );

        //std::cout << "Balanced Sinkhorn divergence: " << balanced_cost.item<double>() << "\n\n";

        return balanced_cost.item<double>();

    }
    else {

        // unbalanced optimal transport
        //std::cout << "Testing unbalanced optimal transport...\n";
        double rho = 1.0;  // Marginal constraint strength

        auto unbalanced_result = UnbalancedSinkhornDivergence::sinkhorn_loop_single_scale(
            UnbalancedSinkhornDivergence::softmin_tensorized,
            a_log,
            b_log,
            C_xx,
            C_yy,
            C_xy,
            C_yx,
            eps_schedule,
            &rho,    // unbalanced case
            true     // debias
        );
        auto f_aa_unbal = std::get<0>(unbalanced_result);
        auto g_bb_unbal = std::get<1>(unbalanced_result);
        auto g_ab_unbal = std::get<2>(unbalanced_result);
        auto f_ba_unbal = std::get<3>(unbalanced_result);

        auto unbalanced_cost = UnbalancedSinkhornDivergence::sinkhorn_cost(
            eps_schedule.back(),  // final epsilon
            &rho,                 // unbalanced
            masses_a,
            masses_b,
            f_aa_unbal,
            g_bb_unbal,
            g_ab_unbal,
            f_ba_unbal,
            false,  // not batch
            true,   // debias
            false   // return cost, not potentials
        );

        //std::cout << "Unbalanced Sinkhorn divergence (rho=" << rho << "): " << unbalanced_cost.item<double>() << "\n\n";
        return unbalanced_cost.item<double>();
    }

    return -1.0;

}


}