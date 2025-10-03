#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "UnbalancedSinkhornDivergence.h"
#include "UBFlashSinkDiv.h"
#include "PMTPositions.h"

using namespace flashmatch::sinkhorn;

// Generate random positions for PMTs in a cylindrical detector geometry
torch::Tensor generate_pmt_positions(int n_pmts, float dist_scale=1.0 ) {
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> theta_dist(0.0, 2.0 * M_PI);
    // std::uniform_real_distribution<float> z_dist(-100.0, 100.0);  // cm

    // auto positions = torch::zeros({n_pmts, 3});
    // const float radius = 128.0;  // MicroBooNE PMT radius in cm

    // for (int i = 0; i < n_pmts; ++i) {
    //     float theta = theta_dist(gen);
    //     float z = z_dist(gen);

    //     positions[i][0] = radius * std::cos(theta);  // x
    //     positions[i][1] = radius * std::sin(theta);  // y
    //     positions[i][2] = z;                         // z
    // }

    auto positions = torch::zeros({n_pmts, 3});

    for (int i = 0; i < n_pmts; ++i) {
        auto pmtpos = flashmatch::PMTPositions::getOpDetPos( i );
        for (int j=0; j<3; j++)
            positions[i][j] = pmtpos[j]/dist_scale;
    }


    return positions;
}

void generate_fixed_masses(float scale, torch::Tensor& masses_a, torch::Tensor& masses_b ) {

    masses_a = torch::zeros({32});
    masses_b = torch::zeros({32});
    
    masses_a[0]  = 1.0486168;
    masses_a[1]  = 1.0491255;
    masses_a[2]  = 1.0487175;
    masses_a[3]  = 1.0486918;
    masses_a[4]  = 1.0484792;
    masses_a[5]  = 1.0487355;
    masses_a[6]  = 1.0484391;
    masses_a[7]  = 23.112646;
    masses_a[8]  = 45.717140;
    masses_a[9]  = 2.8149819;
    masses_a[10] = 101.49279;
    masses_a[11] = 16.468011;
    masses_a[12] = 42.669189;
    masses_a[13] = 181.69705;
    masses_a[14] = 109.77285;
    masses_a[15] = 258.28912;
    masses_a[16] = 52.197849;
    masses_a[17] = 1.4032447;
    masses_a[18] = 99.118621;
    masses_a[19] = 1.0487263;
    masses_a[20] = 1.0484359;
    masses_a[21] = 1.8938610;
    masses_a[22] = 1.0485358;
    masses_a[23] = 1.0486611;
    masses_a[24] = 1.0484833;
    masses_a[25] = 1.0484889;
    masses_a[26] = 1.0484372;
    masses_a[27] = 1.0483601;
    masses_a[28] = 1.0483439;
    masses_a[29] = 1.0485112;
    masses_a[30] = 1.0485589;
    masses_a[31] = 1.0483729;

    masses_b[0]  =  0; 
    masses_b[1]  =  0; 
    masses_b[2]  =  0; 
    masses_b[3]  =  0; 
    masses_b[4]  =  0; 
    masses_b[5]  =  0; 
    masses_b[6]  =  0; 
    masses_b[7]  =  36.42494;
    masses_b[8]  =  0; 
    masses_b[9]  =  0; 
    masses_b[10] =  96.241447;
    masses_b[11] =  28.675005;
    masses_b[12] =  20.811662;
    masses_b[13] =  165.14990;
    masses_b[14] =  106.81942;
    masses_b[15] =  288.03347;
    masses_b[16] =  59.700992;
    masses_b[17] =  0;
    masses_b[18] =  84.7147;
    masses_b[19] =  0;
    masses_b[20] =  0;
    masses_b[21] =  0;
    masses_b[22] =  0;
    masses_b[23] =  0;
    masses_b[24] =  0;
    masses_b[25] =  0;
    masses_b[26] =  0;
    masses_b[27] =  0;
    masses_b[28] =  0;
    masses_b[29] =  0;
    masses_b[30] =  0;
    masses_b[31] =  0;

    masses_b += 1.0e-5;

    masses_a /= scale;
    masses_b /= scale;

}

// Compute squared Euclidean distance matrix
torch::Tensor compute_cost_matrix(const torch::Tensor& x, const torch::Tensor& y) {
    // x: (N, D), y: (M, D)
    // Returns: (N, M) matrix of squared distances

    auto N = x.size(0);
    auto M = y.size(0);
    auto D = x.size(1);

    auto x_expanded = x.unsqueeze(1).expand({N, M, D});  // (N, M, D)
    auto y_expanded = y.unsqueeze(0).expand({N, M, D});  // (N, M, D)

    auto diff = x_expanded - y_expanded;
    return 0.5*torch::sum(diff * diff, 2);  // (N, M)
}

int main() {
    std::cout << "Testing UnbalancedSinkhornDivergence with 32-length random mass vectors\n";
    std::cout << "====================================================================\n";

    // Set random seed for reproducibility
    torch::manual_seed(42);

    const int n_pmts = 32;  // MicroBooNE has 32 PMTs

    // Generate two random mass vectors (simulating PMT photoelectron counts)
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::exponential_distribution<float> mass_dist(1.0);  // Exponential distribution for PE counts

    // auto masses_a = torch::zeros({n_pmts});
    // auto masses_b = torch::zeros({n_pmts});

    // for (int i = 0; i < n_pmts; ++i) {
    //     masses_a[i] = mass_dist(gen) + 0.1;  // Add small offset to avoid zeros
    //     masses_b[i] = mass_dist(gen) + 0.1;
    // }

    flashmatch::UBFlashSinkDiv ubsinkdiv_algo;

    auto masses_a = torch::zeros({n_pmts});
    auto masses_b = torch::zeros({n_pmts});
    generate_fixed_masses( 5000.0, masses_a, masses_b );
    masses_a *= 1.3;

    std::cout << "Generated random mass vectors:\n";
    std::cout << "masses_a: " << masses_a << "\n";
    std::cout << "masses_b: " << masses_b << "\n\n";

    // Normalize masses to make them proper probability measures
    auto pdf_a = masses_a / torch::sum(masses_a);
    auto pdf_b = masses_b / torch::sum(masses_b);

    std::cout << "pdf_a: " << pdf_a << std::endl;
    std::cout << "pdf_b: " << pdf_b << std::endl;

    std::cout << "Normalized masses (sum = 1):\n";
    std::cout << "sum(masses_a) = " << torch::sum(masses_a).item<double>() << "\n";
    std::cout << "sum(masses_b) = " << torch::sum(masses_b).item<double>() << "\n\n";

    // Generate PMT positions
    auto pmt_positions = generate_pmt_positions(n_pmts, 1000.0);
    std::cout << "Generated PMT positions:\n";
    std::cout << pmt_positions << "\n\n";

    // Compute cost matrices (squared distances between PMT positions)
    auto C_xx = compute_cost_matrix(pmt_positions, pmt_positions);
    auto C_yy = C_xx;  // Same positions for both measures
    auto C_xy = C_xx;
    auto C_yx = C_xx;

    std::cout << "Cost matrix shape: " << C_xx.sizes() << "\n";
    std::cout << "Max distance squared: " << torch::max(C_xx).item<double>() << " cm²\n\n";

    // Compute log weights
    auto a_log     = UnbalancedSinkhornDivergence::log_weights(masses_a);
    auto b_log     = UnbalancedSinkhornDivergence::log_weights(masses_b);
    auto pdf_a_log = UnbalancedSinkhornDivergence::log_weights(pdf_a);
    auto pdf_b_log = UnbalancedSinkhornDivergence::log_weights(pdf_b);

    std::cout << "log(pdf_a): " << pdf_a_log << std::endl;
    std::cout << "log(pdf_b): " << pdf_b_log << std::endl;

    // Set up epsilon schedule for annealing
    //double diameter = std::sqrt(UnbalancedSinkhornDivergence::max_diameter(pmt_positions, pmt_positions));
    double diameter = 0.9466; // from python code
    std::cout << "Point cloud diameter: " << diameter << " cm\n";

    auto eps_schedule = UnbalancedSinkhornDivergence::epsilon_schedule(
        2.0,     // p=2 for squared Euclidean distance
        diameter,
        0.05,     // target blur scale
        0.5      // scaling factor
    );

    std::cout << "Epsilon annealing schedule: ";
    for (double eps : eps_schedule) {
        std::cout << eps << " ";
    }
    std::cout << "\n\n";

    // Test balanced optimal transport (rho = nullptr)
    std::cout << "Testing balanced optimal transport...\n";
    std::cout << "Using official softmin_tensorized from geomloss\n";
    auto balanced_result = UnbalancedSinkhornDivergence::sinkhorn_loop_single_scale(
        UnbalancedSinkhornDivergence::softmin_tensorized,
        pdf_a_log,
        pdf_b_log,
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_schedule,
        nullptr,  // balanced case
        true      // debias
    );
    auto f_aa_bal = std::get<0>(balanced_result);
    auto g_bb_bal = std::get<1>(balanced_result);
    auto g_ab_bal = std::get<2>(balanced_result);
    auto f_ba_bal = std::get<3>(balanced_result);

    auto balanced_cost = UnbalancedSinkhornDivergence::sinkhorn_cost(
        eps_schedule.back(),  // final epsilon
        nullptr,              // balanced
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

    std::cout << "Balanced Sinkhorn divergence: " << balanced_cost.item<double>() << "\n\n";

    // Test if interface implementation works
    std::vector< float > fpe_a_v(32);
    std::vector< float > fpe_b_v(32);
    for (size_t ipmt=0; ipmt<32; ipmt++) {
        fpe_a_v[ipmt] = masses_a[ipmt].item<float>();
        fpe_b_v[ipmt] = masses_b[ipmt].item<float>();
    }

    double ubsinkdiv_balanced = ubsinkdiv_algo.calc( fpe_a_v, fpe_b_v, true );
    std::cout << "Balanced Sinkhorn divergence (from UBSinkDiv): "  << ubsinkdiv_balanced << '\n' << std::endl;

    // Test unbalanced optimal transport
    std::cout << "Testing unbalanced optimal transport...\n";
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

    std::cout << "Unbalanced Sinkhorn divergence (rho=" << rho << "): " << unbalanced_cost.item<double>() << "\n\n";

    double ubsinkdiv_unbalanced = ubsinkdiv_algo.calc( fpe_a_v, fpe_b_v, false );
    std::cout << "Unbalanced Sinkhorn divergence (from UBSinkDiv): "  << ubsinkdiv_unbalanced << '\n' << std::endl;

    // Test dampening factor
    for (double eps : {0.1, 1.0, 10.0}) {
        double tau = UnbalancedSinkhornDivergence::dampening(eps, &rho);
        std::cout << "Dampening factor for eps=" << eps << ", rho=" << rho << ": " << tau << "\n";
    }

    std::cout << "\nTest completed successfully!\n";

    return 0;
}