#include "UnbalancedSinkhornDivergence.h"
#include <torch/torch.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace flashmatch {
namespace sinkhorn {

UnbalancedSinkhornDivergence::UnbalancedSinkhornDivergence() {}

UnbalancedSinkhornDivergence::~UnbalancedSinkhornDivergence() {}

double UnbalancedSinkhornDivergence::dampening(double eps, const double* rho) {
    if (rho == nullptr) {
        return 1.0;  // Balanced case
    }
    return 1.0 / (1.0 + eps / (*rho));
}

torch::Tensor UnbalancedSinkhornDivergence::log_weights(const torch::Tensor& a) {
    // Clamp values to avoid numerical issues, similar to Python implementation
    torch::Tensor a_clamped = torch::clamp_min(a, 1e-10);
    return torch::log(a_clamped);
}

torch::Tensor UnbalancedSinkhornDivergence::unbalanced_weight(
    const torch::Tensor& x,
    double eps,
    double rho,
    bool backward
) {
    if (backward) {
        return (rho + eps)*x;
    } else {
        return (rho + eps/2.0)*x;
    }
}

torch::Tensor UnbalancedSinkhornDivergence::scal(
    const torch::Tensor& a,
    const torch::Tensor& f,
    bool batch
) {
    if (batch) {
        auto B = a.size(0);
        auto a_reshaped = a.reshape({B, -1});
        auto f_reshaped = f.reshape({B, -1});
        return (a_reshaped * f_reshaped).sum(1);
    } else {
        return torch::dot(a.reshape(-1), f.reshape(-1));
    }
}

double UnbalancedSinkhornDivergence::max_diameter(const torch::Tensor& x, const torch::Tensor& y) {
    // Compute upper bound on maximum distance between points
    auto x_min = std::get<0>(torch::min(x, 0));
    auto x_max = std::get<0>(torch::max(x, 0));
    auto y_min = std::get<0>(torch::min(y, 0));
    auto y_max = std::get<0>(torch::max(y, 0));

    auto min_coords = torch::min(x_min, y_min);
    auto max_coords = torch::max(x_max, y_max);

    return torch::norm(max_coords - min_coords).item<double>();
}

torch::Tensor UnbalancedSinkhornDivergence::softmin_tensorized(
    double eps,
    const torch::Tensor& C_xy,
    const torch::Tensor& h_y
) {
    // Direct translation from geomloss/sinkhorn_samples.py:softmin_tensorized()
    //
    // Python implementation:
    // def softmin_tensorized(eps, C_xy, h_y):
    //     B = C_xy.shape[0]
    //     return -eps * (h_y.view(B, 1, -1) - C_xy / eps).logsumexp(2).view(B, -1)
    //
    // For non-batched case:
    // - C_xy: (N, M) cost matrix
    // - h_y: (M,) dual potential
    // - Returns: (N,) dual potential
    //
    // Computes: f_i = -ε * log(Σ_j exp[(h_j - C_ij)/ε])

    // Reshape h_y from (M,) to (1, M) for broadcasting
    auto h_y_reshaped = h_y.view({1, -1});

    // Compute (h_y - C_xy / eps), shape: (N, M)
    auto logits = h_y_reshaped - C_xy / eps;

    // Compute log-sum-exp along dimension 1 (over M)
    // torch::logsumexp automatically handles numerical stability
    auto lse = torch::logsumexp(logits, /*dim=*/1);

    // Return -eps * lse, shape: (N,)
    return -eps * lse;
}

std::vector<double> UnbalancedSinkhornDivergence::epsilon_schedule(
    double p,
    double diameter,
    double blur,
    double scaling
) {
    // Direct translation from geomloss/sinkhorn_divergence.py:epsilon_schedule()
    //
    // Python implementation:
    // eps_list = (
    //     [diameter**p]
    //     + [np.exp(e) for e in np.arange(p*np.log(diameter), p*np.log(blur), p*np.log(scaling))]
    //     + [blur**p]
    // )
    //
    // We use an exponential cooling schedule: starting from diameter^p,
    // epsilon is multiplied by scaling^p at each iteration until reaching blur^p

    std::vector<double> eps_list;

    // First element: diameter^p
    eps_list.push_back(std::pow(diameter, p));

    // Middle elements: use np.arange in log space, then exponentiate
    // np.arange goes from p*log(diameter) to p*log(blur) with step p*log(scaling)
    double start_log = p * std::log(diameter);
    double stop_log = p * std::log(blur);
    double step_log = p * std::log(scaling);  // This is negative since scaling < 1

    // Generate values in log space
    // Note: np.arange starts at start_log (not start_log + step_log), which creates
    // a duplicate with the first element. This matches the Python implementation.
    // np.arange doesn't include the stop value, so we use < not <=
    for (double log_eps = start_log;
         (step_log < 0 && log_eps > stop_log) || (step_log > 0 && log_eps < stop_log);
         log_eps += step_log) {
        eps_list.push_back(std::exp(log_eps));
    }

    // Last element: blur^p
    eps_list.push_back(std::pow(blur, p));

    return eps_list;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UnbalancedSinkhornDivergence::sinkhorn_loop(
    const SoftminFunction& softmin,
    const std::vector<torch::Tensor>& a_logs,
    const std::vector<torch::Tensor>& b_logs,
    const std::vector<torch::Tensor>& C_xxs,
    const std::vector<torch::Tensor>& C_yys,
    const std::vector<torch::Tensor>& C_xys,
    const std::vector<torch::Tensor>& C_yxs,
    const std::vector<double>& eps_list,
    const double* rho,
    const std::vector<int>& jumps,
    const TruncationFunction* kernel_truncation,
    int truncate,
    const std::string& cost,
    const ExtrapolationFunction* extrapolate,
    bool debias,
    bool last_extrapolation
) {
    // Initialize dual potentials
    torch::Tensor f_aa, g_bb, g_ab, f_ba;

    // Determine number of scales
    int max_scale = static_cast<int>(a_logs.size()) - 1;

    // Options for making a new tensor
    auto options = torch::TensorOptions().dtype(a_logs[0].dtype()).device(a_logs[0].device());

    // Initialize potentials at coarsest scale
    int k = 0; // scale index
    if (!a_logs.empty()) {
        
        f_aa = torch::zeros_like(a_logs[0], options);
        g_bb = torch::zeros_like(b_logs[0], options);
        g_ab = torch::zeros_like(b_logs[0], options);
        f_ba = torch::zeros_like(a_logs[0], options);
    }
    double eps_start = eps_list[k];
    double tau_start = dampening(eps_start, rho);

    //std::cout << "damping start: " << tau_start << std::endl;

    // Get tensors for current scale
    const auto& a_log = a_logs[k];
    const auto& b_log = b_logs[k];
    const auto& C_xx  = C_xxs[k];
    const auto& C_yy  = C_yys[k];
    const auto& C_xy  = C_xys[k];
    const auto& C_yx  = C_yxs[k];

    g_ab = tau_start*softmin(eps_start,C_yx,a_log);
    f_ba = tau_start*softmin(eps_start,C_xy,b_log);
    if ( debias ) {
        f_aa = tau_start*softmin(eps_start,C_xx,a_log);
        g_bb = tau_start*softmin(eps_start,C_yy,b_log);
    }
    //std::cout << "f_ba[start]: " << std::endl;
    //std::cout << f_ba << std::endl;

    // Main epsilon-scaling loop
    // Lines 4-5: eps-scaling descent ---------------------------------------------------
    // See Fig. 3.25-26 in Jean Feydy's PhD thesis.
    double last_tau = tau_start;
    double last_eps = eps_start;
    for (size_t eps_idx = 0; eps_idx < eps_list.size(); ++eps_idx) {

        double eps = eps_list[eps_idx];

        // Line 6: update the damping coefficient ---------------------------------------
        double tau = dampening(eps,rho);

        // Line 7: "coordinate ascent" on the dual problems -----------------------------
        // N.B.: As discussed in Section 3.3.3 of Jean Feydy's PhD thesis,
        //       we perform "symmetric" instead of "alternate" updates
        //       of the dual potentials "f" and "g".
        //       To this end, we first create buffers "ft", "gt"
        //       (for "f-tilde", "g-tilde") using the standard
        //       Sinkhorn formulas, and update both dual vectors
        //       simultaneously.        
        auto ft_ba = tau * softmin( eps, C_xy, b_log + g_ab/eps );
        auto gt_ab = tau * softmin( eps, C_yx, a_log + f_ba/eps );
        auto ft_aa = torch::zeros_like(a_logs[0], options);
        auto gt_bb = torch::zeros_like(b_logs[0], options);
        if ( debias ) {
            // See Fig. 3.21 in Jean Feydy's PhD thesis to see the importance
            // of debiasing when the target "blur" or "eps**(1/p)" value is larger
            // than the average distance between samples x_i, y_j and their neighbours.
            ft_aa = tau * softmin( eps, C_xx, a_log + f_aa / eps );
            gt_bb = tau * softmin( eps, C_yy, b_log + g_bb / eps );
        }

        // Symmetrized updates - see Fig. 3.24.b in Jean Feydy's PhD thesis:
        // from python: f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)  # OT(a,b) wrt. a, b
        f_ba = 0.5 * (f_ba + ft_ba); // # OT(a,b) wrt. a, b
        g_ab = 0.5 * (g_ab + gt_ab); // # OT(a,b) wrt. a, b
        if ( debias ) {
            // OT(a,a), OT(b,b)
            f_aa = 0.5 * (f_aa + ft_aa);
            g_bb = 0.5 * (g_bb + gt_bb);
        }  

        //std::cout << "f_ba[step=" << eps_idx << "]: " << std::endl;
        //std::cout << f_ba << std::endl;

        // Line 8: jump from a coarse to a finer scale ----------------------------------
        // In multi-scale mode, we work we increasingly detailed representations
        // of the input measures: this type of strategy is known as "multi-scale"
        // in computer graphics, "multi-grid" in numerical analysis,
        // "coarse-to-fine" in signal processing or "divide and conquer"
        // in standard complexity theory (e.g. for the quick-sort algorithm).
        //
        // In the Sinkhorn loop with epsilon-scaling annealing, our
        // representations of the input measures are fine enough to ensure
        // that the typical distance between any two samples x_i, y_j is always smaller
        // than the current value of "blur = eps**(1/p)".
        // As illustrated in Fig. 3.26 of Jean Feydy's PhD thesis, this allows us
        // to reach a satisfying level of precision while speeding up the computation
        // of the Sinkhorn iterations in the first few steps.
        //
        // In practice, different multi-scale representations of the input measures
        // are generated by the "parent" code of this solver and stored in the
        // lists a_logs, b_logs, C_xxs, etc.
        //
        // The switch between different scales is specified by the list of "jump" indices,
        // that is generated in conjunction with the list of temperatures "eps_list".
        //
        // N.B.: In single-scale mode, jumps = []: the code below is never executed
        //       and we retrieve "Algorithm 3.5" from Jean Feydy's PhD thesis.
        
        // TODO: I'm going to skip this for now

        last_tau = tau;
        last_eps = eps;
    }//end of eps loop

    if ( last_extrapolation ) {
        auto g_ab_last = g_ab.clone();
        auto f_ba_last = f_ba.clone();
        f_ba = last_tau * softmin( last_eps, C_xy, (b_log + g_ab_last / last_eps ) ).detach();
        g_ab = last_tau * softmin( last_eps, C_yx, (a_log + f_ba_last / last_eps ) ).detach();
        if ( debias ) {
            auto f_aa_last = f_aa.clone();
            auto g_bb_last = g_bb.clone();
            f_aa = last_tau * softmin( last_eps, C_xx, (a_log + f_aa_last/last_eps)).detach();
            g_bb = last_tau * softmin( last_eps, C_yy, (b_log + g_bb_last/last_eps)).detach();
        }
    }

    //std::cout << "[final f_ba]" << std::endl;
    //std::cout << f_ba << std::endl;

    return std::make_tuple(f_aa, g_bb, g_ab, f_ba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UnbalancedSinkhornDivergence::sinkhorn_loop_single_scale(
    const SoftminFunction& softmin,
    const torch::Tensor& a_log,
    const torch::Tensor& b_log,
    const torch::Tensor& C_xx,
    const torch::Tensor& C_yy,
    const torch::Tensor& C_xy,
    const torch::Tensor& C_yx,
    const std::vector<double>& eps_list,
    const double* rho,
    bool debias
) {
    // Wrapper for single-scale operation
    std::vector<torch::Tensor> a_logs = {a_log};
    std::vector<torch::Tensor> b_logs = {b_log};
    std::vector<torch::Tensor> C_xxs = {C_xx};
    std::vector<torch::Tensor> C_yys = {C_yy};
    std::vector<torch::Tensor> C_xys = {C_xy};
    std::vector<torch::Tensor> C_yxs = {C_yx};

    return sinkhorn_loop(
        softmin, a_logs, b_logs, C_xxs, C_yys, C_xys, C_yxs,
        eps_list, rho, {}, nullptr, 5, "", nullptr, debias, true
    );
}

torch::Tensor UnbalancedSinkhornDivergence::sinkhorn_cost(
    double eps,
    const double* rho,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& f_aa,
    const torch::Tensor& g_bb,
    const torch::Tensor& g_ab,
    const torch::Tensor& f_ba,
    bool batch,
    bool debias,
    bool potentials
) {
    if (potentials) {
        throw std::runtime_error("UnbalancedSinkhornDivergence::sinkhorn_cost( ..., potentials = true) not yet implemented");
        // Return potentials instead of cost
        // if (debias) {
        //     return f_ba - f_aa, g_ab - g_bb
        // }
        // return f_ba;  // Simplified - would need to return both f and g
    }

    if ( debias ) {
        if ( rho==nullptr ) { 
            // Balanced case:
            // See Eq. (3.209) in Jean Feydy's PhD thesis.
            return scal( a, f_ba - f_aa, batch) + scal(b,g_ab-g_bb, batch);
        }
        else {
            // Unbalanced case:
            // See Proposition 12 (Dual formulas for the Sinkhorn costs)
            // in "Sinkhorn divergences for unbalanced optimal transport",
            // Sejourne et al., https://arxiv.org/abs/1910.12958.  
            auto fout = (-f_aa / (*rho)).exp() - (-f_ba / (*rho)).exp();
            auto gout = (-g_bb / (*rho)).exp() - (-g_ab / (*rho)).exp();
            return scal( a, unbalanced_weight( fout, eps, *rho, false ), batch )
                     + scal( b, unbalanced_weight( gout, eps, *rho, false ), batch );

        }

    }
    else {
        if ( rho==nullptr ) {
            return scal( a, f_ba, batch ) + scal(b, g_ab, batch );
        }
        else { 
            auto fout = 1 - (-f_ba / (*rho)).exp();
            auto gout = 1 - (-g_ab / (*rho)).exp();
            return scal( a, unbalanced_weight(fout,eps,*rho,false), batch)
                    + scal( a, unbalanced_weight(gout,eps,*rho,false), batch);
        }
    }

    // should never get here

    torch::Tensor dummy;
    return dummy;
}

} // namespace sinkhorn
} // namespace flashmatch
