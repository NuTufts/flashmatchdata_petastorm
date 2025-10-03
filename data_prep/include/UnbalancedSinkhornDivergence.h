#ifndef __FLASHMATCH_DATAPREP_UNBALANCED_SINKHORN_DIVERGENCE_H__
#define __FLASHMATCH_DATAPREP_UNBALANCED_SINKHORN_DIVERGENCE_H__

#include <torch/torch.h>
#include <vector>
#include <functional>
#include <tuple>

namespace flashmatch {
namespace sinkhorn {

/**
 * @brief Unbalanced Sinkhorn Divergence solver for point sets
 *
 * This class implements the unbalanced Sinkhorn divergence between abstract measures,
 * translated from the GeomLoss Python library (https://www.kernel-operations.io/geomloss/)
 *
 * The Sinkhorn divergence is computed as:
 * S_ε,ρ(α,β) = OT_ε,ρ(α,β) - 0.5*OT_ε,ρ(α,α) - 0.5*OT_ε,ρ(β,β) + ε/2 * ||<α,1> - <β,1>||²
 *
 * where OT_ε,ρ is the entropy-regularized unbalanced optimal transport cost.
 */
class UnbalancedSinkhornDivergence {
public:
    /**
     * @brief Type for the softmin function
     *
     * The softmin function implements the (soft-)C-transform between dual vectors.
     * Given eps (temperature), C_xy (cost matrix), and g (dual potential),
     * it returns: f_i = -ε * log(Σ_j exp[(g_j - C(x_i,y_j))/ε])
     */
    using SoftminFunction = std::function<torch::Tensor(
        double eps,
        const torch::Tensor& C,
        const torch::Tensor& g
    )>;

    /**
     * @brief Type for the extrapolation function
     *
     * Used in multiscale mode to extrapolate dual potentials from coarse to fine resolution
     */
    using ExtrapolationFunction = std::function<torch::Tensor(
        const torch::Tensor& f,
        const torch::Tensor& g,
        double eps,
        double damping,
        const torch::Tensor& C,
        const torch::Tensor& log_weights,
        const torch::Tensor& C_fine
    )>;

    /**
     * @brief Type for the kernel truncation function
     *
     * Implements the kernel truncation trick for efficiency
     */
    using TruncationFunction = std::function<std::pair<torch::Tensor, torch::Tensor>(
        const torch::Tensor& C_xx,
        const torch::Tensor& C_yy,
        const torch::Tensor& C_xx_fine,
        const torch::Tensor& C_yy_fine,
        const torch::Tensor& f,
        const torch::Tensor& g,
        double eps,
        int truncate,
        const std::string& cost
    )>;

    /**
     * @brief Constructor
     */
    UnbalancedSinkhornDivergence();

    /**
     * @brief Destructor
     */
    ~UnbalancedSinkhornDivergence();

    /**
     * @brief Compute dampening factor for unbalanced OT
     * @param eps Temperature parameter
     * @param rho Marginal constraint strength (nullptr for balanced OT)
     * @return Dampening factor (1 for balanced, < 1 for unbalanced)
     */
    static double dampening(double eps, const double* rho);

    /**
     * @brief Compute log weights with numerical stability
     * @param a Input weights tensor
     * @return Log of weights with values clamped to avoid numerical issues
     */
    static torch::Tensor log_weights(const torch::Tensor& a);

    /**
     * @brief Apply unbalanced weight scaling
     *
     * This applies the correct scaling to dual variables in the Sinkhorn formula.
     * Note: forward and backward passes use different scaling factors.
     * @param x Input tensor
     * @param eps Temperature parameter
     * @param rho Marginal constraint strength
     * @param backward Whether this is for backward pass
     * @return Scaled tensor
     */
    static torch::Tensor unbalanced_weight(
        const torch::Tensor& x,
        double eps,
        double rho,
        bool backward = false
    );

    /**
     * @brief Compute scalar product for Sinkhorn cost
     * @param a Weights tensor
     * @param f Dual potential tensor
     * @param batch Whether operating in batch mode
     * @return Scalar product result
     */
    static torch::Tensor scal(
        const torch::Tensor& a,
        const torch::Tensor& f,
        bool batch = false
    );

    /**
     * @brief Compute max diameter of point clouds
     * @param x First point cloud (N, D)
     * @param y Second point cloud (M, D)
     * @return Upper bound on maximum distance between points
     */
    static double max_diameter(const torch::Tensor& x, const torch::Tensor& y);

    /**
     * @brief Soft-C-transform using dense PyTorch tensors (tensorized backend)
     *
     * This is the standard softmin implementation from geomloss, implementing
     * the (soft-)C-transform between dual vectors.
     *
     * Computes: f_i = -ε * log(Σ_j exp[(h_j - C(x_i,y_j))/ε])
     *
     * This is the core computation for Sinkhorn-like optimal transport solvers.
     *
     * @param eps Temperature parameter ε for the Gibbs kernel
     * @param C_xy Cost matrix C(x_i,y_j) of shape (N, M)
     * @param h_y Logarithmic dual values of shape (M,), typically computed as
     *            h_y = b_log + g_j / eps, where b_log is log-weights and g_j
     *            is a dual vector in the Sinkhorn algorithm
     * @return Dual potential f of shape (N,) supported by points x_i
     */
    static torch::Tensor softmin_tensorized(
        double eps,
        const torch::Tensor& C_xy,
        const torch::Tensor& h_y
    );

    /**
     * @brief Create epsilon schedule for annealing
     * @param p Exponent for distance metric
     * @param diameter Maximum diameter of point clouds
     * @param blur Target blur scale
     * @param scaling Ratio between successive blur values
     * @return Vector of epsilon values
     */
    static std::vector<double> epsilon_schedule(
        double p,
        double diameter,
        double blur,
        double scaling
    );

    /**
     * @brief Main Sinkhorn loop implementation
     *
     * Implements the (possibly multiscale) symmetric Sinkhorn loop with
     * epsilon-scaling (annealing) heuristic.
     *
     * @param softmin Soft C-transform function
     * @param a_logs List of log-weights for first measure at different scales
     * @param b_logs List of log-weights for second measure at different scales
     * @param C_xxs List of cost matrices C(x_i, x_j) at different scales
     * @param C_yys List of cost matrices C(y_i, y_j) at different scales
     * @param C_xys List of cost matrices C(x_i, y_j) at different scales
     * @param C_yxs List of cost matrices C(y_i, x_j) at different scales
     * @param eps_list List of temperature values for annealing
     * @param rho Marginal constraint strength (nullptr for balanced OT)
     * @param jumps Iteration indices for scale jumps (empty for single-scale)
     * @param kernel_truncation Optional kernel truncation function
     * @param truncate Truncation parameter
     * @param cost Cost function type
     * @param extrapolate Optional extrapolation function for multiscale
     * @param debias Use debiased Sinkhorn divergence (true) or raw OT cost (false)
     * @param last_extrapolation Perform final full Sinkhorn iteration
     * @return Tuple of four optimal dual potentials (f_aa, g_bb, g_ab, f_ba)
     */
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sinkhorn_loop(
        const SoftminFunction& softmin,
        const std::vector<torch::Tensor>& a_logs,
        const std::vector<torch::Tensor>& b_logs,
        const std::vector<torch::Tensor>& C_xxs,
        const std::vector<torch::Tensor>& C_yys,
        const std::vector<torch::Tensor>& C_xys,
        const std::vector<torch::Tensor>& C_yxs,
        const std::vector<double>& eps_list,
        const double* rho = nullptr,
        const std::vector<int>& jumps = {},
        const TruncationFunction* kernel_truncation = nullptr,
        int truncate = 5,
        const std::string& cost = "",
        const ExtrapolationFunction* extrapolate = nullptr,
        bool debias = true,
        bool last_extrapolation = true
    );

    /**
     * @brief Single-scale convenience wrapper for sinkhorn_loop
     *
     * @param softmin Soft C-transform function
     * @param a_log Log-weights for first measure
     * @param b_log Log-weights for second measure
     * @param C_xx Cost matrix C(x_i, x_j)
     * @param C_yy Cost matrix C(y_i, y_j)
     * @param C_xy Cost matrix C(x_i, y_j)
     * @param C_yx Cost matrix C(y_i, x_j)
     * @param eps_list List of temperature values for annealing
     * @param rho Marginal constraint strength (nullptr for balanced OT)
     * @param debias Use debiased Sinkhorn divergence
     * @return Tuple of four optimal dual potentials (f_aa, g_bb, g_ab, f_ba)
     */
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    sinkhorn_loop_single_scale(
        const SoftminFunction& softmin,
        const torch::Tensor& a_log,
        const torch::Tensor& b_log,
        const torch::Tensor& C_xx,
        const torch::Tensor& C_yy,
        const torch::Tensor& C_xy,
        const torch::Tensor& C_yx,
        const std::vector<double>& eps_list,
        const double* rho = nullptr,
        bool debias = true
    );

    /**
     * @brief Compute Sinkhorn cost from dual potentials
     *
     * @param eps Final temperature
     * @param rho Marginal constraint strength
     * @param a Source measure weights
     * @param b Target measure weights
     * @param f_aa Dual potential for a <-> a problem
     * @param g_bb Dual potential for b <-> b problem
     * @param g_ab Dual potential supported by y_j for a <-> b problem
     * @param f_ba Dual potential supported by x_i for a <-> b problem
     * @param batch Operating in batch mode
     * @param debias Use debiased divergence
     * @param potentials Return potentials instead of cost
     * @return Cost value or pair of dual potentials
     */
    static torch::Tensor sinkhorn_cost(
        double eps,
        const double* rho,
        const torch::Tensor& a,
        const torch::Tensor& b,
        const torch::Tensor& f_aa,
        const torch::Tensor& g_bb,
        const torch::Tensor& g_ab,
        const torch::Tensor& f_ba,
        bool batch = false,
        bool debias = true,
        bool potentials = false
    );
};

} // namespace sinkhorn
} // namespace flashmatch

#endif // __FLASHMATCH_DATAPREP_UNBALANCED_SINKHORN_DIVERGENCE_H__