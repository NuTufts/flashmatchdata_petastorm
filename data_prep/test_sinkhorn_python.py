#!/usr/bin/env python3

import os,sys
sys.path.append(os.environ['FLASHMATCH_BASEDIR'])
import numpy as np
import torch
import math
import geomloss
from flashmatchnet.utils.pmtpos import create_pmtpos_tensor
from geomloss.sinkhorn_divergence import epsilon_schedule

# def generate_pmt_positions(n_pmts, seed=42):
#     """Generate random positions for PMTs in a cylindrical detector geometry"""
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     theta = np.random.uniform(0.0, 2.0 * np.pi, n_pmts)
#     z = np.random.uniform(-100.0, 100.0, n_pmts)  # cm

#     radius = 128.0  # MicroBooNE PMT radius in cm

#     positions = torch.zeros(n_pmts, 3)
#     positions[:, 0] = radius * torch.cos(torch.tensor(theta))  # x
#     positions[:, 1] = radius * torch.sin(torch.tensor(theta))  # y
#     positions[:, 2] = torch.tensor(z)                          # z

#     return positions

def generate_random_masses(n_pmts, seed=42):
    """Generate random mass vectors using exponential distribution"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use exponential distribution for PE counts (similar to C++)
    exp_dist = torch.distributions.exponential.Exponential(1.0)
    masses_a = exp_dist.sample((n_pmts,)) + 0.1
    masses_b = exp_dist.sample((n_pmts,)) + 0.1

    return masses_a, masses_b

def get_fixed_masses(scale=1.0):
    """
*        2 *        0 * 1.0486168 *         0 * 956.57006 * 886.57165 *
*        2 *        1 * 1.0491255 *         0 * 956.57006 * 886.57165 *
*        2 *        2 * 1.0487175 *         0 * 956.57006 * 886.57165 *
*        2 *        3 * 1.0486918 *         0 * 956.57006 * 886.57165 *
*        2 *        4 * 1.0484792 *         0 * 956.57006 * 886.57165 *
*        2 *        5 * 1.0487355 *         0 * 956.57006 * 886.57165 *
*        2 *        6 * 1.0484391 *         0 * 956.57006 * 886.57165 *
*        2 *        7 * 23.112646 * 36.424945 * 956.57006 * 886.57165 *
*        2 *        8 * 45.717140 *         0 * 956.57006 * 886.57165 *
*        2 *        9 * 2.8149819 *         0 * 956.57006 * 886.57165 *
*        2 *       10 * 101.49279 * 96.241447 * 956.57006 * 886.57165 *
Type <CR> to continue or q to quit ==> 
*        2 *       11 * 16.468011 * 28.675005 * 956.57006 * 886.57165 *
*        2 *       12 * 42.669189 * 20.811662 * 956.57006 * 886.57165 *
*        2 *       13 * 181.69705 * 165.14990 * 956.57006 * 886.57165 *
*        2 *       14 * 109.77285 * 106.81942 * 956.57006 * 886.57165 *
*        2 *       15 * 258.28912 * 288.03347 * 956.57006 * 886.57165 *
*        2 *       16 * 52.197849 * 59.700992 * 956.57006 * 886.57165 *
*        2 *       17 * 1.4032447 *         0 * 956.57006 * 886.57165 *
*        2 *       18 * 99.118621 * 84.714767 * 956.57006 * 886.57165 *
*        2 *       19 * 1.0487263 *         0 * 956.57006 * 886.57165 *
*        2 *       20 * 1.0484359 *         0 * 956.57006 * 886.57165 *
*        2 *       21 * 1.8938610 *         0 * 956.57006 * 886.57165 *
*        2 *       22 * 1.0485358 *         0 * 956.57006 * 886.57165 *
*        2 *       23 * 1.0486611 *         0 * 956.57006 * 886.57165 *
*        2 *       24 * 1.0484833 *         0 * 956.57006 * 886.57165 *
*        2 *       25 * 1.0484889 *         0 * 956.57006 * 886.57165 *
*        2 *       26 * 1.0484372 *         0 * 956.57006 * 886.57165 *
*        2 *       27 * 1.0483601 *         0 * 956.57006 * 886.57165 *
*        2 *       28 * 1.0483439 *         0 * 956.57006 * 886.57165 *
*        2 *       29 * 1.0485112 *         0 * 956.57006 * 886.57165 *
*        2 *       30 * 1.0485589 *         0 * 956.57006 * 886.57165 *
*        2 *       31 * 1.0483729 *         0 * 956.57006 * 886.57165 *
    """
    masses_a = torch.zeros( 32 )
    masses_b = torch.zeros( 32 )

    masses_a[0]  = 1.0486168
    masses_a[1]  = 1.0491255
    masses_a[2]  = 1.0487175
    masses_a[3]  = 1.0486918
    masses_a[4]  = 1.0484792
    masses_a[5]  = 1.0487355
    masses_a[6]  = 1.0484391
    masses_a[7]  = 23.112646
    masses_a[8]  = 45.717140
    masses_a[9]  = 2.8149819
    masses_a[10] = 101.49279
    masses_a[11] = 16.468011
    masses_a[12] = 42.669189
    masses_a[13] = 181.69705
    masses_a[14] = 109.77285
    masses_a[15] = 258.28912
    masses_a[16] = 52.197849
    masses_a[17] = 1.4032447
    masses_a[18] = 99.118621
    masses_a[19] = 1.0487263
    masses_a[20] = 1.0484359
    masses_a[21] = 1.8938610
    masses_a[22] = 1.0485358
    masses_a[23] = 1.0486611
    masses_a[24] = 1.0484833
    masses_a[25] = 1.0484889
    masses_a[26] = 1.0484372
    masses_a[27] = 1.0483601
    masses_a[28] = 1.0483439
    masses_a[29] = 1.0485112
    masses_a[30] = 1.0485589
    masses_a[31] = 1.0483729

    masses_b[0]  =  0 
    masses_b[1]  =  0 
    masses_b[2]  =  0 
    masses_b[3]  =  0 
    masses_b[4]  =  0 
    masses_b[5]  =  0 
    masses_b[6]  =  0 
    masses_b[7]  =  36.42494
    masses_b[8]  =  0 
    masses_b[9]  =  0 
    masses_b[10] =  96.241447
    masses_b[11] =  28.675005
    masses_b[12] =  20.811662
    masses_b[13] =  165.14990
    masses_b[14] =  106.81942
    masses_b[15] =  288.03347
    masses_b[16] =  59.700992
    masses_b[17] =  0
    masses_b[18] =  84.7147
    masses_b[19] =  0
    masses_b[20] =  0
    masses_b[21] =  0
    masses_b[22] =  0
    masses_b[23] =  0
    masses_b[24] =  0
    masses_b[25] =  0
    masses_b[26] =  0
    masses_b[27] =  0
    masses_b[28] =  0
    masses_b[29] =  0
    masses_b[30] =  0
    masses_b[31] =  0

    masses_b += 1.0e-5

    masses_a = masses_a/scale
    masses_b = masses_b/scale

    return masses_a, masses_b



def main():
    print("Testing Sinkhorn Divergence with geomloss package")
    print("=================================================")

    # Set random seed for reproducibility (same as C++)
    torch.manual_seed(42)
    np.random.seed(42)

    n_pmts = 32

    # Generate the same random mass vectors as C++ test
    #masses_a, masses_b = generate_random_masses(n_pmts, seed=42)
    masses_a, masses_b = get_fixed_masses(5000.0)
    masses_a = masses_a * 1.3

    print("Generated random mass vectors:")
    print(f"masses_a: {masses_a}")
    print(f"masses_b: {masses_b}")
    print()

    # Normalize masses to make them proper probability measures
    pdf_a = masses_a / torch.sum(masses_a)
    pdf_b = masses_b / torch.sum(masses_b)

    print("Normalized masses (sum = 1):")
    print(f"sum(masses_a) = {torch.sum(masses_a).item():.6f}")
    print(f"sum(masses_b) = {torch.sum(masses_b).item():.6f}")
    print()

    # Generate PMT positions (same as C++)
    #pmt_positions = generate_pmt_positions(n_pmts, seed=42)
    pmt_positions = create_pmtpos_tensor()/1000.0
    print("Generated PMT positions:")
    print(pmt_positions)
    print()

    print(f"PMT positions shape: {pmt_positions.shape}")

    # Compute diameter for epsilon schedule
    x_min = torch.min(pmt_positions, dim=0)[0]
    x_max = torch.max(pmt_positions, dim=0)[0]
    diameter = torch.norm(x_max - x_min).item()
    print(f"Point cloud diameter: {diameter:.4f} cm")

    # Create epsilon schedule
    #def epsilon_schedule(2, diameter, 1.0, 0.5):
    #eps_schedule = compute_epsilon_schedule(diameter, blur=1.0, scaling=0.5, p=2)
    eps_list = epsilon_schedule(2, diameter, 0.05, 0.5)
    print("Epsilon annealing schedule:", [f"{eps:.6f}" for eps in eps_list])
    print()

    # Test 1: Balanced Sinkhorn divergence (equivalent to Wasserstein distance)
    print("Testing balanced optimal transport (Sinkhorn divergence)...")

    # Create SamplesLoss object for balanced case
    loss_balanced = geomloss.SamplesLoss(
        loss="sinkhorn",
        p=2,                   # L2 distance
        blur=0.05,             # final epsilon
        scaling=0.5,           # epsilon scaling factor (0.5 = divide by 2)
        backend="tensorized",  # use tensorized implementation
        debias=True            # compute debiased divergence
    )

    # Compute balanced Sinkhorn divergence
    balanced_cost = loss_balanced(pdf_a, pmt_positions, pdf_b, pmt_positions)
    print(f"Balanced Sinkhorn divergence: {balanced_cost.item():.6f}")
    print()

    # Test 2: Unbalanced Sinkhorn divergence
    print("Testing unbalanced optimal transport...")
    rho = 1.0  # Marginal constraint strength

    # Create SamplesLoss object for unbalanced case
    loss_unbalanced = geomloss.SamplesLoss(
        loss="sinkhorn",
        p=2,                   # L2 distance
        blur=0.05,              # final epsilon
        scaling=0.5,           # epsilon scaling factor
        backend="tensorized",  # use tensorized implementation
        debias=True,           # compute debiased divergence
        reach=rho              # unbalanced parameter (reach = rho)
    )

    # Compute unbalanced Sinkhorn divergence
    unbalanced_cost = loss_unbalanced(masses_a, pmt_positions, masses_b, pmt_positions)
    print(f"Unbalanced Sinkhorn divergence (rho={rho}): {unbalanced_cost.item():.6f}")
    print()

    # Test 3: Different epsilon values and dampening factors
    print("Testing dampening factors:")
    eps_values = [0.1, 1.0, 10.0]
    for eps in eps_values:
        tau = 1.0 / (1.0 + eps / rho)  # dampening formula
        print(f"Dampening factor for eps={eps}, rho={rho}: {tau:.6f}")
    print()

    # Test 4: Raw optimal transport cost (no debiasing)
    print("Testing raw optimal transport cost (no debiasing)...")

    loss_raw = geomloss.SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=1.0,
        scaling=0.5,
        backend="tensorized",
        debias=False           # no debiasing
    )

    raw_cost = loss_raw(masses_a, pmt_positions, masses_b, pmt_positions)
    print(f"Raw optimal transport cost: {raw_cost.item():.6f}")
    print()

    # Test 5: Different blur scales
    print("Testing different blur scales:")
    blur_values = [0.1, 1.0, 10.0]
    for blur in blur_values:
        loss_blur = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            scaling=0.5,
            backend="tensorized",
            debias=True
        )
        cost = loss_blur(masses_a, pmt_positions, masses_b, pmt_positions)
        print(f"Sinkhorn divergence (blur={blur}): {cost.item():.6f}")
    print()

    # Test 6: Check mass conservation
    total_mass_a = torch.sum(masses_a)
    total_mass_b = torch.sum(masses_b)
    mass_difference = abs(total_mass_a - total_mass_b).item()
    print(f"Mass conservation check:")
    print(f"Total mass A: {total_mass_a.item():.6f}")
    print(f"Total mass B: {total_mass_b.item():.6f}")
    print(f"Mass difference: {mass_difference:.6f}")
    print()

    # Summary comparison with C++ implementation
    print("Summary for comparison with C++ implementation:")
    print("=" * 50)
    print(f"Point cloud diameter: {diameter:.4f} cm")
    print(f"Number of PMTs: {n_pmts}")
    print(f"Balanced Sinkhorn divergence: {balanced_cost.item():.6f}")
    print(f"Unbalanced Sinkhorn divergence: {unbalanced_cost.item():.6f}")
    print(f"Raw OT cost (no debias): {raw_cost.item():.6f}")
    print()

    print("Test completed successfully!")
    print()
    print("NOTE: To compare with C++ implementation:")
    print("1. Same random seed (42) should give similar mass vectors")
    print("2. PMT positions should be identical")
    print("3. Balanced/unbalanced costs should be close (within numerical precision)")
    print("4. Different blur scales help validate epsilon-scaling behavior")

if __name__ == "__main__":
    main()