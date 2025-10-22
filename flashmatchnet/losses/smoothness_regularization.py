"""
Smoothness Regularization for SIREN Light Model

This module implements smoothness regularization that penalizes large second
derivatives of the neural network output with respect to its inputs.

The smoothness is computed using finite differences:
- For each sampled point x, we perturb it slightly: x + h*e_i and x - h*e_i
- Second derivative along dimension i: f''_i ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
- Total smoothness penalty: mean of squared second derivatives across all dimensions
"""

import torch
import torch.nn as nn


class SmoothnessRegularization:
    """
    Computes smoothness regularization for a neural network function.

    This encourages the network output to be smooth (have small second derivatives)
    with respect to its inputs, which is physically motivated for light propagation.
    """

    def __init__(self,
                 epsilon=1e-3,
                 num_samples=None,
                 sample_fraction=0.1,
                 derivative_type='second',
                 reduction='mean'):
        """
        Args:
            epsilon: Step size for finite differences (default: 1e-3)
            num_samples: Number of points to sample for smoothness computation.
                        If None, uses sample_fraction instead.
            sample_fraction: Fraction of batch points to sample (default: 0.1)
            derivative_type: Type of derivative penalty
                - 'first': Penalize first derivatives (gradient magnitude)
                - 'second': Penalize second derivatives (Laplacian) - default
                - 'mixed': Penalize both first and second derivatives
            reduction: How to reduce the loss ('mean' or 'sum')
        """
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.sample_fraction = sample_fraction
        self.derivative_type = derivative_type
        self.reduction = reduction

    def compute_smoothness_loss(self, model, vox_feat, q, mask=None):
        """
        Compute smoothness loss by sampling points and computing finite differences.

        Args:
            model: The SIREN model to regularize
            vox_feat: (N, 7) input features
            q: (N, 1) charge values
            mask: Optional (N,) mask indicating valid points (1=valid, 0=invalid)

        Returns:
            smoothness_loss: Scalar tensor with smoothness penalty
        """
        N, D = vox_feat.shape  # N points, D=7 dimensions
        device = vox_feat.device

        # Determine how many points to sample
        if self.num_samples is not None:
            n_samples = min(self.num_samples, N)
        else:
            n_samples = max(1, int(N * self.sample_fraction))

        # Sample random points from the batch
        if mask is not None:
            # Only sample from valid (non-masked) points
            valid_indices = torch.where(mask > 0)[0]
            if len(valid_indices) == 0:
                # No valid points, return zero loss
                return torch.tensor(0.0, device=device)
            if len(valid_indices) < n_samples:
                n_samples = len(valid_indices)
            perm = torch.randperm(len(valid_indices), device=device)[:n_samples]
            sample_idx = valid_indices[perm].long()
        else:
            sample_idx = torch.randperm(N, device=device)[:n_samples].long()

        # Ensure indices are contiguous and on correct device
        sample_idx = sample_idx.contiguous()

        # Get sampled points
        x_samples = vox_feat[sample_idx].contiguous()  # (n_samples, D)
        q_samples = q[sample_idx].contiguous()  # (n_samples, 1)

        # Evaluate at original points
        with torch.set_grad_enabled(True):
            f_x = model(x_samples, q_samples)  # (n_samples, 1) or (n_samples,)

        if self.derivative_type == 'first' or self.derivative_type == 'mixed':
            # First derivative: use finite differences
            gradient_loss = self._compute_gradient_penalty(model, x_samples, q_samples, f_x)
        else:
            gradient_loss = 0.0

        if self.derivative_type == 'second' or self.derivative_type == 'mixed':
            # Second derivative: use finite differences
            laplacian_loss = self._compute_laplacian_penalty(model, x_samples, q_samples, f_x)
        else:
            laplacian_loss = 0.0

        # Combine losses
        if self.derivative_type == 'mixed':
            total_loss = gradient_loss + laplacian_loss
        elif self.derivative_type == 'first':
            total_loss = gradient_loss
        else:  # 'second'
            total_loss = laplacian_loss

        return total_loss

    def _compute_gradient_penalty(self, model, x, q, f_x):
        """
        Compute penalty on first derivatives (gradient magnitude).

        Uses central differences: f'_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        """
        n_samples, D = x.shape
        device = x.device
        dtype = x.dtype
        h = float(self.epsilon)  # Ensure h is a Python float

        gradient_sq_sum = 0.0

        # Clone and detach x to avoid gradient issues
        x_base = x.clone().detach().requires_grad_(False)

        # Compute gradient along each dimension
        for i in range(D):
            # Create perturbation vector
            e_i = torch.zeros(n_samples, D, device=device, dtype=dtype)
            e_i[:, i] = 1.0

            # Forward perturbation: x + h*e_i
            x_plus = x_base + h * e_i
            x_plus.requires_grad_(False)  # Explicitly disable gradient
            f_plus = model(x_plus, q)

            # Backward perturbation: x - h*e_i
            x_minus = x_base - h * e_i
            x_minus.requires_grad_(False)  # Explicitly disable gradient
            f_minus = model(x_minus, q)

            # Central difference approximation of first derivative
            df_di = (f_plus - f_minus) / (2 * h)

            # Add squared gradient component
            gradient_sq_sum = gradient_sq_sum + (df_di ** 2)

        # Mean squared gradient magnitude
        if self.reduction == 'mean':
            return gradient_sq_sum.mean()
        else:
            return gradient_sq_sum.sum()

    def _compute_laplacian_penalty(self, model, x, q, f_x):
        """
        Compute penalty on second derivatives (discrete Laplacian).

        Uses finite differences: f''_i ≈ (f(x + h*e_i) - 2*f(x) + f(x - h*e_i)) / h²
        """
        n_samples, D = x.shape
        device = x.device
        dtype = x.dtype
        h = float(self.epsilon)  # Ensure h is a Python float

        laplacian_sq_sum = 0.0

        # Clone and detach x to avoid gradient issues
        x_base = x.clone().detach().requires_grad_(False)

        # Compute second derivative along each dimension
        for i in range(D):
            # Create perturbation vector
            e_i = torch.zeros(n_samples, D, device=device, dtype=dtype)
            e_i[:, i] = 1.0

            # Forward perturbation: x + h*e_i
            x_plus = x_base + h * e_i
            x_plus.requires_grad_(False)  # Explicitly disable gradient
            f_plus = model(x_plus, q)

            # Backward perturbation: x - h*e_i
            x_minus = x_base - h * e_i
            x_minus.requires_grad_(False)  # Explicitly disable gradient
            f_minus = model(x_minus, q)

            # Second derivative using three-point stencil
            # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            d2f_di2 = (f_plus - 2.0 * f_x + f_minus) / (h ** 2)

            # Add squared second derivative
            laplacian_sq_sum = laplacian_sq_sum + (d2f_di2 ** 2)

        # Mean squared Laplacian
        if self.reduction == 'mean':
            return laplacian_sq_sum.mean()
        else:
            return laplacian_sq_sum.sum()


def compute_smoothness_loss(model, vox_feat, q, mask=None,
                            epsilon=1e-3, num_samples=None,
                            sample_fraction=0.1, derivative_type='second'):
    """
    Convenience function to compute smoothness loss.

    Args:
        model: The SIREN model
        vox_feat: (N, 7) input features
        q: (N, 1) charge values
        mask: Optional (N,) validity mask
        epsilon: Finite difference step size
        num_samples: Number of points to sample (None = use fraction)
        sample_fraction: Fraction of points to sample if num_samples is None
        derivative_type: 'first', 'second', or 'mixed'

    Returns:
        Smoothness loss (scalar tensor)
    """
    regularizer = SmoothnessRegularization(
        epsilon=epsilon,
        num_samples=num_samples,
        sample_fraction=sample_fraction,
        derivative_type=derivative_type
    )
    return regularizer.compute_smoothness_loss(model, vox_feat, q, mask)
