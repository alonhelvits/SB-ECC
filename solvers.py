"""
ODE Solvers for DDECC diffusion models
"""
import torch
from abc import ABC, abstractmethod


class OdeSolver(ABC):
    """Base class for ODE solvers used in diffusion model inference."""
    
    def __init__(self, model, sigma_min: float, sigma_max: float, device: torch.device):
        """
        Initialize the ODE solver.
        
        Args:
            model: The diffusion model (DDECCT_SDE instance)
            sigma_min: Minimum noise standard deviation
            sigma_max: Maximum noise standard deviation 
            device: Device to run computations on
        """
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
    
    def reset_history(self):
        """Reset any internal state/history. Called at the start of each decode sequence."""
        pass
    
    @abstractmethod
    def step(self, xt: torch.Tensor, x0_pred: torch.Tensor, delta_sigma: float, 
             not_converged_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single denoising step.
        
        Args:
            xt: Current noisy state (B, N)
            x0_pred: Previous estimate of clean signal for self-conditioning (B, N) 
            delta_sigma: Step size in sigma space
            not_converged_mask: Boolean mask for samples that haven't converged yet (B,)
            
        Returns:
            tuple: (updated_xt, updated_x0_pred)
        """
        pass


class EulerSolver(OdeSolver):
    """First-order Euler method solver."""
    
    def step(self, xt: torch.Tensor, x0_pred: torch.Tensor, delta_sigma: float,
             not_converged_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform Euler step: x_{t+1} = x_t - delta_sigma * predicted_noise"""
        
        # Extract active (non-converged) samples
        active_xt = xt[not_converged_mask]
        active_x0_pred_cond = x0_pred[not_converged_mask] if x0_pred is not None else None
        
        # Get model prediction
        _, pred_noise = self.model(active_xt, torch.zeros(active_xt.shape[0], 1, device=active_xt.device), 
                                   x0_pred=active_x0_pred_cond)
        
        # Euler update
        update = delta_sigma * pred_noise
        updated_active_xt = active_xt - update
        
        # Update x0_pred for self-conditioning
        updated_active_x0_pred = active_xt - delta_sigma * pred_noise
        
        # Update full tensors
        updated_xt = xt.clone()
        updated_xt[not_converged_mask] = updated_active_xt
        
        if x0_pred is None:
            updated_x0_pred = torch.zeros_like(xt)
        else:
            updated_x0_pred = x0_pred.clone()
        updated_x0_pred[not_converged_mask] = updated_active_x0_pred
        
        return updated_xt, updated_x0_pred


class DpmSolver(OdeSolver):
    """DPM-Solver implementation for diffusion models."""
    
    def __init__(self, model, sigma_min: float, sigma_max: float, device: torch.device, order: int = 2):
        """
        Initialize DPM-Solver.
        
        Args:
            order: Order of the solver (1, 2, or 3)
        """
        super().__init__(model, sigma_min, sigma_max, device)
        self.order = min(order, 3)
        self._reset_history()
    
    def _reset_history(self):
        """Reset the prediction history."""
        self.prev_noise_preds = []
    
    def reset_history(self):
        """Reset any internal state/history. Called at the start of each decode sequence."""
        self._reset_history()
    
    def step(self, xt: torch.Tensor, x0_pred: torch.Tensor, delta_sigma: float,
             not_converged_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform DPM-Solver step"""
        
        # Extract active samples
        active_xt = xt[not_converged_mask]
        active_x0_pred_cond = x0_pred[not_converged_mask] if x0_pred is not None else None
        
        # Get current noise prediction
        _, pred_noise = self.model(active_xt, torch.zeros(active_xt.shape[0], 1, device=active_xt.device),
                                   x0_pred=active_x0_pred_cond)
        
        # Create full-batch noise prediction (with zeros for converged samples)
        full_pred_noise = torch.zeros_like(xt)
        full_pred_noise[not_converged_mask] = pred_noise
        
        # Store full prediction for future higher-order steps
        if len(self.prev_noise_preds) >= self.order:
            self.prev_noise_preds.pop(0)
        self.prev_noise_preds.append(full_pred_noise.clone())
        
        # Choose solver order based on available history
        current_order = min(len(self.prev_noise_preds), self.order)
        
        if current_order == 1:
            # First-order (Euler)
            update = delta_sigma * pred_noise
        elif current_order == 2:
            # Second-order DPM-Solver
            prev_noise = self.prev_noise_preds[-2][not_converged_mask]
            update = delta_sigma * (1.5 * pred_noise - 0.5 * prev_noise)
        else:
            # Third-order DPM-Solver
            prev_noise_1 = self.prev_noise_preds[-2][not_converged_mask]
            prev_noise_2 = self.prev_noise_preds[-3][not_converged_mask]
            update = delta_sigma * (23/12 * pred_noise - 16/12 * prev_noise_1 + 5/12 * prev_noise_2)
        
        updated_active_xt = active_xt - update
        
        # Update x0_pred
        sigma_i = 1.0
        updated_active_x0_pred = active_xt - sigma_i * pred_noise
        
        # Update full tensors
        updated_xt = xt.clone()
        updated_xt[not_converged_mask] = updated_active_xt
        
        if x0_pred is None:
            updated_x0_pred = torch.zeros_like(xt)
        else:
            updated_x0_pred = x0_pred.clone()
        updated_x0_pred[not_converged_mask] = updated_active_x0_pred
        
        return updated_xt, updated_x0_pred


def get_solver(solver_type: str, model, sigma_min: float, sigma_max: float, device: torch.device, **kwargs):
    """
    Factory function to create solver instances.
    
    Args:
        solver_type: Type of solver ('euler' or 'dpm')
        model: The diffusion model
        sigma_min: Minimum noise std
        sigma_max: Maximum noise std
        device: Compute device
        **kwargs: Additional arguments passed to specific solvers
        
    Returns:
        OdeSolver instance
    """
    solvers = {
        'euler': EulerSolver,
        'dpm': DpmSolver,
    }
    
    if solver_type not in solvers:
        available = ', '.join(solvers.keys())
        raise ValueError(f"Unknown solver type: {solver_type}. Available: {available}")
    
    return solvers[solver_type](model, sigma_min, sigma_max, device, **kwargs)
