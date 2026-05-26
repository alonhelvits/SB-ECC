"""
Score-Based Error Correction Codes (SB-ECC) model.
A diffusion-based neural decoder for linear block codes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
import copy
from Codes import sign_to_bin, bin_to_sign, BER


############################################################
#   Utility Functions
############################################################

def clones(module, N):
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def linear_sigma_t(t: torch.Tensor, sigma_min: float, sigma_max: float):
    """Computes σ(t) using a linear schedule."""
    return sigma_min + (sigma_max - sigma_min) * t


def sqrt_sigma_t(t: torch.Tensor, sigma_min: float, sigma_max: float):
    """Computes σ(t) using a square-root schedule."""
    return sigma_min + (sigma_max - sigma_min) * torch.sqrt(t)


############################################################
#   Transformer Components
############################################################

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


############################################################
#   CrossMPT Encoder (Message Passing Transformer)
############################################################

class CrossMPTEncoderLayer(nn.Module):
    """Single layer of CrossMPT: cross-attention between VN and CN."""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(CrossMPTEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, x2, mask):
        # Cross-attention: Query=x, Key=x2, Value=x2
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x2, x2, mask))
        return self.sublayer[1](x, self.feed_forward)


class CrossMPTEncoder(nn.Module):
    """CrossMPT Encoder: alternating updates between VN and CN embeddings."""
    def __init__(self, layer, N):
        super(CrossMPTEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, x2, mask_VN, mask_CN):
        # x is VN features, x2 is CN features
        for idx, layer in enumerate(self.layers, start=1):
            # Update VNs using CNs
            x = layer(x, x2, mask_VN)
            # Update CNs using VNs
            x2 = layer(x2, x, mask_CN)
            
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
                x2 = self.norm2(x2)
        return self.norm(x), self.norm(x2)


############################################################
#   EMA (Exponential Moving Average)
############################################################

class EMA:
    def __init__(self, mu=0.999, flag_run=True):
        self.mu = mu
        self.shadow = {}
        self.flag_run = flag_run

    def register(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    def update(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data


############################################################
#   SB_ECC: Score-Based Error Correction Code Decoder
############################################################

class SB_ECC(nn.Module):
    """
    Score-Based Error Correction Code decoder using CrossMPT backbone.
    Separates Variable Nodes (VN) and Check Nodes (CN) with cross-attention.
    """
    
    def __init__(self, args, device, dropout=0.0):
        super(SB_ECC, self).__init__()
        
        self.device = device
        self.d_model = args.d_model
        self._N = args.code.n  # Block length
        
        # Register parity check matrix
        self.register_buffer('pc_matrix', args.code.pc_matrix.transpose(0, 1).float())
        M = self.pc_matrix.size(1)  # Number of parity checks
        self.max_syndrome = M
        
        # Noise schedule parameters
        self.register_buffer("sigma_min", torch.tensor(args.sigma_min))
        self.register_buffer("sigma_max", torch.tensor(args.sigma_max))
        
        # EMA for training stability
        self.ema = EMA(0.9, flag_run=True)
        
        # Solver caching
        self._solver_cache = {}
        self._current_solver_type = None
        self._current_solver = None
        
        # CrossMPT Encoder
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model, dropout=dropout)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)
        self.decoder = CrossMPTEncoder(
            CrossMPTEncoderLayer(args.d_model, c(attn), c(ff), dropout),
            args.N_dec
        )
        
        # Separate embeddings for VNs and CNs
        self.src_embed_VN = nn.Parameter(torch.empty((self._N, args.d_model)))
        self.src_embed_CN = nn.Parameter(torch.empty((M, args.d_model)))
        nn.init.xavier_uniform_(self.src_embed_VN)
        nn.init.xavier_uniform_(self.src_embed_CN)
        
        # Output layers
        self.oned_final_embed = nn.Linear(args.d_model, 1)
        self.out_fc = nn.Linear(self._N + M, self._N)
        
        # Time embedding (syndrome-based)
        self.time_embed_fn = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, args.d_model)
        )
        
        # Build attention masks
        self._build_masks(args.code)

    def _build_masks(self, code):
        """Build attention masks for CrossMPT based on code structure."""
        def build_mask_VN(code):
            # Mask for VN update (VN attends to connected CNs)
            mask = torch.zeros(code.pc_matrix.size(0), code.n)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    mask[ii, jj] += 1
            mask = mask.transpose(0, 1)  # (N, M)
            return ~(mask > 0).unsqueeze(0).unsqueeze(0)

        def build_mask_CN(code):
            # Mask for CN update (CN attends to connected VNs)
            mask = torch.zeros(code.pc_matrix.size(0), code.n)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    mask[ii, jj] += 1
            return ~(mask > 0).unsqueeze(0).unsqueeze(0)

        self.register_buffer('src_mask_VN', build_mask_VN(code))
        self.register_buffer('src_mask_CN', build_mask_CN(code))

    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma(t) using linear schedule."""
        return linear_sigma_t(t, float(self.sigma_min), float(self.sigma_max))

    def _get_solver(self, solver_type: str):
        """Get a solver instance with caching."""
        if self._current_solver_type == solver_type.lower() and self._current_solver is not None:
            return self._current_solver
        
        if solver_type.lower() in self._solver_cache:
            self._current_solver = self._solver_cache[solver_type.lower()]
            self._current_solver_type = solver_type.lower()
            return self._current_solver
        
        from solvers import get_solver
        solver = get_solver(solver_type, self, float(self.sigma_min), float(self.sigma_max), self.device)
        self._solver_cache[solver_type.lower()] = solver
        self._current_solver = solver
        self._current_solver_type = solver_type.lower()
        return solver

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, y: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor = None):
        """
        Forward pass: predict noise from noisy codeword.
        
        Args:
            y: Noisy codeword (B, N)
            t: Time step (B, 1) - ground truth for training
            x0_pred: Unused (API compatibility)
            
        Returns:
            score: Score function estimate
            predicted_noise: Predicted noise
        """
        # Compute syndrome
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long().float(), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)  # (B, M)
        
        # Prepare VN and CN embeddings
        VN = y.unsqueeze(-1) * self.src_embed_VN.unsqueeze(0)  # (B, N, d_model)
        CN = self.src_embed_CN.unsqueeze(0) * syndrome.unsqueeze(-1)  # (B, M, d_model)
        
        # CrossMPT decoder
        emb_vn, emb_cn = self.decoder(VN, CN, self.src_mask_VN, self.src_mask_CN)
        
        # Concatenate and project to output
        emb = torch.cat([emb_vn, emb_cn], dim=1)  # (B, N+M, d_model)
        predicted_noise = self.out_fc(self.oned_final_embed(emb).squeeze(-1))  # (B, N)
        
        # Compute score
        sigma = self._sigma_t(t)
        score = -predicted_noise / sigma
        
        return score, predicted_noise

    # ------------------------------------------------------------------
    # Loss Function
    # ------------------------------------------------------------------
    def loss(self, x_0, current_epoch: int = 0):
        """Compute denoising score matching loss."""
        B = x_0.size(0)
        x_0 = x_0.to(self.device)
        
        # Sample random time
        t = torch.rand(B, 1, device=x_0.device)
        
        # Sample noise
        z = torch.randn_like(x_0)
        
        # Create noisy input
        sigma = self._sigma_t(t)
        y = x_0 + sigma * z
        
        # Forward pass
        _, predicted_noise = self(y=y, t=t)
        
        # Denoising loss
        loss = 0.5 * (predicted_noise - z).pow(2).mean()
        
        return loss, {}

    # ------------------------------------------------------------------
    # Decoding / Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, y: torch.Tensor, num_steps: int = 10, solver_type: str = 'euler'):
        """
        Decode noisy observation using iterative denoising.
        
        Args:
            y: Noisy channel observation (B, N)
            num_steps: Number of denoising steps
            solver_type: ODE solver type ('euler' or 'dpm')
            
        Returns:
            final_result: Decoded codeword (B, N)
            intermediate_results: List of intermediate results
            convergence_steps: Step at which each sample converged
            syndrome_history: Syndrome counts over iterations
        """
        B, _ = y.shape
        y = y.to(self.device)
        xt = y.clone()
        x0_pred = None

        # Track convergence
        convergence_steps = torch.full((B,), num_steps - 1, dtype=torch.long, device=self.device)
        converged_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        intermediate_results = []
        syndrome_history = []

        # Compute step sizes
        dt = 1.0 / num_steps
        delta_sigma = (self.sigma_max - self.sigma_min) * dt

        # Get solver
        solver = self._get_solver(solver_type.lower())
        solver.reset_history()

        for i in range(num_steps):
            intermediate_results.append(xt.sign().clone())
            
            # Check convergence
            syndrome_count = (torch.matmul(sign_to_bin(xt.sign()), self.pc_matrix) % 2).sum(dim=-1)
            syndrome_history.append(syndrome_count.cpu())
            
            newly_converged = (syndrome_count == 0) & ~converged_mask
            if newly_converged.any():
                convergence_steps[newly_converged] = i
                converged_mask[newly_converged] = True
            
            if i == num_steps - 1 or converged_mask.all():
                break

            # Update non-converged samples
            not_converged_mask = ~converged_mask
            if not_converged_mask.any():
                xt, x0_pred = solver.step(xt, x0_pred, delta_sigma, not_converged_mask)

        final_result = intermediate_results[-1]
        
        # Pad results
        while len(intermediate_results) < num_steps:
            intermediate_results.append(final_result.clone())

        return final_result, intermediate_results, convergence_steps, torch.stack(syndrome_history).t()

    def p_sample_loop(self, cur_y, solver_type: str = 'euler', num_steps: int = 10, **kwargs):
        """Compatibility wrapper for test interface."""
        return self.decode(cur_y, num_steps=num_steps, solver_type=solver_type.lower())
