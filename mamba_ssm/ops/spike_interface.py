# Copyright (c) 2024, Spike Mamba Implementation

from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


# =============================================================================
# LIF Spike State Update (Eqs. 3-5)
# =============================================================================

def spike_scan_ref(
    v_prev: Tensor,      # (B, D) or (B, L, D) membrane potential
    I_t: Tensor,         # (B, D) or (B, L, D) input current
    gate: Tensor,        # (B,) or (B, L) binary gate {0, 1}
    alpha: Tensor,       # (D,) leak coefficient ∈ (0, 1)
    V_th: float = 1.0    # threshold
) -> Tuple[Tensor, Tensor]:
    """
    LIF Spike State Update (Eqs. 3-5)

    if g_t = 0:
        v_t ← v_{t-1}, s_t ← 0
    else:
        ṽ_t ← α ⊙ v_{t-1} + I_t     # decay + input accumulation
        s_t ← Θ[ṽ_t ≥ V_th]          # spike generation
        v_t ← ṽ_t - V_th · s_t       # soft reset

    Returns:
        v_t: Updated membrane potential
        s_t: Spike output {0, 1}
    """
    # Expand gate for broadcasting
    if gate.dim() < v_prev.dim():
        gate = gate.unsqueeze(-1)  # (B,) -> (B, 1) or (B, L) -> (B, L, 1)
    
    # LIF update when gate=1
    v_tilde = alpha * v_prev + I_t
    s_t = (v_tilde >= V_th).float()
    v_updated = v_tilde - V_th * s_t
    
    # Selective update: keep previous state when gate=0
    v_t = torch.where(gate > 0.5, v_updated, v_prev)
    s_t = torch.where(gate > 0.5, s_t, torch.zeros_like(s_t))
    
    return v_t, s_t


# =============================================================================
# State-Dependent Gate (Eq. 6)
# =============================================================================

def state_dependent_gate(
    z_t: Tensor,         # (B, D) encoded input at time t
    v_prev: Tensor,      # (B, D) previous membrane
    v_prev2: Tensor,     # (B, D) v_{t-2} for delta computation
    kappa: float = 0.1,  # rate threshold κ
    delta: float = 0.1,  # state change threshold δ
    lambda1: float = 0.5,
    lambda2: float = 0.5,
    tau_g: float = 0.5,
) -> Tensor:
    """
    Event-driven State-Dependent Gate (Eq. 6)

        r_t ← rate(z_t)                    # spike rate
        Δv_t ← ||v_{t-1} - v_{t-2}||_1     # state change magnitude
        e_t ← λ1 · r_t + λ2 · Δv_t         # energy
        if (r_t ≥ κ) or (Δv_t ≥ δ) or (e_t ≥ τ_g):
            g_t ← 1
        else:
            g_t ← 0

    Returns:
        gate: (B,) binary gate values
    """
    r_t = z_t.abs().mean(dim=-1)  # (B,)
    delta_v_t = (v_prev - v_prev2).abs().sum(dim=-1)  # (B,)
    e_t = lambda1 * r_t + lambda2 * delta_v_t
    gate = ((r_t >= kappa) | (delta_v_t >= delta) | (e_t >= tau_g)).float()
    return gate


class StateDependentGate(nn.Module):
    """
    Learnable State-Dependent Gate (Eq. 6)

    Computes gate based on both input and previous membrane states.
    Sequential processing required due to state dependency.
    """
    def __init__(
        self,
        d_inner: int,
        kappa: float = 0.1,
        delta: float = 0.1,
        lambda1: float = 0.5,
        lambda2: float = 0.5,
        tau_g: float = 0.5,
        learnable: bool = True,
    ):
        super().__init__()
        self.d_inner = d_inner

        if learnable:
            self.kappa = nn.Parameter(torch.tensor(kappa))
            self.delta = nn.Parameter(torch.tensor(delta))
            self.lambda1 = nn.Parameter(torch.tensor(lambda1))
            self.lambda2 = nn.Parameter(torch.tensor(lambda2))
            self.tau_g = nn.Parameter(torch.tensor(tau_g))
        else:
            self.register_buffer('kappa', torch.tensor(kappa))
            self.register_buffer('delta', torch.tensor(delta))
            self.register_buffer('lambda1', torch.tensor(lambda1))
            self.register_buffer('lambda2', torch.tensor(lambda2))
            self.register_buffer('tau_g', torch.tensor(tau_g))

    def forward(
        self,
        z_t: Tensor,
        v_prev: Tensor,
        v_prev2: Tensor,
    ) -> Tensor:
        """
        Args:
            z_t: (B, D) encoded input at timestep t
            v_prev: (B, D) membrane at t-1
            v_prev2: (B, D) membrane at t-2
        Returns:
            gate: (B,) binary gate
        """
        return state_dependent_gate(
            z_t, v_prev, v_prev2,
            kappa=self.kappa.item() if isinstance(self.kappa, nn.Parameter) else self.kappa,
            delta=self.delta.item() if isinstance(self.delta, nn.Parameter) else self.delta,
            lambda1=self.lambda1.item() if isinstance(self.lambda1, nn.Parameter) else self.lambda1,
            lambda2=self.lambda2.item() if isinstance(self.lambda2, nn.Parameter) else self.lambda2,
            tau_g=self.tau_g.item() if isinstance(self.tau_g, nn.Parameter) else self.tau_g,
        )


# =============================================================================
# Sequential Spike Scan with State-Dependent Gate (Eq. 6)
# =============================================================================

def sequential_spike_scan(
    z: Tensor,                        # (B, L, D) encoded input
    alpha: Tensor,                    # (D,) leak coefficient
    gate_module: StateDependentGate,  # gate module
    W_in: nn.Module,                  # input mixing layer
    V_th: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Sequential Spike Scan with State-Dependent Gate (Eq. 6)

    Args:
        z: (B, L, D) encoded input
        alpha: (D,) leak coefficients
        gate_module: StateDependentGate module
        W_in: Input mixing module
        V_th: Spike threshold

    Returns:
        v_out: (B, L, D) membrane potentials
        s_out: (B, L, D) spike outputs
    """
    B, L, D = z.shape
    device = z.device
    dtype = z.dtype

    I = W_in(z)  # (B, L, D)

    v_out = torch.zeros(B, L, D, device=device, dtype=dtype)
    s_out = torch.zeros(B, L, D, device=device, dtype=dtype)

    v_prev = torch.zeros(B, D, device=device, dtype=dtype)
    v_prev2 = torch.zeros(B, D, device=device, dtype=dtype)

    for t in range(L):
        z_t = z[:, t, :]  # (B, D)
        I_t = I[:, t, :]  # (B, D)

        g_t = gate_module(z_t, v_prev, v_prev2)  # (B,)
        v_t, s_t = spike_scan_ref(v_prev, I_t, g_t, alpha, V_th)

        v_out[:, t, :] = v_t
        s_out[:, t, :] = s_t

        v_prev2 = v_prev
        v_prev = v_t

    return v_out, s_out



# =============================================================================
# SSM to Spike Accumulation Coefficient Reparameterization
# =============================================================================

def reparameterize_ssm_to_spike(
    A: Optional[Tensor] = None,  # (D, N) continuous SSM A matrix
    B: Optional[Tensor] = None,  # (D, N) continuous SSM B matrix
    dt: float = 1.0,             # time step Δt
    d_inner: int = None,         # dimension for learnable init
    device=None,
    dtype=None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Reparameterize SSM (A, B) into Spike Accumulation Coefficients (α, W_in).

    Returns:
        alpha: (D,) leak coefficients ∈ (0, 1)
        W_in_init: (D, N) or None
    """
    factory_kwargs = {'device': device, 'dtype': dtype}

    if A is not None:
        A_bar = torch.exp(A * dt)
        if A_bar.dim() == 2:
            alpha = A_bar[:, 0]
        else:
            alpha = A_bar
        alpha = alpha.clamp(0.01, 0.99)
    else:
        assert d_inner is not None, "d_inner required for learnable init"
        # Initialize θ via inverse-softplus so that α = exp(-softplus(θ)) ≈ 0.9
        target_sp = -torch.log(torch.tensor(0.9))
        theta_init = torch.log(torch.exp(target_sp) - 1)
        theta = torch.full((d_inner,), theta_init.item(), **factory_kwargs)
        alpha = torch.exp(-F.softplus(theta))  # ≈ 0.9

    W_in_init = None
    if B is not None:
        W_in_init = B.clone()

    return alpha, W_in_init


class LeakCoefficient(nn.Module):
    """
    Learnable leak coefficient α ∈ (0, 1)

    Parameterized as α = exp(-softplus(θ)) to ensure valid range.
    """
    def __init__(
        self,
        d_inner: int,
        init_value: float = 0.9,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        target_sp = -torch.log(torch.tensor(init_value))
        theta_init = torch.log(torch.exp(target_sp) - 1)  # inverse softplus

        self.theta = nn.Parameter(
            torch.full((d_inner,), theta_init.item(), **factory_kwargs)
        )

    @property
    def alpha(self) -> Tensor:
        return torch.exp(-F.softplus(self.theta))

    def forward(self) -> Tensor:
        return self.alpha


# =============================================================================
# Gate Factory
# =============================================================================

GateType = Literal['state_dependent']


def create_gate(
    gate_type: GateType,
    d_inner: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create gate module.

    Args:
        gate_type: 'state_dependent'
        d_inner: Inner dimension
        **kwargs: Additional gate parameters

    Returns:
        StateDependentGate module
    """
    if gate_type == 'state_dependent':
        return StateDependentGate(
            d_inner=d_inner,
            kappa=kwargs.get('kappa', 0.1),
            delta=kwargs.get('delta', 0.1),
            lambda1=kwargs.get('lambda1', 0.5),
            lambda2=kwargs.get('lambda2', 0.5),
            tau_g=kwargs.get('tau_g', 0.5),
            learnable=kwargs.get('learnable', True),
        )
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}. Use 'state_dependent'")
