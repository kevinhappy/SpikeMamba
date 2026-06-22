# Copyright (c) 2024, Spike Mamba Implementation

import os
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try direct import, fallback to relative path if mamba_ssm.__init__ fails
try:
    from mamba_ssm.ops.spike_interface import (
        spike_scan_ref,
        sequential_spike_scan,
        StateDependentGate,
        LeakCoefficient,
        create_gate,
        GateType,
    )
except ImportError:
    import importlib.util
    _spike_interface_path = os.path.join(
        os.path.dirname(__file__), '..', 'ops', 'spike_interface.py'
    )
    _spec = importlib.util.spec_from_file_location("spike_interface", _spike_interface_path)
    _spike_interface = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_spike_interface)

    spike_scan_ref = _spike_interface.spike_scan_ref
    sequential_spike_scan = _spike_interface.sequential_spike_scan
    StateDependentGate = _spike_interface.StateDependentGate
    LeakCoefficient = _spike_interface.LeakCoefficient
    create_gate = _spike_interface.create_gate
    GateType = _spike_interface.GateType


class TemporalSpikeEncoder(nn.Module):
    """
    φ_enc: Temporal spike encoder / event projection
    
    Encodes input u_t into spike-compatible representation z_t.
    """
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.linear = nn.Linear(d_model, d_inner, bias=bias, **factory_kwargs)
        self.bn = nn.BatchNorm1d(d_inner)
        self.act = nn.SiLU()
    
    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: (B, L, D_model) input sequence
        Returns:
            z: (B, L, D_inner) encoded spike representation
        """
        # Linear projection
        z = self.linear(u)  # (B, L, D_inner)
        
        # BatchNorm (requires transposing)
        z = self.bn(z.transpose(1, 2)).transpose(1, 2)
        
        # Activation
        z = self.act(z)
        
        return z


class SpikeMixingFrontEnd(nn.Module):
    """
    W_in: Input mixing layer
    
    Computes input current I_t = W_in @ z_t
    Supports: linear, dwconv1d, grouped convolution
    """
    def __init__(
        self,
        d_inner: int,
        mixing_type: Literal['linear', 'dwconv1d', 'grouped'] = 'linear',
        kernel_size: int = 4,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.mixing_type = mixing_type
        self.d_inner = d_inner
        
        if mixing_type == 'linear':
            self.mixer = nn.Linear(d_inner, d_inner, bias=bias, **factory_kwargs)
        elif mixing_type == 'dwconv1d':
            # Depthwise Conv1D (similar to original Mamba)
            self.mixer = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=kernel_size,
                groups=d_inner,  # depthwise
                padding=kernel_size - 1,
                bias=bias,
                **factory_kwargs,
            )
        elif mixing_type == 'grouped':
            # Grouped convolution
            self.mixer = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=kernel_size,
                groups=groups if groups > 1 else d_inner // 4,
                padding=kernel_size - 1,
                bias=bias,
                **factory_kwargs,
            )
        else:
            raise ValueError(f"Unknown mixing_type: {mixing_type}")
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, L, D) encoded input
        Returns:
            I: (B, L, D) input current
        """
        if self.mixing_type == 'linear':
            return self.mixer(z)
        else:
            # Conv1D expects (B, D, L)
            z_t = z.transpose(1, 2)  # (B, D, L)
            I_t = self.mixer(z_t)
            # Truncate to original length (causal padding)
            I_t = I_t[..., :z.size(1)]
            return I_t.transpose(1, 2)  # (B, L, D)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with selective update
    
    Implements LIF Spike State Update (Eqs. 3-5)
    """
    def __init__(
        self,
        d_inner: int,
        V_th: float = 1.0,
        alpha_init: float = 0.9,
        learnable_alpha: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_inner = d_inner
        self.V_th = V_th
        
        if learnable_alpha:
            self.leak = LeakCoefficient(d_inner, init_value=alpha_init, **factory_kwargs)
        else:
            self.register_buffer(
                'alpha',
                torch.full((d_inner,), alpha_init, **factory_kwargs)
            )
            self.leak = None
    
    @property
    def alpha(self) -> Tensor:
        if self.leak is not None:
            return self.leak.alpha
        return self._buffers['alpha']
    
    def forward(
        self,
        v_prev: Tensor,
        I_t: Tensor,
        gate: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Single timestep LIF update
        
        Args:
            v_prev: (B, D) previous membrane potential
            I_t: (B, D) input current
            gate: (B,) binary gate
        
        Returns:
            v_t: (B, D) updated membrane
            s_t: (B, D) spike output
        """
        return spike_scan_ref(v_prev, I_t, gate, self.alpha, self.V_th)


class SpikeOutputHead(nn.Module):
    """
    ψ_out: Output readout layer
    
    Converts membrane/spike state to output (spike-rate or logits)
    """
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        output_type: Literal['membrane', 'spike', 'both'] = 'membrane',
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.output_type = output_type
        self.bn = nn.BatchNorm1d(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
    
    def forward(self, v: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            v: (B, L, D_inner) membrane potentials
            s: (B, L, D_inner) spike outputs
        
        Returns:
            y: (B, L, D_model) output
        """
        if self.output_type == 'membrane':
            h = v
        elif self.output_type == 'spike':
            h = s
        elif self.output_type == 'both':
            h = v * s  # gated by spikes
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
        # BatchNorm + projection
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        y = self.out_proj(h)
        
        return y


class SpikeMamba(nn.Module):
    """
    Temporal Spike-based Mamba (Spike-domain SSM)

    SpikeMamba: Spike-domain SSM with State-Dependent Gate (Eq. 6).

    Args:
        d_model: Input/output dimension
        d_state: SSM state dimension (kept for API compatibility)
        d_conv: Convolution kernel size for mixing
        expand: Expansion factor for inner dimension
        V_th: Spike threshold
        gate_type: 'state_dependent' (Eq. 6)
        gate_params: Parameters for the gate module
        mixing_type: 'linear', 'dwconv1d', or 'grouped'
        output_type: 'membrane', 'spike', or 'both'
        device, dtype: Torch device and dtype
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        V_th: float = 1.0,
        gate_type: GateType = 'state_dependent',
        gate_params: Optional[dict] = None,
        mixing_type: Literal['linear', 'dwconv1d', 'grouped'] = 'dwconv1d',
        output_type: Literal['membrane', 'spike', 'both'] = 'membrane',
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.V_th = V_th
        self.gate_type = gate_type
        self.layer_idx = layer_idx
        
        # Temporal spike encoder φ_enc
        self.encoder = TemporalSpikeEncoder(
            d_model, self.d_inner, **factory_kwargs
        )
        
        # Event-driven gate G
        gate_params = gate_params or {}
        self.gate = create_gate(gate_type, self.d_inner, **gate_params)
        
        # Spike mixing front-end W_in
        self.mixing = SpikeMixingFrontEnd(
            self.d_inner,
            mixing_type=mixing_type,
            kernel_size=d_conv,
            **factory_kwargs,
        )
        
        # LIF neuron (leak α and threshold V_th)
        self.lif = LIFNeuron(
            self.d_inner,
            V_th=V_th,
            **factory_kwargs,
        )
        
        # Output head ψ_out
        self.output_head = SpikeOutputHead(
            self.d_inner,
            d_model,
            output_type=output_type,
            **factory_kwargs,
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        return_spikes: bool = False,
    ) -> Tensor:
        """
        Forward pass (Section 3.2)
        
        Args:
            hidden_states: (B, L, D) input sequence
            inference_params: Optional inference parameters (for compatibility)
            return_spikes: If True, also return spike outputs
        
        Returns:
            out: (B, L, D) output sequence
            spikes: (B, L, D_inner) spike outputs (if return_spikes=True)
        """
        B, L, D = hidden_states.shape
        
        # Step 1: Temporal spike encoding z_t = φ_enc(u_t)
        z = self.encoder(hidden_states)  # (B, L, D_inner)
        
        # Sequential spike scan with state-dependent gate (Eq. 6)
        v, s = sequential_spike_scan(
            z=z,
            alpha=self.lif.alpha,
            gate_module=self.gate,
            W_in=self.mixing,
            V_th=self.V_th,
        )
        
        # Step 6: Output readout y_t = ψ_out(v_t, s_t)
        out = self.output_head(v, s)  # (B, L, D)
        
        if return_spikes:
            return out, s
        return out
    
    def get_gate_stats(self, hidden_states: Tensor) -> dict:
        """
        Get approximate gate activation statistics.

        For the state-dependent gate, gate activity is approximated from spike density.
        """
        _, s = self.forward(hidden_states, return_spikes=True)
        gate_approx = (s.abs().sum(dim=-1) > 0).float()
        return {
            'gate_activation_ratio': gate_approx.mean().item(),
            'gate_per_sample': gate_approx.mean(dim=1),  # (B,)
        }


# =============================================================================
# Factory function for easy model creation
# =============================================================================

def create_spike_mamba(
    d_model: int,
    gate_type: GateType = 'state_dependent',
    **kwargs,
) -> SpikeMamba:
    """
    Factory function to create SpikeMamba.

    Args:
        d_model: Model dimension
        gate_type: 'state_dependent' (Eq. 6)
        **kwargs: Additional SpikeMamba parameters

    Returns:
        SpikeMamba module
    """
    return SpikeMamba(d_model=d_model, gate_type=gate_type, **kwargs)
