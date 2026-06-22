# Copyright (c) 2024, Spike Mamba Tests

import pytest
import torch
import torch.nn as nn

from mamba_ssm.ops.spike_interface import (
    spike_scan_ref,
    state_dependent_gate,
    StateDependentGate,
    sequential_spike_scan,
    reparameterize_ssm_to_spike,
    LeakCoefficient,
    create_gate,
)
from mamba_ssm.modules.spike_mamba import (
    SpikeMamba,
    TemporalSpikeEncoder,
    SpikeMixingFrontEnd,
    LIFNeuron,
    SpikeOutputHead,
    create_spike_mamba,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 64


@pytest.fixture
def d_model():
    return 128


@pytest.fixture
def d_inner():
    return 256


# =============================================================================
# Tests for spike_interface.py
# =============================================================================

class TestSpikeScanRef:
    """Tests for LIF spike state update (Eqs. 3-5)"""
    
    def test_gate_zero_preserves_state(self, device):
        """When gate=0, membrane should remain unchanged"""
        B, D = 2, 64
        v_prev = torch.randn(B, D, device=device)
        I_t = torch.randn(B, D, device=device)
        gate = torch.zeros(B, device=device)
        alpha = torch.ones(D, device=device) * 0.9
        
        v_t, s_t = spike_scan_ref(v_prev, I_t, gate, alpha, V_th=1.0)
        
        assert torch.allclose(v_t, v_prev), "Membrane should be preserved when gate=0"
        assert (s_t == 0).all(), "No spikes when gate=0"
    
    def test_gate_one_updates_state(self, device):
        """When gate=1, membrane should be updated via LIF"""
        B, D = 2, 64
        v_prev = torch.zeros(B, D, device=device)
        I_t = torch.ones(B, D, device=device) * 0.5
        gate = torch.ones(B, device=device)
        alpha = torch.ones(D, device=device) * 0.9
        
        v_t, s_t = spike_scan_ref(v_prev, I_t, gate, alpha, V_th=1.0)
        
        expected_v = alpha * v_prev + I_t  # 0.9 * 0 + 0.5 = 0.5
        assert torch.allclose(v_t, expected_v), "LIF update should be applied when gate=1"
    
    def test_spike_generation(self, device):
        """Spikes should be generated when membrane exceeds threshold"""
        B, D = 2, 64
        v_prev = torch.ones(B, D, device=device) * 0.6
        I_t = torch.ones(B, D, device=device) * 0.5  # 0.9*0.6 + 0.5 = 1.04 > 1.0
        gate = torch.ones(B, device=device)
        alpha = torch.ones(D, device=device) * 0.9
        
        v_t, s_t = spike_scan_ref(v_prev, I_t, gate, alpha, V_th=1.0)
        
        assert (s_t == 1).all(), "Spikes should be generated when v ≥ V_th"
        # After soft reset: v = v_tilde - V_th * s = 1.04 - 1.0 = 0.04
        assert (v_t < 0.1).all(), "Membrane should be reset after spike"
    
    def test_batch_processing(self, device):
        """Test with batch dimension and mixed gates"""
        B, D = 4, 32
        v_prev = torch.randn(B, D, device=device)
        I_t = torch.randn(B, D, device=device)
        gate = torch.tensor([0, 1, 0, 1], dtype=torch.float, device=device)
        alpha = torch.ones(D, device=device) * 0.9
        
        v_t, s_t = spike_scan_ref(v_prev, I_t, gate, alpha, V_th=1.0)
        
        # Check gate=0 samples preserved
        assert torch.allclose(v_t[0], v_prev[0])
        assert torch.allclose(v_t[2], v_prev[2])


class TestStateDependentGate:
    """Tests for state-dependent gate (Eq. 6)"""
    
    def test_functional_gate(self, device):
        """Test functional state_dependent_gate"""
        B, D = 2, 64
        z_t = torch.randn(B, D, device=device)
        v_prev = torch.randn(B, D, device=device)
        v_prev2 = torch.randn(B, D, device=device)
        
        gate = state_dependent_gate(z_t, v_prev, v_prev2)
        
        assert gate.shape == (B,), f"Gate shape should be (B,), got {gate.shape}"
        assert ((gate == 0) | (gate == 1)).all(), "Gate should be binary"
    
    def test_rate_threshold_trigger(self, device):
        """Gate should activate when spike rate exceeds kappa"""
        B, D = 2, 64
        z_t = torch.ones(B, D, device=device) * 2.0  # High rate
        v_prev = torch.zeros(B, D, device=device)
        v_prev2 = torch.zeros(B, D, device=device)
        
        gate = state_dependent_gate(z_t, v_prev, v_prev2, kappa=0.5)
        
        assert (gate == 1).all(), "High spike rate should trigger gate"
    
    def test_state_change_trigger(self, device):
        """Gate should activate when state change exceeds delta"""
        B, D = 2, 64
        z_t = torch.zeros(B, D, device=device)  # Low rate
        v_prev = torch.ones(B, D, device=device) * 2.0
        v_prev2 = torch.zeros(B, D, device=device)  # Large delta
        
        gate = state_dependent_gate(z_t, v_prev, v_prev2, kappa=10.0, delta=0.5)
        
        assert (gate == 1).all(), "Large state change should trigger gate"
    
    def test_module_gate(self, device, d_inner):
        """Test StateDependentGate module"""
        B = 2
        z_t = torch.randn(B, d_inner, device=device)
        v_prev = torch.randn(B, d_inner, device=device)
        v_prev2 = torch.randn(B, d_inner, device=device)
        
        gate_module = StateDependentGate(d_inner).to(device)
        gate = gate_module(z_t, v_prev, v_prev2)
        
        assert gate.shape == (B,)


class TestReparameterize:
    """Tests for SSM to spike reparameterization"""
    
    def test_learnable_alpha(self, device):
        """Test learnable alpha initialization"""
        d_inner = 64
        
        alpha, W_init = reparameterize_ssm_to_spike(
            A=None, B=None, d_inner=d_inner, device=device
        )
        
        assert alpha.shape == (d_inner,)
        assert (alpha > 0).all() and (alpha < 1).all(), "Alpha should be in (0, 1)"
        assert torch.allclose(alpha, torch.ones_like(alpha) * 0.9, atol=0.1)
    
    def test_from_pretrained_A(self, device):
        """Test alpha from pretrained A matrix"""
        D, N = 64, 16
        A = -torch.ones(D, N, device=device) * 0.5  # Negative A
        
        alpha, _ = reparameterize_ssm_to_spike(A=A, dt=1.0, device=device)
        
        assert alpha.shape == (D,)
        assert (alpha > 0).all() and (alpha < 1).all()
    
    def test_leak_coefficient_module(self, device):
        """Test LeakCoefficient module"""
        d_inner = 64
        leak = LeakCoefficient(d_inner, init_value=0.9).to(device)
        
        alpha = leak.alpha
        
        assert alpha.shape == (d_inner,)
        assert torch.allclose(alpha, torch.ones(d_inner, device=device) * 0.9, atol=0.05)


class TestSequentialSpikeScan:
    """Tests for sequential spike scan"""
    
    def test_output_shape(self, device, d_inner):
        """Test output shape matches input"""
        B, L = 2, 64
        z = torch.randn(B, L, d_inner, device=device)
        alpha = torch.ones(d_inner, device=device) * 0.9
        gate_module = StateDependentGate(d_inner).to(device)
        W_in = nn.Linear(d_inner, d_inner).to(device)
        
        v, s = sequential_spike_scan(z, alpha, gate_module, W_in)
        
        assert v.shape == (B, L, d_inner)
        assert s.shape == (B, L, d_inner)


# =============================================================================
# Tests for spike_mamba.py
# =============================================================================

class TestTemporalSpikeEncoder:
    """Tests for TemporalSpikeEncoder"""
    
    def test_forward_shape(self, device, batch_size, seq_len, d_model, d_inner):
        encoder = TemporalSpikeEncoder(d_model, d_inner).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        z = encoder(x)
        
        assert z.shape == (batch_size, seq_len, d_inner)


class TestSpikeMixingFrontEnd:
    """Tests for SpikeMixingFrontEnd"""
    
    @pytest.mark.parametrize("mixing_type", ['linear', 'dwconv1d', 'grouped'])
    def test_mixing_types(self, device, batch_size, seq_len, d_inner, mixing_type):
        mixer = SpikeMixingFrontEnd(d_inner, mixing_type=mixing_type).to(device)
        z = torch.randn(batch_size, seq_len, d_inner, device=device)
        
        I = mixer(z)
        
        assert I.shape == (batch_size, seq_len, d_inner)


class TestLIFNeuron:
    """Tests for LIFNeuron module"""
    
    def test_forward(self, device, batch_size, d_inner):
        lif = LIFNeuron(d_inner).to(device)
        v_prev = torch.randn(batch_size, d_inner, device=device)
        I_t = torch.randn(batch_size, d_inner, device=device)
        gate = torch.ones(batch_size, device=device)
        
        v_t, s_t = lif(v_prev, I_t, gate)
        
        assert v_t.shape == (batch_size, d_inner)
        assert s_t.shape == (batch_size, d_inner)


class TestSpikeOutputHead:
    """Tests for SpikeOutputHead"""
    
    @pytest.mark.parametrize("output_type", ['membrane', 'spike', 'both'])
    def test_output_types(self, device, batch_size, seq_len, d_model, d_inner, output_type):
        head = SpikeOutputHead(d_inner, d_model, output_type=output_type).to(device)
        v = torch.randn(batch_size, seq_len, d_inner, device=device)
        s = torch.randint(0, 2, (batch_size, seq_len, d_inner), device=device).float()
        
        y = head(v, s)
        
        assert y.shape == (batch_size, seq_len, d_model)


class TestSpikeMamba:
    """Tests for SpikeMamba module"""

    def test_state_dependent_forward(self, device, batch_size, seq_len, d_model):
        """Test forward with state-dependent gate"""
        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        out = model(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_return_spikes(self, device, batch_size, seq_len, d_model):
        """Test returning spike outputs"""
        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        out, spikes = model(x, return_spikes=True)

        assert out.shape == (batch_size, seq_len, d_model)
        assert spikes.shape == (batch_size, seq_len, d_model * 2)  # d_inner = 2 * d_model

    def test_gradient_flow(self, device, batch_size, seq_len, d_model):
        """Test gradient flows through the model"""
        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("mixing_type", ['linear', 'dwconv1d', 'grouped'])
    def test_mixing_types(self, device, batch_size, seq_len, d_model, mixing_type):
        """Test all mixing types"""
        model = SpikeMamba(d_model, mixing_type=mixing_type).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        out = model(x)
        
        assert out.shape == (batch_size, seq_len, d_model)


class TestCreateSpikeMamba:
    """Tests for factory function"""

    def test_create_state_dependent(self, device, d_model):
        model = create_spike_mamba(d_model, gate_type='state_dependent').to(device)
        assert model.gate_type == 'state_dependent'


class TestGateStats:
    """Tests for gate activation statistics"""

    def test_gate_activation_ratio(self, device, batch_size, seq_len, d_model):
        """Gate activation ratio should be in [0, 1]"""
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        stats = model.get_gate_stats(x)
        assert 0 <= stats['gate_activation_ratio'] <= 1

    def test_determinism(self, device, batch_size, seq_len, d_model):
        """Test deterministic output for same input"""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        model.eval()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_training_step(self, device, batch_size, seq_len, d_model):
        """Test a single training step"""
        model = SpikeMamba(d_model, gate_type='state_dependent').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        target = torch.randn(batch_size, seq_len, d_model, device=device)

        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)

    def test_output_shape(self, device, batch_size, seq_len, d_model):
        """Verify SpikeMamba output shape: (B, L, D)"""
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        model = SpikeMamba(d_model).to(device)
        out = model(x)
        assert out.shape == (batch_size, seq_len, d_model)
