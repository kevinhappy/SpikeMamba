#!/usr/bin/env python3
"""
Theoretical Energy Consumption Analysis for SpikeMamba

Based on methodology from:
  "Spike-driven Transformer" (Yao et al., NeurIPS 2024)
  "Spike-driven Transformer V2" (Yao et al., ICLR 2024)

Energy constants (45nm technology, 32-bit FP):
  E_MAC = 4.6 pJ  (multiply-and-accumulate)
  E_AC  = 0.9 pJ  (addition only)

ANN energy:  E_MAC × FLOPs  (all operations are MAC)
SNN energy:  E_AC × T × R × FLOPs  (spike-driven ops: sparse additions)

For SpikeMamba:
  - T = 1 (direct training, single forward pass, no multi-timestep simulation)
  - R = spike firing rate (measured from trained models on test data)
  - Gate activation ratio determines which timesteps are computed

Components breakdown:
  ANN ops (continuous input):  input_proj, classifier MLP
  Spike-driven ops:           Encoder, Mixing(DWConv), LIF scan, Output head

Usage:
    python energy_analysis.py --dataset ptbxl --size large
    python energy_analysis.py --dataset chbmit --all_sizes
    python energy_analysis.py --dataset mitbih --size large
    python energy_analysis.py --run_all
"""

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

spike_interface = load_module_from_path(
    "spike_interface",
    os.path.join(os.path.dirname(__file__), "mamba_ssm/ops/spike_interface.py")
)
spike_mamba_module = load_module_from_path(
    "spike_mamba",
    os.path.join(os.path.dirname(__file__), "mamba_ssm/modules/spike_mamba.py")
)
SpikeMamba = spike_mamba_module.SpikeMamba


# =============================================================================
# Energy Constants (45nm, 32-bit floating point)
# =============================================================================

E_MAC = 4.6   # pJ per MAC (multiply-and-accumulate)
E_AC  = 0.9   # pJ per AC  (addition)

# SpikeMamba: direct training, T=1
T = 1


# =============================================================================
# Configs
# =============================================================================

MODEL_CONFIGS = {
    'large':  {'d_model': 512, 'n_layers': 8},
    'medium': {'d_model': 256, 'n_layers': 4},
    'tiny':   {'d_model': 64,  'n_layers': 4},
}

DATASET_CONFIGS = {
    'ptbxl': {
        'data_path': './data/PTB-XL',
        'class_names': ['Normal', 'MI', 'STTC', 'CD', 'HYP'],
        'num_classes': 5,
        'ckpt_prefix': 'spike_mamba_ptbxl',
    },
    'ptb': {
        'data_path': './data/PTB',
        'class_names': ['Healthy', 'MI'],
        'num_classes': 2,
        'ckpt_prefix': 'spike_mamba_ptb',
    },
    'adftd': {
        'data_path': './data/ADFTD',
        'class_names': ['CN', 'FTD', 'AD'],
        'num_classes': 3,
        'ckpt_prefix': 'spike_mamba_adftd',
    },
    'chbmit': {
        'data_path': './checkpoints',
        'class_names': ['Inter-ictal', 'Pre-ictal'],
        'num_classes': 2,
        'ckpt_prefix': 'spike_mamba_chbmit_mixed',
    },
    'mitbih': {
        'data_path': './data/MIT-BIH',
        'class_names': ['F', 'N', 'Q', 'S', 'V'],
        'num_classes': 5,
        'ckpt_prefix': 'spike_mamba_mitbih',
    },
}


# =============================================================================
# Data Loaders (minimal, for measuring spike rates)
# =============================================================================

def normalize_batch_ts(batch):
    mean_values = batch.mean(axis=1, keepdims=True)
    std_values = batch.std(axis=1, keepdims=True)
    std_values[std_values == 0] = 1.0
    return (batch - mean_values) / std_values


class PTBXLLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(a, b)
        self.X, self.y = self._load(flag)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]
        self.num_classes = len(np.unique(self.y))

    def _split_ids(self, a, b):
        data_list = np.load(self.label_path)
        ids_by_class = {}
        for c in range(5):
            ids_by_class[c] = list(data_list[np.where(data_list[:, 0] == c)][:, 1])
        train, val, test = [], [], []
        for c in range(5):
            lst = ids_by_class[c]
            train += lst[:int(a * len(lst))]
            val += lst[int(a * len(lst)):int(b * len(lst))]
            test += lst[int(b * len(lst)):]
        return train, val, test

    def _load(self, flag):
        subject_label = np.load(self.label_path)
        filenames = natsorted(os.listdir(self.data_path))
        ids = {'TRAIN': self.train_ids, 'VAL': self.val_ids, 'TEST': self.test_ids}.get(flag, subject_label[:, 1])
        features, labels = [], []
        for j, fname in enumerate(filenames):
            trial_label = subject_label[j]
            for trial in np.load(self.data_path + fname):
                if j + 1 in ids:
                    features.append(trial)
                    labels.append(trial_label)
        X, y = np.array(features), np.array(labels)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(a, b)
        self.X, self.y = self._load(flag)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]
        self.num_classes = len(np.unique(self.y))

    def _split_ids(self, a, b):
        data_list = np.load(self.label_path)
        hc = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        mi = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        train = hc[:int(a*len(hc))] + mi[:int(a*len(mi))]
        val = hc[int(a*len(hc)):int(b*len(hc))] + mi[int(a*len(mi)):int(b*len(mi))]
        test = hc[int(b*len(hc)):] + mi[int(b*len(mi)):]
        return train, val, test

    def _load(self, flag):
        subject_label = np.load(self.label_path)
        filenames = natsorted(os.listdir(self.data_path))
        ids = {'TRAIN': self.train_ids, 'VAL': self.val_ids, 'TEST': self.test_ids}.get(flag, subject_label[:, 1])
        features, labels = [], []
        for j, fname in enumerate(filenames):
            trial_label = subject_label[j]
            for trial in np.load(self.data_path + fname):
                if j + 1 in ids:
                    features.append(trial)
                    labels.append(trial_label)
        X, y = np.array(features), np.array(labels)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(a, b)
        self.X, self.y = self._load(flag)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]
        self.num_classes = len(np.unique(self.y))

    def _split_ids(self, a, b):
        data_list = np.load(self.label_path)
        cn = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        ftd = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        ad = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        train = cn[:int(a*len(cn))] + ftd[:int(a*len(ftd))] + ad[:int(a*len(ad))]
        val = cn[int(a*len(cn)):int(b*len(cn))] + ftd[int(a*len(ftd)):int(b*len(ftd))] + ad[int(a*len(ad)):int(b*len(ad))]
        test = cn[int(b*len(cn)):] + ftd[int(b*len(ftd)):] + ad[int(b*len(ad)):]
        return train, val, test

    def _load(self, flag):
        subject_label = np.load(self.label_path)
        filenames = natsorted(os.listdir(self.data_path))
        ids = {'TRAIN': self.train_ids, 'VAL': self.val_ids, 'TEST': self.test_ids}.get(flag, subject_label[:, 1])
        features, labels = [], []
        for j, fname in enumerate(filenames):
            trial_label = subject_label[j]
            for trial in np.load(self.data_path + fname):
                if j + 1 in ids:
                    features.append(trial)
                    labels.append(trial_label)
        X, y = np.array(features), np.array(labels)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
    def __len__(self):
        return len(self.y)


class CHBMITLoader(Dataset):
    """Loads test set from cached segment-mixed .npz file."""
    def __init__(self, cache_dir, flag=None):
        candidates = sorted([f for f in os.listdir(cache_dir)
                            if f.startswith('chbmit_mixed_dataset_') and f.endswith('.npz')])
        if not candidates:
            raise FileNotFoundError(f"No chbmit_mixed_dataset_*.npz in {cache_dir}")
        npz_path = os.path.join(cache_dir, candidates[0])
        print(f"    CHB-MIT cache: {npz_path}")
        data = np.load(npz_path)

        split_map = {'TRAIN': ('X_train', 'y_train'),
                     'VAL': ('X_val', 'y_val'),
                     'TEST': ('X_test', 'y_test')}
        xk, yk = split_map.get(flag, ('X_test', 'y_test'))
        self.X = data[xk]
        self.y = data[yk]
        self.max_seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]
        self.num_classes = 2

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
    def __len__(self):
        return len(self.y)


def z_score_normalization(data):
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    else:
        data = data.float()
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    return (data - mean) / (std + 1e-8)


class MITBIHLoader(Dataset):
    """MIT-BIH with internal 6:2:2 stratified split."""
    def __init__(self, root_path, flag=None, seed=42):
        data_path = os.path.join(root_path, 'mitdb_data.npy')
        label_path = os.path.join(root_path, 'mitdb_group.npy')

        raw_data = np.load(data_path)
        raw_labels = np.load(label_path)

        if raw_data.ndim == 3 and raw_data.shape[2] == 2:
            raw_data = raw_data.transpose(0, 2, 1)

        unique_labels = np.unique(raw_labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        raw_labels_int = np.array([label_map[l] for l in raw_labels])

        train_x, temp_x, train_y, temp_y = train_test_split(
            raw_data, raw_labels_int, test_size=0.4, random_state=seed, stratify=raw_labels_int)
        vali_x, test_x, vali_y, test_y = train_test_split(
            temp_x, temp_y, test_size=0.5, random_state=seed, stratify=temp_y)

        if flag == 'TRAIN':
            self.X, self.y = train_x, train_y
        elif flag == 'VAL':
            self.X, self.y = vali_x, vali_y
        else:
            self.X, self.y = test_x, test_y

        self.max_seq_len = self.X.shape[-1]  # 360
        self.num_channels = self.X.shape[1]  # 2
        self.num_classes = len(unique_labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        x = z_score_normalization(x)
        x = x.permute(1, 0)  # (2, 360) → (360, 2)
        return x, torch.tensor(self.y[idx], dtype=torch.long)

    def __len__(self):
        return len(self.y)


LOADER_MAP = {
    'ptbxl': PTBXLLoader,
    'ptb': PTBLoader,
    'adftd': ADFTDLoader,
    'chbmit': CHBMITLoader,
    'mitbih': MITBIHLoader,
}

def get_test_loader(dataset_name, batch_size=64):
    cfg = DATASET_CONFIGS[dataset_name]
    LoaderClass = LOADER_MAP[dataset_name]
    test_dataset = LoaderClass(cfg['data_path'], flag='TEST')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    if dataset_name == 'mitbih':
        num_ch = 2
        seq_len = 360
    else:
        num_ch = test_dataset.num_channels
        seq_len = test_dataset.max_seq_len

    info = {
        'num_channels': num_ch,
        'seq_len': seq_len,
        'num_classes': test_dataset.num_classes,
        'test_size': len(test_dataset),
    }
    return test_loader, info


# =============================================================================
# Model (same as training scripts)
# =============================================================================

class SpikeMambaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, n_layers=8,
                 gate_type='state_dependent', dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            SpikeMamba(d_model=d_model, gate_type=gate_type,
                       mixing_type='dwconv1d', d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)
            x = self.dropout(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# =============================================================================
# Spike Rate Measurement
# =============================================================================

def measure_spike_rates(model, data_loader, device, max_batches=50):
    """
    Measure spike firing rates and gate activation ratios from trained model.
    
    Hooks into each SpikeMamba layer to capture spike tensors (s_out)
    and gate activation ratios.
    
    Returns per-layer stats:
        - spike_firing_rate: proportion of non-zero spikes
        - gate_activation_ratio: proportion of timesteps with gate=1
    """
    model.eval()
    
    # Monkey-patch spike_scan_ref to capture spike outputs
    original_spike_scan = spike_interface.spike_scan_ref
    
    spike_stats = {
        'total_spikes': 0,
        'total_elements': 0,
    }
    
    # Collect per-layer stats using return_spikes
    layer_spike_rates = {i: [] for i in range(len(model.layers))}
    layer_gate_rates = {i: [] for i in range(len(model.layers))}
    
    # We use forward hooks on SpikeMamba layers, calling with return_spikes=True
    # But since the classifier's forward doesn't pass return_spikes, we monkey-patch
    original_forwards = []
    for i, layer in enumerate(model.layers):
        original_forwards.append(layer.forward)
        
        def make_hooked_forward(layer_ref, layer_idx):
            orig = layer_ref.forward
            def hooked_forward(hidden_states, inference_params=None, return_spikes=False):
                out, spikes = orig(hidden_states, return_spikes=True)
                
                # Spike firing rate: fraction of non-zero elements
                sfr = (spikes != 0).float().mean().item()
                layer_spike_rates[layer_idx].append(sfr)
                
                # Gate activation ratio (approximated from spike activity for state_dependent)
                gar = (spikes.abs().sum(dim=-1) > 0).float().mean().item()
                layer_gate_rates[layer_idx].append(gar)
                
                return out
            return hooked_forward
        
        layer.forward = make_hooked_forward(layer, i)
    
    # Run inference
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
    
    # Restore original forwards
    for i, layer in enumerate(model.layers):
        layer.forward = original_forwards[i]
    
    # Aggregate stats
    results = {}
    all_sfr = []
    all_gar = []
    
    for i in range(len(model.layers)):
        sfr = np.mean(layer_spike_rates[i]) if layer_spike_rates[i] else 0.0
        gar = np.mean(layer_gate_rates[i]) if layer_gate_rates[i] else 1.0
        results[f'layer_{i}'] = {
            'spike_firing_rate': sfr,
            'gate_activation_ratio': gar,
        }
        all_sfr.append(sfr)
        all_gar.append(gar)
    
    results['avg_spike_firing_rate'] = np.mean(all_sfr)
    results['avg_gate_activation_ratio'] = np.mean(all_gar)
    
    return results


# =============================================================================
# FLOPs Computation
# =============================================================================

def compute_flops(input_dim, num_classes, d_model, n_layers, seq_len, expand=2, d_conv=4):
    """
    Compute FLOPs for each component of SpikeMambaClassifier.
    
    FLOPs are counted as number of multiply-accumulate operations.
    For Linear(in, out): FLOPs = in × out  (per token)
    For DWConv1d(channels, kernel): FLOPs = kernel × channels  (per token)
    
    Returns dict with FLOPs per component and categorized as ANN or spike-driven.
    """
    N = seq_len  # sequence length
    D = d_model
    D_inner = expand * D  # d_inner
    
    flops = OrderedDict()
    
    # =========================================================================
    # 1. Input Projection: Linear(input_dim, d_model)
    # =========================================================================
    flops['input_proj'] = {
        'flops': N * input_dim * D,
        'type': 'ann',  # continuous input
        'desc': f'Linear({input_dim}→{D}) × {N}',
    }
    
    # =========================================================================
    # 2. Per SpikeMamba Layer (× n_layers)
    # =========================================================================
    # Each SpikeMamba layer contains:
    #   a) TemporalSpikeEncoder: Linear(D, D_inner) + BN + SiLU
    #   b) SpikeMixingFrontEnd: DWConv1d(D_inner, kernel=d_conv)
    #   c) LIF Scan: membrane update + spike generation (per timestep)
    #   d) SpikeOutputHead: BN + Linear(D_inner, D)
    
    # a) Encoder: Linear(D → D_inner)
    encoder_flops = N * D * D_inner
    flops['encoder_per_layer'] = {
        'flops': encoder_flops,
        'total': encoder_flops * n_layers,
        'type': 'spike',  # receives residual stream but output feeds into spike pathway
        'desc': f'Linear({D}→{D_inner}) × {N}',
    }
    
    # b) Mixing DWConv1d: depthwise conv with kernel_size=d_conv
    mixing_flops = N * d_conv * D_inner  # depthwise: kernel × channels per position
    flops['mixing_per_layer'] = {
        'flops': mixing_flops,
        'total': mixing_flops * n_layers,
        'type': 'spike',  # operates on encoded spike representation
        'desc': f'DWConv1d(ch={D_inner}, k={d_conv}) × {N}',
    }
    
    # c) LIF Scan operations (per timestep, per neuron):
    #    - Leak: α × v_prev           → 1 MAC per neuron
    #    - Accumulate: + I_t          → 1 AC per neuron
    #    - Threshold comparison       → 1 CMP per neuron
    #    - Reset: v - V_th × s        → 1 MAC per neuron
    #    Total: ~3 MACs + 1 AC per neuron per timestep (simplified as 4 ops)
    lif_flops = N * D_inner * 4
    flops['lif_scan_per_layer'] = {
        'flops': lif_flops,
        'total': lif_flops * n_layers,
        'type': 'spike',  # core spike operation, gated by gate
        'desc': f'LIF({D_inner}) × {N} (leak+accum+threshold+reset)',
    }
    
    # d) Output head: Linear(D_inner → D)
    output_flops = N * D_inner * D
    flops['output_per_layer'] = {
        'flops': output_flops,
        'total': output_flops * n_layers,
        'type': 'spike',  # receives membrane/spike-modulated signal
        'desc': f'Linear({D_inner}→{D}) × {N}',
    }
    
    # Gate computation (small overhead)
    # StateDependentGate: scalar ops per timestep
    gate_flops = N * D_inner * 1
    flops['gate_per_layer'] = {
        'flops': gate_flops,
        'total': gate_flops * n_layers,
        'type': 'ann',  # gate computes on continuous values
        'desc': f'Gate: Linear({D_inner}→1) × {N}',
    }
    
    # =========================================================================
    # 3. Classifier MLP
    # =========================================================================
    # After mean pooling over sequence (no FLOPs), then:
    # Linear(D, D//2) + Linear(D//2, num_classes)
    cls_flops1 = D * (D // 2)
    cls_flops2 = (D // 2) * num_classes
    flops['classifier'] = {
        'flops': cls_flops1 + cls_flops2,
        'type': 'ann',  # continuous pooled features
        'desc': f'Linear({D}→{D//2}) + Linear({D//2}→{num_classes})',
    }
    
    # =========================================================================
    # Totals
    # =========================================================================
    total_ann_flops = 0
    total_spike_flops = 0
    
    for name, info in flops.items():
        f = info.get('total', info['flops'])
        if info['type'] == 'ann':
            total_ann_flops += f
        else:
            total_spike_flops += f
    
    return flops, total_ann_flops, total_spike_flops


# =============================================================================
# Energy Computation
# =============================================================================

def compute_energy(flops_dict, total_ann_flops, total_spike_flops,
                   avg_spike_rate, avg_gate_ratio, n_layers):
    """
    Compute theoretical energy consumption.
    
    ANN baseline (vanilla Mamba equivalent):
        E_ANN = E_MAC × (total_ann_flops + total_spike_flops)
        (All operations use MAC)
    
    SpikeMamba:
        E_spike_ops = E_AC × T × R_eff × total_spike_flops
        E_ann_ops = E_MAC × total_ann_flops
        E_SpikeMamba = E_ann_ops + E_spike_ops
        
    where R_eff = R × G (effective rate = spike rate × gate activation ratio)
    When gate=0, no computation at all (event-driven skip)
    When gate=1 but spike=0, still saves via sparse addition
    """
    total_flops = total_ann_flops + total_spike_flops
    
    # ANN energy: all operations use MAC
    E_ann_total = E_MAC * total_flops  # pJ
    
    # SpikeMamba energy
    # 1. ANN ops (input_proj, classifier, gate): still use MAC
    E_spike_ann_part = E_MAC * total_ann_flops
    
    # 2. Spike-driven ops: use AC × R, and gated by gate ratio
    #    Effective computation ratio = gate_ratio × spike_rate
    #    When gate=0: entire timestep is skipped (0 energy)
    #    When gate=1: MAC → AC, but only R fraction are non-zero additions
    R_eff = avg_gate_ratio * avg_spike_rate
    E_spike_driven_part = E_AC * T * R_eff * total_spike_flops
    
    E_spike_total = E_spike_ann_part + E_spike_driven_part
    
    # Energy ratio
    ratio = E_ann_total / E_spike_total if E_spike_total > 0 else float('inf')
    
    return {
        'E_ann_total': E_ann_total,           # pJ
        'E_spike_total': E_spike_total,       # pJ
        'E_spike_ann_part': E_spike_ann_part, # pJ (non-spike ops in SpikeMamba)
        'E_spike_driven_part': E_spike_driven_part,  # pJ (spike-driven ops)
        'ratio': ratio,                       # E_ANN / E_SpikeMamba
        'total_flops': total_flops,
        'R_eff': R_eff,
    }


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_model(dataset_name, size, gate_type='state_dependent',
                  checkpoint_dir='./checkpoints', device='cuda',
                  batch_size=64, max_batches=50, hp_suffix=''):
    """Run full energy analysis for one model."""
    cfg = DATASET_CONFIGS[dataset_name]
    mcfg = MODEL_CONFIGS[size]
    
    ckpt_path = os.path.join(checkpoint_dir,
                             f'{cfg["ckpt_prefix"]}_{gate_type}_{size}{hp_suffix}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        return None
    
    # Load data
    test_loader, info = get_test_loader(dataset_name, batch_size=batch_size)
    
    # Build model
    model = SpikeMambaClassifier(
        input_dim=info['num_channels'],
        num_classes=info['num_classes'],
        d_model=mcfg['d_model'],
        n_layers=mcfg['n_layers'],
        gate_type=gate_type,
    ).to(device)
    
    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} | {size.upper()} | {mcfg['n_layers']}L-{mcfg['d_model']}D")
    print(f"{'='*70}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Seq len: {info['seq_len']}, Channels: {info['num_channels']}, "
          f"Classes: {info['num_classes']}")
    
    # =========================================================================
    # Step 1: Measure spike firing rates
    # =========================================================================
    print(f"\n  [1/3] Measuring spike firing rates ({max_batches} batches)...")
    spike_stats = measure_spike_rates(model, test_loader, device, max_batches=max_batches)
    
    print(f"\n  {'Layer':<10} {'Spike Rate':>12} {'Gate Ratio':>12}")
    print(f"  {'-'*34}")
    for i in range(mcfg['n_layers']):
        s = spike_stats[f'layer_{i}']
        print(f"  Layer {i:<4} {s['spike_firing_rate']:>11.4f} {s['gate_activation_ratio']:>11.4f}")
    print(f"  {'-'*34}")
    print(f"  {'Average':<10} {spike_stats['avg_spike_firing_rate']:>11.4f} "
          f"{spike_stats['avg_gate_activation_ratio']:>11.4f}")
    
    # =========================================================================
    # Step 2: Compute FLOPs
    # =========================================================================
    print(f"\n  [2/3] Computing FLOPs...")
    flops_dict, total_ann_flops, total_spike_flops = compute_flops(
        input_dim=info['num_channels'],
        num_classes=info['num_classes'],
        d_model=mcfg['d_model'],
        n_layers=mcfg['n_layers'],
        seq_len=info['seq_len'],
    )
    
    total_flops = total_ann_flops + total_spike_flops
    print(f"\n  {'Component':<25} {'FLOPs':>15} {'Type':>8}")
    print(f"  {'-'*50}")
    for name, finfo in flops_dict.items():
        f = finfo.get('total', finfo['flops'])
        print(f"  {name:<25} {f:>15,} {finfo['type']:>8}")
    print(f"  {'-'*50}")
    print(f"  {'Total ANN FLOPs':<25} {total_ann_flops:>15,}")
    print(f"  {'Total Spike FLOPs':<25} {total_spike_flops:>15,}")
    print(f"  {'Total FLOPs':<25} {total_flops:>15,}")
    
    # =========================================================================
    # Step 3: Compute Energy
    # =========================================================================
    print(f"\n  [3/3] Computing theoretical energy...")
    energy = compute_energy(
        flops_dict, total_ann_flops, total_spike_flops,
        avg_spike_rate=spike_stats['avg_spike_firing_rate'],
        avg_gate_ratio=spike_stats['avg_gate_activation_ratio'],
        n_layers=mcfg['n_layers'],
    )
    
    print(f"\n  {'─'*50}")
    print(f"  Energy Consumption Comparison (45nm, 32-bit)")
    print(f"  {'─'*50}")
    print(f"  E_MAC = {E_MAC} pJ,  E_AC = {E_AC} pJ,  T = {T}")
    print(f"  Avg Spike Firing Rate (R): {spike_stats['avg_spike_firing_rate']:.4f}")
    print(f"  Avg Gate Activation Ratio: {spike_stats['avg_gate_activation_ratio']:.4f}")
    print(f"  Effective R (R × G):       {energy['R_eff']:.4f}")
    print(f"  {'─'*50}")
    print(f"  E_ANN (Mamba equiv):    {energy['E_ann_total']:>15.2e} pJ  "
          f"({energy['E_ann_total']/1e6:.2f} mJ)")
    print(f"  E_SpikeMamba:           {energy['E_spike_total']:>15.2e} pJ  "
          f"({energy['E_spike_total']/1e6:.2f} mJ)")
    print(f"  {'─'*50}")
    print(f"  E_ANN / E_SpikeMamba =  {energy['ratio']:.1f}×")
    print(f"  Energy Saving:          {(1 - 1/energy['ratio'])*100:.1f}%")
    
    return {
        'dataset': dataset_name,
        'size': size,
        'config': f"{mcfg['n_layers']}L-{mcfg['d_model']}D",
        'params': num_params,
        'seq_len': info['seq_len'],
        'num_channels': info['num_channels'],
        'spike_stats': spike_stats,
        'flops': {
            'total': total_flops,
            'ann': total_ann_flops,
            'spike': total_spike_flops,
        },
        'energy': energy,
    }


def run_all(args):
    """Run energy analysis for all datasets × all sizes."""
    all_results = {}
    
    all_datasets = ['ptbxl', 'ptb', 'adftd', 'chbmit', 'mitbih']
    for dataset_name in all_datasets:
        all_results[dataset_name] = {}
        for size in ['large', 'medium', 'tiny']:
            result = analyze_model(
                dataset_name, size,
                gate_type=args.gate_type,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
            )
            if result:
                all_results[dataset_name][size] = result
    
    # =========================================================================
    # Summary Tables
    # =========================================================================
    
    # Table 1: Spike Firing Rates
    print("\n\n" + "="*90)
    print(" TABLE 1: Spike Firing Rates and Gate Activation Ratios")
    print("="*90)
    header = f"{'Dataset':<10} {'Size':<8} {'Config':<10} {'Avg SFR':>10} {'Avg GAR':>10} {'R_eff':>10}"
    print(header)
    print("-"*60)
    for ds in all_datasets:
        for sz in ['large', 'medium', 'tiny']:
            r = all_results.get(ds, {}).get(sz)
            if r:
                ss = r['spike_stats']
                print(f"{ds:<10} {sz:<8} {r['config']:<10} "
                      f"{ss['avg_spike_firing_rate']:>9.4f} "
                      f"{ss['avg_gate_activation_ratio']:>9.4f} "
                      f"{r['energy']['R_eff']:>9.4f}")
        print("-"*60)
    
    # Table 2: Energy Comparison
    print("\n" + "="*105)
    print(" TABLE 2: Theoretical Energy Consumption Comparison")
    print("="*105)
    header2 = (f"{'Dataset':<10} {'Config':<10} {'Params':>10} "
               f"{'E_ANN(mJ)':>12} {'E_Spike(mJ)':>12} {'Ratio':>8} {'Saving':>8}")
    print(header2)
    print("-"*72)
    for ds in all_datasets:
        for sz in ['large', 'medium', 'tiny']:
            r = all_results.get(ds, {}).get(sz)
            if r:
                e = r['energy']
                print(f"{ds:<10} {r['config']:<10} {r['params']:>10,} "
                      f"{e['E_ann_total']/1e6:>11.2f} {e['E_spike_total']/1e6:>11.2f} "
                      f"{e['ratio']:>7.1f}× {(1-1/e['ratio'])*100:>6.1f}%")
        print("-"*72)
    
    # Table 3: Per-layer firing rates for largest model
    for ds in all_datasets:
        r = all_results.get(ds, {}).get('large')
        if r:
            n_layers = MODEL_CONFIGS['large']['n_layers']
            print(f"\n  Per-layer SFR for {ds.upper()} Large:")
            layer_strs = [f"L{i}:{r['spike_stats'][f'layer_{i}']['spike_firing_rate']:.3f}" 
                          for i in range(n_layers)]
            print(f"    {' | '.join(layer_strs)}")
    
    # Save
    save_path = os.path.join(args.checkpoint_dir, 'energy_analysis_results.pt')
    torch.save(all_results, save_path)
    print(f"\n\nResults saved to {save_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='SpikeMamba Energy Analysis')
    parser.add_argument('--dataset', type=str, default='ptbxl',
                        choices=['ptbxl', 'ptb', 'adftd', 'chbmit', 'mitbih'])
    parser.add_argument('--size', type=str, default='large',
                        choices=['large', 'medium', 'tiny'])
    parser.add_argument('--gate_type', type=str, default='state_dependent')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Max batches for spike rate measurement')
    parser.add_argument('--hp_suffix', type=str, default='',
                        help='HP suffix for checkpoint name (e.g. _vth0.5, _kappa0.01)')
    parser.add_argument('--all_sizes', action='store_true')
    parser.add_argument('--run_all', action='store_true')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Device: {args.device}")
    print(f"Energy constants: E_MAC={E_MAC}pJ, E_AC={E_AC}pJ (45nm, 32-bit)")
    print(f"SpikeMamba timestep: T={T}")

    if args.run_all:
        run_all(args)
    elif args.all_sizes:
        for size in ['large', 'medium', 'tiny']:
            analyze_model(args.dataset, size, gate_type=args.gate_type,
                          checkpoint_dir=args.checkpoint_dir, device=args.device,
                          batch_size=args.batch_size, max_batches=args.max_batches)
    else:
        analyze_model(args.dataset, args.size, gate_type=args.gate_type,
                      checkpoint_dir=args.checkpoint_dir, device=args.device,
                      batch_size=args.batch_size, max_batches=args.max_batches,
                      hp_suffix=args.hp_suffix)


if __name__ == '__main__':
    main()
