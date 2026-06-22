#!/usr/bin/env python3
"""
SpikeMamba Unified Inference Script (All 5 Datasets)

Evaluates SpikeMamba (Large, Medium, Tiny) on:
  - ADFTD, PTB, PTB-XL, CHB-MIT, MIT-BIH
  - 10 bootstrap runs → mean ± std

Metrics:
  - ADFTD, PTB, PTB-XL, MIT-BIH: accuracy, precision, recall, f1, auroc, auprc
  - CHB-MIT: same + sensitivity (pre-ictal recall)

Usage:
    python inference_all.py --run_all                # All 5 datasets × 3 sizes
    python inference_all.py --dataset ptb --all_sizes # One dataset, 3 sizes
    python inference_all.py --dataset chbmit --size large
    python inference_all.py --run_all --n_runs 20
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

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
spike_mamba = load_module_from_path(
    "spike_mamba",
    os.path.join(os.path.dirname(__file__), "mamba_ssm/modules/spike_mamba.py")
)
SpikeMamba = spike_mamba.SpikeMamba


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
        'data_path': './checkpoints',  # .npz cached dataset
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

BASE_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']


# =============================================================================
# Data Loaders
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
        # Find the .npz file
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
        elif flag == 'TEST':
            self.X, self.y = test_x, test_y
        else:
            self.X, self.y = test_x, test_y

        self.max_seq_len = self.X.shape[-1]  # 360
        self.num_channels = self.X.shape[1]  # 2 (channels-first after transpose)
        self.num_classes = len(unique_labels)
        self._needs_permute = True  # (C, T) → (T, C)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        x = z_score_normalization(x)
        # (2, 360) → (360, 2)
        x = x.permute(1, 0)
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

    # For MIT-BIH the model input_dim is the last dim after permute
    if dataset_name == 'mitbih':
        num_ch = 2  # (T, 2)
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
# Model
# =============================================================================

class SpikeMambaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, n_layers=8,
                 gate_type='state_dependent', dropout=0.1):
        super().__init__()
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
# Inference & Metrics
# =============================================================================

def run_inference(model, data_loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def compute_metrics_single(preds, targets, probs, num_classes, is_chbmit=False):
    """Compute all metrics for a single evaluation."""
    avg = 'binary' if num_classes == 2 else 'macro'
    results = {}
    results['accuracy'] = accuracy_score(targets, preds) * 100
    results['precision'] = precision_score(targets, preds, average=avg, zero_division=0) * 100
    results['recall'] = recall_score(targets, preds, average=avg, zero_division=0) * 100
    results['f1'] = f1_score(targets, preds, average=avg, zero_division=0) * 100

    try:
        if num_classes == 2:
            results['auroc'] = roc_auc_score(targets, probs[:, 1]) * 100
            results['auprc'] = average_precision_score(targets, probs[:, 1]) * 100
        else:
            targets_onehot = label_binarize(targets, classes=list(range(num_classes)))
            results['auroc'] = roc_auc_score(targets_onehot, probs, average='macro',
                                              multi_class='ovr') * 100
            results['auprc'] = average_precision_score(targets_onehot, probs,
                                                        average='macro') * 100
    except ValueError:
        results['auroc'] = 0.0
        results['auprc'] = 0.0

    # CHB-MIT extra: sensitivity = recall of pre-ictal class (label=1)
    if is_chbmit:
        results['sensitivity'] = recall_score(targets, preds, pos_label=1, zero_division=0) * 100

    return results


def bootstrap_metrics(preds, targets, probs, num_classes, n_runs=10, is_chbmit=False):
    """Bootstrap resampling for mean ± std."""
    metric_names = BASE_METRICS + (['sensitivity'] if is_chbmit else [])
    n_samples = len(targets)
    all_runs = {m: [] for m in metric_names}

    for i in range(n_runs):
        rng = np.random.RandomState(seed=i)
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        boot_preds = preds[indices]
        boot_targets = targets[indices]
        boot_probs = probs[indices]

        metrics = compute_metrics_single(boot_preds, boot_targets, boot_probs,
                                         num_classes, is_chbmit=is_chbmit)
        for m in metric_names:
            all_runs[m].append(metrics[m])

    summary = {}
    for m in metric_names:
        vals = np.array(all_runs[m])
        summary[m] = {'mean': np.mean(vals), 'std': np.std(vals), 'all': vals.tolist()}

    return summary


def evaluate_model(dataset_name, size, gate_type='state_dependent',
                   checkpoint_dir='./checkpoints', device='cuda',
                   batch_size=64, n_runs=10, hp_suffix=''):
    """Load checkpoint and evaluate with bootstrap resampling."""
    cfg = DATASET_CONFIGS[dataset_name]
    mcfg = MODEL_CONFIGS[size]
    is_chbmit = (dataset_name == 'chbmit')

    ckpt_path = os.path.join(checkpoint_dir,
                             f'{cfg["ckpt_prefix"]}_{gate_type}_{size}{hp_suffix}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        return None

    # Load data
    test_loader, info = get_test_loader(dataset_name, batch_size=batch_size)
    print(f"\n  Loading {dataset_name.upper()} test set: {info['test_size']} samples, "
          f"{info['num_channels']}ch, {info['seq_len']}len")

    # Build model
    model = SpikeMambaClassifier(
        input_dim=info['num_channels'],
        num_classes=info['num_classes'],
        d_model=mcfg['d_model'],
        n_layers=mcfg['n_layers'],
        gate_type=gate_type,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', '?')}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    # Single inference → bootstrap
    preds, targets, probs = run_inference(model, test_loader, device)

    print(f"  Running {n_runs} bootstrap iterations...")
    summary = bootstrap_metrics(preds, targets, probs, info['num_classes'],
                                n_runs=n_runs, is_chbmit=is_chbmit)

    # Print results
    metric_names = BASE_METRICS + (['sensitivity'] if is_chbmit else [])
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} | {size.upper()} ({n_runs} bootstrap runs)")
    print(f"{'='*60}")
    for m in metric_names:
        label = m.upper() if m in ('auroc', 'auprc') else m.capitalize()
        print(f"  {label:<14}: {summary[m]['mean']:.2f} ± {summary[m]['std']:.2f}")

    return summary


# =============================================================================
# Run All
# =============================================================================

def run_all(args):
    all_datasets = ['adftd', 'ptb', 'ptbxl', 'chbmit', 'mitbih']
    all_results = {}

    for dataset_name in all_datasets:
        all_results[dataset_name] = {}
        for size in ['large', 'medium', 'tiny']:
            result = evaluate_model(
                dataset_name, size,
                gate_type=args.gate_type,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                batch_size=args.batch_size,
                n_runs=args.n_runs,
            )
            if result:
                all_results[dataset_name][size] = result

    # =========================================================================
    # Summary tables
    # =========================================================================

    # Table 1: ADFTD, PTB, PTB-XL, MIT-BIH (standard metrics)
    print("\n\n" + "=" * 120)
    print(f" SUMMARY TABLE — Standard Metrics (mean±std, {args.n_runs} bootstrap runs)")
    print("=" * 120)

    header = (f"{'Dataset':<10} {'Size':<8} {'Accuracy':>14} {'Precision':>14} "
              f"{'Recall':>14} {'F1':>14} {'AUROC':>14} {'AUPRC':>14}")
    print(header)
    print("-" * 120)

    for ds in ['adftd', 'ptb', 'ptbxl', 'mitbih']:
        for size in ['large', 'medium', 'tiny']:
            r = all_results.get(ds, {}).get(size)
            if r:
                cells = []
                for m in BASE_METRICS:
                    cells.append(f"{r[m]['mean']:.2f}±{r[m]['std']:.2f}")
                print(f"{ds.upper():<10} {size:<8} " + " ".join(f"{c:>14}" for c in cells))
        print("-" * 120)

    # Table 2: CHB-MIT (with sensitivity)
    if 'chbmit' in all_results and len(all_results['chbmit']) > 0:
        chbmit_metrics = BASE_METRICS + ['sensitivity']
        print(f"\n{'='*130}")
        print(f" CHB-MIT — With Sensitivity (mean±std, {args.n_runs} bootstrap runs)")
        print("=" * 130)
        header2 = (f"{'Size':<8} {'Accuracy':>14} {'Precision':>14} {'Recall':>14} "
                   f"{'F1':>14} {'AUROC':>14} {'AUPRC':>14} {'Sensitivity':>14}")
        print(header2)
        print("-" * 130)
        for size in ['large', 'medium', 'tiny']:
            r = all_results['chbmit'].get(size)
            if r:
                cells = []
                for m in chbmit_metrics:
                    cells.append(f"{r[m]['mean']:.2f}±{r[m]['std']:.2f}")
                print(f"{size:<8} " + " ".join(f"{c:>14}" for c in cells))
        print("-" * 130)

    # =========================================================================
    # LaTeX output
    # =========================================================================
    print(f"\n{'='*100}")
    print(" LaTeX-ready rows (for paper tables)")
    print("=" * 100)

    for ds in ['adftd', 'ptb', 'ptbxl', 'mitbih']:
        print(f"\n  % {ds.upper()}")
        for size in ['large', 'medium', 'tiny']:
            r = all_results.get(ds, {}).get(size)
            if r:
                name = f"SpikeMamba-{size.capitalize()}"
                parts = []
                for m in BASE_METRICS:
                    parts.append(f"{r[m]['mean']:.2f}$_{{\\pm{r[m]['std']:.2f}}}$")
                print(f"  {name} & {' & '.join(parts)} \\\\")

    if 'chbmit' in all_results and len(all_results['chbmit']) > 0:
        chbmit_metrics_latex = ['accuracy', 'sensitivity', 'f1', 'auroc', 'auprc']
        print(f"\n  % CHB-MIT (with sensitivity)")
        for size in ['large', 'medium', 'tiny']:
            r = all_results['chbmit'].get(size)
            if r:
                name = f"SpikeMamba-{size.capitalize()}"
                parts = []
                for m in chbmit_metrics_latex:
                    parts.append(f"{r[m]['mean']:.2f}$_{{\\pm{r[m]['std']:.2f}}}$")
                print(f"  {name} & {' & '.join(parts)} \\\\")

    # Save
    save_path = os.path.join(args.checkpoint_dir, 'inference_all_5datasets.pt')
    torch.save(all_results, save_path)
    print(f"\nResults saved to {save_path}")

    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SpikeMamba Inference (All 5 Datasets)')
    parser.add_argument('--dataset', type=str, default='ptbxl',
                        choices=['ptbxl', 'ptb', 'adftd', 'chbmit', 'mitbih'])
    parser.add_argument('--size', type=str, default='large',
                        choices=['large', 'medium', 'tiny'])
    parser.add_argument('--gate_type', type=str, default='state_dependent')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of bootstrap resampling runs')
    parser.add_argument('--hp_suffix', type=str, default='',
                        help='HP suffix for checkpoint name (e.g. _vth0.5, _kappa0.01)')
    parser.add_argument('--all_sizes', action='store_true',
                        help='Run all 3 sizes for the specified dataset')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all 5 datasets × all 3 sizes')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Device: {args.device}")
    print(f"Bootstrap runs: {args.n_runs}")

    if args.run_all:
        run_all(args)
    elif args.all_sizes:
        for size in ['large', 'medium', 'tiny']:
            evaluate_model(args.dataset, size, gate_type=args.gate_type,
                           checkpoint_dir=args.checkpoint_dir,
                           device=args.device, batch_size=args.batch_size,
                           n_runs=args.n_runs)
    else:
        evaluate_model(args.dataset, args.size, gate_type=args.gate_type,
                       checkpoint_dir=args.checkpoint_dir,
                       device=args.device, batch_size=args.batch_size,
                       n_runs=args.n_runs, hp_suffix=args.hp_suffix)


if __name__ == '__main__':
    main()
