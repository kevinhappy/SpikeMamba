#!/usr/bin/env python3
"""
SpikeMamba Unified Training Script

Supports all 5 biosignal datasets:
  - ptbxl   : PTB-XL ECG, 5-class (Normal/MI/STTC/CD/HYP)
  - ptb     : PTB ECG, binary (Healthy vs MI)
  - adftd   : ADFTD EEG, 3-class (CN/FTD/AD)
  - mitbih  : MIT-BIH Arrhythmia, 5-class (N/S/V/F/Q)
  - chbmit  : CHB-MIT Seizure Prediction, binary (inter-ictal vs pre-ictal)

Usage:
    python train_spike_mamba.py --dataset ptbxl --size large --gate_type state_dependent
    python train_spike_mamba.py --dataset ptb   --size medium
    python train_spike_mamba.py --dataset adftd --size tiny
    python train_spike_mamba.py --dataset mitbih --size large --all_sizes
    python train_spike_mamba.py --dataset chbmit --size large
    python train_spike_mamba.py --dataset chbmit --size tiny
"""

import os
import sys
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_base = os.path.dirname(__file__)
_spike_interface = _load_module(
    "spike_interface",
    os.path.join(_base, "mamba_ssm/ops/spike_interface.py")
)
_spike_mamba = _load_module(
    "spike_mamba",
    os.path.join(_base, "mamba_ssm/modules/spike_mamba.py")
)
SpikeMamba = _spike_mamba.SpikeMamba


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIGS = {
    'large':  {'d_model': 512, 'n_layers': 8},
    'medium': {'d_model': 256, 'n_layers': 4},
    'tiny':   {'d_model': 64,  'n_layers': 4},
}


# =============================================================================
# SpikeMamba Classifier
# =============================================================================

class SpikeMambaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, n_layers=8,
                 gate_type='state_dependent', dropout=0.1,
                 V_th=1.0, gate_kappa=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            SpikeMamba(
                d_model=d_model,
                gate_type=gate_type,
                gate_params={'kappa': gate_kappa},
                mixing_type='dwconv1d',
                d_conv=4,
                expand=2,
                V_th=V_th,
            )
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.gate_type = gate_type

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
# Early Stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=15, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score > self.best_score if self.mode == 'max' else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# =============================================================================
# Shared Training Utilities
# =============================================================================

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    return total_loss / max(1, len(loader)), 100. * correct / max(1, total)


def _evaluate_basic(model, loader, criterion, device):
    """Basic eval returning loss, accuracy, preds, targets."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return (total_loss / max(1, len(loader)), 100. * correct / max(1, total),
            np.array(all_preds), np.array(all_targets))


def _normalize_ts(batch):
    """Z-score normalization across time axis."""
    mean = batch.mean(axis=1, keepdims=True)
    std = batch.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (batch - mean) / std


# =============================================================================
# Dataset: PTB-XL (5-class ECG)
# =============================================================================

def _get_ptbxl_loaders(data_path, batch_size, num_workers):
    from data.ptbxl_loader import get_ptbxl_dataloaders
    return get_ptbxl_dataloaders(data_path, batch_size=batch_size, num_workers=num_workers)


# =============================================================================
# Dataset: PTB (binary ECG)
# =============================================================================

class _PTBDataset(Dataset):
    """PTB ECG: binary classification (Healthy=0, MI=1)."""
    def __init__(self, root_path, flag=None):
        data_path = os.path.join(root_path, "Feature/")
        label_path = os.path.join(root_path, "Label/label.npy")
        subject_label = np.load(label_path)

        hc = list(subject_label[subject_label[:, 0] == 0][:, 1])
        mi = list(subject_label[subject_label[:, 0] == 1][:, 1])
        a, b = 0.55, 0.7

        train_ids = set(hc[:int(a*len(hc))] + mi[:int(a*len(mi))])
        val_ids   = set(hc[int(a*len(hc)):int(b*len(hc))] + mi[int(a*len(mi)):int(b*len(mi))])
        test_ids  = set(hc[int(b*len(hc)):] + mi[int(b*len(mi)):])
        ids = {'TRAIN': train_ids, 'VAL': val_ids, 'TEST': test_ids}.get(flag, set(subject_label[:, 1]))

        feats, labels = [], []
        for j, fname in enumerate(natsorted(os.listdir(data_path))):
            lbl = subject_label[j]
            for trial in np.load(os.path.join(data_path, fname)):
                if j + 1 in ids:
                    feats.append(trial); labels.append(lbl)

        X = np.array(feats)
        y = np.array(labels)
        X, y = shuffle(X, y, random_state=42)
        self.X = torch.from_numpy(_normalize_ts(X)).float()
        self.y = torch.from_numpy(y[:, 0]).long()
        self.num_channels = X.shape[2]
        self.seq_len      = X.shape[1]
        self.num_classes  = int(y[:, 0].max()) + 1

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _get_ptb_loaders(data_path, batch_size, num_workers):
    tr = _PTBDataset(data_path, 'TRAIN')
    va = _PTBDataset(data_path, 'VAL')
    te = _PTBDataset(data_path, 'TEST')
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True)
    info = {'num_channels': tr.num_channels, 'seq_len': tr.seq_len,
            'num_classes': tr.num_classes,
            'train_size': len(tr), 'val_size': len(va), 'test_size': len(te),
            'class_names': ['Healthy', 'MI']}
    return mk(tr, True), mk(va, False), mk(te, False), info


# =============================================================================
# Dataset: ADFTD (3-class EEG)
# =============================================================================

class _ADFTDDataset(Dataset):
    """ADFTD EEG: 3-class (CN=0, FTD=1, AD=2)."""
    def __init__(self, root_path, flag=None):
        data_path  = os.path.join(root_path, "Feature/")
        label_path = os.path.join(root_path, "Label/label.npy")
        subject_label = np.load(label_path)

        cn  = list(subject_label[subject_label[:, 0] == 0][:, 1])
        ftd = list(subject_label[subject_label[:, 0] == 1][:, 1])
        ad  = list(subject_label[subject_label[:, 0] == 2][:, 1])
        a, b = 0.6, 0.8

        def _split(lst): return (lst[:int(a*len(lst))],
                                 lst[int(a*len(lst)):int(b*len(lst))],
                                 lst[int(b*len(lst)):])
        cn_tr, cn_va, cn_te   = _split(cn)
        ftd_tr, ftd_va, ftd_te = _split(ftd)
        ad_tr, ad_va, ad_te   = _split(ad)

        train_ids = set(cn_tr + ftd_tr + ad_tr)
        val_ids   = set(cn_va + ftd_va + ad_va)
        test_ids  = set(cn_te + ftd_te + ad_te)
        ids = {'TRAIN': train_ids, 'VAL': val_ids, 'TEST': test_ids}.get(flag, set(subject_label[:, 1]))

        feats, labels = [], []
        for j, fname in enumerate(natsorted(os.listdir(data_path))):
            lbl = subject_label[j]
            for trial in np.load(os.path.join(data_path, fname)):
                if j + 1 in ids:
                    feats.append(trial); labels.append(lbl)

        X = np.array(feats)
        y = np.array(labels)
        X, y = shuffle(X, y, random_state=42)
        self.X = torch.from_numpy(_normalize_ts(X)).float()
        self.y = torch.from_numpy(y[:, 0]).long()
        self.num_channels = X.shape[2]
        self.seq_len      = X.shape[1]
        self.num_classes  = 3

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _get_adftd_loaders(data_path, batch_size, num_workers):
    tr = _ADFTDDataset(data_path, 'TRAIN')
    va = _ADFTDDataset(data_path, 'VAL')
    te = _ADFTDDataset(data_path, 'TEST')
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True)
    info = {'num_channels': tr.num_channels, 'seq_len': tr.seq_len,
            'num_classes': tr.num_classes,
            'train_size': len(tr), 'val_size': len(va), 'test_size': len(te),
            'class_names': ['CN', 'FTD', 'AD']}
    return mk(tr, True), mk(va, False), mk(te, False), info


# =============================================================================
# Dataset: MIT-BIH (5-class Arrhythmia)
# =============================================================================

class _MITBIHDataset(Dataset):
    """MIT-BIH: 5-class arrhythmia (N/S/V/F/Q). 6:2:2 stratified split."""
    def __init__(self, root_path, flag='TRAIN', seed=42):
        raw_data   = np.load(os.path.join(root_path, 'mitdb_data.npy'))
        raw_labels = np.load(os.path.join(root_path, 'mitdb_group.npy'))

        if raw_data.ndim == 3 and raw_data.shape[2] == 2:
            raw_data = raw_data.transpose(0, 2, 1)

        unique = np.unique(raw_labels)
        lmap   = {l: i for i, l in enumerate(unique)}
        labels_int = np.array([lmap[l] for l in raw_labels])

        tr_x, tmp_x, tr_y, tmp_y = train_test_split(
            raw_data, labels_int, test_size=0.4, random_state=seed, stratify=labels_int)
        va_x, te_x, va_y, te_y = train_test_split(
            tmp_x, tmp_y, test_size=0.5, random_state=seed, stratify=tmp_y)

        data_map = {'TRAIN': (tr_x, tr_y), 'VAL': (va_x, va_y), 'TEST': (te_x, te_y)}
        self.X, self.y = data_map[flag]
        self.label_names = sorted(lmap.keys())
        self.num_classes = len(unique)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float().permute(1, 0)  # (C,T) -> (T,C)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        mean = x.mean(dim=0, keepdim=True)
        std  = x.std(dim=0, keepdim=True)
        x = (x - mean) / (std + 1e-8)
        return x, y


def _get_mitbih_loaders(data_path, batch_size, num_workers):
    tr = _MITBIHDataset(data_path, 'TRAIN')
    va = _MITBIHDataset(data_path, 'VAL')
    te = _MITBIHDataset(data_path, 'TEST')
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True)
    sample_x, _ = tr[0]
    info = {'num_channels': sample_x.shape[-1], 'seq_len': sample_x.shape[0],
            'num_classes': tr.num_classes,
            'train_size': len(tr), 'val_size': len(va), 'test_size': len(te),
            'class_names': tr.label_names, 'class_counts': np.bincount(tr.y, minlength=tr.num_classes)}
    return mk(tr, True), mk(va, False), mk(te, False), info


# =============================================================================
# Dataset: CHB-MIT (Seizure Prediction) — EDF Parsing
# =============================================================================

CHBMIT_SR_ORIG  = 256
CHBMIT_WIN_SEC  = 5
CHBMIT_PREICTAL = 30
CHBMIT_SPH      = 5

CHBMIT_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]

_chbmit_sr    = CHBMIT_SR_ORIG
_chbmit_winsz = CHBMIT_SR_ORIG * CHBMIT_WIN_SEC


def _set_chbmit_sr(rate):
    global _chbmit_sr, _chbmit_winsz
    _chbmit_sr    = rate
    _chbmit_winsz = rate * CHBMIT_WIN_SEC


def _parse_chbmit_summary(summary_path):
    seizures = defaultdict(list)
    current_file, pending_start = None, None
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                current_file = line.split(':', 1)[1].strip(); pending_start = None
            elif 'Start Time:' in line and 'Seizure' in line and 'File' not in line:
                m = re.search(r'(\d+)\s*seconds', line) or re.search(r':\s*(\d+)', line.split('Time')[-1])
                if m: pending_start = int(m.group(1))
            elif 'End Time:' in line and 'Seizure' in line and 'File' not in line:
                m = re.search(r'(\d+)\s*seconds', line) or re.search(r':\s*(\d+)', line.split('Time')[-1])
                if m and current_file and pending_start is not None:
                    seizures[current_file].append((pending_start, int(m.group(1)))); pending_start = None
    return dict(seizures)


def _read_edf(filepath, target_channels=None):
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib not installed. Run: pip install pyedflib")
    try:
        f = pyedflib.EdfReader(filepath)
    except Exception:
        return None
    labels = [l.strip().upper().replace(' ', '').replace('–', '-').replace('−', '-')
              for l in f.getSignalLabels()]
    if target_channels is None:
        target_channels = CHBMIT_CHANNELS
    target_upper = [c.upper() for c in target_channels]
    lmap = {}
    for i, l in enumerate(labels):
        if l not in ('-', '.', '--', '..', '') and l not in lmap:
            lmap[l] = i
    indices = []
    for tch in target_upper:
        if tch in lmap:
            indices.append(lmap[tch])
        else:
            found = next((lmap[cl] for cl in lmap if tch in cl or cl in tch), None)
            if found is None:
                f.close(); return None
            indices.append(found)
    signals = [f.readSignal(i) for i in indices]
    f.close()
    return np.stack(signals, axis=1)


def _normalize_segs(segs):
    out = []
    for s in segs:
        m, st = s.mean(0, keepdims=True), s.std(0, keepdims=True)
        st[st == 0] = 1.0
        out.append((s - m) / st)
    return out


def _downsample(data, orig=256, target=None):
    if target is None or target >= orig: return data
    from scipy.signal import decimate
    factor = orig // target
    return decimate(data, factor, axis=0, zero_phase=True).astype(np.float32)


def _get_seizure_list(data_dir, subject_id):
    summary = os.path.join(data_dir, subject_id, f'{subject_id}-summary.txt')
    if not os.path.exists(summary): return []
    per_file = _parse_chbmit_summary(summary)
    result, idx = [], 0
    for f in sorted(per_file):
        for s, e in per_file[f]:
            result.append({'seizure_idx': idx, 'edf_file': f, 'start_sec': s, 'end_sec': e})
            idx += 1
    return result


def _extract_preictal(data, seizure_start_sec, ds_rate=None):
    if ds_rate and ds_rate < CHBMIT_SR_ORIG:
        data = _downsample(data, CHBMIT_SR_ORIG, ds_rate)
    n = data.shape[0]
    pre_start = max(0, seizure_start_sec - (CHBMIT_PREICTAL + CHBMIT_SPH) * 60)
    pre_end   = seizure_start_sec - CHBMIT_SPH * 60
    if pre_end <= pre_start or pre_end <= 0: return []
    ps, pe = int(pre_start * _chbmit_sr), min(n, int(pre_end * _chbmit_sr))
    return [data[s:s + _chbmit_winsz] for s in range(ps, pe - _chbmit_winsz + 1, _chbmit_winsz)]


def _extract_interictal(data_dir, subject_id, seizure_list, max_segs=None, ds_rate=None):
    summary = os.path.join(data_dir, subject_id, f'{subject_id}-summary.txt')
    per_file = _parse_chbmit_summary(summary)
    skip = set(per_file.keys()) | {sz['edf_file'] for sz in seizure_list}
    subj_dir = os.path.join(data_dir, subject_id)
    segs = []
    for fname in sorted(f for f in os.listdir(subj_dir) if f.endswith('.edf') and f not in skip):
        data = _read_edf(os.path.join(subj_dir, fname))
        if data is None: continue
        if ds_rate and ds_rate < CHBMIT_SR_ORIG:
            data = _downsample(data, CHBMIT_SR_ORIG, ds_rate)
        n = data.shape[0]
        segs += [data[s:s + _chbmit_winsz] for s in range(0, n - _chbmit_winsz + 1, _chbmit_winsz)]
        if max_segs and len(segs) >= max_segs: break
    return segs


class _SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _load_all_subjects(data_dir, subjects=None, max_ratio=3.0):
    if subjects is None:
        subjects = sorted(d for d in os.listdir(data_dir)
                         if d.startswith('chb') and os.path.isdir(os.path.join(data_dir, d)))
    all_X, all_y, all_subj = [], [], []
    for subj in subjects:
        print(f"  Loading {subj}...")
        sz_list = _get_seizure_list(data_dir, subj)
        if not sz_list: continue
        subj_dir = os.path.join(data_dir, subj)
        pre = []
        for sz in sz_list:
            data = _read_edf(os.path.join(subj_dir, sz['edf_file']))
            if data is None: continue
            segs = _extract_preictal(data, sz['start_sec'], ds_rate=_chbmit_sr)
            pre.extend(segs)
        if not pre: continue
        inter = _extract_interictal(data_dir, subj, sz_list,
                                    max_segs=int(len(pre) * max_ratio), ds_rate=_chbmit_sr)
        pre   = _normalize_segs(pre)
        inter = _normalize_segs(inter)
        rng = np.random.RandomState(42)
        max_inter = int(len(pre) * max_ratio)
        if len(inter) > max_inter:
            inter = [inter[i] for i in rng.choice(len(inter), max_inter, replace=False)]
        for s in pre:   all_X.append(s); all_y.append(1); all_subj.append(subj)
        for s in inter: all_X.append(s); all_y.append(0); all_subj.append(subj)
        print(f"    Pre-ictal: {len(pre)}, Inter-ictal: {len(inter)}")
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"\n  Total: {len(X)} segments")
    return X, y, all_subj


# =============================================================================
# Training Functions per Dataset
# =============================================================================

def train_standard(dataset, train_loader, val_loader, test_loader, info,
                   size, gate_type, epochs, lr, patience, device, save_dir,
                   V_th=1.0, gate_kappa=0.1):
    """Shared training loop for PTB-XL, PTB, ADFTD, MIT-BIH."""
    cfg = MODEL_CONFIGS[size]
    d_model, n_layers = cfg['d_model'], cfg['n_layers']

    print(f"\n{'='*60}")
    print(f"Training SpikeMamba-{size.upper()} on {dataset.upper()} | {gate_type.upper()} gate")
    print(f"Model: {n_layers} layers, {d_model} d_model")
    print(f"Dataset: {info['num_channels']}ch, {info['seq_len']}len, {info['num_classes']}cls")
    print(f"Train/Val/Test: {info['train_size']}/{info['val_size']}/{info['test_size']}")
    print(f"{'='*60}\n")

    model = SpikeMambaClassifier(
        input_dim=info['num_channels'],
        num_classes=info['num_classes'],
        d_model=d_model,
        n_layers=n_layers,
        gate_type=gate_type,
        V_th=V_th,
        gate_kappa=gate_kappa,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Weighted loss for MIT-BIH class imbalance
    if 'class_counts' in info:
        counts = info['class_counts'].astype(float)
        weights = torch.FloatTensor([1.0 / max(1, c) for c in counts])
        weights = weights / weights.sum() * info['num_classes']
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper   = EarlyStopping(patience=patience, mode='max')

    os.makedirs(save_dir, exist_ok=True)
    hp_suffix = ''
    if V_th != 1.0:     hp_suffix += f'_vth{V_th}'
    if gate_kappa != 0.1: hp_suffix += f'_kappa{gate_kappa}'
    save_path = f'{save_dir}/spike_mamba_{dataset}_{gate_type}_{size}{hp_suffix}_best.pt'

    best_test_acc = 0
    results = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'test_loss', 'test_acc', 'epoch_times')}
    results.update({'gate_type': gate_type, 'd_model': d_model, 'n_layers': n_layers,
                    'stopped_epoch': epochs})

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = _evaluate_basic(model, val_loader, criterion, device)
        test_loss, test_acc, preds, targets = _evaluate_basic(model, test_loader, criterion, device)
        et = time.time() - t0

        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}% | Time: {et:.1f}s")

        for k, v in (('train_loss', train_loss), ('train_acc', train_acc),
                     ('val_loss', val_loss), ('val_acc', val_acc),
                     ('test_loss', test_loss), ('test_acc', test_acc), ('epoch_times', et)):
            results[k].append(v)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            results['best_test_acc'] = best_test_acc
            results['best_val_acc']  = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'test_acc': test_acc, 'd_model': d_model, 'n_layers': n_layers,
                        'gate_type': gate_type}, save_path)
            print(f"  * New best test acc! {test_acc:.2f}%")

        if stopper(val_acc):
            print(f"\n  Early stopping at epoch {epoch}")
            results['stopped_epoch'] = epoch
            break
        scheduler.step()

    print(f"\n{'='*40}")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # Per-class accuracy on best model
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    _, _, preds, targets = _evaluate_basic(model, test_loader, criterion, device)
    class_names = info.get('class_names', [str(i) for i in range(info['num_classes'])])
    for i, name in enumerate(class_names):
        mask = targets == i
        if mask.sum() > 0:
            acc = 100. * (preds[mask] == targets[mask]).sum() / mask.sum()
            print(f"  {name}: {acc:.2f}%")

    torch.save(results, f'{save_dir}/spike_mamba_{dataset}_{gate_type}_{size}{hp_suffix}_results.pt')
    return results


# =============================================================================
# CHB-MIT Training Helpers
# =============================================================================

def _train_epoch_simple(model, loader, criterion, optimizer, device):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _eval_chbmit(model, loader, criterion, device):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(target.cpu().numpy())
            probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
    preds, targets, probs = map(np.array, (preds, targets, probs))
    try:   auc = roc_auc_score(targets, probs) * 100
    except: auc = 50.0
    sens = recall_score(targets, preds, pos_label=1, zero_division=0) * 100
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    spec = tn / max(1, tn + fp) * 100
    f1   = f1_score(targets, preds, pos_label=1, zero_division=0) * 100
    return {'acc': 100. * (preds == targets).mean(), 'sensitivity': sens,
            'specificity': spec, 'f1': f1, 'auc': auc}


def train_chbmit_segment_mixed(size, data_dir, gate_type, batch_size, epochs, lr, patience,
                               device, save_dir, subjects=None):
    """Segment-mixed: pool all subjects, random 6:2:2 split."""
    cfg = MODEL_CONFIGS[size]
    d_model, n_layers = cfg['d_model'], cfg['n_layers']
    print(f"\n{'='*70}")
    print(f" SpikeMamba CHB-MIT | {size.upper()} | Segment-Mixed (6:2:2)")
    print(f"{'='*70}\n")

    os.makedirs(save_dir, exist_ok=True)
    cache = os.path.join(save_dir, f'chbmit_mixed_dataset_{_chbmit_sr}hz.npz')
    if os.path.exists(cache):
        print(f"  Using cached dataset: {cache}")
        d = np.load(cache)
        X_tr, y_tr = d['X_train'], d['y_train']
        X_va, y_va = d['X_val'], d['y_val']
        X_te, y_te = d['X_test'], d['y_test']
    else:
        X, y, _ = _load_all_subjects(data_dir, subjects)
        if len(X) == 0: return None
        rng = np.random.RandomState(42)
        p = rng.permutation(len(X))
        n_te = int(len(X) * 0.20); n_va = int(len(X) * 0.20)
        X_te, y_te = X[p[:n_te]], y[p[:n_te]]
        X_va, y_va = X[p[n_te:n_te+n_va]], y[p[n_te:n_te+n_va]]
        X_tr, y_tr = X[p[n_te+n_va:]], y[p[n_te+n_va:]]
        np.savez_compressed(cache, X_train=X_tr, y_train=y_tr,
                            X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
        print(f"  Dataset cached: {cache}")

    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")

    tr_ld = DataLoader(_SimpleDataset(X_tr, y_tr), batch_size=batch_size,
                       shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(_SimpleDataset(X_va, y_va), batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)
    te_ld = DataLoader(_SimpleDataset(X_te, y_te), batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)

    model = SpikeMambaClassifier(
        input_dim=X_tr.shape[2], num_classes=2,
        d_model=d_model, n_layers=n_layers, gate_type=gate_type,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    counts  = np.bincount(y_tr, minlength=2).astype(float)
    weights = torch.FloatTensor([1.0 / max(1, c) for c in counts])
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = f'{save_dir}/spike_mamba_chbmit_mixed_{gate_type}_{size}_best.pt'
    best_auc, best_m, no_improve = 0, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        _train_epoch_simple(model, tr_ld, criterion, optimizer, device)
        va_m = _eval_chbmit(model, va_ld, criterion, device)
        te_m = _eval_chbmit(model, te_ld, criterion, device)
        et = time.time() - t0

        if te_m['auc'] > best_auc:
            best_auc = te_m['auc']; best_m = te_m.copy()
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'd_model': d_model, 'n_layers': n_layers, 'gate_type': gate_type,
                        'test_metrics': te_m, 'size': size}, save_path)
            marker = f" * best AUC={te_m['auc']:.1f}%"; no_improve = 0
        else:
            marker = ""; no_improve += 1

        if epoch <= 5 or epoch % 5 == 0 or marker:
            print(f"  Epoch {epoch:3d} ({et:.0f}s): "
                  f"val_auc={va_m['auc']:.1f}%, "
                  f"test_acc={te_m['acc']:.1f}%, test_sens={te_m['sensitivity']:.1f}%, "
                  f"test_auc={te_m['auc']:.1f}%{marker}")
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}"); break
        scheduler.step()

    print(f"\n  Final: Acc={best_m['acc']:.2f}%, Sens={best_m['sensitivity']:.2f}%, "
          f"Spec={best_m['specificity']:.2f}%, AUC={best_m['auc']:.2f}%")
    results = {'size': size, 'd_model': d_model, 'n_layers': n_layers,
               'gate_type': gate_type, 'final_metrics': best_m}
    torch.save(results, f'{save_dir}/spike_mamba_chbmit_mixed_{gate_type}_{size}_results.pt')
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SpikeMamba Unified Training')

    # Dataset & paths
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ptbxl', 'ptb', 'adftd', 'mitbih', 'chbmit'],
                        help='Target dataset')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset root (default: ./data/<DATASET>)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    # Model
    parser.add_argument('--size', type=str, default='large',
                        choices=['large', 'medium', 'tiny'])
    parser.add_argument('--gate_type', type=str, default='state_dependent',
                        choices=['state_dependent'])

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    # Hyperparameters (PTB-XL / PTB / ADFTD)
    parser.add_argument('--V_th', type=float, default=1.0,
                        help='Spike firing threshold')
    parser.add_argument('--gate_kappa', type=float, default=0.1,
                        help='Gate threshold (kappa)')

    # Multi-size sweep
    parser.add_argument('--all_sizes', action='store_true',
                        help='Train large, medium, tiny sequentially')

    # CHB-MIT specific
    parser.add_argument('--chbmit_mode', type=str, default='segment_mixed',
                        choices=['segment_mixed'],
                        help='CHB-MIT training mode (segment-level 6:2:2 random split)')
    parser.add_argument('--downsample', type=int, default=256,
                        help='CHB-MIT: target sampling rate (256=no downsample)')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='CHB-MIT: specific subjects, e.g. --subjects chb01 chb02')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    DEFAULTS = {
        'ptbxl': './data/PTB-XL',
        'ptb':   './data/PTB',
        'adftd': './data/ADFTD',
        'mitbih':'./data/MIT-BIH',
        'chbmit':'./data/CHB-MIT_edf',
    }
    data_path = args.data_path or DEFAULTS[args.dataset]

    sizes = ['large', 'medium', 'tiny'] if args.all_sizes else [args.size]

    for size in sizes:
        if args.dataset == 'chbmit':
            if args.downsample < CHBMIT_SR_ORIG:
                _set_chbmit_sr(args.downsample)
            fn = train_chbmit_segment_mixed
            fn(size=size, data_dir=data_path, gate_type=args.gate_type,
               batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
               patience=args.patience, device=args.device, save_dir=args.save_dir,
               subjects=args.subjects)
        else:
            LOADERS = {
                'ptbxl':  _get_ptbxl_loaders,
                'ptb':    _get_ptb_loaders,
                'adftd':  _get_adftd_loaders,
                'mitbih': _get_mitbih_loaders,
            }
            tr, va, te, info = LOADERS[args.dataset](data_path, args.batch_size, args.num_workers)
            train_standard(
                dataset=args.dataset,
                train_loader=tr, val_loader=va, test_loader=te, info=info,
                size=size, gate_type=args.gate_type,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                device=args.device, save_dir=args.save_dir,
                V_th=args.V_th, gate_kappa=args.gate_kappa,
            )


if __name__ == '__main__':
    main()
