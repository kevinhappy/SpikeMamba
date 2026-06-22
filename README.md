# SpikeMamba

**SpikeMamba: Spike-Driven State Space Models for Energy-Efficient Biomedical Sequence Modeling**

Accepted at MICCAI 2026.

SpikeMamba reformulates selective SSM dynamics in the spike domain, rewriting the recurrent update as an event-triggered state transition rather than a continuous dense computation. The key contributions are:
- **Spike-domain SSM** implementing LIF membrane equations that replace MAC-heavy transitions with binary accumulations (Eqs. 3–5)
- **State-Dependent Gate** that monitors instantaneous firing activity and membrane potential variance to suppress redundant updates during inactive timesteps (Eq. 6)
- Demonstrated on 5 biosignal benchmarks: PTB-XL ECG, PTB ECG, ADFTD EEG, MIT-BIH Arrhythmia, CHB-MIT Seizure Prediction

---

## Installation

```bash
pip install -e .

# For CHB-MIT (EDF reading):
pip install -e ".[chbmit]"

# For running tests:
pip install -e ".[dev]"
```

---

## Datasets

We do **not** distribute dataset files. Download from the original sources:

| Dataset | Task | Classes | Link |
|---------|------|---------|------|
| PTB-XL | ECG classification | 5 (Normal/MI/STTC/CD/HYP) | [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) |
| PTB | ECG classification | 2 (Healthy/MI) | [PhysioNet PTB](https://physionet.org/content/ptbdb/1.0.0/) |
| ADFTD | EEG classification | 3 (CN/FTD/AD) | [OpenNeuro ADFTD](https://openneuro.org/datasets/ds004504) |
| MIT-BIH | Arrhythmia classification | 5 (N/S/V/F/Q) | [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) |
| CHB-MIT | Seizure prediction | 2 (inter-ictal/pre-ictal) | [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) |

Place each dataset under `./data/<DATASET>/` following the directory structure expected by the data loaders (Feature/ and Label/ subdirectories for PTB-XL, PTB, ADFTD; raw .npy files for MIT-BIH; raw .edf files per subject for CHB-MIT).

---

## Repository Structure

```
spikemamba/
├── mamba_ssm/
│   ├── modules/
│   │   └── spike_mamba.py      # SpikeMamba module (core contribution)
│   └── ops/
│       └── spike_interface.py  # LIF spike scan, gate implementations
├── data/
│   └── ptbxl_loader.py         # PTB-XL data loader
├── tests/
│   └── test_spike_mamba.py     # Unit tests
├── train_spike_mamba.py        # Unified training script (all datasets)
├── inference_all.py            # Unified inference & evaluation
└── energy_analysis.py          # Energy consumption analysis
```

---

## Training

```bash
# PTB-XL (5-class ECG) — Large model
python train_spike_mamba.py --dataset ptbxl --size large

# PTB (binary ECG) — Medium model
python train_spike_mamba.py --dataset ptb --size medium

# ADFTD (3-class EEG) — Tiny model
python train_spike_mamba.py --dataset adftd --size tiny

# MIT-BIH (5-class arrhythmia) — All sizes
python train_spike_mamba.py --dataset mitbih --all_sizes

# CHB-MIT (seizure prediction) — Segment-mixed 6:2:2 split
python train_spike_mamba.py --dataset chbmit --size large
```

Key arguments:
- `--size`: `large` (8L-512D) / `medium` (4L-256D) / `tiny` (4L-64D)
- `--all_sizes`: train all three model sizes sequentially

---

## Inference

```bash
# All 5 datasets × 3 model sizes, 10 bootstrap runs
python inference_all.py --run_all

# Single dataset
python inference_all.py --dataset ptb --all_sizes
python inference_all.py --dataset chbmit --size large
```

---

## Energy Analysis

```bash
python energy_analysis.py
```

---

## Tests

```bash
pytest tests/test_spike_mamba.py -v
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

This project builds on the following open-source code:

- **[Mamba](https://github.com/state-spaces/mamba)** — the selective state space
  model and the `mamba_ssm` package structure that SpikeMamba reformulates into
  the spike domain (Apache 2.0).
- **[Medformer](https://github.com/DL4mHealth/Medformer)** — the subject-independent
  evaluation protocol and biosignal data-loading conventions used in our
  experiments, and a baseline in our comparisons.

The energy analysis follows the spike-vs-ANN accounting methodology of
**Spike-Driven Transformer** (Yao et al., NeurIPS 2024 / ICLR 2024); the 45 nm
CMOS energy constants (E_MAC = 4.6 pJ, E_AC = 0.9 pJ) are taken from
**Horowitz** (IEEE ISSCC 2014). See the paper for full references.

---

## Citation

If you use SpikeMamba, please cite:

```bibtex
@inproceedings{lee2026spikemamba,
  title     = {SpikeMamba: Spike-Driven State Space Models for Energy-Efficient Biomedical Sequence Modeling},
  author    = {Lee, Si Yong and Lee, Ryangjin and Park, Hawon and Kim, Yoora and Kang, Byungkon and Yang, Yoon Seok},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```
