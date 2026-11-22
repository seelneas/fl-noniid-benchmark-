# Benchmarking Federated Learning Under Non-IID Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, reproducible benchmarking suite for studying the effects of **statistical data heterogeneity (non-IID)** on Federated Learning (FL) optimization algorithms.

---

## Overview

In real-world Federated Learning scenarios, client data distributions are inherently non-identically distributed (non-IID). This distribution mismatch introduces **client drift**, causing local model updates to pull the global model away from optimal convergence.

This repository provides an end-to-end framework to evaluate and benchmark classic and state-of-the-art FL algorithms under varying degrees of label skew:

- **FedAvg** (*McMahan et al., 2017*): Baseline Federated Averaging.
- **FedProx** (*Li et al., 2020*): Adds a proximal regularization term $\frac{\mu}{2} \|w - w^t\|^2$ to constrain local client drift.
- **SCAFFOLD** (*Karimireddy et al., 2020*): Employs client and server control variates to correct gradient drift across communication rounds.

Data partitioning uses a **Dirichlet distribution** $\text{Dir}(\alpha)$ across class labels to simulate varying degrees of non-IID severity:
- **IID**: Uniform, random partition across clients.
- **$\alpha = 10.0$**: Mild heterogeneity (near-uniform class representation).
- **$\alpha = 0.5$**: Moderate heterogeneity (standard benchmark setting).
- **$\alpha = 0.1$**: Extreme heterogeneity (severe class imbalance per client).

---

## Project Structure

```text
fl-noniid-benchmark/
├── configs/
│   ├── base.yaml                         # Base default configuration
│   └── experiments/
│       ├── main_experiment.yaml          # Grid benchmark experiment settings
│       └── pilot.yaml                    # Fast pilot run configuration
├── data/                                 # CIFAR-10 data cache (git-ignored)
├── notes/                                # Research and architectural notes
├── results/
│   ├── figures/                          # Generated plots and visualizations
│   ├── summaries/                        # CSV metrics per experiment run
│   └── aggregated_results.csv            # Aggregated seed statistics
├── scripts/
│   ├── check_setup.py                    # Verify PyTorch, device, and config
│   ├── check_dataset.py                  # Verify CIFAR-10 pipeline
│   ├── visualize_partitions.py           # Generate client label distribution plots
│   ├── run_experiment.py                 # Run a single FL experiment
│   ├── run_grid.py                       # Automated experiment grid orchestrator
│   ├── aggregate_results.py              # Aggregate metrics across seeds
│   └── make_figures.py                   # Generate publication-ready figures
├── src/
│   └── flbench/                          # Core benchmark package (src-layout)
│       ├── algorithms/                   # FedAvg, FedProx, SCAFFOLD implementations
│       ├── data/                         # CIFAR-10 loaders & Dirichlet partitioners
│       ├── fl/                           # Client, Server, and Simulation orchestrators
│       ├── models/                       # CNN and ResNet architectures
│       ├── metrics.py                    # Metric tracking and seed aggregation
│       └── utils.py                      # Device detection (MPS/CUDA/CPU) & seed setting
├── tests/                                # Comprehensive pytest suite
├── pyproject.toml                        # Package build definition & dependencies
├── requirements.txt                      # Project dependencies
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- PyTorch 2.0+ (supports CUDA, Apple Silicon MPS, or CPU)

### 2. Clone and Create Virtual Environment
```bash
git clone https://github.com/your-username/fl-noniid-benchmark-.git
cd fl-noniid-benchmark-

python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies & Package
Install the project in editable mode so that `flbench` is importable throughout scripts and tests:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Setup
Check that PyTorch detects your hardware accelerator (Apple Silicon MPS, NVIDIA CUDA, or CPU):
```bash
python scripts/check_setup.py
```

---

## Workflow & Usage

### 1. Inspect & Visualize Data Partitioning
Inspect the CIFAR-10 data partition across clients and plot class distributions:
```bash
# Verify download and dataloaders
python scripts/check_dataset.py

# Generate distribution plot for Dirichlet partitions
python scripts/visualize_partitions.py
```
Output figures are saved in `results/figures/client_class_distributions.png`.

---

### 2. Run a Single Experiment
Run an individual experiment by specifying the algorithm, data distribution, and random seed:

```bash
python scripts/run_experiment.py \
    --config configs/experiments/main_experiment.yaml \
    --algorithm fedavg \
    --distribution dirichlet_alpha_05 \
    --seed 0
```

Supported algorithms:
- `fedavg`
- `fedprox`
- `scaffold`

Supported distributions:
- `iid`
- `dirichlet_alpha_10`
- `dirichlet_alpha_05`
- `dirichlet_alpha_01`

During execution:
- Global models are evaluated after every round on a global validation set.
- Checkpoints of the best-performing model are saved to `checkpoints/`.
- Per-round metrics and final test metrics are saved to `results/summaries/`.

---

### 3. Run the Full Experiment Grid
To execute the entire grid of algorithms $\times$ distributions $\times$ seeds automatically:

```bash
python scripts/run_grid.py --config configs/experiments/main_experiment.yaml
```

You can pass `--dry-run` to inspect the planned execution matrix before starting:
```bash
python scripts/run_grid.py --config configs/experiments/main_experiment.yaml --dry-run
```

---

### 4. Aggregate Results & Generate Figures
Once experiments complete, aggregate results across random seeds and generate summary plots:

```bash
# Aggregate seed metrics (mean ± std) into results/aggregated_results.csv
python scripts/aggregate_results.py --results-dir results/summaries

# Generate comparison plots and convergence curves
python scripts/make_figures.py --results results/aggregated_results.csv
```

---

## Configuration

Experiments are configured using YAML files in `configs/experiments/main_experiment.yaml`. Key configuration blocks:

```yaml
dataset:
  name: cifar10
  num_clients: 20
  batch_size: 64
  train_val_split: 0.8

training:
  num_rounds: 100
  local_epochs: 5
  client_fraction: 1.0

optimizer:
  name: sgd
  learning_rate: 0.01

algorithms:
  fedavg:
    enabled: true
  fedprox:
    enabled: true
    mu: 0.01
  scaffold:
    enabled: true
```

---

## Testing

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

The test suite covers:
- Client data partitioning and Dirichlet concentration consistency
- CNN model forward passes and shapes
- Local client training and gradient updates
- Federated averaging and weighted aggregation logic
- FedProx proximal regularization updates
- SCAFFOLD control variate correction
- Server simulation loops and checkpointing

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
