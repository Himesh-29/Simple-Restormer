# Restormer: Modular Image Restoration

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FHimesh-29%2FSimple-Restormer&label=Visitors&countColor=%23263759)

This repository provides a modernized, modular, and high-performance PyTorch implementation of **[Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)**. 

Designed for both research and production, this refactor prioritizes environment isolation, detailed experiment tracking, and automated workflows.

## 🚀 Key Features

*   **⚡ Modern Dependency Management**: Native support for `uv`, providing significantly faster and completely isolated virtual environments compared to traditional `pip` or `conda`.
*   **🎮 Centralized Launcher (`launch.py`)**: A single entry point to automate the entire lifecycle—environment setup, dependency installation, dataset downloading, training, and evaluation.
*   **📊 Model Complexity Analytics**: Integrated GFLOPs calculation and parameter counting (via `fvcore`). Automatically logs model complexity for various input sizes.
*   **📜 Professional Logging System**: Replaced standard prints with a robust `logging` framework. Logs are timestamped, leveled, and automatically saved to both console and file (`experiments/` and `results/`).
*   **📈 Progressive Learning**: Built-in support for adaptive training where patch sizes and mini-batch sizes evolve throughout the training process to optimize performance on high-resolution images.
*   **🔍 Automatic Dataset Discovery**: The evaluation pipeline automatically discovers available benchmark datasets (e.g., Rain100H, Rain100L, Test100) and executes batch assessments.
*   **🛡️ Model Stability & Augmentation**:
    *   **EMA (Exponential Moving Average)**: Maintains a shadow copy of weights for improved validation stability.
    *   **Mixing Augmentations**: Includes Mixup support to enhance model generalization.
*   **🌐 Distributed Training**: Native support for PyTorch Distributed Data Parallel (DDP) for multi-GPU scaling.
*   **☁️ Cloud Native**: Automatic environment detection and optimized setup for **Google Colab** and **Kaggle**.

## 🛠️ Quick Start

### 1. Project Setup (Recommended: `uv`)
Ensure you have `uv` installed, then simply run:
```powershell
uv python install 3.12
uv sync
```

### 2. Run the Full Pipeline
To set up, download data, train, and evaluate in one go:
```bash
uv run launch.py
```

### 3. Custom Workflows
Use command-line flags to customize the process:
- **Train Only**:
  ```bash
  uv run launch.py --skip-test
  ```
- **Test Only** (using a pre-existing trained model):
  ```bash
  uv run launch.py --skip-train
  ```
- **Skip Installation**:
  ```bash
  uv run launch.py --skip-install
  ```

For a full list of options, use the help flag:
```bash
uv run launch.py --help
```

## ⚙️ Configuration

Control every aspect of your experiment via YAML files located in:
`Options/Restormer.yml`

This file handles network architecture, progressive learning iteration benchmarks, loss functions (L1/MSE), and dataset paths.

## 🔄 Project Workflow

The `launch.py` script follows the workflow illustrated below, automating each step from setup to execution.

![Project Workflow](./assets/workflow.png)

## 📊 Project Structure
- `core/`: Core logic for training, scheduling, and metrics.
- `data/`: Dataset handling and paired image loaders.
- `models/`: Restormer architecture and network definitions.
- `Options/`: Configuration files for various experiments.
- `launch.py`: Unified automation script.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
