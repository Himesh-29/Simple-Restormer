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

## 🖼️ Interactive Visualizer Dashboard

A new tool `visualize_model.py` provides a **model-agnostic Gradio-based dashboard** that
works with any network defined by an Options YAML file.  Features include:

* **Input / Output / Residual** with PSNR and SSIM metrics.
* **Image Channel Explorer** — decompose R/G/B channels of both input and
  output, view histograms, and compare differences.
* **Multi-Scale Comparison** at full, half, and quarter resolution.
* **Block I/O Visualization** — capture the input and output of every top‑level
  block in the model, showing how data flows through the architecture.
* **Feature Flow** and **Channel Explorer** tabs for inspecting activation
  energies and individual feature maps.
* **Context Loss Analysis** — L1 differences between consecutive layers.
* **Automatic tab inclusion** when `capture_all` support is detected (e.g.
  cluster maps and attention heads for ClusterAttentionRestormer).

Launch the dashboard with:
```bash
uv run visualize_model.py --opt Options/Restormer.yml
```

The visualizer was released as part of **v1.4.0**.

## 🚀 What's New in v1.5.0 (Perceptual Metrics)

* 👁️ **LPIPS Integration:** Added Learned Perceptual Image Patch Similarity (LPIPS) to `core/metrics.py` utilizing the standardized `alex` net.
* 📊 **Human-Aligned Evaluation:** `test.py` now calculates and displays LPIPS alongside PSNR and SSIM. This proves that the model creates images that actually *look* better to the human eye, not just mathematically.
* 📈 **Perceptual Gain Tracking:** The beautiful terminal tables now include "In LPIPS", "Out LPIPS", and "+ LPIPS" to accurately track perceptual improvements across entire datasets.

## 🚀 What's New in v1.4.1

* 📊 **Automated Gain Calculation:** Computes the mathematical PSNR/SSIM improvement between the rainy input and derained output per image.
* 🖥️ **Beautiful Terminal Tables:** Replaced standard text logging with clean, aligned ASCII summary tables (via `tabulate`) for cross-dataset comparisons.
* ⚡ **Cleaner Console Output:** Removed noisy per-image tqdm logging in favor of concise per-dataset summaries and a final holistic dashboard.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
