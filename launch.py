import os
import sys
import subprocess
import argparse
import yaml
import tempfile
import torch

def is_colab_or_kaggle():
    """Check if running in a Google Colab or Kaggle notebook environment."""
    return 'COLAB_GPU' in os.environ or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def run_command(command, message=None):
    """Runs a command, streams its output, and checks for errors."""
    if message:
        print(f"\n>>> {message}")
    print(f"--- Running: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8'
        )
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n--- ERROR: Command failed: {e} ---")
        sys.exit(1)

def check_and_download_data(args):
    """Check if data exists and download if it doesn't."""
    if args.skip_data_download_check:
        return

    train_path = os.path.join(args.dataroot, 'train')
    test_path = os.path.join(args.dataroot, 'test')

    train_missing = not os.path.exists(train_path) or not os.listdir(train_path)
    test_missing = not os.path.exists(test_path) or not os.listdir(test_path)

    if train_missing or test_missing:
        print("--- Dataset missing or incomplete. Starting download... ---")
        data_to_download = "train-test" if train_missing and test_missing else ("train" if train_missing else "test")
        
        if os.path.exists("download_data.py"):
            run_command([sys.executable, "download_data.py", "--data", data_to_download], "Downloading data...")
        else:
            print("--- WARNING: download_data.py not found. Please download datasets manually. ---")

def install_dependencies(args):
    """Install dependencies from requirements.txt."""
    if args.skip_install:
        return

    print("--- Step 1: Installing dependencies ---")
    
    # Simple check for torch
    try:
        import torch
        print(f"--- PyTorch {torch.__version__} found. ---")
        if torch.cuda.is_available():
            print(f"--- CUDA is available. Device: {torch.cuda.get_device_name(0)} ---")
        else:
            print("--- WARNING: CUDA is NOT available to PyTorch. Training will be slow on CPU. ---")
    except ImportError:
        print("--- PyTorch not found. You may need to install it manually in your Anaconda environment. ---")

    # Install other requirements
    if os.path.exists("requirements.txt"):
        try:
            # Check if pip is available
            subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, check=True)
            run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "Installing requirements.txt...")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If pip is missing, check if we are in a uv environment
            if 'UV_PYTHON' in os.environ or os.path.exists('.venv'):
                 print("--- [NOTICE] Pip not found in environment (common for uv). ---")
                 print("--- Skipping manual pip install as dependencies should be managed via 'uv sync'. ---")
            else:
                 print("--- WARNING: pip not found. Please install dependencies manually. ---")
    
    print("--- Installation complete ---\n")

def train(args):
    """Run the training process."""
    if args.skip_train:
        return
    
    print("--- Step 2: Starting Model Training ---")
    
    # Load and customize options
    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"--- ERROR: Options file not found at {args.opt} ---")
        sys.exit(1)

    if args.dataroot:
        for phase, dataset in opt.get('datasets', {}).items():
            for key in ['dataroot_gt', 'dataroot_lq']:
                if key in dataset and dataset[key] and './Datasets' in dataset[key]:
                    dataset[key] = dataset[key].replace('./Datasets', args.dataroot)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml', encoding='utf-8') as tmp_file:
        yaml.dump(opt, tmp_file, sort_keys=False)
        temp_opt_file = tmp_file.name

    try:
        run_command([sys.executable, "train.py", "-opt", temp_opt_file], "Training...")
    finally:
        if os.path.exists(temp_opt_file):
            os.remove(temp_opt_file)

def test(args):
    """Run the testing process."""
    if args.skip_test:
        return

    print("--- Step 3: Starting model testing and evaluation ---")
    
    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except Exception:
        sys.exit(1)

    experiment_name = opt.get('name', 'Restormer')
    model_path = args.model_path or f"experiments/{experiment_name}/models/net_g_latest.pth"
    
    if not os.path.exists(model_path):
        print(f"--- ERROR: Model not found at {model_path}. Skipping test. ---")
        return

    command = [sys.executable, "test.py", "-opt", args.opt, f"--model_path={model_path}"]
    if args.test_dataset:
        command.append(f"--dataset={args.test_dataset}")
        
    run_command(command, "Testing...")

def main():
    parser = argparse.ArgumentParser(description="Restormer All-in-One Launcher")
    parser.add_argument('--skip-install', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-train', action='store_true', help='Skip training step')
    parser.add_argument('--skip-test', action='store_true', help='Skip testing step')
    parser.add_argument("--dataroot", type=str, default="Datasets", help="Root directory for datasets")
    parser.add_argument("--opt", type=str, default="Options/Restormer.yml", help="Path to options YAML")
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model for testing')
    parser.add_argument('--test-dataset', type=str, help='Specific test dataset name')
    parser.add_argument('--skip-data-download-check', action='store_true', help='Skip data download check')
    
    args = parser.parse_args()

    if is_colab_or_kaggle():
        print("\n" + "="*50 + "\n🚀 RESTORMER ON CLOUD (Colab/Kaggle)\n" + "="*50)

    install_dependencies(args)
    check_and_download_data(args)
    train(args)
    test(args)

    print("\n--- All steps complete! ---")

if __name__ == '__main__':
    main()