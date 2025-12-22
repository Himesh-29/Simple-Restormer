import os
import sys
import subprocess
import platform
import argparse
import yaml
import tempfile
import torch

# --- Configuration ---
VENV_DIR = "venv"

def check_and_download_data(args):
    """Check if data exists and download if it doesn't."""
    if args.skip_data_download_check:
        print("--- Skipping data download check. ---")
        return

    train_path = os.path.join(args.dataroot, 'train')
    test_path = os.path.join(args.dataroot, 'test')

    # Check if the base directories are missing or empty
    train_missing = not os.path.exists(train_path) or not os.listdir(train_path)
    test_missing = not os.path.exists(test_path) or not os.listdir(test_path)

    # Only download if data is missing
    if train_missing or test_missing:
        print("--- Dataset not found or incomplete. Starting download... ---")
        
        data_to_download = ""
        if train_missing and test_missing:
            print("--- Both training and testing data are missing. ---")
            data_to_download = "train-test"
        elif train_missing:
            print("--- Training data is missing. ---")
            data_to_download = "train"
        else: # test_missing
            print("--- Testing data is missing. ---")
            data_to_download = "test"

        if data_to_download:
            try:
                # Command is now tailored to the user's download_data.py script
                command = [
                    sys.executable,
                    "download_data.py",
                    "--data",
                    data_to_download,
                ]
                run_command(command, "Running data downloader...")
            except FileNotFoundError:
                print("--- ERROR: download_data.py not found. Please ensure it is in the root directory. ---")
                sys.exit(1)
            except Exception as e:
                print(f"--- ERROR: An error occurred during data download: {e} ---")
                sys.exit(1)
    else:
        print("--- Datasets found. Skipping download. ---")

def is_in_virtual_env():
    """Check if currently running in the project's virtual environment."""
    # The second check is for legacy venv activation on some systems.
    return sys.prefix == os.path.abspath(VENV_DIR) or hasattr(sys, 'real_prefix')

def is_colab_or_kaggle():
    """Check if running in a Google Colab or Kaggle notebook environment."""
    return 'COLAB_GPU' in os.environ or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def run_command(command, message=None):
    """Runs a command, streams its output, and checks for errors."""
    if message:
        print(message)
    print(f"--- Running command: {' '.join(command)} ---")
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
        print("--- Command finished successfully ---\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n--- Command failed: {' '.join(command)} ---")
        print(f"--- Error: {e} ---")
        sys.exit(1)

def setup_virtual_environment():
    """Creates a venv and re-launches the script inside it if not already active."""
    if is_colab_or_kaggle() or is_in_virtual_env():
        return # No setup needed for notebooks or if already in venv

    venv_path = os.path.abspath(VENV_DIR)
    if not os.path.isdir(venv_path):
        print(f"--- Creating virtual environment in '{venv_path}' ---")
        run_command([sys.executable, "-m", "venv", venv_path])

    # Determine the Python executable path within the venv
    if platform.system() == "Windows":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")

    print("--- Re-launching script inside the virtual environment ---")
    # Re-launch the script with all original arguments using the venv's Python
    run_command([python_executable, __file__] + sys.argv[1:])
    sys.exit(0) # Exit the outer script gracefully

def install(args):
    """Install dependencies using pip."""
    print("--- Step 1: Installing dependencies ---")

    # Check if we're in Colab and adjust PyTorch installation accordingly
    if is_colab_or_kaggle():
        print("--- Detected Colab/Kaggle environment. Using optimized installation. ---")
        # Colab usually has PyTorch pre-installed, but we'll ensure it's the right version
        try:
            import torch
            print(f"--- PyTorch version: {torch.__version__} ---")
            if torch.cuda.is_available():
                print(f"--- CUDA available: {torch.cuda.get_device_name(0)} ---")
        except ImportError:
            print("--- Installing PyTorch for Colab ---")
            run_command([
                sys.executable, "-m", "pip", "install", "torch", "torchvision"
            ])
    else:
        # Install PyTorch with specific CUDA version for local environments
        print("--- Installing PyTorch with CUDA 11.8 support ---")
        run_command([
            sys.executable, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])

    # Install all other packages from requirements.txt
    print("--- Installing other dependencies ---")
    run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    print("--- Installation complete ---")

def train(args):
    """Run the training process."""
    print("--- Step 3: Starting model training ---")
    train_script = "train.py"
    master_port = "4321"

    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Options file not found at {args.opt}")
        sys.exit(1)

    # The paths in the yml file are like './Datasets/...'.
    # We replace the './Datasets' part with the provided data root to make them correct.
    if args.dataroot:
        print(f"--- Updating dataset paths with root: {args.dataroot} ---")
        for phase, dataset in opt['datasets'].items():
            for key in ['dataroot_gt', 'dataroot_lq']:
                if key in dataset and dataset.get(key) and './Datasets' in dataset[key]:
                    original_path = dataset[key]
                    new_path = original_path.replace('./Datasets', args.dataroot)
                    dataset[key] = new_path
                    print(f"  - Updated {phase} '{key}' to: {new_path}")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml', encoding='utf-8') as tmp_file:
        yaml.dump(opt, tmp_file, sort_keys=False)
        temp_opt_file = tmp_file.name
    
    os.environ.update({
        "MASTER_ADDR": "localhost", "MASTER_PORT": master_port,
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "USE_LIBUV": "0"
    })

    command = [sys.executable, train_script, "-opt", temp_opt_file, "--launcher", "pytorch"]
    run_command(command)
    os.remove(temp_opt_file)
    print("--- Training complete ---")

def test(args):
    """Run the testing process to generate results and metrics."""
    print("--- Step 4: Starting model testing and evaluation ---")

    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Options file not found at {args.opt}")
        sys.exit(1)

    experiment_name = opt.get('name')
    if not experiment_name:
        print(f"Error: Could not find 'name' in options file {args.opt}")
        sys.exit(1)

    # Path to the model trained in the previous step
    model_path = f"experiments/{experiment_name}/models/net_g_latest.pth"
    
    if not os.path.exists(model_path):
        print(f"--- ERROR: Trained model not found at {model_path}. Skipping test. ---")
        return

    command = [
        sys.executable, "test.py", "-opt", args.opt,
        f"--model_path={model_path}"
    ]
    if args.dataset:
        command.append(f"--dataset={args.dataset}")
        
    run_command(command)
    print("--- Testing complete ---")

def main():
    # This must be the first thing to run to ensure we are in the correct venv
    setup_virtual_environment()
    
    # Welcome message for Colab users
    if is_colab_or_kaggle():
        print("="*60)
        print("🚀 WELCOME TO RESTORMER ON COLAB!")
        print("="*60)
        print("This script will automatically:")
        print("  ✅ Install all dependencies")
        print("  ✅ Download training/testing datasets")
        print("  ✅ Train the Restormer model")
        print("  ✅ Test and evaluate the model")
        print("="*60)
        print()
    
    parser = argparse.ArgumentParser(
        description="Restormer all-in-one script for setup, training, and testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--skip-install', action='store_true', 
        help='Skip dependency installation and setup.py.'
    )
    parser.add_argument(
        '--skip-train', action='store_true',
        help='Skip the training step (requires a pre-existing trained model for testing).'
    )
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Skip the final testing and evaluation step.'
    )
    parser.add_argument(
        "--dataroot", type=str, default="Datasets",
        help="Root directory for the datasets for training."
    )
    parser.add_argument(
        "--opt", type=str, default="Options/Restormer.yml",
        help="Path to the options YAML file for training."
    )
    parser.add_argument(
        '--dataset', type=str,
        help='Name of a specific test dataset to evaluate (e.g., Rain100H). If not provided, all test datasets are evaluated.'
    )
    parser.add_argument(
        '--skip-data-download-check',
        action='store_true',
        help='Bypass the check for dataset existence and automatic download.'
    )
    args = parser.parse_args()

    if not args.skip_install:
        install(args)

    # Right after setup, before training/testing, check for data
    check_and_download_data(args)

    if not args.skip_train:
        train(args)
    
    if not args.skip_test:
        test(args)

    print("\n--- All steps complete! ---")

if __name__ == '__main__':
    main() 