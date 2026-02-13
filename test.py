import argparse
import yaml
import torch
import os
import sys
from tqdm import tqdm
from models import define_network
from data.dataset import Dataset_PairedImage
from torch.utils.data import DataLoader
from core.metrics import calculate_psnr, calculate_ssim
import cv2
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model.')
    parser.add_argument('--dataset', type=str, help='Specific test dataset name folder.')
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    os.makedirs('results', exist_ok=True)
    log_file = os.path.join('results', f"test_{opt['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('RestormerTest')

    logger.info(f"Using device: {device}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Option path: {args.opt}")
    
    # Load model
    model = define_network(opt['network_g']).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Successfully loaded model from {args.model_path}")
    model.eval()

    # Get test root and discover datasets
    # In the YAML, val dataroot_gt usually points to something like ./Datasets/test/Rain100H/target
    # We want to find the common root: ./Datasets/test
    val_ds_opt_orig = opt['datasets']['val']
    dataroot_gt_orig = val_ds_opt_orig['dataroot_gt']
    
    # Heuristic to find the test root: parent of the dataset folder
    # e.g. ./Datasets/test/Rain100H/target -> ./Datasets/test
    test_root = os.path.dirname(os.path.dirname(dataroot_gt_orig))
    
    target_datasets = ['Rain100H', 'Rain100L', 'Test100']
    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    else:
        if os.path.exists(test_root):
             all_folders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
             datasets = [d for d in target_datasets if d in all_folders]
             if not datasets:
                 logger.warning(f"Target datasets {target_datasets} not found in {test_root}. Available: {all_folders}")
        else:
            logger.warning(f"Test root {test_root} not found. Fallback to YAML dataset if in target list.")
            yaml_ds = os.path.basename(os.path.dirname(dataroot_gt_orig))
            if yaml_ds in target_datasets:
                datasets = [yaml_ds]
            test_root = os.path.dirname(os.path.dirname(test_root)) # fallback

    # Metric options
    psnr_opt = opt['val'].get('metrics', {}).get('psnr', {})
    crop_border = psnr_opt.get('crop_border', 0)
    test_y_channel = psnr_opt.get('test_y_channel', False)

    print("Starting evaluation on test datasets...")
    logger.info(f"Discovered datasets: {datasets}")
    
    os.makedirs('results', exist_ok=True)

    for dataset_name in datasets:
        # Construct dynamic paths for the dataset
        ds_opt = val_ds_opt_orig.copy()
        cur_ds_root = os.path.join(test_root, dataset_name)
        ds_opt['dataroot_gt'] = os.path.join(cur_ds_root, 'target')
        ds_opt['dataroot_lq'] = os.path.join(cur_ds_root, 'input')
        
        if not os.path.exists(ds_opt['dataroot_gt']):
            logger.info(f"Skipping {dataset_name}: target folder not found.")
            continue

        # Create dataset-specific result folder
        dataset_results_dir = os.path.join('results', dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)

        val_set = Dataset_PairedImage(ds_opt)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

        psnr_total = 0
        ssim_total = 0
        count = 0
        
        pbar = tqdm(total=len(val_set), unit='img', desc=f"Processing {dataset_name}")
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                lq = data['lq'].to(device)
                gt = data['gt'].to(device)
                output = model(lq)
                output = torch.clamp(output, 0, 1)
                
                # Metrics
                psnr = calculate_psnr(output[0], gt[0], crop_border=crop_border, test_y_channel=test_y_channel)
                ssim = calculate_ssim(output[0], gt[0], crop_border=crop_border, test_y_channel=test_y_channel)
                
                psnr_total += psnr
                ssim_total += ssim
                count += 1
                
                # Save processed image
                img_name = os.path.basename(data['lq_path'][0])
                output_img = output[0].detach().cpu().numpy().transpose(1, 2, 0)
                output_img = (output_img * 255.0).clip(0, 255).astype('uint8')
                output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(dataset_results_dir, img_name), output_img)
                
                pbar.update(1)
        
        pbar.close()
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count
        logger.info(f"{dataset_name}: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}")

    logger.info("Evaluation complete.")

if __name__ == '__main__':
    main()
