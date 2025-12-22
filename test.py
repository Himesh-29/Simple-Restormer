import argparse
import yaml
import torch
import os
from models.restormer import Restormer
from data.dataset import Dataset_PairedImage
from torch.utils.data import DataLoader
from core.metrics import calculate_psnr
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model.')
    parser.add_argument('--dataset', type=str, help='Specific test dataset name.')
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    net_opt = opt['network_g'].copy()
    net_opt.pop('type', None)
    model = Restormer(**net_opt).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Dataset
    val_ds_opt = opt['datasets']['val']
    if args.dataset:
        # Override if needed
        pass
    
    val_set = Dataset_PairedImage(val_ds_opt)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    psnr_total = 0
    count = 0
    
    os.makedirs('results', exist_ok=True)

    # Metric options
    psnr_opt = opt['val']['metrics'].get('psnr', {})
    crop_border = psnr_opt.get('crop_border', 0)
    test_y_channel = psnr_opt.get('test_y_channel', False)

    print(f"Testing on {len(val_set)} images...")
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            lq = data['lq'].to(device)
            gt = data['gt'].to(device)
            output = model(lq)
            
            # Clamp and calculate
            output = torch.clamp(output, 0, 1)
            psnr = calculate_psnr(output[0], gt[0], crop_border=crop_border, test_y_channel=test_y_channel)
            psnr_total += psnr
            count += 1
            
            # Save output image
            img_name = os.path.basename(data['lq_path'][0])
            output_img = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            output_img = (output_img * 255.0).clip(0, 255).astype('uint8')
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'results/{img_name}', output_img)

    avg_psnr = psnr_total / count
    print(f"Average PSNR: {avg_psnr:.2f}")

if __name__ == '__main__':
    main()