import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import random
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from copy import deepcopy

from models import define_network
from data.dataset import Dataset_PairedImage
from core.scheduler import CosineAnnealingRestartCyclicLR
from core.metrics import calculate_psnr

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            if random.random() < 0.5:
                return self.mixup(target, input_)
        else:
            return self.mixup(target, input_)
        return target, input_

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        self.is_dist = self.world_size > 1
        if self.is_dist:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            torch.cuda.set_device(self.device)

        self._init_model()
        
        # EMA setup
        self.ema_decay = opt['train'].get('ema_decay', 0.999)
        if self.ema_decay > 0:
            self.net_g_ema = deepcopy(self.net_g).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            self.net_g_ema.eval()

        self._init_optimizer()
        self._init_scheduler()
        self._init_criterion()
        self._init_dataloaders()
        
        if self.rank == 0:
            log_dir = os.path.join('experiments', opt['name'])
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"train_{opt['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger('Restormer')
            self.logger.info(f"Model [{self.opt['network_g'].get('type', 'Restormer')}] is created.")
            self._log_complexity()
            
            self.writer = SummaryWriter(log_dir=os.path.join('tb_logger', opt['name']))
            os.makedirs(os.path.join('experiments', opt['name'], 'models'), exist_ok=True)
            os.makedirs(os.path.join('experiments', opt['name'], 'training_states'), exist_ok=True)

        self.mixing_flag = opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            self.mixing_augmentation = Mixing_Augment(
                opt['train']['mixing_augs'].get('mixup_beta', 1.2),
                opt['train']['mixing_augs'].get('use_identity', False),
                self.device
            )

    def _init_model(self):
        self.net_g = define_network(self.opt['network_g']).to(self.device)

        if self.is_dist:
            self.net_g = DDP(self.net_g, device_ids=[self.local_rank], output_device=self.local_rank)

    def _log_complexity(self):
        if self.rank != 0:
            return

        # Prepare dummy input for GFlops calculation
        # Use gt_size from config if available, else default to 128
        h = w = self.opt['datasets']['train'].get('gt_size', 128)
        dummy_input = torch.randn(1, self.opt['network_g'].get('inp_channels', 3), h, w).to(self.device)
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.net_g.parameters())
        
        # Calculate GFlops using fvcore
        try:
            from fvcore.nn import FlopCountAnalysis
            
            # Suppress fvcore's verbose warnings about unsupported operators (like mul, mean, var)
            # These element-wise ops have negligible impact on GFlops compared to Convs/LLMs.
            logging.getLogger('fvcore').setLevel(logging.ERROR)
            
            flops = FlopCountAnalysis(self.net_g, dummy_input)
            flops.unsupported_ops_warnings(False)
            
            total_flops = flops.total()
            gflops = total_flops / 1e9
            
            self.logger.info(f"Model Parameters: {total_params:,}")
            self.logger.info(f"Model GFlops: {gflops:.2f} (Input: {h}x{w})")
            
        except ImportError:
            self.logger.warning("fvcore not found. GFlops calculation skipped. Install with 'pip install fvcore'")
            self.logger.info(f"Model Parameters: {total_params:,}")
        except Exception as e:
            self.logger.warning(f"Error calculating GFlops: {e}")
            self.logger.info(f"Model Parameters: {total_params:,}")

    def _init_optimizer(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g']['type']
        lr = train_opt['optim_g']['lr']
        weight_decay = train_opt['optim_g'].get('weight_decay', 0)
        betas = train_opt['optim_g'].get('betas', (0.9, 0.999))

        if optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(self.net_g.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=betas)

    def _init_scheduler(self):
        sched_opt = self.opt['train']['scheduler']
        if sched_opt['type'] == 'CosineAnnealingRestartCyclicLR':
            self.scheduler = CosineAnnealingRestartCyclicLR(
                self.optimizer_g,
                periods=sched_opt['periods'],
                restart_weights=sched_opt['restart_weights'],
                eta_mins=sched_opt['eta_mins']
            )

    def _init_criterion(self):
        loss_opt = self.opt['train']['pixel_opt']
        if loss_opt['type'] == 'L1Loss':
            self.criterion = nn.L1Loss().to(self.device)
        elif loss_opt['type'] == 'MSELoss':
            self.criterion = nn.MSELoss().to(self.device)
        else:
            self.criterion = nn.L1Loss().to(self.device)

    def _init_dataloaders(self):
        train_ds_opt = self.opt['datasets']['train'].copy()
        train_ds_opt['phase'] = 'train'
        self.train_set = Dataset_PairedImage(train_ds_opt)
        self.train_sampler = DistributedSampler(self.train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True) if self.is_dist else None
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=train_ds_opt['batch_size_per_gpu'],
            shuffle=(self.train_sampler is None),
            num_workers=train_ds_opt['num_worker_per_gpu'],
            sampler=self.train_sampler,
            pin_memory=True
        )

        val_ds_opt = self.opt['datasets'].get('val')
        if val_ds_opt:
            val_ds_opt = val_ds_opt.copy()
            val_ds_opt['phase'] = 'val'
            self.val_set = Dataset_PairedImage(val_ds_opt)
            self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    def update_ema(self):
        if self.ema_decay <= 0:
            return
        with torch.no_grad():
            module = self.net_g.module if self.is_dist else self.net_g
            for p, p_ema in zip(module.parameters(), self.net_g_ema.parameters()):
                p_ema.copy_(self.ema_decay * p_ema + (1 - self.ema_decay) * p)

    def train(self):
        total_iter = self.opt['train']['total_iter']
        current_iter = 0
        epoch = 0
        start_time = time.time()
        
        # Progressive learning setup
        train_ds_opt = self.opt['datasets']['train']
        iters_list = train_ds_opt.get('iters')
        gt_sizes = train_ds_opt.get('gt_sizes')
        mini_batch_sizes = train_ds_opt.get('mini_batch_sizes')
        gt_size_orig = train_ds_opt['gt_size']
        batch_size_orig = train_ds_opt['batch_size_per_gpu']
        
        groups = np.array([sum(iters_list[0:i + 1]) for i in range(len(iters_list))])
        
        # Resume check
        state_path = self.opt['path'].get('resume_state')
        if state_path:
            self.resume_training(state_path)
            current_iter = self.resume_iter
            epoch = self.resume_epoch

        if self.rank == 0:
            self.logger.info(f"Start training from epoch: {epoch}, iter: {current_iter}")

        data_timer = time.time()
        iter_timer = time.time()
        
        while current_iter < total_iter:
            if self.is_dist:
                self.train_sampler.set_epoch(epoch)
            
            for batch_data in self.train_loader:
                data_time = time.time() - data_timer
                current_iter += 1
                if current_iter > total_iter:
                    break
                
                # Progressive learning update
                idx = np.searchsorted(groups, current_iter)
                if idx >= len(groups): idx = len(groups) - 1
                
                cur_gt_size = gt_sizes[idx]
                cur_batch_size = mini_batch_sizes[idx]

                # Log progressive learning update if it changes
                if current_iter == 1 or (idx > 0 and current_iter == groups[idx-1] + 1):
                    if self.rank == 0:
                        self.logger.info(f"\n Updating Patch_Size to {cur_gt_size} and Batch_Size to {cur_batch_size} \n")
                
                lq = batch_data['lq'].to(self.device)
                gt = batch_data['gt'].to(self.device)
                
                # Dynamic crop for progressive learning if needed
                if cur_gt_size < gt_size_orig:
                    h, w = lq.shape[2:]
                    x = random.randint(0, h - cur_gt_size)
                    y = random.randint(0, w - cur_gt_size)
                    lq = lq[:, :, x:x+cur_gt_size, y:y+cur_gt_size]
                    gt = gt[:, :, x:x+cur_gt_size, y:y+cur_gt_size]
                
                # Dynamic batch size if needed
                if cur_batch_size < batch_size_orig:
                    lq = lq[:cur_batch_size]
                    gt = gt[:cur_batch_size]

                # Mixing augmentation
                if self.mixing_flag:
                    gt, lq = self.mixing_augmentation(gt, lq)

                self.optimizer_g.zero_grad()
                output = self.net_g(lq)
                loss = self.criterion(output, gt)
                loss.backward()
                
                if self.opt['train'].get('use_grad_clip', False):
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                
                self.optimizer_g.step()
                self.scheduler.step()
                self.update_ema()
                
                iter_time = time.time() - iter_timer

                if self.rank == 0 and current_iter % self.opt['logger']['print_freq'] == 0:
                    lr = self.optimizer_g.param_groups[0]['lr']
                    # Calculate ETA
                    remain_iter = total_iter - current_iter
                    avg_iter_time = iter_time # Simplified, for real ETA we'd use a moving average
                    eta_sec = remain_iter * avg_iter_time
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    
                    self.logger.info(
                        f"[{self.opt['name']}][epoch: {epoch:2d}, iter: {current_iter:7,d}, lr:({lr:.3e},)] "
                        f"[eta: {eta_str}, time (data): {iter_time:.3f} ({data_time:.3f})] l_pix: {loss.item():.4e} "
                    )
                    
                    self.writer.add_scalar('Loss/train', loss.item(), current_iter)
                    self.writer.add_scalar('LR', lr, current_iter)
                    self.writer.flush()

                if current_iter % self.opt['val']['val_freq'] == 0:
                    self.validate(current_iter)

                if self.rank == 0 and current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
                    self.logger.info("Saving models and training states.")
                    self.save(epoch, current_iter)
                
                data_timer = time.time()
                iter_timer = time.time()

            epoch += 1

        if self.rank == 0:
            consumed_time = str(timedelta(seconds=int(time.time() - start_time)))
            self.logger.info(f'End of training. Time consumed: {consumed_time}')
            self.logger.info('Save the latest model.')
            self.save(epoch, current_iter)
        
        # Final validation
        self.validate(current_iter)

    @torch.no_grad()
    def validate(self, current_iter):
        model = self.net_g_ema if hasattr(self, 'net_g_ema') else (self.net_g.module if self.is_dist else self.net_g)
        model.eval()
        psnr_total = 0
        count = 0
        
        psnr_opt = self.opt['val']['metrics'].get('psnr', {})
        crop_border = psnr_opt.get('crop_border', 0)
        test_y_channel = psnr_opt.get('test_y_channel', False)

        for val_data in self.val_loader:
            lq = val_data['lq'].to(self.device)
            gt = val_data['gt'].to(self.device)
            output = model(lq)
            # Clamp output to [0, 1] for metric calculation
            output = torch.clamp(output, 0, 1)
            psnr_total += calculate_psnr(output[0], gt[0], crop_border=crop_border, test_y_channel=test_y_channel)
            count += 1
        
        avg_psnr = psnr_total / count
        if self.is_dist:
            avg_psnr_t = torch.tensor(avg_psnr, device=self.device)
            dist.all_reduce(avg_psnr_t, op=dist.ReduceOp.SUM)
            avg_psnr = avg_psnr_t.item() / self.world_size

        if self.rank == 0:
            self.logger.info(f"Validation ValSet, # psnr: {avg_psnr:.4f}")
            self.writer.add_scalar('PSNR/val', avg_psnr, current_iter)
            self.writer.flush()
        
        self.net_g.train()

    def save(self, epoch, current_iter):
        net_g = self.net_g.module if self.is_dist else self.net_g
        save_dict = {'params': net_g.state_dict()}
        if hasattr(self, 'net_g_ema'):
            save_dict['params_ema'] = self.net_g_ema.state_dict()
        
        # Save model
        save_path = os.path.join('experiments', self.opt['name'], 'models', f'net_g_{current_iter}.pth')
        torch.save(save_dict, save_path)
        latest_path = os.path.join('experiments', self.opt['name'], 'models', 'net_g_latest.pth')
        torch.save(save_dict, latest_path)

        # Save training state
        state_dict = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizer_g': self.optimizer_g.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        state_path = os.path.join('experiments', self.opt['name'], 'training_states', f'{current_iter}.state')
        torch.save(state_dict, state_path)

    def resume_training(self, state_path):
        state = torch.load(state_path, map_location=self.device, weights_only=True)
        self.resume_epoch = state['epoch']
        self.resume_iter = state['iter']
        self.optimizer_g.load_state_dict(state['optimizer_g'])
        self.scheduler.load_state_dict(state['scheduler'])
        
        # Load model weights
        model_path = os.path.join('experiments', self.opt['name'], 'models', f"net_g_{self.resume_iter}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            net_g = self.net_g.module if self.is_dist else self.net_g
            net_g.load_state_dict(checkpoint['params'])
            if 'params_ema' in checkpoint and hasattr(self, 'net_g_ema'):
                self.net_g_ema.load_state_dict(checkpoint['params_ema'])
            print(f"Resumed from {model_path}")

if __name__ == '__main__':
    # For testing purposes
    pass
