import argparse
import yaml
import torch
import os
import random
import numpy as np
from core.trainer import Trainer

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    # Distributed setup
    if args.launcher == 'pytorch':
        # os.environ values are already set by torchrun/launcher
        pass
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

    seed = opt.get('manual_seed', 100)
    set_random_seed(seed + int(os.environ['RANK']))

    trainer = Trainer(opt)
    trainer.train()

if __name__ == '__main__':
    main()
