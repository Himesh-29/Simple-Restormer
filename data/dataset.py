import torch
from torch.utils import data as data
from data.data_utils import paired_paths_from_folder, img2tensor, paired_random_crop, augment
import cv2
import numpy as np

class Dataset_PairedImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)
        
        self.phase = opt.get('phase', 'train')
        self.geometric_augs = opt.get('geometric_augs', True)
        self.scale = opt.get('scale', 1)

    def __getitem__(self, index):
        index = index % len(self.paths)
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        if self.phase == 'train' and 'gt_size' in self.opt:
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale)
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = augment([img_gt, img_lq])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
