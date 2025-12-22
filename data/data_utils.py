import os
import cv2
import numpy as np
import torch

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders."""
    paths = []
    lq_folder, gt_folder = folders
    lq_filenames = sorted(os.listdir(lq_folder))
    
    for lq_filename in lq_filenames:
        basename, ext = os.path.splitext(lq_filename)
        gt_filename = filename_tmpl.format(basename) + ext
        gt_path = os.path.join(gt_folder, gt_filename)
        lq_path = os.path.join(lq_folder, lq_filename)
        
        if os.path.exists(gt_path):
            paths.append({'lq_path': lq_path, 'gt_path': gt_path})
    return paths

def imfrombytes(img_bytes, float32=True):
    """Read image from bytes."""
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Convert numpy images to tensors."""
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def paired_random_crop(img_gts, img_lqs, gt_size, scale):
    """Paired random crop."""
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    lq_patch_size = gt_size // scale

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        # pad if necessary
        pad_h = max(0, lq_patch_size - h_lq)
        pad_w = max(0, lq_patch_size - w_lq)
        img_lqs = [cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT) for img in img_lqs]
        img_gts = [cv2.copyMakeBorder(img, 0, pad_h * scale, 0, pad_w * scale, cv2.BORDER_REFLECT) for img in img_gts]
        h_lq, w_lq, _ = img_lqs[0].shape

    top = np.random.randint(0, h_lq - lq_patch_size + 1)
    left = np.random.randint(0, w_lq - lq_patch_size + 1)

    img_lqs = [img[top:top + lq_patch_size, left:left + lq_patch_size, ...] for img in img_lqs]
    top_gt, left_gt = top * scale, left * scale
    img_gts = [img[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...] for img in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def augment(imgs, hflip=True, rotation=True):
    """Augment images with flip and rotation."""
    hflip = hflip and np.random.random() < 0.5
    vflip = rotation and np.random.random() < 0.5
    rot90 = rotation and np.random.random() < 0.5

    def _augment(img):
        if hflip:
            img = cv2.flip(img, 1)
        if vflip:
            img = cv2.flip(img, 0)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if isinstance(imgs, list):
        return [_augment(img) for img in imgs]
    else:
        return _augment(imgs)
