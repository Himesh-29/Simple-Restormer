"""
Model-Agnostic Interactive Visualization Dashboard
===================================================

Launch:
    uv run visualize_model.py --opt Options/Restormer.yml
    uv run visualize_model.py --opt Options/Restormer.yml -c path/to/checkpoint.pth
    uv run visualize_model.py --opt Options/AnyCustomModel.yml

Works with ANY model defined via an Options YAML file.  Uses PyTorch
forward hooks to capture every nn.Module's output automatically, plus
block-level I/O capture on every direct child module.

For models whose forward() accepts a ``capture_all`` kwarg, also
captures model-specific data (cluster IDs, attention weights, etc.).

Tabs
----
  1. INPUT / OUTPUT          — side-by-side with residual & PSNR/SSIM
  2. IMAGE CHANNELS          — R/G/B decomposition + histograms for I/O
  3. MULTI-SCALE VIEW        — input vs output at 1x, 1/2x, 1/4x
  4. FEATURE FLOW            — activation energy at every captured layer
  5. BLOCK I/O               — input vs output for every top-level block
  6. CHANNEL EXPLORER        — browse any layer's channels interactively
  7. CONTEXT LOSS            — L1 diff between consecutive layers
  8+ CLUSTER/ATTENTION TABS  — (auto-shown when model supports capture_all)
"""

import argparse
import inspect
import math
import os
import re
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── Resolve project root ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models import define_network


# ═════════════════════════════════════════════════════════════
#  Model loading  (model-agnostic)
# ═════════════════════════════════════════════════════════════

def load_opt(yml_path):
    """Load an Options YAML and return the full config dict."""
    with open(yml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_from_opt(opt, ckpt_path=None):
    """Instantiate model from Options config, optionally load checkpoint."""
    net_opt = opt['network_g']
    model = define_network(net_opt)

    if ckpt_path is None:
        name = opt.get('name', '')
        candidates = [
            os.path.join(ROOT, 'experiments', name, 'models', 'net_g_latest.pth'),
            os.path.join(ROOT, 'experiments', name, 'models', 'net_g_7000.pth'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'params' in state:
            state = state['params']
        model.load_state_dict(state, strict=True)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found — using random weights")

    model.eval()
    return model


def get_model_type(opt):
    """Return the model class name string from Options."""
    return opt.get('network_g', {}).get('type', 'Unknown')


def _has_capture_all(model):
    """Check whether model.forward() accepts a capture_all kwarg."""
    sig = inspect.signature(model.forward)
    return 'capture_all' in sig.parameters


# ═════════════════════════════════════════════════════════════
#  Hook-based activation capture  (works with ANY model)
# ═════════════════════════════════════════════════════════════

class ActivationCapture:
    """Register forward hooks on all named modules to capture outputs.

    Stores only 4-D ``(B,C,H,W)`` or 3-D ``(B,N,C)`` tensors to keep
    memory manageable.
    """

    def __init__(self, model: nn.Module, max_captures: int = 200):
        self.captures: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks = []
        count = 0
        for name, mod in model.named_modules():
            if count >= max_captures:
                break
            if name == '':
                continue
            h = mod.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)
            count += 1

    def _make_hook(self, name: str):
        def _fn(_module, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() in (3, 4):
                self.captures[name] = out.detach().cpu()
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                t = out[0]
                if isinstance(t, torch.Tensor) and t.dim() in (3, 4):
                    self.captures[name] = t.detach().cpu()
        return _fn

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class BlockIOCapture:
    """Capture input AND output tensors for every direct child module.

    This gives a block-level view of the data flow: what went into each
    top-level architectural component and what came out.
    Works with any ``nn.Module`` — completely model-agnostic.
    """

    def __init__(self, model: nn.Module):
        self.block_io: OrderedDict[str, dict] = OrderedDict()
        self._hooks = []
        for name, child in model.named_children():
            h = child.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _make_hook(self, name: str):
        def _fn(_module, inp, out):
            inp_t = inp[0] if isinstance(inp, tuple) else inp
            out_t = out[0] if isinstance(out, (tuple, list)) else out
            record = {}
            if isinstance(inp_t, torch.Tensor) and inp_t.dim() in (3, 4):
                record['input'] = inp_t.detach().cpu()
            if isinstance(out_t, torch.Tensor) and out_t.dim() in (3, 4):
                record['output'] = out_t.detach().cpu()
            if record:
                self.block_io[name] = record
        return _fn

    def clear(self):
        self.block_io.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═════════════════════════════════════════════════════════════
#  Image helpers
# ═════════════════════════════════════════════════════════════

def load_image(path: str, max_side: int = 256) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        h_new, w_new = int(h * scale), int(w * scale)
        h_new -= h_new % 2; w_new -= w_new % 2
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    else:
        h -= h % 2; w -= w % 2
        img = img[:h, :w]
    return img


def _img_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
    t = t[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


def _fig_to_array(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return arr


# ═════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═════════════════════════════════════════════════════════════

def _channel_heatmap(feat_4d, channel_idx):
    ch = feat_4d[0, channel_idx].numpy()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    vmax = max(abs(ch.max()), abs(ch.min()), 1e-6)
    im = ax.imshow(ch, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f'Channel {channel_idx}  (min={ch.min():.3f} max={ch.max():.3f})', fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_array(fig)


def _feature_overview(feat_4d, title, max_channels=16):
    C = feat_4d.shape[1]
    n = min(C, max_channels)
    cols = min(n, 4)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), dpi=100)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ch = feat_4d[0, i].numpy()
            vmax = max(abs(ch.max()), abs(ch.min()), 1e-6)
            ax.imshow(ch, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_title(f'ch {i}', fontsize=8)
        ax.axis('off')
    fig.suptitle(f'{title}  (shape: {list(feat_4d.shape)})', fontsize=11)
    fig.tight_layout()
    return _fig_to_array(fig)


def _cluster_overlay(img_np, cluster_ids, H, W, num_clusters):
    cmap = plt.cm.get_cmap('tab20', num_clusters)
    cids = cluster_ids[0].numpy().reshape(H, W)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=100)
    axes[0].imshow(img_np); axes[0].set_title('Input Image'); axes[0].axis('off')
    im = axes[1].imshow(cids, cmap=cmap, interpolation='nearest')
    axes[1].set_title(f'Clusters ({len(np.unique(cids))} active / {num_clusters})')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cluster_color = cmap(cids / max(cids.max(), 1))[:, :, :3].astype(np.float32)
    blended = 0.55 * img_np.copy() + 0.45 * cluster_color
    axes[2].imshow(np.clip(blended, 0, 1)); axes[2].set_title('Overlay'); axes[2].axis('off')
    fig.tight_layout()
    return _fig_to_array(fig)


def _per_head_contribution(per_head_out, H, W, num_heads):
    pho = per_head_out[0].numpy()
    head_norms = np.linalg.norm(pho, axis=-1)
    cols = min(num_heads, 4)
    rows = max(1, math.ceil(num_heads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), dpi=100)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_heads:
            hmap = head_norms[:, i].reshape(H, W)
            im = ax.imshow(hmap, cmap='inferno', interpolation='nearest')
            ax.set_title(f'Head {i} (mean={hmap.mean():.3f})', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    fig.suptitle('Per-Head Contribution (L2 norm)', fontsize=11)
    fig.tight_layout()
    return _fig_to_array(fig)


def _attention_heatmap_fig(attn_weights, window_idx=0, head_idx=0):
    attn = attn_weights[0, window_idx, head_idx].numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    ax.imshow(attn, cmap='viridis', interpolation='nearest', vmin=0)
    ax.set_title(f'Attention — Window {window_idx}, Head {head_idx}')
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    fig.tight_layout()
    return _fig_to_array(fig)


def _context_loss_analysis(layer_captures):
    """L1 diff between consecutive 4-D captured layers."""
    keys_4d = [k for k, v in layer_captures.items() if v.dim() == 4]
    pairs = []
    for i in range(len(keys_4d) - 1):
        k1, k2 = keys_4d[i], keys_4d[i + 1]
        f1, f2 = layer_captures[k1], layer_captures[k2]
        if f1.shape[2:] != f2.shape[2:]:
            continue
        c = min(f1.shape[1], f2.shape[1])
        diff = (f1[0, :c] - f2[0, :c]).abs().mean(dim=0).numpy()
        n1 = k1.rsplit('.', 1)[-1] if '.' in k1 else k1
        n2 = k2.rsplit('.', 1)[-1] if '.' in k2 else k2
        pairs.append((f'{n1} -> {n2}', diff))
        if len(pairs) >= 8:
            break

    if not pairs:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=100)
        ax.text(0.5, 0.5, 'No comparable consecutive layers found', ha='center', va='center')
        ax.axis('off')
        return _fig_to_array(fig)

    n = len(pairs)
    cols = min(n, 4)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), dpi=100)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            label, diff = pairs[i]
            im = ax.imshow(diff, cmap='hot', interpolation='nearest')
            ax.set_title(f'{label}\nMean={diff.mean():.4f}', fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    fig.suptitle('Context Change (L1 between consecutive layers)', fontsize=12)
    fig.tight_layout()
    return _fig_to_array(fig)


def _activation_stats_table(layer_captures, model_type=''):
    lines = [f"Model: {model_type}", ""]
    lines.append(f"{'Layer':<50s} {'Shape':>22s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    lines.append("-" * 120)
    for name, t in layer_captures.items():
        t_f = t.float()
        lines.append(
            f"{name:<50s} {str(list(t_f.shape)):>22s} "
            f"{t_f.mean():>10.4f} {t_f.std():>10.4f} "
            f"{t_f.min():>10.4f} {t_f.max():>10.4f}"
        )
    lines.append(f"\nTotal captured layers: {len(layer_captures)}")
    return '\n'.join(lines)


def _channel_decomposition(img_np, title="Image"):
    """Show R, G, B channels separately with per-channel histograms."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)

    # Row 1: original + R, G, B channels
    axes[0, 0].imshow(np.clip(img_np, 0, 1))
    axes[0, 0].set_title(f'{title} (RGB)', fontsize=10)
    axes[0, 0].axis('off')

    ch_names = ['Red', 'Green', 'Blue']
    ch_cmaps = ['Reds', 'Greens', 'Blues']
    ch_colors = ['red', 'green', 'blue']

    for i, (cname, cmap_name) in enumerate(zip(ch_names, ch_cmaps)):
        ch = img_np[:, :, i]
        axes[0, i + 1].imshow(ch, cmap=cmap_name, vmin=0, vmax=1)
        axes[0, i + 1].set_title(
            f'{cname}  \u03bc={ch.mean():.3f}  \u03c3={ch.std():.3f}', fontsize=9)
        axes[0, i + 1].axis('off')

    # Row 2: histograms
    axes[1, 0].hist(img_np.ravel(), bins=64, color='gray', alpha=0.7, density=True)
    axes[1, 0].set_title('Overall Histogram', fontsize=9)
    axes[1, 0].set_xlim(0, 1)

    for i, (cname, color) in enumerate(zip(ch_names, ch_colors)):
        ch = img_np[:, :, i]
        axes[1, i + 1].hist(ch.ravel(), bins=64, color=color, alpha=0.7, density=True)
        axes[1, i + 1].set_title(f'{cname} Distribution', fontsize=9)
        axes[1, i + 1].set_xlim(0, 1)

    fig.suptitle(f'Channel Decomposition \u2014 {title}', fontsize=12)
    fig.tight_layout()
    return _fig_to_array(fig)


def _input_output_channel_comparison(inp_np, out_np):
    """Side-by-side per-channel comparison between input and output."""
    out_float = out_np.astype(np.float32) / 255.0

    ch_names = ['Red', 'Green', 'Blue']
    ch_cmaps = ['Reds', 'Greens', 'Blues']
    ch_colors = ['red', 'green', 'blue']

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), dpi=100)
    for i, (cname, cmap_name, color) in enumerate(
            zip(ch_names, ch_cmaps, ch_colors)):
        inp_ch = inp_np[:, :, i]
        out_ch = out_float[:, :, i]
        diff_ch = np.abs(inp_ch - out_ch)

        axes[i, 0].imshow(inp_ch, cmap=cmap_name, vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input {cname}', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(out_ch, cmap=cmap_name, vmin=0, vmax=1)
        axes[i, 1].set_title(f'Output {cname}', fontsize=9)
        axes[i, 1].axis('off')

        im = axes[i, 2].imshow(diff_ch, cmap='hot', vmin=0,
                               vmax=max(diff_ch.max(), 1e-6))
        axes[i, 2].set_title(
            f'{cname} |Diff| (\u03bc={diff_ch.mean():.4f})', fontsize=9)
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        axes[i, 3].hist(inp_ch.ravel(), bins=64, color=color,
                        alpha=0.4, density=True, label='Input')
        axes[i, 3].hist(out_ch.ravel(), bins=64, color='navy',
                        alpha=0.4, density=True, label='Output')
        axes[i, 3].legend(fontsize=7)
        axes[i, 3].set_title(f'{cname} Histogram', fontsize=9)
        axes[i, 3].set_xlim(0, 1)

    fig.suptitle('Per-Channel Input vs Output Comparison', fontsize=12)
    fig.tight_layout()
    return _fig_to_array(fig)


def _low_scale_comparison(inp_np, out_np):
    """Downscaled view highlighting differences at multiple scales."""
    out_float = out_np.astype(np.float32) / 255.0
    h, w = inp_np.shape[:2]

    scales = [1.0, 0.5, 0.25]
    scale_labels = ['Full', '1/2', '1/4']

    fig, axes = plt.subplots(len(scales), 3,
                             figsize=(15, 5 * len(scales)), dpi=100)
    for si, (scale, label) in enumerate(zip(scales, scale_labels)):
        new_h = max(2, int(h * scale))
        new_w = max(2, int(w * scale))

        inp_s = cv2.resize(inp_np, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)
        out_s = cv2.resize(out_float, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)
        diff = np.abs(inp_s - out_s)
        diff_vis = np.clip(diff * 5, 0, 1)

        axes[si, 0].imshow(np.clip(inp_s, 0, 1))
        axes[si, 0].set_title(
            f'Input @ {label} ({new_h}\u00d7{new_w})', fontsize=10)
        axes[si, 0].axis('off')

        axes[si, 1].imshow(np.clip(out_s, 0, 1))
        axes[si, 1].set_title(
            f'Output @ {label} ({new_h}\u00d7{new_w})', fontsize=10)
        axes[si, 1].axis('off')

        axes[si, 2].imshow(diff_vis)
        axes[si, 2].set_title(
            f'|Diff| x5 @ {label} (\u03bc={diff.mean():.4f})', fontsize=10)
        axes[si, 2].axis('off')

    fig.suptitle('Multi-Scale Input vs Output Comparison', fontsize=13)
    fig.tight_layout()
    return _fig_to_array(fig)


def _block_io_visualization(block_io, block_name):
    """Visualize input and output of a specific block as mean-activation maps."""
    record = block_io[block_name]
    panels, titles = [], []

    for key in ['input', 'output']:
        if key not in record:
            continue
        t = record[key]
        if t.dim() == 4:
            act = t[0].float().abs().mean(dim=0).numpy()
            panels.append(act)
            titles.append(f'{key.title()} (shape: {list(t.shape)})')

    if not panels:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=100)
        ax.text(0.5, 0.5, f'No 4-D tensors for block "{block_name}"',
                ha='center', va='center')
        ax.axis('off')
        return _fig_to_array(fig)

    has_diff = ('input' in record and 'output' in record
                and record['input'].dim() == 4 and record['output'].dim() == 4)
    ncols = len(panels) + (1 if has_diff else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), dpi=100)
    if ncols == 1:
        axes = [axes]

    for i, (panel, title) in enumerate(zip(panels, titles)):
        im = axes[i].imshow(panel, cmap='magma', interpolation='bilinear')
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if has_diff:
        inp_t, out_t = record['input'], record['output']
        if inp_t.shape[2:] == out_t.shape[2:]:
            c = min(inp_t.shape[1], out_t.shape[1])
            diff = (inp_t[0, :c] - out_t[0, :c]).abs().mean(dim=0).numpy()
            im = axes[-1].imshow(diff, cmap='hot', interpolation='bilinear')
            axes[-1].set_title(
                f'Block Change (L1, \u03bc={diff.mean():.4f})', fontsize=9)
            axes[-1].axis('off')
            fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        else:
            axes[-1].text(
                0.5, 0.5,
                f'Shape mismatch:\nIn: {list(inp_t.shape)}\nOut: {list(out_t.shape)}',
                ha='center', va='center', fontsize=9)
            axes[-1].axis('off')

    fig.suptitle(f'Block: {block_name}', fontsize=12)
    fig.tight_layout()
    return _fig_to_array(fig)


def _all_blocks_summary(block_io, model_type=''):
    """Summary view of all blocks' output activation energy."""
    blocks_4d = [(k, v) for k, v in block_io.items()
                 if 'output' in v and v['output'].dim() == 4]
    if not blocks_4d:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=100)
        ax.text(0.5, 0.5, 'No block outputs captured', ha='center', va='center')
        ax.axis('off')
        return _fig_to_array(fig)

    n = len(blocks_4d)
    cols = min(n, 5)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 4 * rows), dpi=100)
    if rows == 1:
        axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            name, record = blocks_4d[i]
            out = record['output']
            act = out[0].float().abs().mean(dim=0).numpy()
            im = ax.imshow(act, cmap='magma', interpolation='bilinear')
            ax.set_title(f'{name}\n{list(out.shape)}', fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    fig.suptitle(f'{model_type} \u2014 All Blocks Output Energy', fontsize=12)
    fig.tight_layout()
    return _fig_to_array(fig)


def _compute_psnr(img1, img2):
    """Compute PSNR between two float images in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def _compute_ssim_simple(img1, img2):
    """Simplified SSIM (mean over channels) between two [0,1] images."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(img1.shape[2]):
        a, b = img1[:, :, c], img2[:, :, c]
        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.var(), b.var()
        sig_ab = np.mean((a - mu_a) * (b - mu_b))
        num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
        den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
        vals.append(num / den)
    return float(np.mean(vals))


# ═════════════════════════════════════════════════════════════
#  Inference  (hook-based + optional native capture_all)
# ═════════════════════════════════════════════════════════════

def run_inference(model, img_np, model_type=''):
    """Run model with hook-based, block-level, and (if available) native capture."""
    inp = _img_to_tensor(img_np)

    ac = ActivationCapture(model)
    bio = BlockIOCapture(model)

    native_captures = {}
    use_native = _has_capture_all(model)

    with torch.no_grad():
        if use_native:
            out = model(inp, capture_all=native_captures)
        else:
            out = model(inp)

    hook_captures = OrderedDict(ac.captures)
    block_io = OrderedDict(bio.block_io)
    ac.remove()
    bio.remove()

    # Prepend input, append output
    ordered = OrderedDict()
    ordered['__input__'] = inp.detach().cpu()
    ordered.update(hook_captures)
    ordered['__output__'] = out.detach().cpu()

    out_np = _tensor_to_img(out.cpu())
    return out_np, ordered, native_captures, block_io


# ═════════════════════════════════════════════════════════════
#  Gradio Dashboard
# ═════════════════════════════════════════════════════════════

def build_dashboard(model, model_type, default_image_path=None):
    """Build and return a Gradio Blocks app (model-agnostic)."""
    import gradio as gr

    _state = {
        'hook_captures': None,     # OrderedDict  — all layer outputs
        'native_captures': None,   # dict         — model-specific (capture_all)
        'block_io': None,          # OrderedDict  — top-level block I/O
        'img_np': None,
        'out_np': None,
        'H': 0, 'W': 0,
        'model_type': model_type,
    }
    # Capability-based detection: show extra tabs if model supports capture_all
    _has_native_capture = _has_capture_all(model)

    # ── Callbacks ────────────────────────────────────────────

    def process_image(img_input):
        if img_input is None:
            return None, "No image provided"
        img_np = img_input.astype(np.float32) / 255.0
        h, w = img_np.shape[:2]
        h -= h % 2; w -= w % 2
        img_np = img_np[:h, :w]

        out_np, hook_caps, native_caps, block_io = run_inference(
            model, img_np, model_type)
        _state['hook_captures'] = hook_caps
        _state['native_captures'] = native_caps
        _state['block_io'] = block_io
        _state['img_np'] = img_np
        _state['out_np'] = out_np
        _state['H'] = h; _state['W'] = w

        # I/O comparison figure with metrics
        out_float = out_np.astype(np.float32) / 255.0
        residual = np.abs(img_np - out_float)
        residual_vis = (np.clip(residual * 5, 0, 1) * 255).astype(np.uint8)

        psnr_val = _compute_psnr(img_np, out_float)
        ssim_val = _compute_ssim_simple(img_np, out_float)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        axes[0].imshow(img_np)
        axes[0].set_title(f'Input ({h}\u00d7{w})', fontsize=10)
        axes[0].axis('off')
        axes[1].imshow(out_np)
        axes[1].set_title(f'Output (Restored)', fontsize=10)
        axes[1].axis('off')
        axes[2].imshow(residual_vis)
        axes[2].set_title(f'Residual (x5)', fontsize=10)
        axes[2].axis('off')
        fig.suptitle(
            f'PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  '
            f'Blocks captured: {len(block_io)}',
            fontsize=11, y=0.02)
        fig.tight_layout()
        io_img = _fig_to_array(fig)

        stats = _activation_stats_table(hook_caps, model_type)
        return io_img, stats

    # ── Feature Flow ──
    def view_feature_flow():
        if _state['hook_captures'] is None:
            return None
        caps = _state['hook_captures']
        items_4d = [(k, v) for k, v in caps.items() if v.dim() == 4]
        if not items_4d:
            return None
        if len(items_4d) > 15:
            step = max(1, len(items_4d) // 15)
            items_4d = items_4d[::step]
        n = len(items_4d)
        cols = min(n, 5)
        rows = max(1, math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), dpi=100)
        if rows == 1:
            axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < n:
                name, feat = items_4d[i]
                act = feat[0].float().abs().mean(dim=0).numpy()
                im = ax.imshow(act, cmap='magma', interpolation='bilinear')
                short = name.rsplit('.', 1)[-1] if '.' in name else name
                ax.set_title(f'{short}\n{list(feat.shape)}', fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        fig.suptitle(f'{model_type} — Feature Flow (Mean |Activation|)', fontsize=12)
        fig.tight_layout()
        return _fig_to_array(fig)

    # ── Channel Explorer helpers ──
    def _layer_names_4d():
        if _state['hook_captures'] is None:
            return []
        return [k for k, v in _state['hook_captures'].items() if v.dim() == 4]

    def refresh_layer_list():
        names = _layer_names_4d()
        return gr.Dropdown(choices=names, value=names[0] if names else None)

    def view_overview(layer_name):
        if not _state['hook_captures'] or layer_name not in _state['hook_captures']:
            return None
        feat = _state['hook_captures'][layer_name]
        return _feature_overview(feat, layer_name, max_channels=16) if feat.dim() == 4 else None

    def view_channel(layer_name, ch_idx):
        if not _state['hook_captures'] or layer_name not in _state['hook_captures']:
            return None
        feat = _state['hook_captures'][layer_name]
        if feat.dim() != 4:
            return None
        ch = max(0, min(int(ch_idx), feat.shape[1] - 1))
        return _channel_heatmap(feat, ch)

    # ── Context loss ──
    def view_ctx():
        if _state['hook_captures'] is None:
            return None
        return _context_loss_analysis(_state['hook_captures'])

    # ── Image Channels ──
    def view_input_channels():
        if _state['img_np'] is None:
            return None
        return _channel_decomposition(_state['img_np'], 'Input')

    def view_output_channels():
        if _state['out_np'] is None:
            return None
        out_float = _state['out_np'].astype(np.float32) / 255.0
        return _channel_decomposition(out_float, 'Output')

    def view_channel_comparison():
        if _state['img_np'] is None or _state['out_np'] is None:
            return None
        return _input_output_channel_comparison(_state['img_np'], _state['out_np'])

    # ── Multi-Scale View ──
    def view_multiscale():
        if _state['img_np'] is None or _state['out_np'] is None:
            return None
        return _low_scale_comparison(_state['img_np'], _state['out_np'])

    # ── Block I/O ──
    def _block_names():
        bio = _state.get('block_io')
        if not bio:
            return []
        return list(bio.keys())

    def refresh_block_list():
        names = _block_names()
        return gr.Dropdown(choices=names, value=names[0] if names else None)

    def view_block_detail(block_name):
        bio = _state.get('block_io')
        if not bio or block_name not in bio:
            return None
        return _block_io_visualization(bio, block_name)

    def view_blocks_summary():
        bio = _state.get('block_io')
        if not bio:
            return None
        return _all_blocks_summary(bio, model_type)

    # ── Cluster-specific callbacks (models with capture_all) ──
    def _cluster_block_names():
        nc = _state.get('native_captures')
        if not nc:
            return []
        return sorted(
            k for k in nc
            if '_block' in k and isinstance(nc[k], dict) and 'cluster_ids' in nc[k]
        )

    def refresh_cluster_blocks():
        names = _cluster_block_names()
        return gr.Dropdown(choices=names, value=names[0] if names else None)

    def view_clusters(block_name):
        nc = _state.get('native_captures')
        if not nc or block_name not in nc:
            return None
        cap = nc[block_name]
        if 'cluster_ids' not in cap:
            return None
        cids = cap['cluster_ids']
        inp_4d = cap.get('input', nc.get('input_image'))
        if inp_4d is None:
            return None
        H, W = inp_4d.shape[2], inp_4d.shape[3]
        if inp_4d.shape[1] == 3:
            inp_rgb = np.clip(inp_4d[0].permute(1, 2, 0).numpy(), 0, 1)
        else:
            grey = np.clip(inp_4d[0].mean(dim=0).numpy(), 0, 1)
            inp_rgb = np.stack([grey] * 3, axis=-1)
        return _cluster_overlay(inp_rgb, cids, H, W, num_clusters=64)

    def view_heads(block_name):
        nc = _state.get('native_captures')
        if not nc or block_name not in nc:
            return None
        cap = nc[block_name]
        if 'per_head_out' not in cap:
            return None
        pho = cap['per_head_out']
        inp_4d = cap.get('input')
        if inp_4d is None:
            return None
        H, W = inp_4d.shape[2], inp_4d.shape[3]
        nh = pho.shape[2]
        result = _per_head_contribution(pho, H, W, nh)
        if 'temperature' in cap:
            temp = cap['temperature']
            temp_str = ', '.join(f'H{i}={temp[i,0,0].item():.3f}' for i in range(temp.shape[0]))
            fig, ax = plt.subplots(1, 1,
                                   figsize=(result.shape[1] / 100,
                                            result.shape[0] / 100 + 0.4),
                                   dpi=100)
            ax.imshow(result); ax.axis('off')
            ax.set_title(f'Temperatures: {temp_str}', fontsize=9, color='navy')
            fig.tight_layout()
            return _fig_to_array(fig)
        return result

    def view_attn(block_name, win_idx, head_idx):
        nc = _state.get('native_captures')
        if not nc or block_name not in nc:
            return None
        cap = nc[block_name]
        if 'attn_weights' not in cap:
            return None
        aw = cap['attn_weights']
        wi = max(0, min(int(win_idx), aw.shape[1] - 1))
        hi = max(0, min(int(head_idx), aw.shape[2] - 1))
        return _attention_heatmap_fig(aw, wi, hi)

    # ── Build Gradio UI ──────────────────────────────────────
    with gr.Blocks(title=f"{model_type} Visualizer") as app:
        gr.Markdown(f"""
# {model_type} — Interactive Model Visualizer
Upload an image to trace it through the model.  Explore per-channel
characteristics, multi-scale comparisons, block-level I/O, feature maps,
and activation statistics — fully model-agnostic.
{('*Model-specific tabs: Cluster Maps, Attention Heads, Attention Heatmaps*' if _has_native_capture else '')}
        """)

        with gr.Row():
            img_input = gr.Image(label="Upload Input Image", type="numpy")
            run_btn = gr.Button("Run Inference & Capture", variant="primary", scale=0)

        io_output = gr.Image(label="Input / Output / Residual", interactive=False)
        stats_output = gr.Textbox(
            label="Activation Statistics (all layers)", lines=20,
            interactive=False, buttons=["copy"],
        )

        run_btn.click(process_image, inputs=[img_input], outputs=[io_output, stats_output])

        # ── Tab: Image Channels ──
        with gr.Tab("Image Channels"):
            gr.Markdown(
                "Per-channel (R/G/B) decomposition with histograms for both "
                "input and output, plus a side-by-side diff per channel.")
            with gr.Row():
                inp_ch_btn = gr.Button("Input Channels")
                out_ch_btn = gr.Button("Output Channels")
                cmp_ch_btn = gr.Button("Channel Comparison (I/O)")
            inp_ch_out = gr.Image(label="Input Channel Decomposition",
                                  interactive=False)
            out_ch_out = gr.Image(label="Output Channel Decomposition",
                                  interactive=False)
            cmp_ch_out = gr.Image(label="Per-Channel I/O Comparison",
                                  interactive=False)
            inp_ch_btn.click(view_input_channels, outputs=[inp_ch_out])
            out_ch_btn.click(view_output_channels, outputs=[out_ch_out])
            cmp_ch_btn.click(view_channel_comparison, outputs=[cmp_ch_out])

        # ── Tab: Multi-Scale View ──
        with gr.Tab("Multi-Scale View"):
            gr.Markdown(
                "Input vs output at full, 1/2, and 1/4 resolution — "
                "reveals how restoration behaves at different scales.")
            ms_btn = gr.Button("Generate Multi-Scale Comparison")
            ms_out = gr.Image(label="Multi-Scale View", interactive=False)
            ms_btn.click(view_multiscale, outputs=[ms_out])

        # ── Tab: Feature Flow ──
        with gr.Tab("Feature Flow"):
            gr.Markdown("Mean absolute activation at sampled layers throughout the network.")
            flow_btn = gr.Button("Generate Flow Map")
            flow_img = gr.Image(label="Feature Flow", interactive=False)
            flow_btn.click(view_feature_flow, outputs=[flow_img])

        # ── Tab: Block I/O ──
        with gr.Tab("Block I/O"):
            gr.Markdown(
                "Input vs output for every top-level block (direct children "
                "of the model). Shows what data entered each block and what "
                "came out, with a change heatmap.")
            blk_summary_btn = gr.Button("All Blocks Summary")
            blk_summary_out = gr.Image(label="All Blocks Overview",
                                        interactive=False)
            blk_summary_btn.click(view_blocks_summary, outputs=[blk_summary_out])
            gr.Markdown("---")
            with gr.Row():
                blk_dd = gr.Dropdown(choices=[], label="Block (Refresh after inference)")
                blk_ref_btn = gr.Button("Refresh Block List")
            blk_detail_btn = gr.Button("Show Block Detail")
            blk_detail_out = gr.Image(label="Block Input / Output / Change",
                                       interactive=False)
            blk_ref_btn.click(refresh_block_list, outputs=[blk_dd])
            blk_detail_btn.click(view_block_detail, inputs=[blk_dd],
                                 outputs=[blk_detail_out])

        # ── Tab: Channel Explorer ──
        with gr.Tab("Channel Explorer"):
            gr.Markdown("Browse individual feature channels at any layer.")
            with gr.Row():
                layer_dd = gr.Dropdown(choices=[], label="Layer (Refresh after inference)")
                layer_refresh_btn = gr.Button("Refresh Layer List")
            ch_slider = gr.Slider(0, 511, step=1, value=0, label="Channel Index")
            with gr.Row():
                overview_btn = gr.Button("Show Overview (first 16 ch)")
                single_btn = gr.Button("Show Single Channel")
            overview_out = gr.Image(label="Channel Overview", interactive=False)
            single_out = gr.Image(label="Single Channel", interactive=False)
            layer_refresh_btn.click(refresh_layer_list, outputs=[layer_dd])
            overview_btn.click(view_overview, inputs=[layer_dd], outputs=[overview_out])
            single_btn.click(view_channel, inputs=[layer_dd, ch_slider], outputs=[single_out])

        # ── Tab: Context Loss Analysis ──
        with gr.Tab("Context Loss Analysis"):
            gr.Markdown("L1 difference between consecutive layers — bright = high change.")
            ctx_btn = gr.Button("Analyse Context Changes")
            ctx_out = gr.Image(label="Context Change Heatmaps", interactive=False)
            ctx_btn.click(view_ctx, outputs=[ctx_out])

        # ── Model-specific tabs (auto-shown when model supports capture_all) ──
        if _has_native_capture:
            with gr.Tab("Cluster Maps"):
                gr.Markdown("Semantic cluster assignments — which pixels the router groups together.")
                cl_dd = gr.Dropdown(choices=[], label="Block (Refresh after inference)")
                cl_ref = gr.Button("Refresh Block List")
                cl_btn = gr.Button("Show Cluster Map")
                cl_out = gr.Image(label="Cluster Map", interactive=False)
                cl_ref.click(refresh_cluster_blocks, outputs=[cl_dd])
                cl_btn.click(view_clusters, inputs=[cl_dd], outputs=[cl_out])

            with gr.Tab("Attention Heads"):
                gr.Markdown("Per-head contribution (L2 norm) + learned temperature values.")
                hd_dd = gr.Dropdown(choices=[], label="Block")
                hd_ref = gr.Button("Refresh Block List")
                hd_btn = gr.Button("Show Head Contributions")
                hd_out = gr.Image(label="Head Contributions", interactive=False)
                hd_ref.click(refresh_cluster_blocks, outputs=[hd_dd])
                hd_btn.click(view_heads, inputs=[hd_dd], outputs=[hd_out])

            with gr.Tab("Attention Heatmaps"):
                gr.Markdown("Raw attention weight matrices within sorted-cluster windows.")
                with gr.Row():
                    at_dd = gr.Dropdown(choices=[], label="Block")
                    at_ref = gr.Button("Refresh")
                with gr.Row():
                    win_sl = gr.Slider(0, 100, step=1, value=0, label="Window Index")
                    hd_sl = gr.Slider(0, 7, step=1, value=0, label="Head Index")
                at_btn = gr.Button("Show Attention Heatmap")
                at_out = gr.Image(label="Attention Matrix", interactive=False)
                at_ref.click(refresh_cluster_blocks, outputs=[at_dd])
                at_btn.click(view_attn, inputs=[at_dd, win_sl, hd_sl], outputs=[at_out])

        if default_image_path and os.path.isfile(default_image_path):
            gr.Markdown(f"*Default test image: `{os.path.basename(default_image_path)}`*")

    return app


# ═════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════

def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description="Model-Agnostic Interactive Visualizer")
    parser.add_argument('--opt', '-o', type=str, required=True,
                        help='Path to Options YAML (e.g. Options/Restormer.yml)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='Default input image path')
    parser.add_argument('--port', '-p', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    opt = load_opt(args.opt)
    model_type = get_model_type(opt)
    model = load_model_from_opt(opt, args.checkpoint)

    param_count = sum(p.numel() for p in model.parameters())

    # Auto-discover default image
    default_img = args.image
    if default_img is None:
        for d in ['Rain100H', 'Rain100L', 'Test100']:
            p = os.path.join(ROOT, 'Datasets', 'test', d, 'input')
            if os.path.isdir(p):
                imgs = sorted(f for f in os.listdir(p)
                              if f.lower().endswith(('.png', '.jpg')))
                if imgs:
                    default_img = os.path.join(p, imgs[0])
                    break

    print("=" * 60)
    print(f"  {model_type} — Interactive Visualizer")
    print("=" * 60)
    print(f"  Options    : {args.opt}")
    print(f"  Model      : {model_type}")
    print(f"  Parameters : {param_count:,}")
    print(f"  Checkpoint : {args.checkpoint or '(auto-discovered)'}")
    print(f"  Default img: {default_img or '(none)'}")
    print(f"  Port       : {args.port}")
    print("=" * 60)

    app = build_dashboard(model, model_type, default_img)
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == '__main__':
    main()
