#!/usr/bin/env python3
"""
CDG-DFD Phase 2 trainer: E1 (held-out generator test), E2 ladder and E3 seeds.

Trains on a Phase 1 fold (train.csv), selects the checkpoint on val.csv only, and evaluates
test.csv exactly once at the end. Every test prediction is saved, so all metrics, confusion
matrices and Figure 4(b) can be regenerated without re-running the model.

Architecture is the paper's (EfficientNet-B4 + SE-style SAM + DAT with GRL), unchanged from
df-1-first-edition except where a flag says otherwise:
  --init imagenet          (default) ImageNet weights, no CelebDF warm start
  --input dct|rgb          DCT (signed log-magnitude, as in the first edition) or raw RGB
  --spectral band|none     revised SAM (attention over DCT frequency bands) on or off
  --channel-attn se        add the first edition's channel attention (squeeze-and-excitation)
  --no-dat                 drop domain-adversarial training
  --lambda-dom, --warmup   DAT weight and GRL warm-up (E8)

Example (Kaggle):
  python cdg_train.py --data /kaggle/input/<phase1-output>/phase1 --fold A --seed 1 --out runs/A_s1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class SplitDataset(Dataset):
    """Holds only numpy arrays (no pandas/Python lists), so forked DataLoader workers do not slowly
    copy the parent's memory. Metadata for reporting is kept separately in .meta (main process only)."""

    def __init__(self, csv_path, root, transform):
        meta = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
        self.set_rows(meta, root)
        self.transform = transform

    def set_rows(self, meta, root):
        self.meta = meta.reset_index(drop=True)
        self.paths = np.array([os.path.join(root, p) for p in self.meta.crop_path])   # fixed-width str array
        self.labels = self.meta.label.astype(int).to_numpy()
        self.domains = self.meta.domain.astype(int).to_numpy()

    def __len__(self):
        return len(self.paths)

    degrade_mode = 'resize_jpeg'   # or 'wang': CNNDetection (Wang et al., CVPR 2020) blur + JPEG
    degrade = 0.0   # set on the training set only: probability of a random resize + JPEG round trip
    corrupt = ''    # set on the test set only (E5): a fixed corruption, e.g. 'jpeg:50', 'resize:0.5', 'blur:1', 'noise:5'

    def __getitem__(self, i):
        img = Image.open(str(self.paths[i])).convert('RGB')
        if self.degrade and random.random() < self.degrade:
            img = degrade_image(img, self.degrade_mode)
        if self.corrupt:
            img = corrupt_image(img, self.corrupt, seed=i)
        return self.transform(img), int(self.labels[i]), int(self.domains[i]), i

    def __getstate__(self):           # workers never need the metadata frame
        st = self.__dict__.copy()
        st['meta'] = None
        return st


def degrade_image(img, mode='resize_jpeg'):
    """mode 'wang' = CNNDetection augmentation (Wang et al., CVPR 2020): Gaussian blur sigma ~ U[0, 3] with p=0.5,
    then JPEG quality ~ U[30, 100] with p=0.5. Default mode: Real-world storage damage, applied to real and fake training images alike: downscale by a random factor
    in [0.5, 1] and back (bilinear), then JPEG at a random quality in [40, 95]. Removes the shortcut of a fixed
    file history and teaches robustness to recompression."""
    import io
    if mode == 'wang':
        from PIL import ImageFilter
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 3)))
        if random.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=random.randint(30, 100))
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img
    w, h = img.size
    s = random.uniform(0.5, 1.0)   # python random: PyTorch seeds it per loader worker
    if s < 0.98:
        img = img.resize((max(16, int(w * s)), max(16, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(40, 95))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def corrupt_image(img, spec, seed=0):
    """Deterministic test-time corruption for the robustness test (E5), applied to real and fake images alike.
    Several can be chained with '+', e.g. 'resize:0.5+jpeg:70'."""
    import io
    from PIL import ImageFilter
    for part in spec.split('+'):
        kind, val = part.split(':')
        v = float(val)
        if kind == 'jpeg':
            buf = io.BytesIO(); img.save(buf, format='JPEG', quality=int(v)); buf.seek(0)
            img = Image.open(buf).convert('RGB')
        elif kind == 'resize':          # downscale by v and back up (bilinear)
            w, h = img.size
            img = img.resize((max(16, int(w * v)), max(16, int(h * v))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        elif kind == 'blur':            # Gaussian blur, radius v pixels
            img = img.filter(ImageFilter.GaussianBlur(v))
        elif kind == 'noise':           # additive Gaussian noise, sigma v on the 0-255 scale
            a = np.asarray(img).astype(np.float32)
            a = a + np.random.default_rng(seed).normal(0, v, a.shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        else:
            raise ValueError(f'unknown corruption {kind}')
    return img


def make_transforms():
    """CPU side only decodes and converts; the random augmentation runs on the GPU (gpu_augment)."""
    from torchvision import transforms
    base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return base, base


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def gpu_augment(x, gen=None):
    """First-edition augmentation, batched on the GPU: horizontal flip (p=0.5), rotation in [-10, 10]
    degrees (bilinear, black fill), then brightness/contrast in [0.7, 1.3] and saturation in [0.8, 1.2].
    x: (B, 3, H, W) in [0, 1]."""
    B, dev = x.shape[0], x.device
    u = lambda *s: torch.rand(*s, device=dev, generator=gen)
    flip = u(B) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    ang = (u(B) * 2 - 1) * (10 * np.pi / 180)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    b = 0.7 + 0.6 * u(B, 1, 1, 1)
    x = (x * b).clamp(0, 1)
    c = 0.7 + 0.6 * u(B, 1, 1, 1)
    m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * c + m).clamp(0, 1)
    s_ = 0.8 + 0.4 * u(B, 1, 1, 1)
    g = _gray(x)
    return ((x - g) * s_ + g).clamp(0, 1)


# ----------------------------------------------------------------------------
# Model (as in df-1-first-edition)
# ----------------------------------------------------------------------------
class BatchDCT(nn.Module):
    """Orthonormal 2-D DCT-II per channel, then sign(D)*log(1+|D|). Always computed in float32."""

    def __init__(self, size=224):
        super().__init__()
        n = torch.arange(size, dtype=torch.float64)
        M = torch.cos(np.pi / size * (n.unsqueeze(0) + 0.5) * n.unsqueeze(1))
        alpha = torch.ones(size, dtype=torch.float64)
        alpha[0] = 1 / np.sqrt(2)
        self.register_buffer('M', (M * alpha.unsqueeze(1) * np.sqrt(2.0 / size)).float())

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            D = self.M @ x.float() @ self.M.T
            return torch.sign(D) * torch.log1p(D.abs())


class SpectralAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // 4, 1)
        self.conv2 = nn.Conv2d(c // 4, c, 1)
        self.bn = nn.BatchNorm2d(c // 4)

    def forward(self, x):
        a = torch.sigmoid(self.conv2(F.relu(self.bn(self.conv1(F.adaptive_avg_pool2d(x, 1))))))
        return x * a


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None


class DomainClassifier(nn.Module):
    def __init__(self, d=1792):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, x, alpha):
        return self.classifier(GRL.apply(x, alpha))


def radial_bands(size=224, bands=16):
    """Band index of every DCT coefficient: normalised radial frequency sqrt(u^2+v^2), split into equal-width
    rings from 0 (DC, lowest frequency) to bands-1 (highest frequency)."""
    k = torch.arange(size, dtype=torch.float64)
    r = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2) / (np.sqrt(2) * (size - 1))
    return torch.clamp((r * bands).long(), max=bands - 1)


class SpectralBandAttention(nn.Module):
    """Spectral Attention Module (revised): attention over DCT frequency bands, applied to the DCT map
    before the backbone.

    For each image and colour channel it measures the mean magnitude in each of K radial frequency bands,
    passes these C*K numbers through a small MLP, and rescales every coefficient of a band by
    w = 2*sigmoid(z), w in (0, 2). The last layer starts at zero, so w = 1 everywhere at initialisation
    (the module begins as the identity). The weights are returned, so the bands the model relies on can be
    reported per image, per class and per generator."""

    def __init__(self, size=224, bands=16, channels=3, hidden=32):
        super().__init__()
        idx = radial_bands(size, bands)
        onehot = F.one_hot(idx, bands).permute(2, 0, 1).float()          # K, H, W
        self.register_buffer('band_idx', idx, persistent=False)
        self.register_buffer('onehot', onehot, persistent=False)
        self.register_buffer('area', onehot.sum((1, 2)), persistent=False)
        self.bands, self.channels = bands, channels
        self.fc1 = nn.Linear(channels * bands, hidden)
        self.fc2 = nn.Linear(hidden, channels * bands)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        B, C = x.shape[:2]
        with torch.autocast(device_type=x.device.type, enabled=False):   # small and cheap: keep it in fp32
            xf = x.float()
            energy = torch.einsum('bchw,khw->bck', xf.abs(), self.onehot) / self.area     # B, C, K
            z = self.fc2(F.relu(self.fc1(energy.flatten(1))))
            w = 2 * torch.sigmoid(z).view(B, C, self.bands)
            return xf * w[:, :, self.band_idx], w


class Detector(nn.Module):
    """spectral='band': revised SAM on the DCT map (default). channel_attn='se': the first edition's
    squeeze-and-excitation block on backbone channels (kept for ablation and for loading old checkpoints)."""

    def __init__(self, pretrained=True, spectral='band', channel_attn='off', input_mode='dct', bands=16,
                 backbone='efficientnet_b4'):
        super().__init__()
        import timm
        self.input_mode = input_mode
        self.dct = BatchDCT(224)
        self.band_attention = SpectralBandAttention(224, bands) if spectral == 'band' else None
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        nf = self.backbone.num_features   # 1792 for EfficientNet-B4, 2048 for ResNet-50 (E10 baseline)
        # attribute name kept as in the first edition, so old checkpoints still load
        self.spectral_attention = SpectralAttention(nf) if channel_attn == 'se' else nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        self.domain_classifier = DomainClassifier(nf)

    def forward(self, x, alpha=1.0):
        if self.input_mode == 'dct':
            x = self.dct(x)
        if self.band_attention is not None:
            x, w = self.band_attention(x)
            w = w.flatten(1)
        else:
            w = x.new_zeros(x.shape[0], 0)
        x = x.contiguous(memory_format=torch.channels_last)   # much faster depthwise convs on T4
        f = self.spectral_attention(self.backbone(x))
        f = self.dropout(F.adaptive_avg_pool2d(f, 1).flatten(1))
        return self.classifier(f), self.domain_classifier(f, alpha), w


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if ('bias' in n or 'bn' in n or p.ndim == 1) else decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan')


def eer(y, s):
    from sklearn.metrics import roc_curve
    if len(set(y)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def balanced_acc(y, s, thr):
    y, p = np.asarray(y), (np.asarray(s) >= thr).astype(int)
    tpr = (p[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (p[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([tpr, tnr])), float(tpr), float(tnr)


def bootstrap_auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if not len(pos) or not len(neg):
        return [float('nan')] * 2
    vals = []
    for _ in range(n):   # stratified resampling keeps both classes present
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluation_pairs(test_sets):
    """Which reals are the negatives for which fakes. Returns (name, real sets, fake sets)."""
    sets = set(test_sets)
    pairs = []
    by_ds = {}
    for t in sets:
        ds = t.split(':')[0]
        by_ds.setdefault(ds, []).append(t)
    for ds, ts in sorted(by_ds.items()):
        reals = [t for t in ts if t.split(':')[1] == 'real' and not t.endswith(':linked')]
        reals_all = [t for t in ts if t.split(':')[1] == 'real']
        fakes = sorted(t for t in ts if t.split(':')[1] not in ('real', 'real_matched'))
        for f in fakes:
            if f.endswith(':linked'):
                continue
            pairs.append((f'{f} vs {ds}:real', reals, [f]))
            if ds == 'CelebDF' and any(t.endswith(':linked') for t in ts):
                pairs.append((f'{f} (+linked) vs {ds}:real (+linked)', reals_all,
                              [f] + ([f + ':linked'] if (f + ':linked') in sets else [])))
            if ds == 'DFF' and 'DFF:real_matched' in sets:
                pairs.append((f'{f} vs DFF:real_matched', ['DFF:real_matched'], [f]))
    for t in sorted(sets):
        if t.startswith('ext_') and t.split(':')[1] != 'real' and 'DFF:real' in sets:
            pairs.append((f'{t} vs DFF:real', ['DFF:real'], [t]))
    return pairs


def compute_metrics(pred, thr):
    out = []
    for name, reals, fakes in evaluation_pairs(pred.test_set):
        sub = pred[pred.test_set.isin(reals + fakes)]
        y, s = sub.label.to_numpy(), sub.prob.to_numpy()
        ba, tpr, tnr = balanced_acc(y, s, thr)
        lo, hi = bootstrap_auc_ci(y, s)
        out.append(dict(comparison=name, n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
                        AUC=round(roc_auc(y, s), 4), AUC_CI95=f'[{lo:.4f}, {hi:.4f}]', EER=round(eer(y, s), 4),
                        balanced_acc=round(ba, 4), TPR=round(tpr, 4), TNR=round(tnr, 4)))
    # Shortcut check: share of each real set flagged as fake at the chosen threshold.
    for t in ('DFF:real', 'DFF:real_matched'):
        sub = pred[pred.test_set == t]
        if len(sub):
            out.append(dict(comparison=f'{t} flagged as fake', n_real=len(sub), n_fake=0,
                            FPR=round(float((sub.prob >= thr).mean()), 4)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except Exception:  # noqa: BLE001
        return float('nan')


def resources():
    try:
        import psutil
        vm = psutil.virtual_memory()
        out = f'RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB'
    except Exception:  # noqa: BLE001
        out = 'RAM n/a'
    if torch.cuda.is_available():
        out += ' | GPU ' + ', '.join(f'{torch.cuda.max_memory_allocated(i) / 1e9:.1f}'
                                     for i in range(torch.cuda.device_count())) + ' GB peak'
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp, return_weights=False):
    predict.nonfinite, predict.refp32 = 0, 0
    model.eval()
    probs, idxs, ws = [], [], []
    for x, _, _, i in loader:
        xd = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits, _, w = model(xd, alpha=0.0)
        p = F.softmax(logits.float(), 1)[:, 1]
        if amp and not torch.isfinite(p).all():   # rare fp16 overflow: score this batch again in fp32
            predict.refp32 += 1
            logits, _, w = model(xd, alpha=0.0)
            p = F.softmax(logits.float(), 1)[:, 1]
        bad = ~torch.isfinite(p)
        if bad.any():   # still not finite in fp32: report, never hide
            predict.nonfinite += int(bad.sum())
            p = torch.where(bad, torch.full_like(p, 0.5), p)
        probs.append(p.cpu())
        ws.append(w.float().cpu())
        idxs.append(i)
    order = torch.cat(idxs).argsort()
    p = torch.cat(probs)[order].numpy()
    if return_weights:
        return p, torch.cat(ws)[order].numpy()
    return p


def val_threshold(y, s):
    """Threshold maximising balanced accuracy on validation (applied unchanged to test)."""
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))):
        b = balanced_acc(y, s, t)[0]
        if b > best:
            best, thr = b, float(t)
    return thr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='Phase 1 output folder (contains splits/ and crops/)')
    ap.add_argument('--fold', default='A')
    ap.add_argument('--degrade', type=float, default=0.0,
                    help='probability of a random resize + JPEG round trip per training image (E3)')
    ap.add_argument('--splits', default='splits',
                    help="split folder inside --data: 'splits' (original DFF reals) or 'splits_matched'")
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=20, help='0 = evaluate --init checkpoint only')
    ap.add_argument('--backbone', default='efficientnet_b4', help='timm model name (E10: resnet50 for CNNDetection)')
    ap.add_argument('--degrade-mode', choices=['resize_jpeg', 'wang'], default='resize_jpeg')
    ap.add_argument('--test-corrupt', default='',
                    help="E5: corrupt the TEST images only (val stays clean, so the threshold is the clean one), "
                         "e.g. 'jpeg:50', 'resize:0.5', 'blur:1', 'noise:5', chained with '+'")
    ap.add_argument('--epoch-size', type=int, default=24000, help='samples drawn per epoch (first edition: 24000)')
    ap.add_argument('--batch', type=int, default=64, help='images per GPU step')
    ap.add_argument('--accum', type=int, default=2,
                    help='steps accumulated per update: 64 x 2 = 128, the paper\'s batch size, on one 16 GB T4')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=2e-4)
    ap.add_argument('--lambda-dom', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=10, help='GRL alpha = min(1, epoch/warmup); 0 = constant 1')
    ap.add_argument('--input', choices=['dct', 'rgb'], default='dct')
    ap.add_argument('--spectral', choices=['band', 'none'], default='band',
                    help='band: revised SAM, attention over DCT frequency bands (default)')
    ap.add_argument('--bands', type=int, default=16, help='number of radial DCT bands for --spectral band')
    ap.add_argument('--channel-attn', choices=['off', 'se'], default='off',
                    help="se: first edition's squeeze-and-excitation on backbone channels (ablation / old checkpoints)")
    ap.add_argument('--no-dat', action='store_true', help='drop the domain loss (lambda=0)')
    ap.add_argument('--init', default='imagenet', help="'imagenet', 'none', or a checkpoint path")
    ap.add_argument('--workers', type=int, default=2, help='loader processes (2 runs share 4 CPUs on Kaggle)')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--gpus', choices=['all', '1'], default='1',
                    help="1 (default): one GPU per run; 'all' = DataParallel (not recommended on Kaggle T4s)")
    ap.add_argument('--pin-memory', action='store_true', help='pinned host memory for the loader (off by default)')
    ap.add_argument('--clip', type=float, default=5.0, help='gradient-norm clipping (0 = off)')
    ap.add_argument('--benchmark', type=int, default=0,
                    help='time this many training batches, write benchmark.json and exit')
    ap.add_argument('--limit-eval', type=int, default=0, help='testing only: cap val/test rows')
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not a.no_amp
    split_dir = os.path.join(a.data, a.splits, f'fold_{a.fold}')
    tr_tf, te_tf = make_transforms()
    tr = SplitDataset(os.path.join(split_dir, 'train.csv'), a.data, tr_tf)
    tr.degrade = a.degrade
    tr.degrade_mode = a.degrade_mode
    va = SplitDataset(os.path.join(split_dir, 'val.csv'), a.data, te_tf)
    te = SplitDataset(os.path.join(split_dir, 'test.csv'), a.data, te_tf)
    te.corrupt = a.test_corrupt
    if a.limit_eval:
        for d in (va, te):
            # a few rows of every (label, test set) group, so the smoke test exercises every metric
            keep = d.meta.groupby(['label', 'test_set'], group_keys=False).apply(
                lambda g: g.head(max(4, a.limit_eval // 20))).index
            d.set_rows(d.meta.loc[keep], a.data)
    print(f'fold {a.fold}: train={len(tr)} val={len(va)} test={len(te)} | device={device} amp={amp}', flush=True)

    # Same balancing as the first edition: inverse frequency over (label, domain) groups.
    keys = list(zip(tr.labels, tr.domains))
    cnt = pd.Series(keys).value_counts().to_dict()
    weights = torch.tensor([1.0 / cnt[k] for k in keys], dtype=torch.double)
    g = torch.Generator().manual_seed(a.seed)
    sampler = WeightedRandomSampler(weights, num_samples=a.epoch_size, replacement=True, generator=g)
    pin = device.type == 'cuda' and a.pin_memory
    tl = DataLoader(tr, batch_size=a.batch, sampler=sampler, drop_last=True, num_workers=a.workers,
                    pin_memory=pin, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    eval_kw = dict(batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=pin, persistent_workers=False)
    vl = DataLoader(va, **eval_kw)

    pretrained = a.init == 'imagenet'
    model = Detector(pretrained=pretrained, spectral=a.spectral, channel_attn=a.channel_attn,
                     input_mode=a.input, bands=a.bands, backbone=a.backbone)
    n_band = sum(p.numel() for p in model.band_attention.parameters()) if model.band_attention is not None else 0
    print(f'parameters: total {sum(p.numel() for p in model.parameters()):,} | band attention {n_band:,}', flush=True)
    if a.init not in ('imagenet', 'none'):
        sd = torch.load(a.init, map_location='cpu', weights_only=True)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f'init from {a.init}: missing={len(miss)} unexpected={len(unexp)}')
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    net = nn.DataParallel(model) if (torch.cuda.device_count() > 1 and a.gpus == 'all') else model
    print(f'GPUs used: {torch.cuda.device_count() if net is not model else 1}', flush=True)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr)
    steps_per_epoch = len(tl) // a.accum
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.max_lr, total_steps=a.epochs * steps_per_epoch)
             if a.epochs > 0 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    lam = 0.0 if a.no_dat else a.lambda_dom
    cfg = dict(vars(a), effective_batch=a.batch * a.accum, n_train=len(tr), n_val=len(va), n_test=len(te), group_counts={str(k): v for k, v in cnt.items()},
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=2)

    history, best_auc, t0, start_epoch = [], -1.0, time.time(), 0
    last_path = os.path.join(a.out, 'last.pth')
    aug_gen = torch.Generator(device=device).manual_seed(a.seed + 1000)
    if a.epochs > 0 and os.path.exists(last_path):   # resume after a crash or a killed process
        ck = torch.load(last_path, map_location='cpu', weights_only=False)   # RNG states must stay on the CPU
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])        # moves optimizer state to the parameters' device
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        g.set_state(ck['sampler_gen'].cpu())
        aug_gen.set_state(ck['aug_gen'].cpu())
        history, best_auc, start_epoch = ck['history'], ck['best_auc'], ck['epoch']
        del ck
        print(f'resumed from {last_path} after epoch {start_epoch} (best val AUC {best_auc:.4f})', flush=True)
    if a.epochs == 0:   # evaluation only: score an existing checkpoint (--init) on this fold's val and test
        torch.save(model.state_dict(), os.path.join(a.out, 'best.pth'))
        pv = predict(net, vl, device, amp)
        np.save(os.path.join(a.out, 'val_probs_best.npy'), pv)
        best_auc = roc_auc(va.labels, pv)
        history.append(dict(epoch=0, val_auc=best_auc))
    for epoch in range(start_epoch, a.epochs):
        net.train()
        alpha = 1.0 if a.warmup == 0 else min(1.0, epoch / a.warmup)
        tot, correct, n, dom_correct = 0.0, 0, 0, 0
        te0, wait, nb, bad_steps, group_bad = time.time(), 0.0, 0, 0, False
        opt.zero_grad(set_to_none=True)
        tw = time.time()
        for x, y, d, _ in tl:
            wait += time.time() - tw
            nb += 1
            x, y, d = x.to(device, non_blocking=True), y.to(device), d.to(device)
            x = gpu_augment(x, aug_gen)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                cls, dom, _ = net(x, alpha=alpha)
                loss = F.cross_entropy(cls, y)
                if lam > 0:
                    loss = loss + lam * F.cross_entropy(dom, d)
            if torch.isfinite(loss):
                scaler.scale(loss / a.accum).backward()
                tot += loss.item() * len(y)
                n += len(y)
                correct += (cls.argmax(1) == y).sum().item()
                dom_correct += (dom.argmax(1) == d).sum().item()
            else:
                bad_steps += 1
                group_bad = True
                if bad_steps > 50:
                    raise RuntimeError('more than 50 training steps with a non-finite loss: training diverged')
            if nb % a.accum == 0:   # one optimizer update per a.accum batches (effective batch a.batch * a.accum)
                if group_bad:
                    opt.zero_grad(set_to_none=True)
                else:
                    if a.clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                group_bad = False
                sched.step()
            if nb == min(10, max(a.benchmark // 3, 1)) or (nb == 10 and not a.benchmark):
                ram0, t_first, nb0 = ram_gb(), time.time(), nb   # after warm-up (optimizer state, cudnn tuning)
            if nb % 20 == 0 and (nb <= 100 or nb % 100 == 0):
                el = time.time() - te0
                print(f'   batch {nb}/{len(tl)}: {nb * a.batch / el:.0f} img/s, waiting for data {100 * wait / el:.0f}% '
                      f'of the time | {resources()}', flush=True)
            if a.benchmark and nb == a.benchmark:
                el = time.time() - t_first
                res = dict(gpus=a.gpus, pin_memory=a.pin_memory, batches=nb, img_per_s=round((nb - nb0) * a.batch / el, 1),
                           ram_growth_mb_per_batch=round(1000 * (ram_gb() - ram0) / max(nb - nb0, 1), 1),
                           data_wait_pct=round(100 * wait / (time.time() - te0), 1), skipped_nonfinite=bad_steps)
                json.dump(res, open(os.path.join(a.out, 'benchmark.json'), 'w'))
                print('BENCHMARK', json.dumps(res), flush=True)
                return
            tw = time.time()
        opt.zero_grad(set_to_none=True)   # drop a leftover partial accumulation at the end of the epoch
        train_min = (time.time() - te0) / 60
        if n == 0:
            raise RuntimeError('every training step this epoch had a non-finite loss')
        if bad_steps:
            print(f'   WARNING: skipped {bad_steps} steps with a non-finite loss this epoch', flush=True)
        pv = predict(net, vl, device, amp)
        if predict.nonfinite:
            print(f'   WARNING: {predict.nonfinite} validation scores were not finite (set to 0.5)', flush=True)
        if predict.refp32:
            print(f'   note: {predict.refp32} validation batches re-scored in fp32 after an fp16 overflow', flush=True)
        yv, dv = va.labels, va.domains
        row = dict(epoch=epoch + 1, alpha=round(alpha, 2), loss=tot / n, train_acc=correct / n,
                   train_domain_acc=dom_correct / n, val_auc=roc_auc(yv, pv),
                   val_bal_acc=balanced_acc(yv, pv, 0.5)[0], minutes=(time.time() - t0) / 60,
                   train_minutes=train_min, data_wait_pct=100 * wait / max(time.time() - te0, 1e-9),
                   skipped_steps=bad_steps, val_nonfinite=predict.nonfinite)
        for dom_id, name in ((0, 'gan'), (1, 'diff')):
            m = dv == dom_id
            row[f'val_auc_{name}'] = roc_auc(yv[m], pv[m]) if m.any() else float('nan')
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(a.out, 'history.csv'), index=False)
        flag = ''
        if row['val_auc'] > best_auc:   # selection on validation AUC only; test is never touched here
            best_auc = row['val_auc']
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pth'))
            np.save(os.path.join(a.out, 'val_probs_best.npy'), pv)
            flag = '  * best'
        print(f"ep {row['epoch']:02d} a={row['alpha']:.2f} loss={row['loss']:.4f} train={row['train_acc']:.3f} "
              f"dom={row['train_domain_acc']:.3f} | val AUC={row['val_auc']:.4f} (GAN {row['val_auc_gan']:.4f}, "
              f"diff {row['val_auc_diff']:.4f}) balacc={row['val_bal_acc']:.3f} | {row['minutes']:.1f} min{flag}",
              flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), sampler_gen=g.get_state().cpu(),
                        aug_gen=aug_gen.get_state().cpu(),
                        history=history, best_auc=best_auc, epoch=epoch + 1), last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)   # atomic: a kill mid-save never corrupts it

    # ---- single test evaluation with the selected checkpoint ----
    model.load_state_dict(torch.load(os.path.join(a.out, 'best.pth'), map_location=device, weights_only=True))
    pv = np.load(os.path.join(a.out, 'val_probs_best.npy'))
    thr = val_threshold(va.labels, pv)
    tel = DataLoader(te, **eval_kw)
    pt, wt = predict(net, tel, device, amp, return_weights=True)
    test_nonfinite, test_refp32 = predict.nonfinite, predict.refp32
    pred = te.meta[['uid', 'test_set', 'label', 'source', 'generator']].copy()
    pred['prob'] = pt
    if wt.shape[1]:   # band weights: mean per test set, per colour channel and band (for the SAM figure)
        np.save(os.path.join(a.out, 'test_band_weights.npy'), wt.astype(np.float16))
        K = model.band_attention.bands
        cols = [f'{c}_b{k:02d}' for c in 'RGB' for k in range(K)]
        bw = pd.DataFrame(wt, columns=cols)
        bw['test_set'] = pred.test_set.values
        bw.groupby('test_set').mean().round(4).to_csv(os.path.join(a.out, 'band_weights_by_test_set.csv'))
    pred.to_csv(os.path.join(a.out, 'test_predictions.csv'), index=False)
    met = compute_metrics(pred, thr)
    met.to_csv(os.path.join(a.out, 'test_metrics.csv'), index=False)
    hist = pd.DataFrame(history)
    best_ep = int(hist.loc[hist.val_auc.idxmax(), 'epoch'])
    summary = dict(fold=a.fold, seed=a.seed, splits=a.splits, best_epoch=best_ep, best_val_auc=best_auc,
                   threshold=thr, minutes=(time.time() - t0) / 60, test_nonfinite=test_nonfinite,
                   test_batches_rescored_fp32=test_refp32)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    if os.path.exists(last_path):
        os.remove(last_path)   # finished: keep only best.pth
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(met.to_string(index=False))


if __name__ == '__main__':
    main()
