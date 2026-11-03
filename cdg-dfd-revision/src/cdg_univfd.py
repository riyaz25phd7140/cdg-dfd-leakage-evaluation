#!/usr/bin/env python3
"""
E10 baseline: UnivFD (Ojha, Li and Lee, "Towards Universal Fake Image Detectors that Generalize Across
Generative Models", CVPR 2023) under the leakage-controlled protocol.

A frozen CLIP ViT-L/14 image encoder (OpenAI weights) embeds every face crop; a linear classifier on the
L2-normalised embedding is trained on the fold's train.csv only. The regularisation strength and the decision
threshold are chosen on val.csv; test.csv is scored once. Training rows are weighted by inverse frequency over
(label, domain) groups, the same balancing as cdg_train.py. Metrics come from cdg_train.compute_metrics, so the
numbers are directly comparable with every other run.

Stages:
  features   embed every crop of the matched split (both folds) and, if given, the ArtiFact crops
  fit        fit + evaluate per fold -> runs/e10_univfd_<fold>_s1/{test_predictions,test_metrics}.csv, summary.json
  corrupt    embed the test crops after a fixed corruption and re-score with the fitted classifiers
  artifact   append ArtiFact scores (run e10_univfd_<fold>_s1) to <external>/scores.csv
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdg_train import compute_metrics, corrupt_image, roc_auc, val_threshold  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class Crops(Dataset):
    def __init__(self, paths, preprocess, corrupt=''):
        self.paths, self.pre, self.corrupt = np.array(paths), preprocess, corrupt

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(str(self.paths[i])).convert('RGB')
        if self.corrupt:
            img = corrupt_image(img, self.corrupt, seed=i)
        return self.pre(img), i


def load_clip():
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    return model.eval(), pre


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.encode_image(x)


@torch.no_grad()
def embed(paths, model, pre, corrupt='', batch=128, workers=4):
    dev = torch.device('cuda')
    net = Encoder(model).to(dev).half()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    dl = DataLoader(Crops(paths, pre, corrupt), batch_size=batch, num_workers=workers, shuffle=False)
    out = np.zeros((len(paths), 768), dtype=np.float16)
    t, n = time.time(), 0
    for x, i in dl:
        f = net(x.to(dev).half()).float()
        f = f / f.norm(dim=1, keepdim=True)
        out[i.numpy()] = f.cpu().numpy().astype(np.float16)
        n += len(i)
        if n % (batch * 50) < batch:
            log(f'   {n}/{len(paths)} images, {n / (time.time() - t):.0f} img/s')
    return out


def split_rows(data, splits):
    rows = {}
    for fold in 'AB':
        for sp in ('train', 'val', 'test'):
            d = pd.read_csv(os.path.join(data, splits, f'fold_{fold}', f'{sp}.csv'), keep_default_na=False, low_memory=False)
            rows[(fold, sp)] = d
    return rows


def stage_features(a):
    rows = split_rows(a.data, a.splits)
    paths = sorted({p for d in rows.values() for p in d.crop_path})
    log(f'{len(paths)} unique crops in the matched split (both folds)')
    model, pre = load_clip()
    feats = embed([os.path.join(a.data, p) for p in paths], model, pre)
    np.save(os.path.join(a.out, 'feats_split.npy'), feats)
    pd.Series(paths).to_csv(os.path.join(a.out, 'feats_split_paths.csv'), index=False, header=['crop_path'])
    log(f'split features done | {(time.time() - T0) / 60:.1f} min')
    if a.external:
        c = pd.read_csv(os.path.join(a.external, 'crops.csv'), keep_default_na=False)
        c = c[c.ok.astype(int) == 1].reset_index(drop=True)
        fx = embed(c.crop.tolist(), model, pre)
        np.save(os.path.join(a.out, 'feats_artifact.npy'), fx)
        c.to_csv(os.path.join(a.out, 'feats_artifact_rows.csv'), index=False)
        log(f'ArtiFact features done ({len(c)} crops) | {(time.time() - T0) / 60:.1f} min')


def lookup(feats, index, paths):
    return feats[index.loc[paths].to_numpy()].astype(np.float32)


def group_weights(d):
    keys = list(zip(d.label.astype(int), d.domain.astype(int)))
    cnt = pd.Series(keys).value_counts().to_dict()
    w = np.array([1.0 / cnt[k] for k in keys])
    return w * len(w) / w.sum()


def stage_fit(a):
    from sklearn.linear_model import LogisticRegression
    rows = split_rows(a.data, a.splits)
    feats = np.load(os.path.join(a.out, 'feats_split.npy'))
    idx = pd.read_csv(os.path.join(a.out, 'feats_split_paths.csv')).reset_index().set_index('crop_path')['index']
    for fold in 'AB':
        tr, va, te = rows[(fold, 'train')], rows[(fold, 'val')], rows[(fold, 'test')]
        Xtr, Xva, Xte = (lookup(feats, idx, d.crop_path) for d in (tr, va, te))
        ytr, yva = tr.label.astype(int).to_numpy(), va.label.astype(int).to_numpy()
        best = None
        for C in (0.01, 0.1, 1.0, 10.0, 100.0):
            clf = LogisticRegression(C=C, max_iter=3000)
            clf.fit(Xtr, ytr, sample_weight=group_weights(tr))
            auc = roc_auc(yva, clf.predict_proba(Xva)[:, 1])
            log(f'fold {fold}: C={C} val AUC {auc:.4f}')
            if best is None or auc > best[0]:
                best = (auc, C, clf)
        auc, C, clf = best
        pv = clf.predict_proba(Xva)[:, 1]
        thr = val_threshold(yva, pv)
        out = os.path.join(a.runs, f'e10_univfd_{fold}_s1')
        os.makedirs(out, exist_ok=True)
        pred = te[['uid', 'test_set', 'label', 'source', 'generator']].copy()
        pred['prob'] = clf.predict_proba(Xte)[:, 1]
        pred.to_csv(os.path.join(out, 'test_predictions.csv'), index=False)
        met = compute_metrics(pred, thr)
        met.to_csv(os.path.join(out, 'test_metrics.csv'), index=False)
        json.dump(dict(fold=fold, seed=1, method='UnivFD (CLIP ViT-L/14 + linear probe)', C=C, best_val_auc=auc,
                       threshold=thr), open(os.path.join(out, 'summary.json'), 'w'), indent=2)
        pickle.dump(dict(clf=clf, thr=thr), open(os.path.join(a.out, f'clf_{fold}.pkl'), 'wb'))
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(met.to_string(index=False), flush=True)


def stage_corrupt(a):
    rows = split_rows(a.data, a.splits)
    tests = {f: rows[(f, 'test')] for f in 'AB'}
    paths = sorted(set(tests['A'].crop_path) | set(tests['B'].crop_path))
    idx = pd.Series(range(len(paths)), index=paths)
    model, pre = load_clip()
    for cond in a.conditions:
        tag = cond.replace(':', '').replace('+', '_').replace('.', 'p')
        done = all(os.path.exists(os.path.join(a.runs, f'e10c_univfd_{tag}_{f}_s1', 'test_metrics.csv')) for f in 'AB')
        if done:
            continue
        feats = embed([os.path.join(a.data, p) for p in paths], model, pre, corrupt=cond)
        for fold in 'AB':
            m = pickle.load(open(os.path.join(a.out, f'clf_{fold}.pkl'), 'rb'))
            te = tests[fold]
            pred = te[['uid', 'test_set', 'label', 'source', 'generator']].copy()
            pred['prob'] = m['clf'].predict_proba(lookup(feats, idx, te.crop_path))[:, 1]
            out = os.path.join(a.runs, f'e10c_univfd_{tag}_{fold}_s1')
            os.makedirs(out, exist_ok=True)
            compute_metrics(pred, m['thr']).to_csv(os.path.join(out, 'test_metrics.csv'), index=False)
        log(f'UnivFD {cond}: done | {(time.time() - T0) / 60:.1f} min')


def stage_artifact(a):
    rows = pd.read_csv(os.path.join(a.out, 'feats_artifact_rows.csv'), keep_default_na=False)
    fx = np.load(os.path.join(a.out, 'feats_artifact.npy')).astype(np.float32)
    path = os.path.join(a.external, 'scores.csv')
    for fold in 'AB':
        m = pickle.load(open(os.path.join(a.out, f'clf_{fold}.pkl'), 'rb'))
        res = rows[['set', 'uid', 'view']].copy()
        res['run'], res['prob'], res['threshold'] = f'e10_univfd_{fold}_s1', m['clf'].predict_proba(fx)[:, 1], m['thr']
        if os.path.exists(path):
            old = pd.read_csv(path)
            res = pd.concat([old[old.run != f'e10_univfd_{fold}_s1'], res])
        res.to_csv(path, index=False)
        log(f'ArtiFact scored with UnivFD fold {fold}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['features', 'fit', 'corrupt', 'artifact'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--splits', default='splits_matched')
    ap.add_argument('--out', required=True, help='feature cache and fitted classifiers')
    ap.add_argument('--runs', default='/kaggle/working/runs')
    ap.add_argument('--external', default='', help='folder with ArtiFact crops.csv (and scores.csv)')
    ap.add_argument('--conditions', nargs='*', default=[])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.runs, exist_ok=True)
    dict(features=stage_features, fit=stage_fit, corrupt=stage_corrupt, artifact=stage_artifact)[a.stage](a)


if __name__ == '__main__':
    main()
