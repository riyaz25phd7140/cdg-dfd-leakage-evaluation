#!/usr/bin/env python3
"""
CDG-DFD E3: multi-family training with leave-one-family-out tests (ArtiFact faces + Phase 1 data).

Stages (all outputs inside the Phase 1 data folder, so crop paths stay relative to one root):
  prep    sample ArtiFact face images per generator, MTCNN-crop them exactly as Phase 1 (224, margin 40),
          read with many threads (Kaggle dataset storage is slow per file)  -> crops_artifact/, artifact_index.csv
  split   build splits_e3/fold_{SG,DIF,OTH}/{train,val,test}.csv:
            Phase 1 rows: union of splits_matched folds A and B, so both DFF generators are in training
            ArtiFact reals: 70/10/20 by image, the same in every fold
            ArtiFact fakes: the held-out family goes to test only; the other families 70/10/20
Families held out in turn:
  SG   StyleGAN family: StyleGAN 1/2/3, ProjectedGAN, GANformer, SFHQ
  DIF  diffusion: Stable Diffusion faces, denoising-diffusion GAN, Diffusion-StyleGAN2
  OTH  other: CIPS, StarGAN, Taming Transformer, LaMa, FaceSynthetics
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdg_external import AF_FACES, _norm, _resolve, find_artifact  # noqa: E402

FAMILIES = {
    'SG': ['stylegan1', 'stylegan2', 'stylegan3', 'projected_gan', 'gansformer', 'sfhq'],
    'DIF': ['stable_diffusion', 'denoising_diffusion_gan', 'diffusion_gan'],
    'OTH': ['cips', 'star_gan', 'taming_transformer', 'lama', 'face_synthetics'],
}
DIFFUSION_TYPE = {'stable_diffusion', 'denoising_diffusion_gan', 'diffusion_gan'}   # DAT domain 1; others 0
REALS = ['ffhq', 'celebahq']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def uid_of(*parts):
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
def stage_prep(a):
    import torch
    from facenet_pytorch import MTCNN
    gens = REALS + [g for f in FAMILIES.values() for g in f]
    metas = {_norm(os.path.basename(os.path.dirname(p))): p for p in find_artifact(a.artifact_root)}
    missing = [g for g in gens if _norm(g) not in metas]
    log(f'{len(metas)} ArtiFact folders found; missing: {missing or "none"}')
    if len(missing) == len(gens):
        sys.exit('ArtiFact not found: attach the Kaggle dataset awsaf49/artifact-dataset')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=224, margin=40, post_process=False, keep_all=False, device=device)
    out_csv = os.path.join(a.data, 'artifact_index.csv')
    done = pd.read_csv(out_csv, keep_default_na=False) if os.path.exists(out_csv) else pd.DataFrame()
    rows = done.to_dict('records')
    have = set(done.generator) if len(done) else set()
    for g in gens:
        if g in have or _norm(g) not in metas:
            continue
        t0 = time.time()
        mpath = metas[_norm(g)]
        folder = os.path.dirname(mpath)
        m = pd.read_csv(mpath, keep_default_na=False, dtype=str)
        t = m['target'].astype(str).str.strip()
        m = m[(t == '0') if g in REALS else (t != '0')]
        sel = AF_FACES[_norm(g)]
        m = m[[bool(sel(str(c))) for c in m['category']]]
        n = a.n_real if g in REALS else a.n_fake
        m = m.sample(n=min(n, len(m)), random_state=a.seed)
        paths = [_resolve(folder, rel) for rel in m['image_path']]

        def load(p):
            try:
                with open(p, 'rb') as f:
                    return Image.open(io.BytesIO(f.read())).convert('RGB')
            except Exception:  # noqa: BLE001
                return None

        with ThreadPoolExecutor(a.threads) as ex:
            imgs = list(ex.map(load, [p for p in paths if p]))
        rels = [r for r, p in zip(m['image_path'], paths) if p]
        out_dir = os.path.join(a.data, 'crops_artifact', g)
        os.makedirs(out_dir, exist_ok=True)
        ok = 0
        # batch the face detector over images of the same size (ArtiFact images are 200x200)
        by_size = {}
        for i, im in enumerate(imgs):
            if im is not None:
                by_size.setdefault(im.size, []).append(i)
        for size, idx in by_size.items():
            for s in range(0, len(idx), a.batch):
                chunk = idx[s:s + a.batch]
                try:
                    faces = mtcnn([imgs[i] for i in chunk])
                except ValueError:
                    # facenet_pytorch + new numpy: a batch where images have different
                    # numbers of detections breaks select_boxes -> fall back to one by one
                    faces = []
                    for i in chunk:
                        try:
                            faces.append(mtcnn(imgs[i]))
                        except Exception:  # noqa: BLE001
                            faces.append(None)
                for i, face in zip(chunk, faces):
                    if face is None:
                        continue
                    u = uid_of('artifact', g, rels[i])
                    rel = f'crops_artifact/{g}/{u}.png'
                    Image.fromarray(face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(
                        os.path.join(a.data, rel))
                    rows.append(dict(crop_path=rel, label=0 if g in REALS else 1, generator=g, uid=u,
                                     src_path=rels[i]))
                    ok += 1
        log(f'{g}: {len(m)} sampled, {sum(im is not None for im in imgs)} read, {ok} faces '
            f'({time.time() - t0:.0f} s)')
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # resumable per generator
    log(f'artifact_index.csv: {len(rows)} crops')


# ----------------------------------------------------------------------------
def stage_split(a):
    base = {}
    for s in ('train', 'val', 'test'):
        parts = [pd.read_csv(os.path.join(a.data, a.base_splits, f'fold_{f}', f'{s}.csv'), keep_default_na=False,
                             low_memory=False) for f in 'AB']
        base[s] = pd.concat(parts, ignore_index=True).drop_duplicates('uid')
    # a row in two different splits across folds A and B would be a leak: check
    seen = {}
    for s, d in base.items():
        for u in d.uid:
            assert seen.setdefault(u, s) == s, f'Phase 1 row {u} is in two splits'
    af = pd.read_csv(os.path.join(a.data, 'artifact_index.csv'), keep_default_na=False)
    af['source'] = 'artifact_' + af.generator
    af['dataset'] = 'ArtiFact'
    af['family'] = af.generator.map(lambda g: 'real' if g in REALS else next(k for k, v in FAMILIES.items() if g in v))
    af['domain'] = [(int(u, 16) % 2) if g in REALS else (1 if g in DIFFUSION_TYPE else 0)
                    for u, g in zip(af.uid, af.generator)]
    af['generator'] = af.generator.where(af.label == 1, 'real')
    tr, va, _ = (float(x) for x in a.ratios.split(','))

    def role(u):
        r = random.Random(f'{a.seed}:{u}').random()
        return 'train' if r < tr else ('val' if r < tr + va else 'test')

    af['role'] = af.uid.map(role)
    cols = list(base['train'].columns)
    summary = []
    for fold, held in FAMILIES.items():
        out = os.path.join(a.data, 'splits_e3', f'fold_{fold}')
        os.makedirs(out, exist_ok=True)
        is_held = af.family == fold
        for s in ('train', 'val', 'test'):
            if s == 'test':
                held_rows = af[is_held]
                held_rows = held_rows.sample(frac=1, random_state=a.seed).groupby('generator').head(a.held_test)
                part = pd.concat([af[~is_held & (af.role == 'test')], held_rows])
            else:
                part = af[~is_held & (af.role == s)]
            part = part.copy()
            part['test_set'] = ('ArtiFact:' + part.generator) if s == 'test' else ''
            part['group'] = 'af:' + part.uid
            for c in cols:
                if c not in part:
                    part[c] = ''
            d = pd.concat([base[s], part[cols]], ignore_index=True)
            d.to_csv(os.path.join(out, f'{s}.csv'), index=False)
            summary.append(dict(fold=fold, split=s, rows=len(d), artifact=len(part),
                                artifact_fake=int(part.label.astype(int).sum())))
        # no ArtiFact image of the held-out family may be in train/val
        trv = pd.concat([pd.read_csv(os.path.join(out, f'{s}.csv'), keep_default_na=False, low_memory=False)
                         for s in ('train', 'val')])
        assert not trv.generator.isin(held).any(), f'fold {fold}: held-out generator in train/val'
        te = pd.read_csv(os.path.join(out, 'test.csv'), keep_default_na=False, low_memory=False)
        assert not set(trv.uid) & set(te.uid), f'fold {fold}: uid in both train/val and test'
        assert te.test_set.str.contains(':').all(), f'fold {fold}: test rows without a test-set name'
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.data, 'splits_e3', 'summary.csv'), index=False)
    print(s.to_string(index=False), flush=True)
    log('E3 splits written and checked (held-out families never in train/val; no uid overlap)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['prep', 'split'])
    ap.add_argument('--data', required=True, help='Phase 1 data folder (has splits_matched/)')
    ap.add_argument('--artifact-root', nargs='+', default=['/kaggle/input'])
    ap.add_argument('--n-fake', type=int, default=2500)
    ap.add_argument('--n-real', type=int, default=6000, help='per real source (FFHQ, CelebA-HQ)')
    ap.add_argument('--held-test', type=int, default=1000, help='test images per held-out generator')
    ap.add_argument('--ratios', default='0.7,0.1,0.2')
    ap.add_argument('--base-splits', default='splits_matched')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args(argv)
    {'prep': stage_prep, 'split': stage_split}[a.stage](a)


if __name__ == '__main__':
    main()
