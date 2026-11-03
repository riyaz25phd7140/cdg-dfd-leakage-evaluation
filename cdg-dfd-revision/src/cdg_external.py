#!/usr/bin/env python3
"""
CDG-DFD external test: score trained checkpoints on DiffusionFace (DiffFace; Chen et al., 2024; Zenodo, CC BY 4.0),
whose generators were never used in training.

Stages (all outputs under --out):
  fetch   download each archive, keep a random sample of N images, delete the archive   -> raw/<set>/...
  crop    MTCNN crop (Phase 1 settings) in two versions:                                  -> crops/<view>/<set>/...
            asis     the files as distributed
            matched  every image (real and fake) first resized to 512x512 and saved as JPEG quality 75,
                     the DFF fakes' storage, so both classes share one file history
  score   every checkpoint (best.pth + config.json) on every crop                        -> scores.csv
  report  AUC of each generator against DiffFace reals, per view, mean and spread over seeds,
          plus a file-format table per set                                               -> report.md, report.csv
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
ZENODO = {'part1': '10865300', 'part2': '10865065'}   # record ids
URL_FORMS = ['https://zenodo.org/records/{rec}/files/{f}?download=1',
             'https://zenodo.org/api/records/{rec}/files/{f}/content',
             'https://zenodo.org/record/{rec}/files/{f}?download=1']
# name -> (zenodo part, archive, label, family)
SETS = {
    'Real':           ('part1', 'Real.tar', 0, 'real'),
    'DDPM':           ('part1', 'DDPM.tar', 1, 'unconditional diffusion'),
    'DDIM':           ('part1', 'DDIM.tar', 1, 'unconditional diffusion'),
    'PNDM':           ('part1', 'PNDM.tar', 1, 'unconditional diffusion'),
    'LDM':            ('part1', 'LDM.tar', 1, 'latent diffusion'),
    'ADM':            ('part1', 'ADM.tar', 1, 'unconditional diffusion'),
    'SDv21_img2img':  ('part1', 'SDv21_DS0.5.tar', 1, 'Stable Diffusion 2.1 (image-to-image)'),
    'SDv21_text2img': ('part2', 'stable_diffusion_v_2_1_text2img_p1g7.tar', 1, 'Stable Diffusion 2.1 (text-to-image)'),
}
MATCHED_SIZE, MATCHED_QUALITY = (512, 512), 75


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ----------------------------------------------------------------------------
def _why(url):
    """First bytes of the server's answer, to explain a refusal."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return f'HTTP {r.status}, {r.headers.get("Content-Type")}'
    except urllib.error.HTTPError as e:
        body = e.read(300).decode('utf-8', 'replace').replace('\n', ' ')
        return f'HTTP {e.code}: {body[:200]}'
    except Exception as e:  # noqa: BLE001
        return f'{type(e).__name__}: {e}'


def download(rec, fname, dest):
    """Try plain download tools (no browser identity) and several Zenodo URL forms; keep the first real tar."""
    import shutil
    import subprocess
    tmp = dest + '.part'
    tried = []
    for form in URL_FORMS:
        url = form.format(rec=rec, f=fname)
        tools = []
        if shutil.which('curl'):
            tools.append(('curl', ['curl', '-fL', '--retry', '3', '--retry-delay', '5', '-sS', '-o', tmp, url]))
        if shutil.which('wget'):
            tools.append(('wget', ['wget', '-q', '--tries=3', '-O', tmp, url]))
        tools.append(('python', None))
        for tool, cmd in tools:
            if os.path.exists(tmp):
                os.remove(tmp)
            t0 = time.time()
            try:
                if cmd:
                    subprocess.run(cmd, check=True, timeout=3 * 3600)
                else:
                    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
                        shutil.copyfileobj(r, f, 1 << 22)
                if tarfile.is_tarfile(tmp):
                    os.replace(tmp, dest)
                    gb = os.path.getsize(dest) / 1e9
                    log(f'   {fname}: {gb:.2f} GB with {tool} in {(time.time() - t0) / 60:.1f} min')
                    return
                tried.append(f'{tool} {url}: got a non-tar file ({os.path.getsize(tmp)} bytes)')
            except Exception as e:  # noqa: BLE001
                tried.append(f'{tool} {url}: {type(e).__name__}')
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError('all download attempts failed:\n   ' + '\n   '.join(tried) +
                       f'\n   server says: {_why(URL_FORMS[0].format(rec=rec, f=fname))}')


class _Meter:
    """File-like wrapper: counts bytes, reports speed every 30 s, stops at a deadline."""

    def __init__(self, f, label, deadline):
        self.f, self.label, self.deadline = f, label, deadline
        self.n, self.t0, self.last = 0, time.time(), time.time()

    def read(self, size=-1):
        if time.time() > self.deadline:
            raise TimeoutError('time limit for this set reached')
        b = self.f.read(size)
        self.n += len(b)
        if time.time() - self.last > 30:
            self.last = time.time()
            log(f'   {self.label}: {self.n / 1e9:.2f} GB read, {self.n / 1e6 / (time.time() - self.t0):.1f} MB/s')
        return b


def stream_sample(rec, fname, out_dir, n, stride, minutes):
    """Read the .tar while it downloads and stop once n images are kept (every stride-th image in archive order),
    so only a fraction of the archive is transferred. Returns (kept, note)."""
    agents = [None, 'curl/8.5.0', 'Wget/1.21.4']   # never a browser identity (those get bot challenges)
    notes = []
    for form in URL_FORMS:
        url = form.format(rec=rec, f=fname)
        for ua in agents:
            req = urllib.request.Request(url, headers={'User-Agent': ua} if ua else {})
            try:
                r = urllib.request.urlopen(req, timeout=60)
            except Exception as e:  # noqa: BLE001
                notes.append(f'{url} [{ua or "python"}]: {getattr(e, "code", type(e).__name__)}')
                continue
            os.makedirs(out_dir, exist_ok=True)
            kept, seen, meter = 0, 0, _Meter(r, fname, time.time() + 60 * minutes)
            try:
                with tarfile.open(fileobj=meter, mode='r|') as tf:
                    for m in tf:
                        if not (m.isfile() and m.name.lower().endswith(IMG_EXT)) or \
                                os.path.basename(m.name).startswith('.'):
                            continue
                        seen += 1
                        if (seen - 1) % stride:
                            continue
                        data = tf.extractfile(m).read()
                        key = hashlib.sha1(m.name.encode()).hexdigest()[:16]
                        with open(os.path.join(out_dir, key + os.path.splitext(m.name)[1].lower()), 'wb') as f:
                            f.write(data)
                        kept += 1
                        if kept >= n:
                            break
                note = 'complete'
            except TimeoutError:
                note = f'stopped at the {minutes}-minute limit'
            except Exception as e:  # noqa: BLE001
                note = f'stream ended early ({type(e).__name__})'
            finally:
                r.close()
            log(f'   {fname}: kept {kept} of the first {seen} images, {meter.n / 1e9:.2f} GB read in '
                f'{(time.time() - meter.t0) / 60:.1f} min ({note})')
            return kept, note
    return 0, 'no URL accepted: ' + '; '.join(notes)


def stage_fetch(a):
    failed = []
    for name in a.sets:
        rng = random.Random(f'{a.seed}:{name}')   # per set, so one failed set does not change the others
        part, archive, label, family = SETS[name]
        out_dir = os.path.join(a.out, 'raw', name)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= a.min_keep:
            log(f'{name}: already sampled ({len(os.listdir(out_dir))} files)')
            continue
        src = os.path.join(a.local, archive) if a.local else None
        tar_path = src if src and os.path.exists(src) else os.path.join(a.out, archive)
        if not os.path.exists(tar_path):
            # stream from Zenodo and stop early (about n*stride images are transferred, not the whole archive)
            log(f'{name}: streaming {archive} from Zenodo')
            kept, note = stream_sample(ZENODO[part], archive, out_dir, a.n, a.stride, a.minutes)
            if kept < a.min_keep:
                log(f'{name}: FAILED ({kept} images; {note})')
                failed.append(name)
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)   # a partial set is not used
            continue
        log(f'{name}: indexing {archive}')
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(IMG_EXT)
                       and not os.path.basename(m.name).startswith('.')]
            members.sort(key=lambda m: m.name)
            pick = rng.sample(members, min(a.n, len(members)))
            os.makedirs(out_dir, exist_ok=True)
            for m in pick:
                data = tf.extractfile(m).read()
                key = hashlib.sha1(m.name.encode()).hexdigest()[:16]
                with open(os.path.join(out_dir, key + os.path.splitext(m.name)[1].lower()), 'wb') as f:
                    f.write(data)
        log(f'{name}: {len(pick)} of {len(members)} images kept')
        if tar_path.startswith(a.out):
            os.remove(tar_path)   # keep the disk free
    have = [n for n in a.sets if os.path.isdir(os.path.join(a.out, 'raw', n)) and os.listdir(os.path.join(a.out, 'raw', n))]
    log(f'sampled sets: {have}; failed: {failed}')
    if 'Real' not in have or len(have) < 2:
        sys.exit('Not enough sets downloaded (need Real and at least one generator). If Zenodo refuses Kaggle, '
                 'download Real.tar and a few generator .tar files in a browser, upload them as a Kaggle dataset, '
                 'attach it, and run again: the notebook passes it with --local.')


# ----------------------------------------------------------------------------
# ArtiFact (Rahman et al., ICIP 2023; Kaggle awsaf49/artifact-dataset): already on Kaggle, attach as an input.
# One folder per source, each with metadata.csv (image_path, target 0/1, category). Every image (real and fake)
# went through the same random crop, downscale and JPEG impairments and is 200x200, so the two classes share
# one file history by construction (no compression shortcut).
AF_REAL = ('ffhq', 'celebahq')                     # real face sources
AF_FAMILY = {
    'ddpm': 'diffusion (not SD)', 'latent_diffusion': 'diffusion (not SD)', 'glide': 'diffusion (not SD)',
    'denoising_diffusion_gan': 'diffusion-GAN hybrid', 'diffusion_gan': 'diffusion-GAN hybrid',
    'vq_diffusion': 'diffusion (not SD)', 'palette': 'diffusion (not SD)',
    'stable_diffusion': 'Stable Diffusion (seen family)',
    'stylegan1': 'GAN', 'stylegan2': 'GAN', 'stylegan3': 'GAN', 'star_gan': 'GAN', 'stargan': 'GAN', 'cips': 'GAN',
    'projected_gan': 'GAN', 'gansformer': 'GAN', 'big_gan': 'GAN', 'biggan': 'GAN', 'pro_gan': 'GAN',
    'progan': 'GAN', 'cycle_gan': 'GAN', 'gau_gan': 'GAN', 'taming_transformer': 'transformer',
    'generative_inpainting': 'inpainting', 'lama': 'inpainting', 'mat': 'inpainting',
    'sfhq': 'mixed (SFHQ)', 'face_synthetics': 'rendered (3D)',
}
# Face subsets, from the folder inventory on Kaggle (category = the generator's training set or prompt group)
_ALL = lambda c: True
AF_FACES = {
    'ffhq': _ALL, 'celebahq': _ALL,                                   # reals
    'stylegan1': _ALL,                                               # '1 million fake faces'
    'stylegan2': lambda c: c == 'ffhq',
    'stylegan3': lambda c: 'ffhq' in c.lower(),
    'cips': lambda c: c == 'cips-ffhq',
    'gansformer': lambda c: c == 'ffhq_images',
    'projectedgan': lambda c: c == 'ffhq',
    'tamingtransformer': lambda c: c == 'tt-ffhq',
    'diffusiongan': lambda c: 'ffhq' in c.lower(),                   # Diffusion-StyleGAN2-FFHQ
    'denoisingdiffusiongan': _ALL,                                   # content not labelled: MTCNN rate shows it
    'lama': lambda c: c == 'celeb',                                  # inpainted CelebA-HQ
    'stargan': _ALL, 'sfhq': _ALL, 'facesynthetics': _ALL,
    'stablediffusion': lambda c: c in ('Male', 'Female'),
}
FACE_RX = r'face|celeb|ffhq|portrait'
FACE_RX2 = r'human|person|people'


def _norm(x):
    return x.lower().replace('_', '').replace('-', '')


def af_family(folder):
    k = _norm(folder)
    if k in [_norm(x) for x in AF_REAL] or k in ('imagenet', 'coco', 'lsun', 'afhq', 'metfaces', 'landscape'):
        return 'real'
    return {_norm(x): v for x, v in AF_FAMILY.items()}.get(k, 'other')


def find_artifact(roots, max_depth=6):
    """Folders holding metadata.csv under the ArtiFact input; glob by fixed depth (a recursive glob would walk
    2.5 million files)."""
    found = []
    for root in roots:
        for d in range(1, max_depth + 1):
            found += glob.glob(os.path.join(root, *(['*'] * d), 'metadata.csv'))
    return sorted(set(found))


def _resolve(folder, rel):
    for c in (os.path.join(folder, rel), os.path.join(os.path.dirname(folder), rel),
              os.path.join(folder, os.path.basename(rel))):
        if os.path.exists(c):
            return c
    parts = rel.replace('\\', '/').split('/')
    for k in range(1, len(parts)):   # path written from a different root: drop leading parts
        for base in (folder, os.path.dirname(folder)):
            c = os.path.join(base, *parts[k:])
            if os.path.exists(c):
                return c
    return None


def stage_artifact(a):
    import shutil
    metas = find_artifact(a.artifact_root)
    log(f'{len(metas)} metadata.csv files found under {a.artifact_root}')
    if not metas:
        sys.exit('ArtiFact not found: attach the Kaggle dataset "awsaf49/artifact-dataset" as an input')
    summary, used = [], set()
    for mpath in metas:
        folder = os.path.dirname(mpath)
        name = os.path.basename(folder)
        if name in used:   # same folder name at two places: add the parent's name
            name = os.path.basename(os.path.dirname(folder)) + '_' + name
        used.add(name)
        fam = af_family(name)
        try:
            m = pd.read_csv(mpath, keep_default_na=False, dtype=str)
        except Exception as e:  # noqa: BLE001
            log(f'{name}: metadata unreadable ({type(e).__name__}), skipped'); continue
        if 'category' not in m or 'image_path' not in m:
            log(f'{name}: no category/image_path column, skipped'); continue
        cats = m['category'].value_counts()
        row = dict(folder=name, family=fam, rows=len(m),
                   categories='; '.join(f'{k} {v}' for k, v in cats.head(8).items()))
        if 'target' in m:   # 0 = real; ArtiFact numbers the fake generator types (1, 2, 6, ...)
            t = m['target'].astype(str).str.strip()
            m = m[(t == '0') if fam == 'real' else (t != '0')]
        sel = AF_FACES.get(_norm(name))
        if sel is None:
            summary.append(dict(row, face_rows=0, kept=0, note='not a face source')); continue
        face = m[[bool(sel(str(c))) for c in m['category']]]
        row['face_rows'] = len(face)
        n_want = a.n_real if fam == 'real' else a.n
        if face.empty:
            summary.append(dict(row, kept=0, note='no face images')); continue
        pick = face.sample(n=min(n_want, len(face)), random_state=a.seed)
        set_name = 'Real' if fam == 'real' else 'AF_' + name
        out_dir = os.path.join(a.out, 'raw', set_name)
        os.makedirs(out_dir, exist_ok=True)
        kept = 0
        for rel in pick['image_path']:
            p = _resolve(folder, rel)
            if p is None:
                continue
            key = (name + '_' if fam == 'real' else '') + hashlib.sha1(rel.encode()).hexdigest()[:16]
            dst = os.path.join(out_dir, key + os.path.splitext(p)[1].lower())
            if not os.path.exists(dst):
                shutil.copyfile(p, dst)
            kept += 1
        summary.append(dict(row, kept=kept, note='' if kept else 'image_path did not resolve'))
        log(f'{name} [{fam}]: {len(face):,} face images of {len(m):,}; kept {kept}')
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.out, 'artifact_inventory.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
    print(s.to_string(index=False), flush=True)
    sets = sorted(os.listdir(os.path.join(a.out, 'raw'))) if os.path.isdir(os.path.join(a.out, 'raw')) else []
    json.dump({x: (0 if x == 'Real' else 1) for x in sets}, open(os.path.join(a.out, 'sets.json'), 'w'), indent=1)
    if 'Real' not in sets or len(sets) < 2:
        sys.exit('ArtiFact: no real faces or no fake faces were found; the inventory above shows what is there')


# ----------------------------------------------------------------------------
def estimate_jpeg_quality(im):
    from cdg_phase1 import estimate_jpeg_quality as est   # same estimator as Phase 1
    return est(im)


def to_matched(img):
    buf = io.BytesIO()
    img.resize(MATCHED_SIZE, Image.Resampling.BICUBIC).save(buf, format='JPEG', quality=MATCHED_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def stage_crop(a):
    import torch
    from facenet_pytorch import MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=224, margin=40, post_process=False, keep_all=False, device=device)
    rows = []
    sets = a.sets or sorted(os.listdir(os.path.join(a.out, 'raw')))
    for name in sets:
        files = sorted(glob.glob(os.path.join(a.out, 'raw', name, '*')))
        if not files:
            log(f'{name}: not downloaded, skipped')
            continue
        n_ok = {'asis': 0, 'matched': 0}
        for p in files:
            uid = os.path.splitext(os.path.basename(p))[0]
            try:
                with Image.open(p) as im:
                    fmt, w, h = im.format, im.size[0], im.size[1]
                    q = estimate_jpeg_quality(im)
                    img = im.convert('RGB')
            except Exception as e:  # noqa: BLE001
                rows.append(dict(set=name, uid=uid, view='asis', ok=0, reason=f'read:{type(e).__name__}'))
                continue
            for view, im_v in (('asis', img), ('matched', to_matched(img))):
                out = os.path.join(a.out, 'crops', view, name, uid + '.png')
                ok = os.path.exists(out)
                if not ok:
                    try:
                        face = mtcnn(im_v)
                    except Exception:  # noqa: BLE001
                        face = None
                    if face is not None:
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        Image.fromarray(face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(out)
                        ok = True
                n_ok[view] += int(ok)
                rows.append(dict(set=name, uid=uid, view=view, ok=int(ok), crop=out if ok else '',
                                 orig_format=fmt, orig_w=w, orig_h=h, jpeg_q=q))
        log(f'{name}: {len(files)} images, faces found asis {n_ok["asis"]}, matched {n_ok["matched"]}')
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'crops.csv'), index=False)


# ----------------------------------------------------------------------------
def _walk_inputs(roots):
    """(best.pth paths, zip paths) under roots, without entering image datasets: a folder whose sub-folders hold
    metadata.csv (ArtiFact) or that has thousands of entries is not searched."""
    pths, zips = [], []
    for root in roots:
        for d, dirs, files in os.walk(root, followlinks=True):
            if 'metadata.csv' in files or len(files) > 5000 or \
                    any(os.path.exists(os.path.join(d, x, 'metadata.csv')) for x in dirs[:50]):
                dirs[:] = []
                continue
            pths += [os.path.join(d, f) for f in files if f == 'best.pth']
            zips += [os.path.join(d, f) for f in files if f.endswith('.zip')]
    return pths, zips


def find_checkpoints(roots):
    """Every runs/<name>/best.pth with its config.json; unpacks zipped notebook outputs if needed."""
    import zipfile
    found = {}
    pths, zips = _walk_inputs(roots)
    for p in pths:
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, 'config.json')):
            found[os.path.basename(d)] = d
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith(('best.pth', 'config.json', 'summary.json'))
                         and '/runs/' in '/' + n]
                for n in names:
                    run = os.path.basename(os.path.dirname(n))
                    if run in found:
                        continue
                    target = os.path.join('/tmp/ckpts', run)
                    os.makedirs(target, exist_ok=True)
                    if not os.path.exists(os.path.join(target, os.path.basename(n))):
                        with zf.open(n) as src, open(os.path.join(target, os.path.basename(n)), 'wb') as dst:
                            dst.write(src.read())
        except zipfile.BadZipFile:
            pass
    for d in glob.glob('/tmp/ckpts/*'):
        if os.path.exists(os.path.join(d, 'best.pth')) and os.path.exists(os.path.join(d, 'config.json')):
            found.setdefault(os.path.basename(d), d)
    return dict(sorted(found.items()))


def stage_score(a):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from cdg_train import Detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crops = pd.read_csv(os.path.join(a.out, 'crops.csv'), keep_default_na=False)
    crops = crops[crops.ok == 1].reset_index(drop=True)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    class DS(Dataset):
        def __init__(self, paths):
            self.paths = np.array(paths)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return tf(Image.open(str(self.paths[i])).convert('RGB')), i

    ckpts = find_checkpoints(a.checkpoints)
    if a.only:
        ckpts = {k: v for k, v in ckpts.items() if any(k.startswith(p) for p in a.only)}
    log(f'{len(ckpts)} checkpoints: {", ".join(ckpts)}')
    if not ckpts:
        sys.exit('no checkpoints found: attach the E1 / E1b notebook outputs')
    out_path = os.path.join(a.out, 'scores.csv')
    done = set(pd.read_csv(out_path).run.unique()) if os.path.exists(out_path) else set()
    loader = DataLoader(DS(crops.crop.tolist()), batch_size=128, shuffle=False, num_workers=2)
    for run, d in ckpts.items():
        if run in done:
            continue
        cfg = json.load(open(os.path.join(d, 'config.json')))
        model = Detector(pretrained=False, spectral=cfg.get('spectral', 'band'),
                         channel_attn=cfg.get('channel_attn', 'off'), input_mode=cfg.get('input', 'dct'),
                         bands=cfg.get('bands', 16),
                         backbone=cfg.get('backbone', 'efficientnet_b4'))
        model.load_state_dict(torch.load(os.path.join(d, 'best.pth'), map_location='cpu', weights_only=True))
        model = model.to(device).eval()
        probs = np.zeros(len(crops), dtype=np.float32)
        with torch.no_grad():
            for x, i in loader:
                x = x.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits = model(x, alpha=0.0)[0]
                p = F.softmax(logits.float(), 1)[:, 1]
                if not torch.isfinite(p).all():
                    p = F.softmax(model(x, alpha=0.0)[0].float(), 1)[:, 1]
                probs[i.numpy()] = p.cpu().numpy()
        thr = json.load(open(os.path.join(d, 'summary.json'))).get('threshold', 0.5) \
            if os.path.exists(os.path.join(d, 'summary.json')) else 0.5
        res = crops[['set', 'uid', 'view']].copy()
        res['run'], res['prob'], res['threshold'] = run, probs, thr
        res.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
        log(f'scored {run}')


# ----------------------------------------------------------------------------
def stage_report(a):
    from sklearn.metrics import roc_auc_score
    from cdg_phase1 import md_table
    s = pd.read_csv(os.path.join(a.out, 'scores.csv'))
    rows = []
    for (run, view), g in s.groupby(['run', 'view']):
        real = g[g.set == 'Real']
        if real.empty:
            continue
        thr = g.threshold.iloc[0]
        rows.append(dict(run=run, view=view, fake_set='(reals flagged as fake)', n_real=len(real), n_fake=0,
                         AUC=np.nan, flagged=(real.prob >= thr).mean()))
        for fs, fg in g[g.set != 'Real'].groupby('set'):
            y = np.r_[np.zeros(len(real)), np.ones(len(fg))]
            p = np.r_[real.prob.values, fg.prob.values]
            rows.append(dict(run=run, view=view, fake_set=fs, n_real=len(real), n_fake=len(fg),
                             AUC=roc_auc_score(y, p), flagged=(fg.prob >= thr).mean()))
    r = pd.DataFrame(rows)
    parts = r.run.str.extract(r'^(?P<mode>[a-z]+)_(?P<sam>[a-z]+)_(?P<fold>[AB])_s(?P<seed>\d+)$')
    r = pd.concat([r, parts], axis=1)
    r['training_reals'] = r['mode'].map({'m': 'matched', 'o': 'original', 'band': 'original', 'se': 'original'})
    # E1 runs are named band_A_s1 / se_A_s1 (original reals)
    e1 = r.run.str.extract(r'^(?P<sam2>band|se)_(?P<fold2>[AB])_s(?P<seed2>\d+)$')
    r.loc[e1.sam2.notna(), 'sam'] = e1.sam2
    r.loc[e1.sam2.notna(), 'fold'] = e1.fold2
    r.loc[e1.sam2.notna(), 'seed'] = e1.seed2
    r.loc[e1.sam2.notna(), 'training_reals'] = 'original'
    # E2 ablation runs are named e2_rgb_A_s1 / e2_band_A_s1 (matched reals, no DAT)
    e2 = r.run.str.extract(r'^(?P<cfg3>e\d+_[a-z0-9]+)_(?P<fold3>[AB])_s(?P<seed3>\d+)$')   # e2_rgb, e10_resnet50, ...
    r.loc[e2.cfg3.notna(), 'sam'] = e2.cfg3
    r.loc[e2.cfg3.notna(), 'fold'] = e2.fold3
    r.loc[e2.cfg3.notna(), 'seed'] = e2.seed3
    r.loc[e2.cfg3.notna(), 'training_reals'] = 'matched'
    fam = {k: v[3] for k, v in SETS.items()}
    r['family'] = r.fake_set.map(lambda x: af_family(x[3:]) if x.startswith('AF_') else fam.get(x, ''))
    r.to_csv(os.path.join(a.out, 'report_per_run.csv'), index=False)
    agg = (r[(r.sam == 'band') | r.sam.astype(str).str.match(r'^e\d+_')].groupby(['sam', 'view', 'training_reals', 'fold', 'family', 'fake_set'])
           .agg(n_seeds=('run', 'nunique'), AUC_mean=('AUC', 'mean'), AUC_sd=('AUC', 'std'),
                flagged_mean=('flagged', 'mean')).round(4).reset_index())
    agg.to_csv(os.path.join(a.out, 'report.csv'), index=False)
    c = pd.read_csv(os.path.join(a.out, 'crops.csv'), keep_default_na=False)
    c = c[c.view == 'asis']
    fmt = c.groupby('set').agg(n=('uid', 'count'), faces=('ok', 'sum'),
                               formats=('orig_format', lambda x: ', '.join(f'{k} {v}' for k, v in x.value_counts().items())),
                               median_w=('orig_w', 'median'), median_h=('orig_h', 'median'),
                               jpeg_q=('jpeg_q', lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 1))).reset_index()
    md = [f'# External test: {a.title} (unseen generators)', '',
          'AUC of each generator against the same dataset\'s real faces. `asis` = files as distributed; `matched` = real and fake '
          'both re-saved as 512x512 JPEG quality 75 before cropping (no compression difference between classes). '
          '`flagged` = share scored as fake at the threshold chosen on the model\'s own validation set.', '',
          '## File formats per set (as distributed)', '', md_table(fmt), '',
          '## AUC, revised SAM, mean over seeds', '', md_table(agg)]
    open(os.path.join(a.out, 'report.md'), 'w').write('\n'.join(md))
    log('report written')
    return agg, fmt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fetch', 'artifact', 'crop', 'score', 'report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', nargs='*', default=[], help='default: every sampled set')
    ap.add_argument('--n', type=int, default=1000, help='images kept per set')
    ap.add_argument('--stride', type=int, default=2, help='streaming: keep every stride-th image in archive order')
    ap.add_argument('--minutes', type=float, default=25, help='streaming: time limit per set')
    ap.add_argument('--min-keep', type=int, default=200, help='a set with fewer images counts as failed')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--local', default='', help='folder that already holds the .tar archives (skip download)')
    ap.add_argument('--artifact-root', nargs='+', default=['/kaggle/input'])
    ap.add_argument('--title', default='ArtiFact faces')
    ap.add_argument('--n-real', type=int, default=1500, help='ArtiFact: images per real face source')
    ap.add_argument('--checkpoints', nargs='+', default=['/kaggle/input'])
    ap.add_argument('--only', nargs='*', default=[], help='score only runs whose name starts with these')
    a = ap.parse_args(argv)
    if a.stage == 'fetch' and not a.sets:
        a.sets = list(SETS)
    os.makedirs(a.out, exist_ok=True)
    {'fetch': stage_fetch, 'artifact': stage_artifact, 'crop': stage_crop, 'score': stage_score, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
