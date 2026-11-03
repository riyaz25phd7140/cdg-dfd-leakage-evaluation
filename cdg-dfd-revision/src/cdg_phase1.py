#!/usr/bin/env python3
"""
CDG-DFD Phase 1: data and protocol rebuild.

Rebuilds every split from the raw files, so each number in the paper can be
traced to code.

Stages (run in order; every stage reads and writes under --out):

  scan            index raw files / sampled video frames      -> raw_index.csv
  crop            MTCNN crops, lossless PNG, resumable         -> crop_status.csv, crops/
  dedup           exact + near-duplicate detection             -> dedup.csv, near_dup_pairs.csv
  split           group-disjoint, leave-one-generator-out      -> splits/fold_*/{train,val,test}.csv
  audit           leakage checks, accounting + domain tables   -> reports/*.md, reports/*.csv
  audit-original  re-creates the first-edition 85/15 split and
                  measures how much leaked across it          -> reports/audit_original.md

Label conventions (unchanged from the paper):
  y = 1 fake, 0 real
  d = 0 GAN-era family (HardFake, CelebDF-v2), 1 diffusion family (DFF),
      -1 = never used for training (test-only sets; no domain label needed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
VID_EXT = ('.mp4', '.avi', '.mov', '.mkv')

# ----------------------------------------------------------------------------
# Source definitions
# ----------------------------------------------------------------------------
# family: real | gan | faceswap_ae | diffusion | faceswap
SOURCES = {
    'hardfake_real':  dict(dataset='HardFake', generator='real',          family='real',        label=0, domain=0),
    'hardfake_fake':  dict(dataset='HardFake', generator='hardfake_gan',  family='gan',         label=1, domain=0),
    'celebdf_real':   dict(dataset='CelebDF',  generator='real',          family='real',        label=0, domain=0),
    'celebdf_fake':   dict(dataset='CelebDF',  generator='celebdf_swap',  family='faceswap_ae', label=1, domain=0),
    'dff_wiki':       dict(dataset='DFF',      generator='real',          family='real',        label=0, domain=1),
    'dff_inpainting': dict(dataset='DFF',      generator='sd_inpainting', family='diffusion',   label=1, domain=1),
    'dff_text2img':   dict(dataset='DFF',      generator='sd15_text2img', family='diffusion',   label=1, domain=1),
    # InsightFace is a face-swap toolbox, not diffusion: test-only, never labelled d=1.
    'dff_insight':    dict(dataset='DFF',      generator='insightface',   family='faceswap',    label=1, domain=-1),
    # Shortcut diagnostic: the same wiki reals passed through the fakes' file pipeline (512x512, JPEG q75)
    # before cropping. Test-only. If a detector flags these as fake, it is reading compression, not faces.
    'dff_wiki_matched': dict(dataset='DFF',    generator='real_matched',  family='real',        label=0, domain=-1),
}
MATCHED_SIZE, MATCHED_QUALITY = (512, 512), 75   # DFF fakes: all 512x512, IJG quality 75 (PIL's default)
DFF_SUBSETS = {'dff_wiki': 'wiki', 'dff_inpainting': 'inpainting',
               'dff_text2img': 'text2img', 'dff_insight': 'insight'}

# Leave-one-generator-out folds over the two DFF diffusion generators.
FOLDS = {
    'A': dict(train_gens=['sd_inpainting'], heldout_gens=['sd15_text2img']),
    'B': dict(train_gens=['sd15_text2img'], heldout_gens=['sd_inpainting']),
}
# Generators that are only ever tested on (in every fold).
TEST_ONLY_GENS = {'insightface', 'real_matched'}

# Standard IJG luminance quantisation table (quality 50), used to estimate JPEG quality.
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def uid_of(*parts) -> str:
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:16]


def md5_file(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def list_files(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def estimate_jpeg_quality(im) -> float:
    """Estimate IJG quality from the luminance quantisation table; NaN if not JPEG."""
    q = getattr(im, 'quantization', None)
    if not q or 0 not in q:
        return float('nan')
    table = np.array(q[0], dtype=np.float64)
    if table.size != 64:
        return float('nan')
    scale = float(np.mean(table / _IJG_LUMA) * 100.0)
    qual = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return round(float(np.clip(qual, 1, 100)), 1)


def image_header(path):
    try:
        with Image.open(path) as im:
            return (im.format or os.path.splitext(path)[1][1:].upper(),
                    im.size[0], im.size[1], estimate_jpeg_quality(im))
    except Exception:
        return ('UNREADABLE', -1, -1, float('nan'))


def find_subdir(root, name):
    """DFF zips extract either to root/name or root/name/name; return the deepest match."""
    cands = [os.path.join(root, name, name), os.path.join(root, name)]
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


# Identity parsing --------------------------------------------------------------
_WIKI_ID = re.compile(r'^(\d+)_')          # IMDB-WIKI wiki files:  10049200_1891-09-16_1958.jpg
_IMDB_ID = re.compile(r'^(nm\d+)_')        # IMDB files:            nm0000100_rm1001569280_1955-1-6_2011.jpg
_CDF_REAL = re.compile(r'^(id\d+)_\d+$')   # Celeb-real/id0_0000
_CDF_FAKE = re.compile(r'^(id\d+)_(id\d+)_\d+$')  # Celeb-synthesis/id0_id16_0000


def dff_identity(basename):
    stem = os.path.basename(basename)
    for rx, tag in ((_WIKI_ID, 'wiki'), (_IMDB_ID, 'imdb')):
        m = rx.match(stem)
        if m:
            return f'{tag}:{m.group(1)}'
    return ''


def celebdf_identities(video_id):
    folder, stem = video_id.split('/', 1) if '/' in video_id else ('', video_id)
    m = _CDF_FAKE.match(stem)
    if m:
        return [f'cdf:{m.group(1)}', f'cdf:{m.group(2)}']
    m = _CDF_REAL.match(stem)
    if m:
        return [f'cdf:{m.group(1)}']
    return []  # YouTube-real videos carry no identity label


# ----------------------------------------------------------------------------
# Stage 1: scan
# ----------------------------------------------------------------------------
def row_template(source, raw_path, **kw):
    meta = SOURCES.get(source, None)
    r = dict(uid='', source=source, raw_path=raw_path, frame_idx=-1, rel_key='', video_id='',
             identities='', celebdf_official_test='', orig_format='', orig_w=-1, orig_h=-1,
             jpeg_q=float('nan'), md5_raw='')
    if meta:
        r.update(meta)
    r.update(kw)
    return r


def scan_images(source, root, rel_root, with_identity):
    rows = []
    for p in list_files(root, IMG_EXT):
        rel = os.path.relpath(p, rel_root)
        rel_key = os.path.splitext(rel)[0].lower().replace('\\', '/')
        fmt, w, h, q = image_header(p)
        ident = dff_identity(p) if with_identity else ''
        rows.append(row_template(source, p, uid=uid_of(source, rel), rel_key=rel_key, identities=ident,
                                 orig_format=fmt, orig_w=w, orig_h=h, jpeg_q=q, md5_raw=md5_file(p)))
    return rows


def read_celebdf_test_list(path):
    vids = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rel = parts[-1]
            vids.add(os.path.splitext(rel)[0].replace('\\', '/'))
    return vids


def scan_celebdf(root, test_list, frames_real, frames_fake, max_fake_train_videos, seed):
    import cv2
    test_vids = read_celebdf_test_list(test_list)
    log(f'CelebDF official test list: {len(test_vids)} videos')
    vids = []
    for folder, source in (('Celeb-real', 'celebdf_real'), ('YouTube-real', 'celebdf_real'),
                           ('Celeb-synthesis', 'celebdf_fake')):
        d = os.path.join(root, folder)
        if not os.path.isdir(d):
            log(f'  WARNING: {d} not found')
            continue
        for p in list_files(d, VID_EXT):
            vid = f'{folder}/{os.path.splitext(os.path.basename(p))[0]}'
            vids.append((source, p, vid))
    # Optional cap on fake training videos (test videos are always kept).
    if max_fake_train_videos:
        rng = random.Random(seed)
        fake_train = [v for v in vids if v[0] == 'celebdf_fake' and v[2] not in test_vids]
        keep = set(x[2] for x in rng.sample(fake_train, min(max_fake_train_videos, len(fake_train))))
        vids = [v for v in vids if not (v[0] == 'celebdf_fake' and v[2] not in test_vids and v[2] not in keep)]
    missing = test_vids - set(v[2] for v in vids)
    if missing:
        log(f'  WARNING: {len(missing)} official test videos not found on disk (e.g. {sorted(missing)[:3]})')
    rows = []
    for source, p, vid in vids:
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        k = frames_real if source == 'celebdf_real' else frames_fake
        if n <= 0:
            idxs = [-2]  # unreadable video; recorded so it shows up in the accounting
        else:
            idxs = sorted(set(np.linspace(0, n - 1, num=k + 2)[1:-1].round().astype(int).tolist()))
        for fi in idxs:
            rows.append(row_template(source, p, uid=uid_of(source, vid, fi), frame_idx=fi, video_id=vid,
                                     identities=';'.join(celebdf_identities(vid)),
                                     celebdf_official_test=int(vid in test_vids),
                                     orig_format='video:' + os.path.splitext(p)[1][1:].lower(),
                                     orig_w=w, orig_h=h))
    return rows


def scan_external(name, root):
    """External test set: <root>/real/** and <root>/fake/** (either may be missing)."""
    rows = []
    for sub, label in (('real', 0), ('fake', 1)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        src = f'ext_{name}_{sub}'
        for p in list_files(d, IMG_EXT):
            rel = os.path.relpath(p, root)
            fmt, w, h, q = image_header(p)
            gen = 'real' if label == 0 else f'ext_{name}'
            rows.append(row_template(src, p, uid=uid_of(src, rel), rel_key=rel.lower(), dataset=f'ext_{name}',
                                     generator=gen, family='real' if label == 0 else 'external',
                                     label=label, domain=-1, orig_format=fmt, orig_w=w, orig_h=h,
                                     jpeg_q=q, md5_raw=md5_file(p)))
    return rows


def matched_rows(wiki_rows):
    out = []
    for x in wiki_rows:
        y = dict(x)
        y.update(SOURCES['dff_wiki_matched'], source='dff_wiki_matched',
                 uid=uid_of('dff_wiki_matched', x['rel_key']),
                 md5_raw='matched:' + str(x['md5_raw']), orig_format='JPEG(matched)',
                 orig_w=MATCHED_SIZE[0], orig_h=MATCHED_SIZE[1], jpeg_q=float(MATCHED_QUALITY))
        out.append(y)
    return out


def stage_add_matched(a):
    """Add the matched-real rows to an existing rebuild (needs the wiki images at their scanned paths).
    Then run crop (only the new rows are cropped), split and audit. dedup need not be re-run: the new
    rows share a person and source photo with their wiki originals, so they always join those groups."""
    path = os.path.join(a.out, 'raw_index.csv')
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if (df.source == 'dff_wiki_matched').any():
        log('matched rows already present; nothing to do')
        return
    wiki = df[df.source == 'dff_wiki']
    missing = (~wiki.raw_path.map(os.path.exists)).sum()
    if missing:
        sys.exit(f'{missing} of {len(wiki)} wiki images are not at their scanned paths; re-download wiki.zip first')
    m = pd.DataFrame(matched_rows(wiki.to_dict('records')))
    pd.concat([df, m[df.columns]], ignore_index=True).to_csv(path, index=False)
    log(f'added {len(m)} dff_wiki_matched rows; raw_index.csv now {len(df) + len(m)} rows')


def stage_scan(a):
    os.makedirs(a.out, exist_ok=True)
    rows = []
    if a.hardfake_root:
        for sub, src in (('real', 'hardfake_real'), ('fake', 'hardfake_fake')):
            d = os.path.join(a.hardfake_root, sub)
            r = scan_images(src, d, d, with_identity=False)
            log(f'{src}: {len(r)} images from {d}')
            rows += r
    if a.dff_root:
        for src, sub in DFF_SUBSETS.items():
            d = find_subdir(a.dff_root, sub)
            if d is None:
                log(f'{src}: not found under {a.dff_root} (skipped)')
                continue
            r = scan_images(src, d, d, with_identity=True)
            if src == 'dff_insight' and a.dff_insight_sample and len(r) > a.dff_insight_sample:
                r = random.Random(a.seed).sample(r, a.dff_insight_sample)  # test-only set: a sample is enough
            log(f'{src}: {len(r)} images from {d}')
            rows += r
            if src == 'dff_wiki' and a.dff_matched_real:
                m = matched_rows(r)
                log(f'dff_wiki_matched: {len(m)} pipeline-matched copies of the wiki reals (test-only)')
                rows += m
    if a.celebdf_root:
        tl = a.celebdf_test_list or os.path.join(a.celebdf_root, 'List_of_testing_videos.txt')
        r = scan_celebdf(a.celebdf_root, tl, a.celebdf_frames_real, a.celebdf_frames_fake,
                         a.celebdf_max_fake_train_videos, a.seed)
        log(f'celebdf: {len(r)} frames from {len(set(x["video_id"] for x in r))} videos')
        rows += r
    for spec in a.external or []:
        name, root = spec.split('=', 1)
        r = scan_external(name, root)
        log(f'external {name}: {len(r)} images')
        rows += r
    if not rows:
        sys.exit('Nothing scanned: give at least one of --hardfake-root/--dff-root/--celebdf-root/--external')
    df = pd.DataFrame(rows)
    dupe_uid = df.uid.duplicated()
    if dupe_uid.any():
        sys.exit(f'uid collision on {dupe_uid.sum()} rows; check input paths')
    df.to_csv(os.path.join(a.out, 'raw_index.csv'), index=False)
    cfg = {k: v for k, v in vars(a).items() if k != 'func'}
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    log(f'raw_index.csv: {len(df)} rows')
    if 'dff_wiki' in set(df.source):
        wiki_keys = set(df.loc[df.source == 'dff_wiki', 'rel_key'])
        for src in ('dff_inpainting', 'dff_text2img', 'dff_insight'):
            sub = df[df.source == src]
            if len(sub):
                log(f'  {src}: {sub.rel_key.isin(wiki_keys).mean()*100:.1f}% share a relative path with a wiki '
                    f'real image; identity parsed for {(sub.identities != "").mean()*100:.1f}%')


# ----------------------------------------------------------------------------
# Stage 2: crop (MTCNN, identical settings to the first edition)
# ----------------------------------------------------------------------------
def match_fake_pipeline(img):
    """Resize to 512x512 and JPEG-encode at quality 75, as every DFF fake file is stored."""
    import io
    buf = io.BytesIO()
    img.resize(MATCHED_SIZE, Image.Resampling.BICUBIC).save(buf, format='JPEG', quality=MATCHED_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def stage_crop(a):
    import torch
    from facenet_pytorch import MTCNN
    df = pd.read_csv(os.path.join(a.out, 'raw_index.csv'), keep_default_na=False, low_memory=False)
    status_path = os.path.join(a.out, 'crop_status.csv')
    done = set()
    if os.path.exists(status_path):
        done = set(pd.read_csv(status_path, usecols=['uid']).uid)
    todo = df[~df.uid.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    log(f'crop: {len(done)} already done, {len(todo)} to do')
    if todo.empty:
        return
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Same settings as df-1-first-edition: 224 px, 40 px margin, no post-processing, largest face.
    mtcnn = MTCNN(image_size=224, margin=40, post_process=False, keep_all=False, device=device)
    crop_root = os.path.join(a.out, 'crops')
    header = not os.path.exists(status_path)
    buf, n_ok, n_fail, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal buf, header
        if buf:
            pd.DataFrame(buf).to_csv(status_path, mode='a', header=header, index=False)
            header, buf = False, []

    def handle(row, pil_img, err=''):
        nonlocal n_ok, n_fail
        out_dir = os.path.join(crop_root, row.source)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{row.uid}.png')
        ok, prob = 0, float('nan')
        if pil_img is not None:
            try:
                face, prob = mtcnn(pil_img, return_prob=True)
                if face is not None:
                    arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
                    Image.fromarray(arr).save(out_path)  # PNG: no extra lossy step on either class
                    ok = 1
                else:
                    err = 'no_face'
            except Exception as e:  # noqa: BLE001
                err = f'mtcnn_error:{type(e).__name__}'
        crop_md5 = md5_file(out_path) if ok else ''
        buf.append(dict(uid=row.uid, mtcnn_ok=ok, mtcnn_prob=prob if prob is not None else float('nan'),
                        crop_path=os.path.relpath(out_path, a.out) if ok else '', crop_md5=crop_md5,
                        fail_reason=err))
        n_ok += ok
        n_fail += 1 - ok
        if (n_ok + n_fail) % 2000 == 0:
            flush()
            rate = (n_ok + n_fail) / max(time.time() - t0, 1e-6)
            log(f'  {n_ok + n_fail}/{len(todo)}  ok={n_ok} fail={n_fail}  {rate:.1f} img/s')

    img_rows = todo[todo.frame_idx.astype(int) == -1]
    vid_rows = todo[todo.frame_idx.astype(int) != -1]
    for row in img_rows.itertuples(index=False):
        try:
            img = Image.open(row.raw_path).convert('RGB')
            if row.source == 'dff_wiki_matched':
                img = match_fake_pipeline(img)
            handle(row, img)
        except Exception as e:  # noqa: BLE001
            handle(row, None, f'read_error:{type(e).__name__}')
    if len(vid_rows):
        import cv2
        for path, grp in vid_rows.groupby('raw_path'):
            cap = cv2.VideoCapture(path)
            for row in grp.sort_values('frame_idx').itertuples(index=False):
                fi = int(row.frame_idx)
                if fi < 0:
                    handle(row, None, 'video_unreadable')
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    handle(row, None, 'frame_read_error')
                    continue
                handle(row, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
    flush()
    log(f'crop done: ok={n_ok} fail={n_fail}')


# ----------------------------------------------------------------------------
# Stage 3: dedup
# ----------------------------------------------------------------------------
def image_signature(path):
    """64-bit dHash (for candidate search) and a 32x32 grayscale thumbnail (for confirmation).

    Computed on the whole raw image for stills: two MTCNN crops of the same photo can be offset by a
    few pixels, which hides a re-encoded or resized copy, while the full-image thumbnails still match."""
    with Image.open(path) as im:
        g = im.convert('L')
        d = np.asarray(g.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        thumb = np.asarray(g.resize((32, 32), Image.Resampling.BOX), dtype=np.uint8)
    bits = (d[:, 1:] > d[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2), thumb


def near_dup_pairs(hashes, max_dist):
    """All index pairs with Hamming distance <= max_dist (pigeonhole over max_dist+1 chunks)."""
    n_chunks = max_dist + 1
    edges = [int(round(x)) for x in np.linspace(0, 64, n_chunks + 1)]
    pairs = set()
    for c in range(n_chunks):
        lo, hi = edges[c], edges[c + 1]
        mask = (1 << (hi - lo)) - 1
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> lo) & mask].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                hx = hashes[members[x]]
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if (hx ^ hashes[j]).bit_count() <= max_dist:
                        pairs.add((i, j) if i < j else (j, i))
    return sorted(pairs)


def load_merged(out):
    raw = pd.read_csv(os.path.join(out, 'raw_index.csv'), keep_default_na=False, low_memory=False)
    st = pd.read_csv(os.path.join(out, 'crop_status.csv'), keep_default_na=False)
    st = st.drop_duplicates('uid', keep='last')
    df = raw.merge(st, on='uid', how='left')
    df['mtcnn_ok'] = pd.to_numeric(df['mtcnn_ok'], errors='coerce').fillna(-1).astype(int)  # -1 = not cropped yet
    for c in ('label', 'domain', 'frame_idx'):
        df[c] = df[c].astype(int)
    return df


def stage_dedup(a):
    df = load_merged(a.out)
    if (df.mtcnn_ok == -1).any():
        sys.exit(f'{(df.mtcnn_ok == -1).sum()} rows not cropped yet; finish the crop stage first')
    ok = df[df.mtcnn_ok == 1].copy()
    log(f'dedup: hashing {len(ok)} crops')
    # stills: signature of the raw image; video frames (raw not stored): signature of the crop.
    # Signatures are cached so dedup can be re-run after the raw downloads are deleted.
    cache_path = os.path.join(a.out, 'signatures.npz')
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        cache = {u: (int(h, 16), t) for u, h, t in zip(z['uid'], z['dhash'], z['thumb'])}
    sigs, missing = [], 0
    for r in ok.itertuples(index=False):
        if r.uid in cache:
            sigs.append(cache[r.uid])
            continue
        src = r.raw_path if r.frame_idx == -1 else os.path.join(a.out, r.crop_path)
        if r.frame_idx == -1 and not os.path.exists(src):
            missing += 1
            src = os.path.join(a.out, r.crop_path)   # weaker fallback: crop-based signature
        sigs.append(image_signature(src))
    if missing:
        log(f'  WARNING: {missing} raw files missing and not cached; used crop-based signatures for them')
    np.savez(cache_path, uid=ok.uid.to_numpy(dtype=str), dhash=np.array([f'{h:016x}' for h, _ in sigs]),
             thumb=np.stack([t for _, t in sigs]))
    ok['dhash'] = pd.Series([h for h, _ in sigs], index=ok.index, dtype=object)
    ok['_thumb'] = pd.Series([t for _, t in sigs], index=ok.index, dtype=object)
    # Exact duplicates: identical raw file (images) or identical crop file. Keep the first by a fixed order.
    order = list(SOURCES) + sorted(set(ok.source) - set(SOURCES))
    ok['_ord'] = ok.source.map({s: i for i, s in enumerate(order)})
    ok = ok.sort_values(['_ord', 'raw_path', 'frame_idx']).reset_index(drop=True)
    first_raw, first_crop, dup_of = {}, {}, []
    for r in ok.itertuples(index=False):
        d = ''
        if r.md5_raw and r.md5_raw in first_raw:
            d = first_raw[r.md5_raw]
        elif r.crop_md5 in first_crop:
            d = first_crop[r.crop_md5]
        else:
            if r.md5_raw:
                first_raw[r.md5_raw] = r.uid
            first_crop[r.crop_md5] = r.uid
        dup_of.append(d)
    ok['exact_dup_of'] = dup_of
    keep = ok[ok.exact_dup_of == ''].reset_index(drop=True)
    log(f'  exact duplicates removed: {(ok.exact_dup_of != "").sum()}')
    hashes = [int(h) for h in keep.dhash.tolist()]
    thumbs = keep._thumb.tolist()
    cand = near_dup_pairs(hashes, a.near_dup_hamming)
    # Unrelated images can share a 64-bit dHash, so every candidate is confirmed by pixel distance on
    # 32x32 thumbnails. Only near-identical images (re-encodes, resizes, small edits) survive.
    rows = []
    for i, j in cand:
        mad = float(np.abs(thumbs[i].astype(np.int16) - thumbs[j].astype(np.int16)).mean())
        if mad <= a.near_dup_mad:
            rows.append((keep.uid[i], keep.uid[j], keep.source[i], keep.source[j],
                         (hashes[i] ^ hashes[j]).bit_count(), round(mad, 2)))
    pr = pd.DataFrame(rows, columns=['uid_a', 'uid_b', 'source_a', 'source_b', 'hamming', 'thumb_mad'])
    pr.to_csv(os.path.join(a.out, 'near_dup_pairs.csv'), index=False)
    log(f'  near-duplicate candidates (dHash Hamming <= {a.near_dup_hamming}): {len(cand)}; '
        f'confirmed (32x32 mean abs diff <= {a.near_dup_mad}): {len(pr)}')
    if len(pr):
        log('  confirmed pairs by source: ' + ', '.join(
            f'{k[0]}~{k[1]}: {v}' for k, v in Counter(zip(pr.source_a, pr.source_b)).most_common(8)))
    ok['dhash'] = ok.dhash.map(lambda h: f'{h:016x}')
    ok[['uid', 'dhash', 'exact_dup_of']].to_csv(os.path.join(a.out, 'dedup.csv'), index=False)


# ----------------------------------------------------------------------------
# Stage 4: split
# ----------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                ra, rb = rb, ra
            self.p[ra] = rb


def load_eligible(out):
    df = load_merged(out)
    dd = pd.read_csv(os.path.join(out, 'dedup.csv'), keep_default_na=False)
    df = df.merge(dd, on='uid', how='left')
    df['exact_dup_of'] = df['exact_dup_of'].fillna('')
    df['dhash'] = df['dhash'].fillna('')
    df['eligible'] = (df.mtcnn_ok == 1) & (df.exact_dup_of == '')
    return df


def build_groups(df, out, celebdf_identity_mode):
    uf = UnionFind()
    for r in df.itertuples(index=False):
        u = 'u:' + r.uid
        uf.find(u)
        if r.video_id:
            uf.union(u, 'v:' + r.video_id)
        if r.dataset == 'DFF':
            if r.rel_key:
                uf.union(u, 'k:' + r.rel_key)          # same source photo across wiki/text2img/inpainting
            if r.identities:
                uf.union(u, 'i:' + r.identities)       # same person
        elif r.dataset == 'CelebDF' and celebdf_identity_mode == 'link':
            for ident in filter(None, r.identities.split(';')):
                uf.union(u, 'i:' + ident)
    pr_path = os.path.join(out, 'near_dup_pairs.csv')
    if os.path.exists(pr_path):
        for r in pd.read_csv(pr_path).itertuples(index=False):
            uf.union('u:' + r.uid_a, 'u:' + r.uid_b)
    return df.uid.map(lambda u: uf.find('u:' + u))


def stage_split(a):
    df = load_eligible(a.out)
    df = df[df.eligible].copy()
    ext = df.dataset.str.startswith('ext_')
    pool = df[~ext].copy()
    pool['group'] = build_groups(pool, a.out, a.celebdf_identity)
    tr, va, te = (float(x) for x in a.ratios.split(','))
    assert abs(tr + va + te - 1) < 1e-6, '--ratios must sum to 1'
    # One split per group, shared by all folds, so a person/video/photo never crosses splits.
    g = pool.groupby('group').agg(forced_test=('celebdf_official_test', lambda s: any(str(x) == '1' for x in s)),
                                  datasets=('dataset', lambda s: '|'.join(sorted(set(s)))))
    rng = random.Random(a.seed)
    split_of = {}
    for gid in sorted(g.index):
        row = g.loc[gid]
        if row.forced_test:
            split_of[gid] = 'test'
        elif 'CelebDF' in row.datasets:
            # CelebDF already has an official test set; remaining videos are divided train/val.
            split_of[gid] = 'val' if rng.random() < va / (tr + va) else 'train'
        else:
            x = rng.random()
            split_of[gid] = 'train' if x < tr else ('val' if x < tr + va else 'test')
    pool['gsplit'] = pool.group.map(split_of)
    sizes = pool.group.value_counts()
    log(f'groups: {len(sizes)}; largest sizes {sizes.head(5).tolist()}; '
        f'singletons {int((sizes == 1).sum())}')
    biggest = sizes.iloc[0] / len(pool)
    if biggest > a.max_group_share:
        sys.exit(f'largest group holds {biggest:.1%} of all images (> {a.max_group_share:.0%}). Linking has '
                 f'chained unrelated images together; inspect near_dup_pairs.csv or lower --near-dup-mad.')
    for s_ in ('train', 'val', 'test'):
        log(f'  group split {s_}: {(pool.gsplit == s_).mean():.1%} of images')
    out_root = os.path.join(a.out, a.split_name)
    os.makedirs(out_root, exist_ok=True)
    matched = a.dff_real == 'matched'
    # Test-only sources. 'original': DFF reals as downloaded train the model; the matched copies are a test-only
    # control. 'matched': the matched copies (stored like the fakes) train the model; the originals become test-only.
    test_only_src = {'dff_insight', 'dff_wiki' if matched else 'dff_wiki_matched'}
    log(f'DFF training reals: {a.dff_real}; test-only sources: {sorted(test_only_src)}; writing {out_root}')
    long_rows = []
    cols = ['crop_path', 'label', 'domain', 'source', 'dataset', 'generator', 'family', 'test_set',
            'uid', 'group', 'video_id', 'identities', 'rel_key']
    for fold, spec in FOLDS.items():
        f = pool.copy()
        role = []
        for r in f.itertuples(index=False):
            gen = r.generator
            if gen in spec['heldout_gens'] or r.source in test_only_src:
                role.append('test' if r.gsplit == 'test' else 'excluded')
            elif r.family == 'real' or gen in spec['train_gens'] or r.dataset in ('HardFake', 'CelebDF'):
                role.append(r.gsplit)
            else:
                role.append('excluded')
        f['role'] = role
        if matched:
            # A matched copy is used only if its original photo is usable here: not an exact duplicate (those were
            # removed before the copies existed) and face-cropped successfully. Otherwise two identical copies could
            # sit on both sides of the split. Rows are excluded after the split, so the groups stay as in E1.
            usable_wiki = set(pool.rel_key[pool.source == 'dff_wiki'])
            orphan = (f.source == 'dff_wiki_matched') & ~f.rel_key.isin(usable_wiki)
            f.loc[orphan, 'role'] = 'excluded'
            if fold == 'A':
                log(f'matched copies excluded (original duplicate or not cropped): {int(orphan.sum())}')
            # the matched reals now carry the DFF (diffusion) domain label; the originals are test-only
            f.loc[f.source == 'dff_wiki_matched', 'domain'] = 1
            f.loc[f.source == 'dff_wiki', 'domain'] = -1
        f['test_set'] = np.where(f.role == 'test', f.dataset + ':' + f.generator, '')
        # CelebDF videos outside the official list can land in test because they share frames (same
        # target clip) with an official test video. Label them so results on the official 518 videos
        # stay comparable with published numbers.
        linked = (f.dataset == 'CelebDF') & (f.role == 'test') & (f.celebdf_official_test.astype(str) != '1')
        f.loc[linked, 'test_set'] = f.loc[linked, 'test_set'] + ':linked'
        # Domain labels only matter for training rows; test-only generators keep d=-1.
        x = df[ext].copy()
        x['group'], x['gsplit'], x['role'] = 'ext:' + x.uid, 'test', 'test'
        x['test_set'] = x.dataset + ':' + x.generator
        f = pd.concat([f, x], ignore_index=True)
        d = os.path.join(out_root, f'fold_{fold}')
        os.makedirs(d, exist_ok=True)
        for s in ('train', 'val', 'test'):
            f.loc[f.role == s, cols].to_csv(os.path.join(d, f'{s}.csv'), index=False)
        log(f'fold {fold}: train={int((f.role == "train").sum())} val={int((f.role == "val").sum())} '
            f'test={int((f.role == "test").sum())} excluded={int((f.role == "excluded").sum())}')
        f['fold'] = fold
        long_rows.append(f[['fold', 'uid', 'role', 'test_set', 'group']])
    pd.concat(long_rows).to_csv(os.path.join(out_root, 'split_manifest_long.csv'), index=False)
    with open(os.path.join(out_root, 'protocol.json'), 'w') as fh:
        json.dump(dict(folds=FOLDS, test_only_sources=sorted(test_only_src), dff_real=a.dff_real,
                       ratios=a.ratios, seed=a.seed, celebdf_identity=a.celebdf_identity), fh, indent=2)


# ----------------------------------------------------------------------------
# Stage 5: audit
# ----------------------------------------------------------------------------
def md_table(df):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def overlap_report(fold_df, celebdf_identity_mode, near_pairs=None):
    """Count keys shared between splits. Returns (rows, hard_failures)."""
    rows, failures = [], 0
    keys = {
        'group': lambda s: s.group,
        'video_id': lambda s: s.video_id[s.video_id != ''],
        'identity': lambda s: s.identities[s.identities != ''].str.split(';').explode(),
        'source photo (rel_key, DFF)': lambda s: s.rel_key[(s.dataset == 'DFF') & (s.rel_key != '')],
        'raw file md5': lambda s: s.md5_raw[s.md5_raw != ''],
        'crop md5': lambda s: s.crop_md5[s.crop_md5 != ''],
    }
    by = {s: fold_df[fold_df.role == s] for s in ('train', 'val', 'test')}
    role_of = dict(zip(fold_df.uid, fold_df.role))
    near = near_pairs if near_pairs is not None else pd.DataFrame(columns=['uid_a', 'uid_b'])
    by_noext = {s: v[~v.dataset.str.startswith('ext_')] for s, v in by.items()}
    for name, fn in keys.items():
        for a_, b_ in (('train', 'test'), ('val', 'test'), ('train', 'val')):
            ka, kb = set(fn(by_noext[a_])), set(fn(by_noext[b_]))
            n = len(ka & kb)
            soft = name == 'identity' and celebdf_identity_mode == 'report'
            if n and soft:
                # Only CelebDF identities may overlap in report mode; DFF identities never.
                cdf_only = all(k.startswith('cdf:') for k in ka & kb)
                status = 'WARN (CelebDF identities, report mode)' if cdf_only else 'FAIL'
                failures += 0 if cdf_only else 1
            else:
                status = 'ok' if n == 0 else 'FAIL'
                failures += int(n > 0)
            rows.append(dict(key=name, splits=f'{a_} vs {b_}', shared=n, status=status))
    for a_, b_ in (('train', 'test'), ('val', 'test'), ('train', 'val')):
        n = sum(1 for r in near.itertuples(index=False)
                if {role_of.get(r.uid_a), role_of.get(r.uid_b)} == {a_, b_})
        failures += int(n > 0)
        rows.append(dict(key='near-duplicate pair', splits=f'{a_} vs {b_}', shared=n,
                         status='ok' if n == 0 else 'FAIL'))
    return pd.DataFrame(rows), failures


def stage_audit(a):
    cfg = json.load(open(os.path.join(a.out, 'config.json')))
    cdf_mode = a.celebdf_identity or cfg.get('celebdf_identity', 'report')
    df = load_eligible(a.out)
    split_dir = os.path.join(a.out, a.split_name)
    mf = pd.read_csv(os.path.join(split_dir, 'split_manifest_long.csv'), keep_default_na=False)
    proto = json.load(open(os.path.join(split_dir, 'protocol.json')))
    test_only_src = set(proto.get('test_only_sources', ['dff_insight', 'dff_wiki_matched']))
    real_src = 'dff_wiki_matched' if proto.get('dff_real') == 'matched' else 'dff_wiki'
    rep = os.path.join(a.out, 'reports' if a.split_name == 'splits' else f'reports_{a.split_name}')
    os.makedirs(rep, exist_ok=True)
    md = ['# Phase 1 data audit', '',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")} from `{os.path.abspath(a.out)}`.', '']

    # --- 1. Sample accounting (Comment 2) ---
    unit = df.apply(lambda r: r.video_id if r.video_id else r.raw_path, axis=1)
    acc = df.assign(unit=unit).groupby('source').agg(
        dataset=('dataset', 'first'), generator=('generator', 'first'), label=('label', 'first'),
        raw_files=('unit', 'nunique'), sampled=('uid', 'count'),
        mtcnn_ok=('mtcnn_ok', lambda s: int((s == 1).sum())),
        mtcnn_fail=('mtcnn_ok', lambda s: int((s == 0).sum())),
        exact_dups_removed=('exact_dup_of', lambda s: int((s != '').sum())),
        usable=('eligible', 'sum'))
    for fold in sorted(mf.fold.unique()):
        m = mf[mf.fold == fold].merge(df[['uid', 'source']], on='uid')
        piv = m.pivot_table(index='source', columns='role', values='uid', aggfunc='count', fill_value=0)
        for role in ('train', 'val', 'test', 'excluded'):
            acc[f'{fold}:{role}'] = piv[role] if role in piv else 0
    acc = acc.fillna(0)
    for c in acc.columns:
        if c not in ('dataset', 'generator'):
            acc[c] = acc[c].astype(int)
    acc = acc.reset_index()
    total = acc.select_dtypes('number').sum()
    tot_row = {c: (int(total[c]) if c in total else '') for c in acc.columns}
    tot_row['source'] = 'TOTAL'
    tot_row['label'] = ''
    acc = pd.concat([acc, pd.DataFrame([tot_row])], ignore_index=True)
    acc.to_csv(os.path.join(rep, 'accounting.csv'), index=False)
    md += ['## 1. Sample accounting (R2 Comment 2)', '',
           '`raw_files` counts files (or videos for CelebDF); `sampled` counts images/frames taken from them. '
           'Every later column is computed from the manifest, so the rows add up by construction: '
           '`sampled = mtcnn_ok + mtcnn_fail`, `usable = mtcnn_ok - exact_dups_removed`, and for each fold '
           '`train + val + test + excluded = usable`. `excluded` rows are held-out-generator images whose '
           'person or source photo was assigned to train/val, so they cannot be used as unseen test data.', '',
           md_table(acc), '']
    bad = []
    for rd in acc[acc.source != 'TOTAL'].to_dict('records'):
        if rd['sampled'] != rd['mtcnn_ok'] + rd['mtcnn_fail']:
            bad.append(f'{rd["source"]}: sampled != ok + fail')
        if rd['usable'] != rd['mtcnn_ok'] - rd['exact_dups_removed']:
            bad.append(f'{rd["source"]}: usable mismatch')
        for fold in sorted(mf.fold.unique()):
            if sum(rd[f'{fold}:{x}'] for x in ('train', 'val', 'test', 'excluded')) != rd['usable']:
                bad.append(f'{rd["source"]}: fold {fold} rows != usable')
    md += ['Arithmetic check: ' + ('**all rows add up**' if not bad else '**FAILED** ' + '; '.join(bad)), '']

    # --- 2. Leakage (Comments 3 and 10) ---
    md += ['## 2. Leakage checks (R2 Comments 3 and 10)', '',
           'Shared keys between splits, after removing exact duplicates. `group` is the union of every '
           'link below plus near-duplicates, so a zero there implies zero for each key. Any `FAIL` means '
           'the split is not usable.', '']
    total_fail = 0
    np_path = os.path.join(a.out, 'near_dup_pairs.csv')
    near_pairs = pd.read_csv(np_path) if os.path.exists(np_path) else None
    for fold in sorted(mf.fold.unique()):
        m = mf[mf.fold == fold].drop(columns=['group']).merge(df, on='uid')
        m = m.merge(mf[mf.fold == fold][['uid', 'group']], on='uid')
        if real_src == 'dff_wiki_matched':   # domain labels as written to this split's CSVs
            m.loc[m.source == 'dff_wiki_matched', 'domain'] = 1
            m.loc[m.source == 'dff_wiki', 'domain'] = -1
        rpt, nfail = overlap_report(m, cdf_mode, near_pairs)
        total_fail += nfail
        tr = m[m.role.isin(['train', 'val'])]
        spec = FOLDS[fold]
        leak_gen = (tr.generator.isin(spec['heldout_gens']) | tr.source.isin(test_only_src)).sum()
        total_fail += int(leak_gen > 0)
        md += [f'### Fold {fold}: train on {", ".join(spec["train_gens"])}; held out: '
               f'{", ".join(spec["heldout_gens"])}', '', md_table(rpt), '',
               f'Held-out or test-only generator rows in train/val: **{leak_gen}** (must be 0).', '']
        dom = m[m.role != 'excluded'].groupby(['dataset', 'generator', 'family', 'label', 'domain', 'role']) \
            .size().unstack('role', fill_value=0).reset_index()
        for c in ('train', 'val', 'test'):
            if c not in dom:
                dom[c] = 0
        dom['DAT label used'] = np.where((dom.train > 0) & (dom.domain >= 0), 'yes (d=' + dom.domain.astype(str) + ')',
                                         'no (test only)')
        md += [f'Domain-label table, fold {fold} (R2 Comment 7):', '',
               md_table(dom[['dataset', 'generator', 'family', 'label', 'domain', 'train', 'val', 'test',
                             'DAT label used']]), '']
        if 'CelebDF' in set(m.dataset) and cdf_mode == 'report':
            te_ids = set(m[(m.role == 'test') & (m.dataset == 'CelebDF')].identities.str.split(';').explode()) - {''}
            trv = m[(m.role.isin(['train', 'val'])) & (m.dataset == 'CelebDF')]
            shared = trv.identities.str.split(';').apply(lambda ids: bool(set(ids) & te_ids))
            md += [f'CelebDF identity overlap (report mode): {int(shared.sum())} of {len(trv)} train/val frames '
                   f'({trv[shared].video_id.nunique()} videos) show a person who also appears in the official '
                   f'test list. Videos never cross splits. A fully identity-disjoint CelebDF split is usually not '
                   f'possible: swap pairs connect most of the 59 celebrities into one component, so '
                   f'`--celebdf-identity link` tends to move all of CelebDF into test (the group-size guard '
                   f'stops the run if so). Video-disjoint with the official list is the standard protocol; '
                   f'report this overlap in the paper.', '']
    md += [f'**Overall leakage verdict: {"PASS" if total_fail == 0 else f"FAIL ({total_fail} problems)"}**', '']

    # --- 3. File format / compression shortcuts ---
    fmt = df[df.source != 'dff_wiki_matched'].groupby(['dataset', 'source', 'label']).agg(
        n=('uid', 'count'),
        formats=('orig_format', lambda s: ', '.join(f'{k} {v}' for k, v in Counter(s).most_common(3))),
        median_w=('orig_w', 'median'), median_h=('orig_h', 'median'),
        jpeg_q_mean=('jpeg_q', lambda s: round(pd.to_numeric(s, errors='coerce').mean(), 1))).reset_index()
    fmt.to_csv(os.path.join(rep, 'formats.csv'), index=False)
    warn = []
    for ds, sub in fmt.groupby('dataset'):
        if sub.label.nunique() < 2:
            continue
        f0 = set(sub[sub.label == 0].formats.str.split(' ').str[0])
        f1 = set(sub[sub.label == 1].formats.str.split(' ').str[0])
        if f0 != f1:
            warn.append(f'{ds}: real files are {sorted(f0)}, fake files are {sorted(f1)}')
        q = sub.groupby('label').jpeg_q_mean.mean()
        if q.notna().all() and abs(q.iloc[0] - q.iloc[-1]) > 5:
            warn.append(f'{ds}: mean JPEG quality real {q.iloc[0]:.0f} vs fake {q.iloc[-1]:.0f}')
        s = sub.groupby('label')[['median_w', 'median_h']].mean()
        if len(s) == 2 and not np.allclose(s.iloc[0].values, s.iloc[1].values, rtol=0.1):
            warn.append(f'{ds}: median size real {s.iloc[0].values.tolist()} vs fake {s.iloc[1].values.tolist()}')
    md += ['## 3. File format and size by class', '',
           'A frequency-domain detector can separate classes by compression history alone, so real and fake '
           'images within one dataset should share format, JPEG quality and size. All crops in this rebuild '
           'are saved as lossless PNG, which removes differences introduced by our own pipeline, but '
           'differences already present in the source files survive cropping.', '', md_table(fmt), '',
           ('**Warnings:**\n\n' + '\n'.join(f'- {w}' for w in warn)) if warn else 'No format, quality or size '
           'mismatches between classes.', '']
    md += metadata_baseline(df, mf, rep, real_src)
    with open(os.path.join(rep, 'audit_report.md'), 'w') as fh:
        fh.write('\n'.join(md))
    log(f'audit written to {rep}/audit_report.md  (leakage: {"PASS" if total_fail == 0 else "FAIL"})')
    for w in warn:
        log('  FORMAT WARNING: ' + w)
    if total_fail:
        sys.exit(1)


def matched_quality_estimate():
    """What estimate_jpeg_quality() reads from a file saved like the matched copies (PIL, quality 75)."""
    import io
    buf = io.BytesIO()
    Image.new('RGB', (16, 16)).save(buf, format='JPEG', quality=MATCHED_QUALITY)
    buf.seek(0)
    with Image.open(buf) as im:
        return estimate_jpeg_quality(im)


def metadata_features(d):
    q = pd.to_numeric(d.jpeg_q, errors='coerce')
    if 'source' in d:   # matched copies were recorded as 75.0; use the value measured on a real q75 file
        q = q.where(d.source != 'dff_wiki_matched', matched_quality_estimate())
    fmt = d.orig_format.astype(str).str.upper()
    return np.column_stack([q.fillna(-1), pd.to_numeric(d.orig_w, errors='coerce'),
                            pd.to_numeric(d.orig_h, errors='coerce'),
                            fmt.str.startswith('JPEG'), fmt.str.startswith('PNG')]).astype(float)


def metadata_baseline(df, mf, rep, real_src='dff_wiki'):
    """How well does file metadata alone (JPEG quality, width, height, format) separate DFF reals from
    fakes? A high AUC means the dataset carries a non-face shortcut that the detector could also use."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rows = []
    for fold in sorted(mf.fold.unique()):
        spec = FOLDS[fold]
        m = mf[mf.fold == fold].merge(df.drop(columns=['test_set'], errors='ignore'), on='uid')
        other_real = 'dff_wiki' if real_src == 'dff_wiki_matched' else 'dff_wiki_matched'
        m = m[(m.dataset == 'DFF') & (m.source != other_real)]
        tr = m[(m.role == 'train') & ((m.label == 0) | m.generator.isin(spec['train_gens']))]
        if tr.label.nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(metadata_features(tr), tr.label)
        te_real = m[(m.role == 'test') & (m.source == real_src)]
        for gen in spec['train_gens'] + spec['heldout_gens'] + ['insightface']:
            te = pd.concat([te_real, m[(m.role == 'test') & (m.generator == gen)]])
            if te.label.nunique() < 2:
                continue
            p = clf.predict_proba(metadata_features(te))[:, 1]
            kind = 'held-out' if gen in spec['heldout_gens'] else ('seen' if gen in spec['train_gens'] else 'test-only')
            rows.append(dict(fold=fold, fakes=f'{gen} ({kind})', n_real=len(te_real), n_fake=int(te.label.sum()),
                             metadata_AUC=round(roc_auc_score(te.label, p), 4)))
    # A one-line rule: "512x512 JPEG at quality ~75 => fake"
    d = df[(df.dataset == 'DFF') & (df.source != 'dff_wiki_matched') & (df.mtcnn_ok == 1)]
    q = pd.to_numeric(d.jpeg_q, errors='coerce')
    rule = (q.between(74, 76)) & (pd.to_numeric(d.orig_w) == 512) & (pd.to_numeric(d.orig_h) == 512)
    rule_txt = (f'Rule "512x512 JPEG with quality 74-76 means fake": flags {rule[d.label == 1].mean():.1%} of DFF '
                f'fakes and {rule[d.label == 0].mean():.1%} of DFF reals.')
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(rep, 'metadata_baseline.csv'), index=False)
    log('metadata baseline: ' + rule_txt)
    return ['## 4. Metadata-only shortcut baseline (DFF)', '',
            'A gradient-boosted classifier trained on the fold\'s training rows using only file metadata '
            '(JPEG quality, width, height, format), never pixels, and tested on the test rows. An AUC near 1 '
            'means real and fake DFF files can be told apart without looking at a face, so a detector\'s DFF '
            'result needs the matched-real check (`DFF:real_matched` test set) before it can be read as face '
            'forgery detection.', '', md_table(r) if len(r) else '(not enough data)', '', rule_txt, '']


# ----------------------------------------------------------------------------
# Stage 6: audit-original — how leaky was the first-edition split?
# ----------------------------------------------------------------------------
def stage_audit_original(a):
    """Re-create df-1-first-edition cell 9 (stratified 85/15 by (class, domain), random.seed(42)) on the
    same pool of images and measure what the validation set shared with the training set.

    The original lists came from os.walk order on Kaggle, which is not recorded, so the exact
    membership cannot be reproduced; we use sorted paths and repeat over several seeds. Overlap rates
    are properties of an image-level random split, so they barely move between seeds."""
    df = load_merged(a.out)
    hf = df[df.dataset == 'HardFake']                       # originally used uncropped
    dff = df[(df.source.isin(['dff_wiki', 'dff_text2img', 'dff_inpainting'])) & (df.mtcnn_ok == 1)]
    pool = pd.concat([hf, dff], ignore_index=True)
    pool = pool.sort_values(['dataset', 'label', 'raw_path']).reset_index(drop=True)
    near = pd.read_csv(os.path.join(a.out, 'near_dup_pairs.csv')) if os.path.exists(
        os.path.join(a.out, 'near_dup_pairs.csv')) else pd.DataFrame(columns=['uid_a', 'uid_b'])
    res = []
    for seed in [42] + list(range(1, a.repeats)):
        groups = {}
        order = [('HardFake', 0), ('HardFake', 1), ('DFF', 0), ('DFF', 1)]   # notebook insertion order
        for key in order:
            groups[key] = pool[(pool.dataset == key[0]) & (pool.label == key[1])].index.tolist()
        random.seed(seed)
        split = {}
        for key in order:
            items = groups[key]
            random.shuffle(items)
            cut = int(0.85 * len(items))
            for i in items[:cut]:
                split[i] = 'train'
            for i in items[cut:]:
                split[i] = 'val'
        p = pool.assign(split=pool.index.map(split))
        tr, va = p[p.split == 'train'], p[p.split == 'val']
        va_dff = va[va.dataset == 'DFF']
        ids_tr = set(tr.identities) - {''}
        keys_tr = set(tr[tr.dataset == 'DFF'].rel_key)
        va_fake = va_dff[va_dff.label == 1]
        va_real = va_dff[va_dff.label == 0]
        tr_uids, va_uids = set(tr.uid), set(va.uid)
        cross_near = sum(1 for r in near.itertuples(index=False)
                         if (r.uid_a in tr_uids and r.uid_b in va_uids) or (r.uid_b in tr_uids and r.uid_a in va_uids))
        exact = len(set(tr.md5_raw) & set(va.md5_raw) - {''})
        res.append(dict(
            seed=seed, total=len(p), train=len(tr), val=len(va),
            val_dff_identity_in_train=round(100 * va_dff.identities.isin(ids_tr).mean(), 2),
            val_fake_same_photo_in_train=round(100 * va_fake.rel_key.isin(keys_tr).mean(), 2),
            val_real_same_photo_in_train=round(100 * va_real.rel_key.isin(keys_tr).mean(), 2),
            exact_dup_pairs_across=exact, near_dup_pairs_across=cross_near))
    r = pd.DataFrame(res)
    rep = os.path.join(a.out, 'reports')
    os.makedirs(rep, exist_ok=True)
    r.to_csv(os.path.join(rep, 'audit_original.csv'), index=False)
    md = ['# Leakage in the first-edition split', '',
          'Re-creates the stratified 85/15 split from df-1-first-edition (cell 9: groups by (class, domain), '
          '`random.seed(42)`, image-level shuffle) on the same images: HardFake uncropped, DFF wiki/text2img/'
          'inpainting with a successful MTCNN crop. The first edition logged 87,530 images (74,399 train / '
          '13,131 val).', '',
          'Columns, as % of DFF validation images: `val_dff_identity_in_train` = the same person appears in '
          'training; `val_fake_same_photo_in_train` = a validation fake was generated from a real photo whose '
          'own image or other fake version is in training; `val_real_same_photo_in_train` = a validation real '
          'has a fake made from it in training.', '', md_table(r), '']
    with open(os.path.join(rep, 'audit_original.md'), 'w') as fh:
        fh.write('\n'.join(md))
    log('audit-original:\n' + r.to_string(index=False))


# ----------------------------------------------------------------------------
# Phase 2 helper: dataset that reads the split CSVs
# ----------------------------------------------------------------------------
class ManifestDataset:
    """torch Dataset over a split CSV. Returns (image, label, domain, index).

    Usage in the training notebook:
        from cdg_phase1 import ManifestDataset
        ds = ManifestDataset('/kaggle/working/phase1/splits/fold_A/train.csv',
                             root='/kaggle/working/phase1', transform=train_transform)
    """

    def __init__(self, csv_path, root, transform=None):
        self.df = pd.read_csv(csv_path, keep_default_na=False)
        self.root, self.transform = root, transform
        self.paths = [os.path.join(root, p) for p in self.df.crop_path]
        self.labels = self.df.label.astype(int).tolist()
        self.domains = self.df.domain.astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i], self.domains[i], i


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)

    def common(p):
        p.add_argument('--out', required=True, help='working directory for this rebuild')
        p.add_argument('--seed', type=int, default=2026)
        return p

    p = common(sub.add_parser('scan'))
    p.add_argument('--hardfake-root', help='folder with real/ and fake/')
    p.add_argument('--dff-root', help='folder with wiki/, text2img/, inpainting/ (and optionally insight/)')
    p.add_argument('--dff-matched-real', action='store_true',
                   help='also crop wiki reals after 512x512 + JPEG q75 (test-only shortcut diagnostic)')
    p.add_argument('--dff-insight-sample', type=int, default=0,
                   help='keep a random N of the InsightFace images (test-only); 0 = all')
    p.add_argument('--celebdf-root', help='folder with Celeb-real/, Celeb-synthesis/, YouTube-real/')
    p.add_argument('--celebdf-test-list', help='List_of_testing_videos.txt (default: inside --celebdf-root)')
    p.add_argument('--celebdf-frames-real', type=int, default=10)
    p.add_argument('--celebdf-frames-fake', type=int, default=2,
                   help='fewer frames per fake video, more fake videos: roughly balances classes')
    p.add_argument('--celebdf-max-fake-train-videos', type=int, default=0, help='0 = all')
    p.add_argument('--external', action='append', metavar='NAME=PATH',
                   help='test-only set with PATH/real and/or PATH/fake (repeatable)')
    p.set_defaults(func=stage_scan)

    p = common(sub.add_parser('add-matched'))
    p.set_defaults(func=stage_add_matched)

    p = common(sub.add_parser('crop'))
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int, default=0, help='crop at most N rows this call (for testing)')
    p.set_defaults(func=stage_crop)

    p = common(sub.add_parser('dedup'))
    p.add_argument('--near-dup-hamming', type=int, default=6, help='candidate threshold on 64-bit dHash')
    p.add_argument('--near-dup-mad', type=float, default=6.0,
                   help='confirmation threshold: mean abs pixel difference (0-255) on 32x32 grayscale')
    p.set_defaults(func=stage_dedup)

    p = common(sub.add_parser('split'))
    p.add_argument('--dff-real', choices=['original', 'matched'], default='original',
                   help='which DFF reals train the model; matched = stored like the fakes (compression control)')
    p.add_argument('--split-name', default='splits', help='output folder for this split')
    p.add_argument('--ratios', default='0.7,0.1,0.2', help='train,val,test share of groups')
    p.add_argument('--celebdf-identity', choices=['report', 'link'], default='report',
                   help='report (default): video-disjoint, identity overlap reported; link: also '
                        'identity-disjoint (usually collapses CelebDF into one group)')
    p.add_argument('--max-group-share', type=float, default=0.05,
                   help='abort if one linked group exceeds this share of all images')
    p.set_defaults(func=stage_split)

    p = common(sub.add_parser('audit'))
    p.add_argument('--split-name', default='splits')
    p.add_argument('--celebdf-identity', choices=['report', 'link'], default=None)
    p.set_defaults(func=stage_audit)

    p = common(sub.add_parser('audit-original'))
    p.add_argument('--repeats', type=int, default=5, help='seed 42 plus this many-1 other seeds')
    p.set_defaults(func=stage_audit_original)

    a = ap.parse_args(argv)
    if a.stage == 'split':
        # remember the CelebDF mode for the audit stage
        cfg_path = os.path.join(a.out, 'config.json')
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        cfg['celebdf_identity'] = a.celebdf_identity
        json.dump(cfg, open(cfg_path, 'w'), indent=2)
    a.func(a)


if __name__ == '__main__':
    main()
