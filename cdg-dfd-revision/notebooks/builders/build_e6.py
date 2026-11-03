"""Builds phase2_E6_rgb_followup.ipynb: test-only follow-up to E2. The RGB (and DCT+band, no DAT) checkpoints of E2 are
scored on (1) the 14 ArtiFact generators, as in E1c, and (2) corrupted test images, as in E5. Also: Celeb-DF audit
lines and T4 GPU latency (E9). No training."""
import json
import re

b5 = open('build_e5.py').read()
FIND_PHASE1 = re.search(r"FIND_PHASE1 = '''(.*?)'''\n", open('build_phase2_notebooks.py').read(), re.S).group(1)
e1b = open('build_e1b.py').read()
e3b = open('build_e3.py').read()
RUN_PARALLEL = re.search(r"(KEEP = .*?)\nSPLITS = ", e1b, re.S).group(1)
SH = re.search(r"(def sh\(cmd\):.*?raise RuntimeError\(f'failed \(exit \{p.returncode\}\):' \+ chr\(10\) \+ ''.join\(tail\)\)\n)",
               e3b, re.S).group(1)
train = open('cdg_train.py').read()
phase1 = open('cdg_phase1.py').read()
ext = open('cdg_external.py').read()
assert 'test_corrupt' in train and 'face = None' in ext


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


setup = '''
!pip install -q facenet-pytorch --no-deps
import os, re, sys, json, glob, shutil, time, subprocess, zipfile
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown
T0 = time.time()
''' + FIND_PHASE1 + '''
def tmp_is_ram():
    for line in open('/proc/mounts'):
        parts = line.split()
        if len(parts) > 2 and parts[1] == '/tmp':
            return parts[2] == 'tmpfs'
    return False

UNPACK = '/kaggle/working/_data' if tmp_is_ram() else '/tmp/phase1_input'
DATA = find_phase1(extract_to=UNPACK)
if not DATA:
    show_inputs()
assert DATA, 'No Phase 1 output found: add the oct18 notebook output as an input'
print('Phase 1 data:', DATA, flush=True)
NGPU = torch.cuda.device_count()
print('GPUs:', NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)], flush=True)
assert NGPU >= 1, 'No GPU: set Accelerator to GPU T4 x2'
RUNS, LOGS = '/kaggle/working/runs', '/kaggle/working/logs'
os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
CK = '/kaggle/working/_ckpt'

# E2 checkpoints: runs/e2_<cfg>_<fold>_s<seed>/{best.pth,config.json,summary.json}, either as files, inside
# e2_checkpoints.zip, or inside the notebook's _output_.zip (a zip holding that zip)
WANT = re.compile(r'(e2_(?:rgb|band)_[AB]_s\\d)/(best\\.pth|config\\.json|summary\\.json)$')
def take(zf):
    n_found = 0
    for n in zf.namelist():
        m = WANT.search(n)
        if m:
            os.makedirs(f'{CK}/{m.group(1)}', exist_ok=True)
            with zf.open(n) as fi, open(f'{CK}/{m.group(1)}/{m.group(2)}', 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            n_found += 1
        elif n.endswith('e2_checkpoints.zip'):
            tmp = '/kaggle/working/_e2ck.zip'
            with zf.open(n) as fi, open(tmp, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            with zipfile.ZipFile(tmp) as inner:
                n_found += take(inner)
            os.remove(tmp)
    return n_found
for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    if 'crops' in root or 'metadata.csv' in files or len(files) > 5000 or root.count(os.sep) > 9:
        dirs[:] = []; continue
    m = re.search(r'(e2_(?:rgb|band)_[AB]_s\\d)$', root)
    if m and 'best.pth' in files:
        shutil.copytree(root, f'{CK}/{m.group(1)}', dirs_exist_ok=True)
    for f in files:
        if f.endswith('.zip') and 'oct18' not in root:
            try:
                with zipfile.ZipFile(os.path.join(root, f)) as zf:
                    print(f, 'checkpoint files taken:', take(zf), flush=True)
            except zipfile.BadZipFile:
                pass
CKPT = {}
for d in sorted(glob.glob(f'{CK}/e2_*')):
    if os.path.exists(f'{d}/best.pth') and os.path.exists(f'{d}/config.json'):
        cfg_, fold_, seed_ = re.match(r'e2_(rgb|band)_([AB])_s(\\d)$', os.path.basename(d)).groups()
        CKPT[(cfg_, fold_, int(seed_))] = d
print('E2 checkpoints:', sorted(CKPT), flush=True)
assert any(k[0] == 'rgb' for k in CKPT), 'No E2 RGB checkpoints: add the phase2-e2-ablation notebook output as an input'

''' + SH.replace("{OUT}/log.txt", "{LOGS}/sh_log.txt") + '''
''' + RUN_PARALLEL + '''
FLAGS = {'rgb': '--input rgb --spectral none --no-dat', 'band': '--input dct --spectral band --no-dat'}
CONDITIONS = ['none', 'jpeg:70', 'jpeg:30', 'resize:0.5', 'blur:1', 'noise:10', 'resize:0.5+jpeg:70']
tag = lambda c: c.replace(':', '').replace('+', '_').replace('.', 'p')

def eval_cmd(cond, cfg, fold, seed):
    corrupt = '' if cond == 'none' else f"--test-corrupt '{cond}'"
    return (f'python cdg_train.py --data {DATA} --splits splits_matched --fold {fold} --seed {seed} {FLAGS[cfg]} '
            f'--epochs 0 --init {CKPT[(cfg, fold, seed)]}/best.pth {corrupt} --out {RUNS}/e6_{cfg}_{tag(cond)}_{fold}_s{seed}')

def run_condition(cond, cfgs=('rgb',)):
    jobs = [(f'e6_{c}_{tag(cond)}_{f}_s{s}', eval_cmd(cond, c, f, s)) for (c, f, s) in sorted(CKPT) if c in cfgs
            and not os.path.exists(f'{RUNS}/e6_{c}_{tag(cond)}_{f}_s{s}/test_metrics.csv')]
    for i in range(0, len(jobs), NGPU):
        run_parallel([(n, c, k % NGPU) for k, (n, c) in enumerate(jobs[i:i + NGPU])])
    print(f'{cond}: done | elapsed {(time.time() - T0) / 3600:.2f} h', flush=True)
'''

audit = '''
rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
assert '**Overall leakage verdict: PASS**' in rep, 'matched split: leakage audit failed'
print('leakage audit: PASS')
os.makedirs('/kaggle/working/e6_results', exist_ok=True)
shutil.copy(f'{DATA}/reports_splits_matched/audit_report.md', '/kaggle/working/e6_results/audit_report_matched.md')
# Celeb-DF: how identities are kept apart, and how many there are per split (does RGB 0.985 come from identity overlap?)
print('\\n'.join(l for l in rep.splitlines() if re.search('celeb', l, re.I))[:6000])
for fold in 'AB':
    parts = {}
    for sp in ('train', 'val', 'test'):
        t = pd.read_csv(f'{DATA}/splits_matched/fold_{fold}/{sp}.csv', keep_default_na=False)
        parts[sp] = t[t.astype(str).apply(lambda r: 'celeb' in ' '.join(r).lower(), axis=1)]
    cols = [c for c in parts['test'].columns if re.search('ident|person|group|subject|video|src', c, re.I)]
    print(f'fold {fold}: Celeb-DF rows', {k: len(v) for k, v in parts.items()}, '| id-like columns:', cols)
    for c in cols:
        tr, te = set(parts['train'][c]), set(parts['test'][c])
        print(f'   {c}: train {len(tr)} unique, test {len(te)} unique, shared {len(tr & te)}')
'''

latency = '''
# E9 on the Kaggle GPU: inference latency and throughput per configuration (random weights; fp16 autocast as in testing)
sys.path.insert(0, '.')
import cdg_train as T
dev = torch.device('cuda:0')
rows = []
for name, kw in {'rgb': dict(spectral='none', input_mode='rgb'), 'dct': dict(spectral='none', input_mode='dct'),
                 'dct+se': dict(spectral='none', input_mode='dct', channel_attn='se'),
                 'dct+band (proposed)': dict(spectral='band', input_mode='dct')}.items():
    m = T.Detector(pretrained=False, **kw).to(dev).eval().to(memory_format=torch.channels_last)
    for bs in (1, 64):
        x = torch.rand(bs, 3, 224, 224, device=dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for _ in range(10): m(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter(); n = 50 if bs == 1 else 20
            for _ in range(n): m(x)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / n * 1000
        rows.append(dict(config=name, batch=bs, ms_per_batch=round(ms, 2), images_per_s=round(bs * 1000 / ms, 1),
                         peak_mem_MB=round(torch.cuda.max_memory_allocated() / 2**20)))
    del m; torch.cuda.empty_cache()
lat = pd.DataFrame(rows); lat.to_csv('/kaggle/working/e6_results/E9_gpu_latency.csv', index=False); display(lat)
'''

artifact_check = '''
from cdg_external import find_artifact
metas = find_artifact(['/kaggle/input'])
print(len(metas), 'ArtiFact folders found')
assert metas, 'ArtiFact not found: add the dataset awsaf49/artifact-dataset as an input'
OUT_X = '/kaggle/working/external'
os.makedirs(OUT_X, exist_ok=True)
'''

summary = '''
OUT = '/kaggle/working/e6_results'
# (a) corruption, RGB checkpoints
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/e6_*/test_metrics.csv')):
    name = os.path.basename(os.path.dirname(mpath))
    m = re.match(r'e6_(rgb|band)_(.+)_([AB])_s(\\d)$', name)
    t = pd.read_csv(mpath)
    t['config'], t['condition'] = m.group(1), {tag(c): c for c in CONDITIONS}[m.group(2)]
    t['fold'], t['seed'] = m.group(3), int(m.group(4))
    rows.append(t)
allm = pd.concat(rows, ignore_index=True); allm.to_csv(f'{OUT}/E6_corrupt_all_metrics.csv', index=False)
UNSEEN = {'A': 'DFF:sd15_text2img vs DFF:real_matched', 'B': 'DFF:sd_inpainting vs DFF:real_matched'}
main = allm[[c == UNSEEN[f] for c, f in zip(allm.comparison, allm.fold)]].assign(comparison='unseen DFF generator')
other = allm[allm.comparison.isin(['CelebDF:celebdf_swap vs CelebDF:real', 'HardFake:hardfake_gan vs HardFake:real'])]
flag = allm[allm.comparison.isin(['DFF:real_matched flagged as fake'])].assign(AUC=lambda d: d.FPR)
agg = (pd.concat([main, other, flag]).groupby(['comparison', 'fold', 'config', 'condition'])
         .agg(n=('AUC', 'count'), mean=('AUC', 'mean'), sd=('AUC', 'std')).round(4).reset_index())
agg.to_csv(f'{OUT}/E6_corrupt_summary.csv', index=False)
wide = agg.pivot_table(index='condition', columns=['comparison', 'fold'], values='mean').round(3)
wide = wide.reindex([c for c in CONDITIONS if c in wide.index]); wide.to_csv(f'{OUT}/E6_corrupt_wide.csv')
display(wide)
# (b) ArtiFact: copy the external report
for f in glob.glob(f'{OUT_X}/*.csv') + glob.glob(f'{OUT_X}/*.md'):
    shutil.copy(f, OUT)
print('saved:', sorted(os.listdir(OUT)), f'| total {(time.time() - T0) / 3600:.2f} h')
shutil.rmtree('/kaggle/working/_data', ignore_errors=True)
shutil.rmtree(CK, ignore_errors=True)
shutil.rmtree(f'{OUT_X}/raw', ignore_errors=True)
'''

cells = [
    md('''# CDG-DFD Phase 2, E6: does plain RGB input transfer better? (follow-up to E2)

E2 showed that, under the leakage-controlled protocol, an RGB EfficientNet-B4 beats the DCT pipeline on the held-out
SD generator (0.99/0.90 vs 0.92/0.79). This notebook checks whether that holds outside the training family and under
degradation. **Test only, no training.**

1. Leakage audit + Celeb-DF identity check (is RGB's Celeb-DF 0.985 an identity shortcut?)
2. E9: GPU latency and memory on the T4
3. ArtiFact (14 unseen generators, as E1c): E2 RGB vs E2 DCT+band checkpoints (both without DAT, 2 seeds x 2 folds)
4. Corrupted test images (as E5) for the RGB checkpoints: 7 conditions
5. Summary in `e6_results/`

**Inputs:** the **oct18** notebook output, the **phase2-e2-ablation** notebook output, the **awsaf49/artifact-dataset** dataset.
**Settings:** GPU **T4 x2**, Internet on.'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    code('%%writefile cdg_external.py\n' + ext),
    md('## 1. Leakage audit and Celeb-DF identities'),
    code(audit),
    md('## 2. GPU latency (E9)'),
    code(latency),
    md('## 3. ArtiFact: 14 unseen generators'),
    code(artifact_check),
    code("sh(f'python cdg_external.py artifact --out {OUT_X} --artifact-root /kaggle/input --n 800 --n-real 1500')"),
    code("sh(f'python cdg_external.py crop --out {OUT_X}')"),
    code("sh(f'python cdg_external.py score --out {OUT_X} --checkpoints {CK} --only e2_rgb e2_band')"),
    code("""
sh(f'python cdg_external.py report --out {OUT_X} --title "ArtiFact faces: E2 RGB vs DCT+band"')
display(Markdown(open(f'{OUT_X}/report.md').read()))
"""),
    md('## 4. Corrupted test images, RGB checkpoints (smoke test first)'),
    code("""
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
c, f, s = sorted(k for k in CKPT if k[0] == 'rgb')[0]
run_parallel([('smoke_e6', eval_cmd('resize:0.5+jpeg:70', c, f, s).replace(f'{RUNS}/e6_rgb_resize0p5_jpeg70_{f}_s{s}', SMOKE)
               + ' --limit-eval 300', 0)], retries=0, poll=10)
assert os.path.exists(f'{SMOKE}/test_metrics.csv'), 'smoke test: no metrics written'
print('SMOKE TEST PASSED'); shutil.rmtree(SMOKE, ignore_errors=True)
"""),
    code("""
for cond in CONDITIONS:
    run_condition(cond)
"""),
    md('## 5. Summary'),
    code(summary),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E6_rgb_followup.ipynb', 'w'), indent=1)
print('wrote phase2_E6_rgb_followup.ipynb')
