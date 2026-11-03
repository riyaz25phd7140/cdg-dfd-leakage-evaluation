"""Builds phase2_E5_robustness.ipynb: test-time corruption of the E1b checkpoints (R1 AA, R2 C11). No training."""
import json
import re

e2 = open('build_e2.py').read()
SETUP_HEAD = '''
import os, re, json, glob, shutil, time, subprocess, zipfile
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown
T0 = time.time()
'''
FIND_PHASE1 = re.search(r"FIND_PHASE1 = '''(.*?)'''\n", open('build_phase2_notebooks.py').read(), re.S).group(1)
e1b = open('build_e1b.py').read()
e3b = open('build_e3.py').read()
RUN_PARALLEL = re.search(r"(KEEP = .*?)\nSPLITS = ", e1b, re.S).group(1)
SH = re.search(r"(def sh\(cmd\):.*?raise RuntimeError\(f'failed \(exit \{p.returncode\}\):' \+ chr\(10\) \+ ''.join\(tail\)\)\n)",
               e3b, re.S).group(1)
train = open('cdg_train.py').read()
phase1 = open('cdg_phase1.py').read()
assert 'test_corrupt' in train


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


setup = SETUP_HEAD + FIND_PHASE1 + '''
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
print('Phase 1 data:', DATA)
NGPU = torch.cuda.device_count()
print('GPUs:', NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, 'No GPU: set Accelerator to GPU T4 x2'
RUNS, LOGS = '/kaggle/working/runs', '/kaggle/working/logs'
os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

# E1b checkpoints (matched-reals training, band SAM + DAT): any attached input holding m_band_<fold>_s<seed>/best.pth
CKPT = {}
for root, dirs, files in os.walk('/kaggle/input'):
    if root.count(os.sep) > 9 or 'crops' in root:
        dirs[:] = []; continue
    m = re.search(r'm_band_([AB])_s(\\d)$', root)
    if m and 'best.pth' in files:
        CKPT[(m.group(1), int(m.group(2)))] = os.path.join(root, 'best.pth')
if not CKPT:   # checkpoints saved inside a zip (a notebook output kept as one file)
    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
        if 'output' in z and os.path.getsize(z) > 2e9:
            continue   # the Phase 1 data zip
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                m = re.search(r'm_band_([AB])_s(\\d)/best\\.pth$', n)
                if m:
                    dst = f'/kaggle/working/_ckpt/m_band_{m.group(1)}_s{m.group(2)}'
                    os.makedirs(dst, exist_ok=True)
                    with zf.open(n) as fi, open(f'{dst}/best.pth', 'wb') as fo:
                        shutil.copyfileobj(fi, fo)
                    CKPT[(m.group(1), int(m.group(2)))] = f'{dst}/best.pth'
print('E1b checkpoints found:', sorted(CKPT))
assert CKPT, 'No E1b checkpoints: add the cdg-e1b-checkpoints dataset (or the oct-23-1 notebook output) as an input'

''' + SH + '''
''' + RUN_PARALLEL + '''
CONDITIONS = ['none', 'jpeg:90', 'jpeg:70', 'jpeg:50', 'jpeg:30', 'resize:0.75', 'resize:0.5', 'resize:0.25',
              'blur:1', 'blur:2', 'noise:5', 'noise:10', 'resize:0.5+jpeg:70']
tag = lambda c: c.replace(':', '').replace('+', '_').replace('.', 'p')

def eval_cmd(cond, fold, seed):
    corrupt = '' if cond == 'none' else f"--test-corrupt '{cond}'"
    return (f'python cdg_train.py --data {DATA} --splits splits_matched --fold {fold} --seed {seed} --spectral band '
            f'--epochs 0 --init {CKPT[(fold, seed)]} {corrupt} --out {RUNS}/e5_{tag(cond)}_{fold}_s{seed}')

def run_condition(cond):
    jobs = [(f'e5_{tag(cond)}_{f}_s{s}', eval_cmd(cond, f, s)) for (f, s) in sorted(CKPT)
            if not os.path.exists(f'{RUNS}/e5_{tag(cond)}_{f}_s{s}/test_metrics.csv')]
    for i in range(0, len(jobs), NGPU):   # one evaluation per GPU at a time
        run_parallel([(n, c, k % NGPU) for k, (n, c) in enumerate(jobs[i:i + NGPU])])
    print(f'{cond}: done | elapsed {(time.time() - T0) / 3600:.2f} h', flush=True)
'''

summary = '''
import re as _re
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/e5_*/test_metrics.csv')):
    name = os.path.basename(os.path.dirname(mpath))          # e5_jpeg50_A_s2
    m = _re.match(r'e5_(.+)_([AB])_s(\\d)$', name)
    cond_tag, fold, seed = m.group(1), m.group(2), int(m.group(3))
    t = pd.read_csv(mpath)
    t['condition'] = {tag(c): c for c in CONDITIONS}[cond_tag]
    t['fold'], t['seed'] = fold, seed
    rows.append(t)
allm = pd.concat(rows, ignore_index=True)
OUT = '/kaggle/working/e5_results'; os.makedirs(OUT, exist_ok=True)
allm.to_csv(f'{OUT}/E5_all_metrics.csv', index=False)
UNSEEN = {'A': 'DFF:sd15_text2img vs DFF:real_matched', 'B': 'DFF:sd_inpainting vs DFF:real_matched'}
main = allm[[c == UNSEEN[f] for c, f in zip(allm.comparison, allm.fold)]].assign(comparison='unseen DFF generator')
other = allm[allm.comparison.isin(['CelebDF:celebdf_swap vs CelebDF:real', 'HardFake:hardfake_gan vs HardFake:real'])]
flag = allm[allm.comparison.isin(['DFF:real_matched flagged as fake'])].assign(AUC=lambda d: d.FPR)
tab = pd.concat([main, other, flag])
agg = (tab.groupby(['comparison', 'fold', 'condition'])
          .agg(n=('AUC', 'count'), mean=('AUC', 'mean'), sd=('AUC', 'std')).round(4).reset_index())
agg['condition'] = pd.Categorical(agg.condition, CONDITIONS); agg = agg.sort_values(['comparison', 'fold', 'condition'])
agg.to_csv(f'{OUT}/E5_summary.csv', index=False)
wide = agg.pivot_table(index='condition', columns=['comparison', 'fold'], values='mean', observed=False).round(3)
wide.to_csv(f'{OUT}/E5_wide.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('Mean over seeds: AUC, or the share of matched DFF reals flagged as fake (clean-val threshold)')
display(wide)
print('saved:', os.listdir(OUT), f'| total {(time.time() - T0) / 3600:.2f} h')
shutil.rmtree('/kaggle/working/_data', ignore_errors=True)
'''

cells = [
    md('''
# CDG-DFD Phase 2, E5: robustness to real-world degradation (Reviewer 1 AA, Reviewer 2 Comment 11)

Test only, no training. The E1b checkpoints (matched-reals training, band SAM + DAT, 2 folds x 3 seeds) score the
same leakage-audited test sets after a fixed corruption of every **test** image, real and fake alike. Validation stays
clean, so the decision threshold is the one a deployed model would use.

Conditions: clean; JPEG quality 90/70/50/30; downscale x0.75/0.5/0.25 and back; Gaussian blur radius 1/2;
Gaussian noise sigma 5/10 (0-255 scale); downscale x0.5 then JPEG 70. 13 conditions x 6 checkpoints = 78
evaluations, two at a time: about 1.5-2 hours.

**Inputs:** the **oct18** notebook output and the **cdg-e1b-checkpoints** dataset.
**Settings:** GPU **T4 x2**. **Main number:** unseen-generator AUC per condition (`E5_wide.csv`).
'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    md('## 1. Matched split (rebuilt if absent, same seed as E1b) and leakage audit'),
    code('''
if not os.path.exists(f'{DATA}/splits_matched/fold_A/train.csv'):
    sh(f'python cdg_phase1.py split --out {DATA} --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched')
    sh(f'python cdg_phase1.py audit --out {DATA} --split-name splits_matched')
rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
assert '**Overall leakage verdict: PASS**' in rep, 'matched split: leakage audit failed'
print('leakage audit: PASS')
'''),
    md('## 2. Smoke test'),
    code('''
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
f, s = sorted(CKPT)[0]
run_parallel([('smoke_e5', eval_cmd('resize:0.5+jpeg:70', f, s).replace(f'{RUNS}/e5_resize0p5_jpeg70_{f}_s{s}', SMOKE)
               + ' --limit-eval 300', 0)], retries=0, poll=10)
assert os.path.exists(f'{SMOKE}/test_metrics.csv'), 'smoke test: no metrics written'
print('SMOKE TEST PASSED'); shutil.rmtree(SMOKE, ignore_errors=True)
'''),
    md('## 3. Evaluations (clean first)'),
    code('''
for cond in CONDITIONS:
    run_condition(cond)
'''),
    md('## 4. Summary'),
    code(summary),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E5_robustness.ipynb', 'w'), indent=1)
print('wrote phase2_E5_robustness.ipynb')
