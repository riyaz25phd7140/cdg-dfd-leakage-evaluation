"""Builds phase2_E3_multifamily.ipynb: multi-family training, leave-one-family-out tests, +/- degradation aug."""
import json
import re

src = open('build_phase2_notebooks.py').read()
FIND_PHASE1 = re.search(r"FIND_PHASE1 = '''(.*?)'''\n", src, re.S).group(1)
train = open('cdg_train.py').read()
ext = open('cdg_external.py').read()
e3 = open('cdg_e3.py').read()
phase1 = open('cdg_phase1.py').read()
e1b = open('build_e1b.py').read()
RUN_PARALLEL = re.search(r"(KEEP = .*?)\nSPLITS = ", e1b, re.S).group(1)


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


setup = '''
!pip install -q facenet-pytorch --no-deps
import os, json, glob, shutil, time, subprocess, zipfile
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown
''' + FIND_PHASE1 + '''
def tmp_is_ram():
    for line in open('/proc/mounts'):
        parts = line.split()
        if len(parts) > 2 and parts[1] == '/tmp':
            return parts[2] == 'tmpfs'
    return False

UNPACK = '/kaggle/working/_data' if tmp_is_ram() else '/tmp/phase1_input'   # never unpack into RAM
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

def sh(cmd):
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        if 'Warning' in line:
            continue
        print(line, end='', flush=True)
        tail = (tail + [line])[-30:]
    if p.wait() != 0:
        raise RuntimeError(f'failed (exit {p.returncode}):' + chr(10) + ''.join(tail))

''' + RUN_PARALLEL + '''
RECIPES = {'deg': '--degrade 0.8', 'plain': ''}

def train_cmd(recipe, fold, seed, extra=''):
    return (f'python cdg_train.py --data {DATA} --splits splits_e3 --fold {fold} --seed {seed} --spectral band '
            f'{RECIPES[recipe]} --out {RUNS}/e3_{recipe}_{fold}_s{seed} {extra}')

def round_(pairs, seed=1):
    # pairs: [(recipe, fold), ...], one per GPU, in parallel; finished runs are skipped
    jobs = [(f'e3_{r}_{f}_s{seed}', train_cmd(r, f, seed), i % NGPU) for i, (r, f) in enumerate(pairs)
            if not os.path.exists(f'{RUNS}/e3_{r}_{f}_s{seed}/summary.json')]
    if jobs:
        run_parallel(jobs)
    else:
        print('already done')
'''

cells = [
    md('''
# CDG-DFD Phase 2, E3: multi-family training, leave-one-family-out test

E1c showed the E1/E1b models do not recognise fakes from generator families they never saw (ArtiFact AUC near 0.5).
E3 tests the two obvious remedies, each against a **family the model has never seen**:

1. **More generator families in training**: Phase 1 data (HardFake, CelebDF, DFF with matched reals, both DFF
   generators) plus ArtiFact faces from two of three families; the third family is held out for testing.
2. **Storage-damage augmentation** (`deg`): each training image, real or fake alike, gets a random downscale
   (x0.5-1) and JPEG (quality 40-95) with probability 0.8. `plain` = without it.

| Fold | Held out (test only) | Trained on (ArtiFact part) |
|---|---|---|
| SG | StyleGAN 1/2/3, ProjectedGAN, GANformer, SFHQ | diffusion + other |
| DIF | Stable Diffusion faces, denoising-diffusion GAN, Diffusion-StyleGAN2 | StyleGAN family + other |
| OTH | CIPS, StarGAN, Taming Transformer, LaMa, FaceSynthetics | StyleGAN family + diffusion |

ArtiFact reals (FFHQ, CelebA-HQ) are split 70/10/20 by image, identically in all folds; held-out-family images never
enter train/val (checked). Six runs, seed 1, two at a time: about 3.5-4.5 hours in all.

**Inputs:** the **oct18** notebook output (Phase 1 data) and the dataset **awsaf49/artifact-dataset**.
**Settings:** GPU **T4 x2**, Internet on (for one pip install and ImageNet weights).
**Main number:** AUC of each held-out generator against ArtiFact reals (`E3_heldout.csv`), deg vs plain.
'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    code('%%writefile cdg_external.py\n' + ext),
    code('%%writefile cdg_e3.py\n' + e3),
    md('## 1. ArtiFact faces: sample, crop (threaded reads, batched face detector), and E3 splits'),
    code('''
# the matched-reals split from E1b (same seed, so identical people/photos/videos per split); rebuilt if absent
if not os.path.exists(f'{DATA}/splits_matched/fold_A/train.csv'):
    sh(f'python cdg_phase1.py split --out {DATA} --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched')
    sh(f'python cdg_phase1.py audit --out {DATA} --split-name splits_matched')
    rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
    assert '**Overall leakage verdict: PASS**' in rep, 'matched split: leakage audit failed'
sh(f'python cdg_e3.py prep --data {DATA} --artifact-root /kaggle/input --n-fake 2500 --n-real 6000')
sh(f'python cdg_e3.py split --data {DATA} --held-test 1000')
'''),
    md('## 2. Smoke test (a few minutes)'),
    code('''
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
run_parallel([('smoke_e3', f'python cdg_train.py --data {DATA} --splits splits_e3 --fold DIF --seed 1 --spectral band '
               f'--degrade 0.8 --out {SMOKE} --epochs 1 --epoch-size 1280 --limit-eval 400', 0)], retries=0, poll=10)
assert os.path.exists(f'{SMOKE}/test_metrics.csv'), 'smoke test: no metrics written'
print('SMOKE TEST PASSED')
shutil.rmtree(SMOKE, ignore_errors=True)
'''),
    md('## 3. Training: augmentation runs first'),
    code("round_([('deg', 'DIF'), ('deg', 'SG')])"),
    code("round_([('deg', 'OTH'), ('plain', 'DIF')])"),
    code("round_([('plain', 'SG'), ('plain', 'OTH')])"),
    md('## 4. Summary'),
    code('''
import sys; sys.path.insert(0, '.')
from cdg_e3 import FAMILIES
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/e3_*/test_metrics.csv')):
    name = os.path.basename(os.path.dirname(mpath))          # e3_deg_DIF_s1
    _, recipe, fold, seed = name.split('_')
    m = pd.read_csv(mpath)
    m['recipe'], m['fold'], m['seed'] = recipe, fold, int(seed[1:])
    rows.append(m)
assert rows, 'no finished runs'
allm = pd.concat(rows, ignore_index=True)
OUT = '/kaggle/working/e3_results'; os.makedirs(OUT, exist_ok=True)
allm.to_csv(f'{OUT}/E3_all_metrics.csv', index=False)
af = allm[allm.comparison.str.match(r'^ArtiFact:\\w+ vs ArtiFact:real$')].copy()
af['generator'] = af.comparison.str.extract(r'^ArtiFact:(\\w+) ')[0]
af['held_out'] = [g in FAMILIES[f] for g, f in zip(af.generator, af.fold)]
held = af[af.held_out].pivot_table(index=['fold', 'generator'], columns='recipe', values='AUC').round(3)
seen = af[~af.held_out].groupby(['fold', 'recipe']).AUC.mean().unstack().round(3)
other = allm[allm.comparison.str.contains('vs DFF:real_matched|CelebDF:celebdf_swap vs CelebDF:real$|HardFake:hardfake_gan vs')]
other = other.pivot_table(index=['fold', 'comparison'], columns='recipe', values='AUC').round(3)
held.to_csv(f'{OUT}/E3_heldout.csv'); seen.to_csv(f'{OUT}/E3_seen_mean.csv'); other.to_csv(f'{OUT}/E3_other.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('AUC of each HELD-OUT generator vs ArtiFact reals (never seen in training):'); display(held)
print('Mean AUC over SEEN ArtiFact generators (test split):'); display(seen)
print('Phase 1 test sets:'); display(other)
with zipfile.ZipFile('/kaggle/working/e3_checkpoints.zip', 'w') as z:   # one file, easy to attach later
    for d in glob.glob(f'{RUNS}/e3_*'):
        for f in ('best.pth', 'config.json', 'summary.json'):
            if os.path.exists(f'{d}/{f}'):
                z.write(f'{d}/{f}', f'runs/{os.path.basename(d)}/{f}')
print('saved:', os.listdir(OUT), '+ e3_checkpoints.zip')
'''),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E3_multifamily.ipynb', 'w'), indent=1)
print('wrote phase2_E3_multifamily.ipynb')
