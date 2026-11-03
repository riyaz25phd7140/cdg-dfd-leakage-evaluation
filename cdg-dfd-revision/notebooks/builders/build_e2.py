"""Builds phase2_E2_ablation.ipynb: controlled ablation under the matched protocol (R2 C8).
RGB -> DCT -> DCT + SE (standard channel attention) -> DCT + band SAM, all without DAT.
The full model (band SAM + DAT) is E1b (3 seeds), trained with the same code, data and settings."""
import json
import re

src = open('build_phase2_notebooks.py').read()
FIND_PHASE1 = re.search(r"FIND_PHASE1 = '''(.*?)'''\n", src, re.S).group(1)
train = open('cdg_train.py').read()
phase1 = open('cdg_phase1.py').read()
e1b = open('build_e1b.py').read()
e3b = open('build_e3.py').read()
RUN_PARALLEL = re.search(r"(KEEP = .*?)\nSPLITS = ", e1b, re.S).group(1)
SH = re.search(r"(def sh\(cmd\):.*?raise RuntimeError\(f'failed \(exit \{p.returncode\}\):' \+ chr\(10\) \+ ''.join\(tail\)\)\n)",
               e3b, re.S).group(1)


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


setup = '''
import os, json, glob, shutil, time, subprocess, zipfile
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

''' + SH + '''
''' + RUN_PARALLEL + '''
CONFIGS = {   # all without DAT; the full model (band + DAT) is E1b
    'rgb':   '--input rgb --spectral none --no-dat',
    'dct':   '--input dct --spectral none --no-dat',
    'dctse': '--input dct --spectral none --channel-attn se --no-dat',
    'band':  '--input dct --spectral band --no-dat',
}
BUDGET_H = 9.8    # Kaggle stops a session at 12 h and then keeps no output: stop starting rounds after this

def train_cmd(cfg, fold, seed):
    return (f'python cdg_train.py --data {DATA} --splits splits_matched --fold {fold} --seed {seed} '
            f'{CONFIGS[cfg]} --out {RUNS}/e2_{cfg}_{fold}_s{seed}')

def round_(cfg, seed):
    # both folds in parallel, one per GPU; finished runs are skipped; no new round after the time budget
    hours = (time.time() - T0) / 3600
    if hours > BUDGET_H:
        print(f'skipped e2_{cfg} seed {seed}: {hours:.1f} h used, budget {BUDGET_H} h'); return
    jobs = [(f'e2_{cfg}_{f}_s{seed}', train_cmd(cfg, f, seed), i % NGPU) for i, f in enumerate('AB')
            if not os.path.exists(f'{RUNS}/e2_{cfg}_{f}_s{seed}/summary.json')]
    if jobs:
        run_parallel(jobs)
    else:
        print('already done')
    print(f'elapsed {(time.time() - T0) / 3600:.2f} h')
'''

summary = '''
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/e2_*/test_metrics.csv')):
    name = os.path.basename(os.path.dirname(mpath))          # e2_dctse_A_s2
    _, cfg, fold, seed = name.split('_')
    m = pd.read_csv(mpath)
    m['config'], m['fold'], m['seed'] = cfg, fold, int(seed[1:])
    s = json.load(open(f'{RUNS}/{name}/summary.json'))
    m['best_epoch'], m['minutes'] = s.get('best_epoch'), s.get('minutes')
    rows.append(m)
assert rows, 'no finished runs'
allm = pd.concat(rows, ignore_index=True)
OUT = '/kaggle/working/e2_results'; os.makedirs(OUT, exist_ok=True)
allm.to_csv(f'{OUT}/E2_all_metrics.csv', index=False)

# unseen DFF generator vs matched reals: fold A holds out SD1.5 text2img, fold B holds out SD inpainting
UNSEEN = {'A': 'DFF:sd15_text2img vs DFF:real_matched', 'B': 'DFF:sd_inpainting vs DFF:real_matched'}
main = allm[[c == UNSEEN[f] for c, f in zip(allm.comparison, allm.fold)]].assign(comparison='unseen DFF generator')
other = allm[allm.comparison.isin(['CelebDF:celebdf_swap vs CelebDF:real', 'HardFake:hardfake_gan vs HardFake:real',
                                   'DFF:insightface vs DFF:real_matched'])]
tab = pd.concat([main, other])
order = list(CONFIGS)
agg = (tab.groupby(['comparison', 'fold', 'config'])
          .agg(n_seeds=('AUC', 'count'), AUC_mean=('AUC', 'mean'), AUC_sd=('AUC', 'std'),
               balacc_mean=('balanced_acc', 'mean'), EER_mean=('EER', 'mean'))
          .round(4).reset_index())
agg['config'] = pd.Categorical(agg.config, order); agg = agg.sort_values(['comparison', 'fold', 'config'])
agg.to_csv(f'{OUT}/E2_summary.csv', index=False)
wide = agg.pivot_table(index=['comparison', 'fold'], columns='config', values='AUC_mean', observed=False)[order].round(3)
wide.to_csv(f'{OUT}/E2_auc_wide.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('Mean test AUC over seeds (no DAT in any row; the full model with DAT is E1b):'); display(wide)
display(agg)
bw = sorted(glob.glob(f'{RUNS}/e2_band_*/band_weights_by_test_set.csv'))
if bw:
    pd.concat([pd.read_csv(p).assign(run=os.path.basename(os.path.dirname(p))) for p in bw]
              ).to_csv(f'{OUT}/E2_band_weights.csv', index=False)
with zipfile.ZipFile('/kaggle/working/e2_checkpoints.zip', 'w') as z:
    for d in glob.glob(f'{RUNS}/e2_*'):
        for f in ('best.pth', 'config.json', 'summary.json', 'test_predictions.csv'):
            if os.path.exists(f'{d}/{f}'):
                z.write(f'{d}/{f}', f'runs/{os.path.basename(d)}/{f}')
print('saved:', os.listdir(OUT), '+ e2_checkpoints.zip', f'| total {(time.time() - T0) / 3600:.2f} h')
shutil.rmtree('/kaggle/working/_data', ignore_errors=True)
'''

cells = [
    md('''
# CDG-DFD Phase 2, E2: controlled ablation under the matched protocol (Reviewer 2, Comment 8)

Each row adds one component; everything else is identical: matched DFF reals (no compression shortcut), the same
leakage-audited people/photo/video splits as E1b, ImageNet initialisation, 20 epochs, same sampling and seeds.

| Config | Input | Attention | DAT |
|---|---|---|---|
| `rgb` | cropped RGB | none | off |
| `dct` | cropped DCT (signed log) | none | off |
| `dctse` | DCT | standard channel SE (first-edition SAM, 1.6M params) | off |
| `band` | DCT | spectral band SAM (16 radial bands, 3,152 params) | off |
| full model | DCT | band SAM | on (= E1b, 3 seeds) |

Two folds (A: SD1.5 text2img unseen, B: SD inpainting unseen), seeds 1 then 2: 16 runs, two at a time, about 9-10 h.
A time guard stops starting new rounds after 9.8 h so the summary and outputs are always saved.

**Input:** the **oct18** notebook output. **Settings:** GPU **T4 x2**, Internet on (ImageNet weights).
**Main number:** AUC of the unseen DFF generator against matched reals, per config (`E2_auc_wide.csv`).
'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    md('## 1. Matched split (rebuilt if absent) and leakage audit'),
    code('''
if not os.path.exists(f'{DATA}/splits_matched/fold_A/train.csv'):
    sh(f'python cdg_phase1.py split --out {DATA} --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched')
    sh(f'python cdg_phase1.py audit --out {DATA} --split-name splits_matched')
rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
assert '**Overall leakage verdict: PASS**' in rep, 'matched split: leakage audit failed'
print('leakage audit: PASS')
'''),
    md('## 2. Smoke test (every config, a few minutes)'),
    code('''
SMOKE = '/kaggle/working/smoke'
for i, cfg in enumerate(CONFIGS):
    shutil.rmtree(SMOKE, ignore_errors=True)
    run_parallel([(f'smoke_{cfg}', f'python cdg_train.py --data {DATA} --splits splits_matched --fold A --seed 1 '
                   f'{CONFIGS[cfg]} --out {SMOKE} --epochs 1 --epoch-size 640 --limit-eval 200', 0)], retries=0, poll=10)
    assert os.path.exists(f'{SMOKE}/test_metrics.csv'), f'smoke test {cfg}: no metrics written'
shutil.rmtree(SMOKE, ignore_errors=True)
print('SMOKE TESTS PASSED')
'''),
    md('## 3. Seed 1 (all configs)'),
    code("round_('rgb', 1)"),
    code("round_('dct', 1)"),
    code("round_('dctse', 1)"),
    code("round_('band', 1)"),
    md('## 4. Seed 2 (all configs, while the time budget allows)'),
    code("round_('rgb', 2)"),
    code("round_('dct', 2)"),
    code("round_('dctse', 2)"),
    code("round_('band', 2)"),
    md('## 5. Summary'),
    code(summary),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E2_ablation.ipynb', 'w'), indent=1)
print('wrote phase2_E2_ablation.ipynb')
