"""Builds phase2_E1b_matched.ipynb: training with matched DFF reals (compression control) + extra seeds."""
import json
import re

src = open('build_phase2_notebooks.py').read()
FIND_PHASE1 = re.search(r"FIND_PHASE1 = '''(.*?)'''\n", src, re.S).group(1)
phase1 = open('cdg_phase1.py').read()
train = open('cdg_train.py').read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


setup = '''
import os, json, glob, shutil, time, subprocess
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
assert DATA, 'No Phase 1 output found under /kaggle/input. Add the add-on notebook output (oct18) as input.'
print('Phase 1 data:', DATA)
NGPU = torch.cuda.device_count()
print('GPUs:', NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, 'No GPU: set Accelerator to GPU T4 x2'
RUNS, LOGS = '/kaggle/working/runs', '/kaggle/working/logs'
os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

KEEP = ('fold ', 'parameters', 'GPUs used', '   batch 300', 'ep ', 'resumed', '   WARNING', '   note', 'Traceback',
        'Error', '"best_epoch"', 'vs DFF:real_matched', 'vs DFF:real ', 'flagged as fake')

def run_parallel(jobs, retries=2, poll=30):
    # jobs: list of (name, command, gpu). One job per GPU at the same time; prints the key log lines;
    # restarts a job killed for memory (it resumes from its last epoch); raises if any job fails.
    state, failed = {}, []
    def start(name, cmd, gpu, tries):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        log = open(f'{LOGS}/{name}.log', 'a')
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        pos = state[name]['pos'] if name in state else 0
        state[name] = dict(p=p, cmd=cmd, gpu=gpu, tries=tries, pos=pos, log=log)
        print(f'[{name}] started on GPU {gpu}', flush=True)
    for name, cmd, gpu in jobs:
        start(name, cmd, gpu, 0)
    while state:
        time.sleep(poll)
        for name in list(state):
            st = state[name]
            with open(f'{LOGS}/{name}.log') as f:
                f.seek(st['pos']); new = f.read(); st['pos'] = f.tell()
            for line in new.splitlines():
                if line.startswith(KEEP) or 'Error' in line:
                    print(f'[{name}] {line}', flush=True)
            code = st['p'].poll()
            if code is None:
                continue
            st['log'].close()
            if code == 0:
                print(f'[{name}] finished', flush=True); del state[name]
            elif code in (137, -9) and st['tries'] < retries:
                print(f'[{name}] killed (exit {code}): restarting from its last finished epoch', flush=True)
                start(name, st['cmd'], st['gpu'], st['tries'] + 1)
            else:
                tail = open(f'{LOGS}/{name}.log').read().splitlines()[-25:]
                print(f'[{name}] FAILED (exit {code}). Last lines:', *tail, sep='\\n', flush=True)
                failed.append(name); del state[name]
    if failed:
        raise RuntimeError('failed: ' + ', '.join(failed) + ' (the last lines of each log are printed above)')

SPLITS = {'m': 'splits_matched',   # DFF reals stored like the fakes (512x512, JPEG q75) train the model
          'o': 'splits'}           # original DFF reals, as in E1

def train_cmd(mode, fold, seed, extra=''):
    return (f'python cdg_train.py --data {DATA} --splits {SPLITS[mode]} --fold {fold} --seed {seed} '
            f'--spectral band --out {RUNS}/{mode}_band_{fold}_s{seed} {extra}')

def round_(mode, seed):
    # both folds in parallel, one per GPU; runs that already finished are skipped
    jobs = [(f'{mode}_band_{f}_s{seed}', train_cmd(mode, f, seed), i % NGPU) for i, f in enumerate('AB')
            if not os.path.exists(f'{RUNS}/{mode}_band_{f}_s{seed}/summary.json')]
    if jobs:
        run_parallel(jobs)
    else:
        print('already done')

!free -g | head -2; nproc
'''

cells = [
    md('''
# CDG-DFD Phase 2, E1b: compression-controlled training (+ seeds)

E1 showed that most of the unseen-generator score came from file compression: every DFF fake is a 512x512 JPEG at
quality 75 and the original reals are not. Against reals stored like the fakes (`real_matched`), the unseen-generator
AUC fell from 0.93 to 0.59 (fold A) and from 0.91 to 0.71 (fold B).

Here the model is **trained with the matched reals**, so real and fake DFF files are stored the same way and the
compression shortcut is gone from training. Same people, photos and splits as E1; only the DFF real images change.
The original reals become a test-only set.

| Step | What | Time |
|---|---|---|
| 1 | build the matched split, audit it, check it uses E1's exact people/photo split | ~2 min, CPU |
| 2 | smoke test on the GPU | ~5 min |
| 3 | **matched training, seed 1, folds A and B** (the decisive result) | ~1 h |
| 4 | matched training, seeds 2 and 3 | ~2 h |
| 5 | original-reals training (as E1), seeds 2 and 3, for error bars on E1 | ~2 h |
| 6 | summary: mean and spread over seeds | |

All runs use the revised SAM (`band`). **Settings:** GPU **T4 x2**, Internet on. **Input:** the add-on output (`oct18`).
If the session stops, run it again: finished runs are skipped and unfinished ones resume from their last epoch only
within the same session.

**Main number:** `DFF:sd15_text2img vs DFF:real_matched` in fold A and `DFF:sd_inpainting vs DFF:real_matched` in fold B.
'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    md('## 1. Matched split and audit'),
    code('''
!python cdg_phase1.py split --out {DATA} --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched
!python cdg_phase1.py audit --out {DATA} --split-name splits_matched

# Same people/photos/videos in test as E1: every non-DFF-real test row must match exactly
for f in 'AB':
    a = pd.read_csv(f'{DATA}/splits/fold_{f}/test.csv', keep_default_na=False)
    b = pd.read_csv(f'{DATA}/splits_matched/fold_{f}/test.csv', keep_default_na=False)
    keep = lambda d: set(d.uid[~d.source.isin(['dff_wiki', 'dff_wiki_matched'])])
    assert keep(a) == keep(b), f'fold {f}: the matched split does not reuse E1 test groups'
    print(f'fold {f}: test set identical to E1 apart from the DFF reals ({len(keep(a)):,} rows checked)')
rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
if '**Overall leakage verdict: PASS**' not in rep:
    print(chr(10).join(l for l in rep.splitlines() if 'FAIL' in l or 'must be 0' in l))
    raise AssertionError('leakage audit failed: the failing checks are listed above')
display(Markdown(rep[rep.index('## 4.'):]))
'''),
    md('## 2. Smoke test'),
    code('''
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
run_parallel([('smoke_m', f'python cdg_train.py --data {DATA} --splits splits_matched --fold A --seed 1 --spectral band '
               f'--out {SMOKE} --epochs 1 --epoch-size 1280 --limit-eval 400', 0)], retries=0, poll=10)
assert os.path.exists(f'{SMOKE}/test_metrics.csv'), 'smoke test: no metrics written'
print('SMOKE TEST PASSED')
shutil.rmtree(SMOKE, ignore_errors=True)
'''),
    md('## 3. Matched training, seed 1 (decisive result)'),
    code("round_('m', 1)"),
    md('## 4. Matched training, seeds 2 and 3'),
    code("round_('m', 2)"),
    code("round_('m', 3)"),
    md('## 5. Original-reals training (as E1), seeds 2 and 3'),
    code("round_('o', 2)"),
    code("round_('o', 3)"),
    md('## 6. Summary'),
    code('''
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/*/test_metrics.csv')):
    name = os.path.basename(os.path.dirname(mpath))          # e.g. m_band_A_s2
    mode, _, fold, seed = name.split('_')
    m = pd.read_csv(mpath)
    m['training_reals'] = {'m': 'matched', 'o': 'original'}[mode]
    m['fold'], m['seed'] = fold, int(seed[1:])
    rows.append(m)
assert rows, 'no finished runs'
allm = pd.concat(rows, ignore_index=True)
allm.to_csv(f'{RUNS}/E1b_all_metrics.csv', index=False)
main = allm[allm.comparison.str.contains('DFF:sd15_text2img vs|DFF:sd_inpainting vs|flagged|CelebDF:celebdf_swap vs CelebDF:real$')]
main = main.assign(value=main.AUC.fillna(main.FPR))
agg = (main.groupby(['training_reals', 'fold', 'comparison'])['value']
       .agg(n_seeds='count', mean='mean', sd='std').round(4).reset_index())
agg.to_csv(f'{RUNS}/E1b_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('value = AUC, or the share flagged as fake for the "flagged" rows')
display(agg)
shutil.rmtree('/kaggle/working/_data', ignore_errors=True)
'''),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E1b_matched.ipynb', 'w'), indent=1)
print('wrote phase2_E1b_matched.ipynb')
