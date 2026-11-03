"""Builds phase1_addon_matched.ipynb and phase2_E1_train.ipynb with the scripts embedded."""
import json


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


def save(cells, path):
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}}}
    json.dump(nb, open(path, 'w'), indent=1)
    print('wrote', path)


phase1 = open('cdg_phase1.py').read()
train = open('cdg_train.py').read()

FIND_PHASE1 = '''
import zipfile

def _walk(root):
    # followlinks: Kaggle mounts some inputs through symlinks; skip the 100k-file crops folders
    for dp, dns, fns in os.walk(root, followlinks=True):
        dns[:] = [d for d in dns if d != 'crops']
        yield dp, dns, fns

def find_phase1(root='/kaggle/input', extract_to='/tmp/phase1_input'):
    """Folder holding raw_index.csv. Kaggle often saves a big notebook output as a single zip;
    then the zip is unpacked to extract_to first (a few minutes for ~7 GB)."""
    for dp, dns, fns in _walk(root):
        if 'raw_index.csv' in fns:
            return dp
    for dp, dns, fns in _walk(root):
        for fn in sorted(fns):
            if not fn.lower().endswith('.zip'):
                continue
            zpath = os.path.join(dp, fn)
            try:
                with zipfile.ZipFile(zpath) as zf:
                    hits = [n for n in zf.namelist() if n.endswith('raw_index.csv')]
                    if not hits:
                        continue
                    prefix = os.path.dirname(min(hits, key=len))
                    target = os.path.join(extract_to, prefix)
                    if not os.path.exists(os.path.join(target, 'raw_index.csv')):
                        print(f'unpacking {zpath} ({os.path.getsize(zpath)/1e9:.1f} GB) -> {extract_to} ...')
                        zf.extractall(extract_to)
                        print('unpacked')
                    return target
            except zipfile.BadZipFile:
                print('not a readable zip:', zpath)
    return None

def show_inputs(root='/kaggle/input', max_depth=6):
    print('Inputs attached to this notebook:')
    for dp, dns, fns in _walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth <= max_depth:
            print('  ' * depth + (os.path.basename(dp) or root) + '/', fns[:5])
'''

# ---------------------------------------------------------------------------
addon = [
    md('''
# CDG-DFD Phase 1 add-on: matched reals, metadata check, Celeb-DF labels

Adds three things to the finished Phase 1 rebuild (`df-1-rebuild-oct-16`) without re-running it:

1. **Matched reals** (`DFF:real_matched`, test-only): every DFF real image, first resized to 512x512 and saved as
   JPEG quality 75, the way every DFF fake file is stored. If a detector flags these as fake, it is reacting to
   compression, not to the face.
2. **Metadata-only baseline** in the audit: how well file size, format and JPEG quality alone separate DFF reals
   from fakes.
3. **Celeb-DF test labels**: test frames from outside the official 518-video list are marked `:linked`, so
   results on the official list stay comparable with published numbers.

The existing splits do not change: the new images join the groups of their originals.

**Settings:** GPU T4, Internet on. **Input:** add the output of `df-1-rebuild-oct-16` (Add Input → Your Work).
About 45 minutes: 1.8 GB download, 30,000 new crops.
'''),
    code('''
!pip install -q facenet-pytorch --no-deps
!pip install -q huggingface_hub
import os, shutil, zipfile, subprocess
''' + FIND_PHASE1 + '''
SRC = find_phase1(extract_to='/kaggle/working/_unpacked')   # unpack next to the output: no second copy
if not SRC:
    show_inputs()
assert SRC, ('No raw_index.csv under /kaggle/input. Add the df-1-rebuild-oct-16 output (Add Input -> Your Work) '
             'and check the listing above.')
assert os.path.isdir(os.path.join(SRC, 'splits')), f'{SRC} has no splits/ folder: not a finished Phase 1 output'
OUT = '/kaggle/working/phase1'
if SRC.startswith('/kaggle/working/'):
    if os.path.abspath(SRC) != OUT:
        shutil.move(SRC, OUT)                     # same disk: instant
    shutil.rmtree('/kaggle/working/_unpacked', ignore_errors=True)
else:
    print('copying', SRC, '->', OUT)
    shutil.copytree(SRC, OUT, dirs_exist_ok=True)   # inputs are read-only
print('Phase 1 data ready in', OUT, '|', len(os.listdir(OUT)), 'items')
'''),
    code('''
# The wiki images must be back at the paths recorded in raw_index.csv
from huggingface_hub import hf_hub_download
DFF_ROOT = '/kaggle/working/dff'
if not os.path.isdir(f'{DFF_ROOT}/wiki'):
    p = hf_hub_download(repo_id='OpenRL/DeepFakeFace', filename='wiki.zip', repo_type='dataset', local_dir=DFF_ROOT)
    with zipfile.ZipFile(p) as zf:
        zf.extractall(f'{DFF_ROOT}/wiki')
    os.remove(p)
print(len(os.listdir(f'{DFF_ROOT}/wiki/wiki')), 'wiki folders')
'''),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('''
!python cdg_phase1.py add-matched --out {OUT}
!python cdg_phase1.py crop --out {OUT}
!python cdg_phase1.py split --out {OUT} --ratios 0.7,0.1,0.2
!python cdg_phase1.py audit --out {OUT}
'''),
    code('''
from IPython.display import Markdown, display
display(Markdown(open(f'{OUT}/reports/audit_report.md').read()))
'''),
    code('''
shutil.rmtree(DFF_ROOT, ignore_errors=True)
shutil.copy('cdg_phase1.py', OUT)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
print(f'{OUT}: {total/1e9:.1f} GB')
'''),
]
save(addon, 'phase1_addon_matched.ipynb')

# ---------------------------------------------------------------------------
e1 = [
    md("""
# CDG-DFD Phase 2, E1: held-out generator test, both SAM designs

The experiment that decides the paper's title (R2 Comment 1). EfficientNet-B4 on DCT input + DAT, trained from
ImageNet weights, checkpoint chosen on validation AUC only, test scored once at the end. Two SAM designs:

| Name | SAM | Parameters |
|---|---|---|
| `band` (option 2) | attention over 16 radial DCT frequency bands, on the DCT map before the backbone | 3,152 |
| `se` (option 1) | the first edition's squeeze-and-excitation on backbone channels | 1,608,768 |

Each design is trained on **fold A** (train SD-Inpainting, test unseen SD v1.5 text-to-image) and **fold B** (the reverse).
That is four runs; each uses **one GPU**, and the two GPUs run two of them at the same time.

**Settings:** GPU **T4 x2**, Internet on. **Input:** the Phase 1 add-on output. Optional: `phase3_best.pth` as a dataset.

**Order:** (1) a 5-minute smoke test runs every step (training, validation, test, metrics, band weights) on a few
hundred images for both designs, so any error appears in minutes; (2) round 1 trains `band` on folds A and B;
(3) round 2 trains `se` on folds A and B; (4) summary. Each epoch is saved; a run killed for memory restarts from its
last finished epoch.

**Main number:** `DFF:sd15_text2img vs DFF:real_matched` in fold A (unseen generator, reals stored like the fakes).
"""),
    code("""
import os, json, glob, shutil, time, subprocess
import pandas as pd
import torch
from IPython.display import display
""" + FIND_PHASE1 + """
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
assert DATA, 'No Phase 1 output found under /kaggle/input. Add the add-on notebook output (Add Input -> Your Work).'
print('Phase 1 data:', DATA)
for f in 'AB':
    print(f'  fold {f}:', ', '.join(f"{s} {sum(1 for _ in open(f'{DATA}/splits/fold_{f}/{s}.csv')) - 1:,}"
                                     for s in ('train', 'val', 'test')))
NGPU = torch.cuda.device_count()
print('GPUs:', NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, 'No GPU: set Accelerator to GPU T4 x2'
RUNS, LOGS = '/kaggle/working/runs', '/kaggle/working/logs'
os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

VARIANTS = {'band': '--spectral band',                    # option 2: revised SAM (frequency bands)
            'se':   '--spectral none --channel-attn se'}  # option 1: first-edition channel attention
KEEP = ('fold ', 'parameters', 'GPUs used', '   batch', 'ep ', 'resumed', '   WARNING', 'Traceback', 'Error',
        'BENCHMARK', '"best_epoch"', 'vs DFF:real_matched', 'flagged as fake')

def run_parallel(jobs, retries=2, poll=30):
    # jobs: list of (name, command, gpu). Runs them at the same time, one per GPU, prints the key log lines,
    # restarts a job killed for memory (it resumes from its last epoch), and raises if any job fails.
    state, failed = {}, []
    def start(name, cmd, gpu, tries):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        log = open(f'{LOGS}/{name}.log', 'a')
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        pos = state[name]['pos'] if name in state else 0
        state[name] = dict(p=p, cmd=cmd, gpu=gpu, tries=tries, pos=pos, log=log)
        print(f'[{name}] started on GPU {gpu}: {cmd}', flush=True)
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

def train_cmd(variant, fold, seed, out, extra=''):
    return f'python cdg_train.py --data {DATA} --fold {fold} --seed {seed} --out {out} {VARIANTS[variant]} {extra}'

!free -g | head -2; nproc
"""),
    code('%%writefile cdg_train.py\n' + train),
    md("""
## 1. Smoke test (about 5 minutes)
Both designs, both GPUs at once: 20 training batches, a small validation and test set, metrics and band weights.
Checks speed and RAM too. Nothing here is used for results.
"""),
    code("""
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
jobs = [(f'smoke_{v}', train_cmd(v, 'A', 1, f'{SMOKE}/{v}', '--epochs 1 --epoch-size 1280 --limit-eval 400'), i % NGPU)
        for i, v in enumerate(VARIANTS)]
run_parallel(jobs, retries=0, poll=10)
for v in VARIANTS:
    assert os.path.exists(f'{SMOKE}/{v}/test_metrics.csv'), f'smoke test {v}: no metrics written'
print('band weights saved:', os.path.exists(f'{SMOKE}/band/band_weights_by_test_set.csv'))
print('SMOKE TEST PASSED: every step runs on the GPU')
shutil.rmtree(SMOKE, ignore_errors=True)
"""),
    md("## 2. Round 1: revised SAM (`band`), folds A and B in parallel"),
    code("""
run_parallel([(f'band_{f}_s1', train_cmd('band', f, 1, f'{RUNS}/band_{f}_s1'), i % NGPU) for i, f in enumerate('AB')])
"""),
    md("## 3. Round 2: first-edition SAM (`se`), folds A and B in parallel"),
    code("""
run_parallel([(f'se_{f}_s1', train_cmd('se', f, 1, f'{RUNS}/se_{f}_s1'), i % NGPU) for i, f in enumerate('AB')])
"""),
    md("## 4. Summary"),
    code("""
rows = []
for mpath in sorted(glob.glob(f'{RUNS}/*/test_metrics.csv')):
    d = os.path.dirname(mpath)
    m = pd.read_csv(mpath)
    s = json.load(open(os.path.join(d, 'summary.json')))
    m.insert(0, 'run', os.path.basename(d))
    m.insert(1, 'best_epoch', s['best_epoch'])
    rows.append(m)
assert rows, 'No finished runs in ' + RUNS
summary = pd.concat(rows, ignore_index=True)
summary.to_csv(f'{RUNS}/E1_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20); pd.set_option('display.max_rows', 200)
key = summary[summary.comparison.str.contains('DFF:sd15_text2img vs|DFF:sd_inpainting vs|flagged|CelebDF:celebdf_swap vs')]
display(key[['run', 'best_epoch', 'comparison', 'n_real', 'n_fake', 'AUC', 'AUC_CI95', 'EER', 'balanced_acc', 'FPR']])
shutil.rmtree('/kaggle/working/_data', ignore_errors=True)   # keep the saved output small
"""),
    md("""
## Optional: the first-edition checkpoint on the new test sets
Its DFF numbers are inflated by leakage (about 96% of these test people were in its training data); the useful row is
how many matched reals it flags as fake.
"""),
    code("""
OLD = next(iter(glob.glob('/kaggle/input/**/phase3_best.pth', recursive=True)), None)
print('old checkpoint:', OLD)
if OLD:
    run_parallel([('old_checkpoint_A', train_cmd('se', 'A', 1, f'{RUNS}/old_checkpoint_A', f'--init {OLD} --epochs 0'), 0)])
"""),
]
save(e1, 'phase2_E1_train.ipynb')
