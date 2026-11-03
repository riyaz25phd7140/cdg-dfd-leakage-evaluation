"""Builds phase1_data_rebuild.ipynb with cdg_phase1.py embedded via %%writefile."""
import json

src = open('cdg_phase1.py').read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


cells = [
    md('''
# CDG-DFD Phase 1: data and protocol rebuild

Rebuilds every split from the raw files so each number in the revised paper is traceable to code.
It answers Reviewer 2 Comments 2 (sample accounting), 3 (video/identity leakage), 7 (domain labels),
10 (separate test set) and the split-release half of 12.

**What it produces** (in `/kaggle/working/phase1`, saved as this notebook's output):

| Output | Use |
|---|---|
| `crops/<source>/<uid>.png` | MTCNN face crops, same settings as the first edition (224 px, margin 40), saved **lossless** |
| `splits/fold_A/{train,val,test}.csv`, `fold_B/...` | leave-one-generator-out splits for Phase 2 (E1) |
| `reports/audit_report.md` | accounting table, leakage checks, domain-label table, format check |
| `reports/audit_original.md` | how much the first-edition 85/15 split leaked |
| `raw_index.csv`, `crop_status.csv`, `dedup.csv`, `near_dup_pairs.csv` | full trail from raw file to split |

**Protocol.** Fold A trains on SD-Inpainting and tests on SD v1.5 text-to-image; Fold B swaps them.
HardFake and CelebDF-v2 (GAN-era) are in training in both folds. DFF is split by **person**
(IMDB-WIKI id) and **source photo**, so a test fake never shares a person or photo with anything in
training. CelebDF uses the official test list, split by **video**. InsightFace (face swap, not
diffusion) is test-only.

**Settings.** GPU T4 (MTCNN), Internet **on** (DFF download from Hugging Face).
Add these inputs: `hamzaboulahia/hardfakevsrealfaces`, and a Celeb-DF-v2 dataset that contains
`Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/` and `List_of_testing_videos.txt`.
Expect about 3 hours for cropping (about 150,000 images at 15 per second). Every stage resumes where it stopped if the session ends.

**Disk.** About 5 GB of extracted DFF during the run plus about 8.5 GB of crops. The raw DFF folder is
deleted in the last cell so only the crops and CSVs are saved.
'''),
    code('''
!pip install -q facenet-pytorch --no-deps
!pip install -q huggingface_hub
import os, glob, zipfile, subprocess, sys
import torch
print(torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
'''),
    md('## 1. Paths\nThe cell finds the inputs automatically; set a path by hand if it prints `None`.'),
    code('''
OUT = '/kaggle/working/phase1'
DFF_ROOT = '/kaggle/working/dff'

def find_dir_containing(name, root='/kaggle/input'):
    for dp, dns, fns in os.walk(root):
        if name in dns or name in fns:
            return dp
    return None

HARDFAKE_ROOT = None
for dp, dns, _ in os.walk('/kaggle/input'):
    if 'hardfakevsrealfaces' in dp.lower() and {'real', 'fake'} <= set(dns):
        HARDFAKE_ROOT = dp
        break
CELEBDF_ROOT = find_dir_containing('List_of_testing_videos.txt')

# Test-only sets for later (DiFF, DF40 diffusion subsets, Deepfake-Eval-2024): each as NAME=PATH with
# PATH/real and/or PATH/fake. Leave empty for now; they can be added in a later run.
EXTERNAL = []   # e.g. ['df40_pixart=/kaggle/input/df40-pixart']

print('HardFake :', HARDFAKE_ROOT)
print('CelebDF  :', CELEBDF_ROOT)
missing = [n for n, v in (('hamzaboulahia/hardfakevsrealfaces', HARDFAKE_ROOT), ('Celeb-DF-v2', CELEBDF_ROOT)) if not v]
assert not missing, f'Add these inputs (Add Input, top right) and re-run this cell: {missing}'
'''),
    md('## 2. Download DeepFakeFace (including `insight.zip`, used only as a test set)'),
    code('''
from huggingface_hub import hf_hub_download
os.makedirs(DFF_ROOT, exist_ok=True)
for fname in ['wiki.zip', 'text2img.zip', 'inpainting.zip', 'insight.zip']:
    sub = fname[:-4]
    if os.path.isdir(os.path.join(DFF_ROOT, sub)):
        print(sub, 'already extracted'); continue
    path = hf_hub_download(repo_id='OpenRL/DeepFakeFace', filename=fname, repo_type='dataset',
                           local_dir=DFF_ROOT)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(os.path.join(DFF_ROOT, sub))
    os.remove(path)              # keep disk free: zips are not needed after extraction
    print(sub, 'extracted')
'''),
    md('## 3. Pipeline code\nThe same file is attached to the paper\'s code release.'),
    code('%%writefile cdg_phase1.py\n' + src),
    md('''
## 4. Scan
Indexes every file (format, size, JPEG quality, MD5) and samples CelebDF frames: 10 per real video and
2 per fake video (more fake videos, roughly balanced classes). A random 8,000 InsightFace images are
kept; they are test-only.

`--dff-matched-real` adds a second, test-only copy of every DFF real image that first goes through the
fakes' file pipeline (resized to 512x512, saved as JPEG quality 75). Every DFF fake file is stored that way and
the reals are not, so this copy tells us whether a detector reacts to the face or to the compression.
'''),
    code('''
args = ['python', 'cdg_phase1.py', 'scan', '--out', OUT, '--dff-root', DFF_ROOT, '--dff-insight-sample', '8000',
        '--dff-matched-real']
if HARDFAKE_ROOT: args += ['--hardfake-root', HARDFAKE_ROOT]
if CELEBDF_ROOT:  args += ['--celebdf-root', CELEBDF_ROOT]
for e in EXTERNAL: args += ['--external', e]
if not os.path.exists(os.path.join(OUT, 'raw_index.csv')):
    subprocess.run(args, check=True)
else:
    print('raw_index.csv exists; delete it to rescan')
'''),
    md('## 5. MTCNN crop (long step; resumable)\nRe-run this cell after a restart and it continues where it stopped.'),
    code("!python cdg_phase1.py crop --out {OUT}"),
    md('## 6. Duplicates, splits, audit'),
    code('''
!python cdg_phase1.py dedup --out {OUT}
!python cdg_phase1.py split --out {OUT} --ratios 0.7,0.1,0.2
!python cdg_phase1.py audit --out {OUT}
!python cdg_phase1.py audit-original --out {OUT}
'''),
    code('''
from IPython.display import Markdown, display
display(Markdown(open(f'{OUT}/reports/audit_report.md').read()))
display(Markdown(open(f'{OUT}/reports/audit_original.md').read()))
'''),
    md('## 7. Clean up\nDeletes the raw DFF images (re-downloadable) so the saved output is only crops and CSVs.'),
    code('''
import shutil
shutil.rmtree(DFF_ROOT, ignore_errors=True)
shutil.copy('cdg_phase1.py', OUT)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
print(f'{OUT}: {total/1e9:.1f} GB')
'''),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase1_data_rebuild.ipynb', 'w'), indent=1)
print('notebook written')
