"""Builds phase2_E1c_external.ipynb: test-only evaluation on ArtiFact faces (unseen generators)."""
import json

phase1 = open('cdg_phase1.py').read()
train = open('cdg_train.py').read()
ext = open('cdg_external.py').read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


cells = [
    md("""
# CDG-DFD Phase 2, E1c: external test on unseen generators (ArtiFact faces)

Test only, no training, **no downloads**. The E1 and E1b checkpoints are scored on the face images of **ArtiFact**
(Rahman et al., ICIP 2023), which is already hosted on Kaggle. None of its generators was used in training
(training saw only Stable Diffusion 1.5 inpainting and text-to-image from DFF).

- Reals: the FFHQ and CelebA-HQ folders. Fakes: the face subsets of the other folders: StyleGAN 1/2/3, CIPS,
  GANformer, ProjectedGAN and Taming Transformer (their FFHQ models), Diffusion-StyleGAN2 (FFHQ), denoising-diffusion
  GAN, LaMa (inpainted CelebA-HQ), StarGAN, SFHQ, FaceSynthetics, and Stable Diffusion's Male/Female prompts.
  The first table printed lists every folder, what it holds and how many images were taken.
- ArtiFact put **every** image (real and fake) through the same random crop, downscale and JPEG compression and
  stored it at 200x200, so the compression shortcut we found in DFF is absent here by construction.
- Faces are cropped exactly as in Phase 1 (MTCNN, 224, margin 40), in two versions: `asis` and `matched`
  (both classes re-saved as 512x512 JPEG q75 first).

**Inputs (Add Input):** the dataset **awsaf49/artifact-dataset** (search "ArtiFact"), plus the outputs of
**oct-22** (E1) and **oct-23-1** (E1b) for the checkpoints. **Settings:** GPU T4 (one is enough), Internet on
(only for one small pip install). **Time:** about 1 hour.

**Please run it with "Run All" in the open editor, not "Save Version"**: a saved version that fails shows no output
at all, which is why we could not see the last two failures. If a cell fails, the error stays visible under it.
"""),
    code("""
!pip install -q facenet-pytorch --no-deps
import os, glob, json, subprocess, shutil
import pandas as pd
import torch
from IPython.display import display, Markdown
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
OUT = '/kaggle/working/external'
os.makedirs(OUT, exist_ok=True)

def sh(cmd):
    # runs a step, shows its output live and also keeps it in external/log.txt
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    with open(f'{OUT}/log.txt', 'a') as logf:
        for line in p.stdout:
            if 'Warning' in line:
                continue
            print(line, end='', flush=True); logf.write(line)
            tail = (tail + [line])[-30:]
    if p.wait() != 0:
        raise RuntimeError(f'failed (exit {p.returncode}):' + chr(10) + ''.join(tail))
"""),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    code('%%writefile cdg_external.py\n' + ext),
    md('## 1. Inputs: ArtiFact and the checkpoints'),
    code("""
import sys; sys.path.insert(0, '.')
from cdg_external import find_checkpoints, find_artifact
metas = find_artifact(['/kaggle/input'])
print(len(metas), 'ArtiFact folders found', '(e.g. ' + os.path.dirname(metas[0]) + ')' if metas else '')
def show_tree(root='/kaggle/input', depth=4, per_dir=12):
    # what is attached: folder names to 4 levels (12 entries per folder) and every .csv near the top
    for d, dirs, files in os.walk(root, followlinks=True):
        lvl = d[len(root):].count(os.sep)
        if lvl >= depth:
            dirs[:] = []
        dirs.sort(); files.sort()
        print('  ' * lvl + os.path.basename(d) + '/', f'({len(dirs)} folders, {len(files)} files)',
              ', '.join([f for f in files if not f.lower().endswith(('.jpg', '.jpeg', '.png'))][:per_dir]))
        del dirs[per_dir:]
if not metas:
    show_tree()
assert metas, 'ArtiFact not found: the folder tree of the inputs is printed above (send it to Claude)'
ck = find_checkpoints(['/kaggle/input'])
ck = {k: v for k, v in ck.items() if k.startswith(('m_band', 'o_band', 'band_'))}   # the revised-SAM runs
print(len(ck), 'checkpoints:', ', '.join(ck))
assert ck, 'No checkpoints found: add the oct-22 and oct-23-1 notebook outputs as inputs'
"""),
    md('## 2. Take a sample of the face images (copied from the attached dataset, nothing downloaded)'),
    code("sh(f'python cdg_external.py artifact --out {OUT} --artifact-root /kaggle/input --n 800 --n-real 1500')"),
    md('## 3. Face crops (as distributed, and compression-matched)'),
    code("sh(f'python cdg_external.py crop --out {OUT}')"),
    md('## 4. Score every checkpoint'),
    code("sh(f'python cdg_external.py score --out {OUT} --checkpoints /kaggle/input /tmp/ckpts --only m_band o_band band_')"),
    md('## 5. Report'),
    code("""
sh(f'python cdg_external.py report --out {OUT} --title "ArtiFact faces"')
display(Markdown(open(f'{OUT}/report.md').read()))
shutil.rmtree(f'{OUT}/raw', ignore_errors=True)   # keep crops, scores and reports; drop the copied originals
"""),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E1c_external.ipynb', 'w'), indent=1)
print('wrote phase2_E1c_external.ipynb')
