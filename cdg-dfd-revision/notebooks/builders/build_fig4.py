"""Builds phase2_FIG4_roc.ipynb: replacement for the old Figure 4 (Reviewer 2, Comment 9).
CPU only. Reads the saved test predictions + validation thresholds of the E2 runs (and of the E1b full model,
if its checkpoints input is attached) and draws ROC curves and confusion matrices inside Kaggle."""
import json


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


load = r'''
import os, re, io, json, zipfile, shutil, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
T0 = time.time()
OUT = '/kaggle/working/fig4'; os.makedirs(OUT, exist_ok=True)

# runs/<name>/{test_predictions.csv, summary.json}: E2 = e2_<cfg>_<fold>_s<seed>, E1b full model = m_band_<fold>_s<seed>
WANT = re.compile(r'(e2_rgb|e2_dct|e2_dctse|e2_band|m_band)_([AB])_s(\d)/(test_predictions\.csv|summary\.json)$')
NAME = {'e2_rgb': 'rgb', 'e2_dct': 'dct', 'e2_dctse': 'dctse', 'e2_band': 'band', 'm_band': 'full'}
runs = {}

def add(name, data):
    m = WANT.search(name)
    if not m:
        return
    d = runs.setdefault((NAME[m.group(1)], m.group(2), int(m.group(3))), {})
    if m.group(4) == 'summary.json':
        d['thr'] = float(json.loads(data)['threshold'])
    else:
        d['pred'] = pd.read_csv(io.BytesIO(data))

def scan_zip(path, depth=0):
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return
    with zf:
        for n in zf.namelist():
            if WANT.search(n):
                add(n, zf.read(n))
            elif n.endswith('.zip') and depth < 2:          # e2_checkpoints.zip inside a notebook's _output_.zip
                tmp = f'/kaggle/working/_inner{depth}.zip'
                with zf.open(n) as fi, open(tmp, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                scan_zip(tmp, depth + 1); os.remove(tmp)

for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    if 'crops' in root or 'metadata.csv' in files or len(files) > 5000 or root.count(os.sep) > 10:
        dirs[:] = []; continue
    for f in files:
        p = os.path.join(root, f)
        if WANT.search(p):
            add(p, open(p, 'rb').read())
        elif f.endswith('.zip') and 'oct18' not in root and os.path.getsize(p) < 6e9:
            scan_zip(p)

runs = {k: v for k, v in runs.items() if 'pred' in v and 'thr' in v}
print('runs with predictions + threshold:', sorted(runs))
assert any(k[0] == 'rgb' for k in runs) and any(k[0] == 'band' for k in runs), \
    'E2 predictions not found: add the phase2-e2-ablation notebook output as an input'
'''

plot = r'''
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
HELD = {'A': 'DFF:sd15_text2img', 'B': 'DFF:sd_inpainting'}
TESTS = [('held', 'Held-out diffusion generator\nvs matched DFF reals'),
         ('DFF:insightface', 'DFF face swap (test only)\nvs matched DFF reals'),
         ('CelebDF:celebdf_swap', 'Celeb-DF swap\nvs Celeb-DF real'),
         ('HardFake:hardfake_gan', 'HardFake GAN\nvs HardFake real')]
REAL = {'DFF': 'DFF:real_matched', 'CelebDF': 'CelebDF:real', 'HardFake': 'HardFake:real'}
HAS_FULL = any(k[0] == 'full' for k in runs)
M = [('rgb', 'RGB EfficientNet-B4', '#4C72B0', '-'), ('dct', 'DCT input', '#55A868', '--'),
     ('band', 'DCT + band SAM', '#DD8452', '-.')]
if HAS_FULL:
    M.append(('full', 'DCT + band SAM + DAT (proposed)', '#C44E52', '-'))
MAIN = 'full' if HAS_FULL else 'band'

def subset(pred, test, fold):
    fake = HELD[fold] if test == 'held' else test
    real = REAL[fake.split(':')[0]]
    s = pred[pred.test_set.isin([fake, real])]
    return s.label.to_numpy(), s.prob.to_numpy()

def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', dpi=200)
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); plt.show(); print('saved', f'{OUT}/{name}.png / .pdf')

# ---- Figure 4a: ROC curves (seed 1), rows = folds, columns = test sets ----
rows = []
fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True, sharey=True)
for i, fold in enumerate('AB'):
    for j, (test, title) in enumerate(TESTS):
        ax = axes[i, j]
        ax.plot([0, 1], [0, 1], color='#BBBBBB', lw=1, ls=':')
        for key, label, col, ls in M:
            r = runs.get((key, fold, 1))
            if r is None:
                continue
            y, s = subset(r['pred'], test, fold)
            fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
            p = (s >= r['thr']).astype(int)
            tp_r, tn_r = (p[y == 1] == 1).mean(), (p[y == 0] == 0).mean()
            ax.plot(fpr, tpr, color=col, ls=ls, lw=2, label=f'{label}  {auc:.3f}')
            ax.plot(1 - tn_r, tp_r, 'o', color=col, ms=7, mec='white', mew=1.5)   # operating point at the val threshold
            rows.append(dict(fold=fold, test=title.replace(chr(10), ' '), method=label, AUC=round(auc, 4),
                             threshold=round(r['thr'], 4), TPR=round(tp_r, 4), TNR=round(tn_r, 4),
                             balanced_acc=round((tp_r + tn_r) / 2, 4), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum())))
        if i == 0: ax.set_title(title, fontsize=10)
        if j == 0: ax.set_ylabel(f'Fold {fold}\nTrue positive rate (fakes caught)')
        if i == 1: ax.set_xlabel('False positive rate (reals flagged)')
        ax.legend(loc='lower right', fontsize=7.5, frameon=False, title='AUC', title_fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.6)
fig.suptitle('Figure 4. ROC curves on the leakage-controlled test sets (seed 1). Dots: operating point at the '
             'validation-selected threshold. Fold A holds out SD 1.5 text-to-image, fold B holds out SD inpainting.',
             fontsize=10, y=1.0)
save(fig, 'fig4a_roc')
op = pd.DataFrame(rows); op.to_csv(f'{OUT}/fig4_operating_points.csv', index=False)
display(op)

# ---- Figure 4b: confusion matrices of the proposed model at the validation threshold ----
fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
for i, fold in enumerate('AB'):
    r = runs[(MAIN, fold, 1)]
    for j, (test, title) in enumerate(TESTS):
        ax = axes[i, j]
        y, s = subset(r['pred'], test, fold)
        p = (s >= r['thr']).astype(int)
        cm = np.array([[((y == a) & (p == b)).sum() for b in (0, 1)] for a in (0, 1)])
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
        for a in (0, 1):
            for b in (0, 1):
                ax.text(b, a, f'{pct[a, b]:.0%}\n({cm[a, b]:,})', ha='center', va='center', fontsize=9,
                        color='white' if pct[a, b] > 0.6 else '#222222')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['pred. real', 'pred. fake'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['real', 'fake'])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0: ax.set_title(title, fontsize=10)
        if j == 0: ax.set_ylabel(f'Fold {fold}\ntrue class')
fig.suptitle(f'Figure 4b. Confusion matrices of {dict((k, l) for k, l, _, _ in M)[MAIN]} (seed 1) at the '
             f'validation-selected threshold; rows sum to 100% (recall per class).', fontsize=10, y=1.0)
save(fig, 'fig4b_confusion')
print(f'done in {(time.time() - T0) / 60:.1f} min | files:', sorted(os.listdir(OUT)))
'''

cells = [
    md('''
# CDG-DFD: Figure 4 replacement (Reviewer 2, Comment 9)

The old Figure 4 (recall bars) is replaced by **ROC curves** and **confusion matrices**, computed directly from the
saved test predictions of the runs in the paper; nothing is retrained. CPU only.

- **4a**: ROC curves per test set and fold for RGB, DCT, DCT + band SAM (E2) and, if attached, the full model with
  DAT (E1b). Dots mark the operating point at the threshold chosen on validation data.
- **4b**: confusion matrices of the proposed model at that threshold (rows = true class, so the diagonal is recall).

**Inputs:** the `phase2-e2-ablation` notebook output; optionally the E1b checkpoints input (`m_band_*` runs).
**Settings:** Accelerator None, Internet off. Output: `fig4/` (PNG + PDF + operating-point CSV).
'''),
    md('## 1. Load saved predictions and validation thresholds'),
    code(load),
    md('## 2. Figures'),
    code(plot),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_FIG4_roc.ipynb', 'w'), indent=1)
print('wrote phase2_FIG4_roc.ipynb')
