"""Builds phase2_E10_baselines.ipynb: two published detectors under the leakage-controlled protocol
(R2 Comments 4 and 8): UnivFD (Ojha et al., CVPR 2023) and CNNDetection (Wang et al., CVPR 2020)."""
import json

head = open('/tmp/e10_head.py').read()
shrp = open('/tmp/e10_shrp.py').read()
train = open('cdg_train.py').read()
phase1 = open('cdg_phase1.py').read()
ext = open('cdg_external.py').read()
uf = open('cdg_univfd.py').read()
plots = open('e10/plots_cell.py').read()
assert "backbone" in train and 'e\\d+_' in ext


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(keepends=True)}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': t.strip('\n').splitlines(keepends=True)}


setup = head + '''
# ArtiFact face crops already made by E6 (phase2-e6-rgb-followup output): crops.csv + crops/ (+ scores.csv of the
# E2 RGB and DCT+band runs, so one report holds every method). Taken as files, or out of the output zip.
OUT_X = '/kaggle/working/external'
os.makedirs(OUT_X, exist_ok=True)
src = None
for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    dirs[:] = [d for d in dirs if d not in ('crops', 'datasets')]
    if 'crops.csv' in files and 'scores.csv' in files and os.path.isdir(os.path.join(root, 'crops')):
        src = root; break
if src is None:
    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
        dirs[:] = [d for d in dirs if d not in ('crops', 'datasets')]
        for f in files:
            if f.endswith('.zip') and 'oct18' not in root:
                try:
                    with zipfile.ZipFile(os.path.join(root, f)) as zf:
                        names = [n for n in zf.namelist() if n.startswith('external/') and not n.startswith('external/raw/')]
                        if any(n == 'external/crops.csv' for n in names):
                            print('extracting', len(names), 'files from', f, flush=True)
                            for n in names:
                                zf.extract(n, '/kaggle/working/_x')
                            src = '/kaggle/working/_x/external'
                except zipfile.BadZipFile:
                    pass
assert src, 'E6 ArtiFact crops not found: add the phase2-e6-rgb-followup notebook output as an input'
c = pd.read_csv(f'{src}/crops.csv', keep_default_na=False)
c['crop'] = c.crop.astype(str).str.replace('/kaggle/working/external', src, regex=False)
c.to_csv(f'{OUT_X}/crops.csv', index=False)
shutil.copy(f'{src}/scores.csv', f'{OUT_X}/scores.csv')
ok = c[c.ok.astype(str) == '1']
print('ArtiFact crops:', len(ok), '| exist:', all(os.path.exists(p) for p in ok.crop.head(50)), '| sets:', c.set.nunique(), flush=True)
''' + shrp + '''
RES = '--input rgb --spectral none --no-dat --backbone resnet50'
CONDITIONS = ['none', 'jpeg:30', 'resize:0.5', 'blur:1', 'noise:10', 'resize:0.5+jpeg:70']
tag = lambda c: c.replace(':', '').replace('+', '_').replace('.', 'p')
BUDGET_H = 5.5
'''

audit = '''
if not os.path.exists(f'{DATA}/splits_matched/fold_A/train.csv'):   # same seed as E1b/E2/E5/E6
    sh(f'python cdg_phase1.py split --out {DATA} --ratios 0.7,0.1,0.2 --dff-real matched --split-name splits_matched')
    sh(f'python cdg_phase1.py audit --out {DATA} --split-name splits_matched')
rep = open(f'{DATA}/reports_splits_matched/audit_report.md').read()
assert '**Overall leakage verdict: PASS**' in rep, 'matched split: leakage audit failed'
print('leakage audit: PASS')
'''

univfd = '''
UF = '/kaggle/working/_univfd'
sh(f'python cdg_univfd.py features --data {DATA} --out {UF} --runs {RUNS} --external {OUT_X}')
sh(f'python cdg_univfd.py fit --data {DATA} --out {UF} --runs {RUNS}')
sh(f'python cdg_univfd.py artifact --data {DATA} --out {UF} --runs {RUNS} --external {OUT_X}')
sh(f"python cdg_univfd.py corrupt --data {DATA} --out {UF} --runs {RUNS} --conditions " + ' '.join(f"'{c}'" for c in CONDITIONS[1:]))
print(f'UnivFD done | elapsed {(time.time() - T0) / 3600:.2f} h')
'''

smoke = '''
SMOKE = '/kaggle/working/smoke'
shutil.rmtree(SMOKE, ignore_errors=True)
run_parallel([('smoke_resnet', f'python cdg_train.py --data {DATA} --splits splits_matched --fold A --seed 1 {RES} '
               f'--degrade 1.0 --degrade-mode wang --epochs 1 --epoch-size 256 --limit-eval 300 --out {SMOKE}', 0)], retries=0, poll=10)
assert os.path.exists(f'{SMOKE}/test_metrics.csv'), 'smoke test: no metrics written'
print('SMOKE TEST PASSED'); shutil.rmtree(SMOKE, ignore_errors=True)
'''

train_cell = '''
# CNNDetection recipe: ImageNet ResNet-50 on RGB, blur (sigma U[0,3], p=0.5) + JPEG (q U[30,100], p=0.5) augmentation.
# Everything else (data, balancing, schedule, epochs, checkpoint selection) is identical to E2.
for seed in (1, 2):
    if seed == 2 and (time.time() - T0) / 3600 > BUDGET_H - 1.6:
        print('time budget: seed 2 skipped'); break
    jobs = [(f'e10_resnet50_{f}_s{seed}', f'python cdg_train.py --data {DATA} --splits splits_matched --fold {f} --seed {seed} '
             f'{RES} --degrade 1.0 --degrade-mode wang --out {RUNS}/e10_resnet50_{f}_s{seed}', k % NGPU)
            for k, f in enumerate('AB') if not os.path.exists(f'{RUNS}/e10_resnet50_{f}_s{seed}/summary.json')]
    if jobs:
        run_parallel(jobs)
    print(f'ResNet-50 seed {seed}: done | elapsed {(time.time() - T0) / 3600:.2f} h', flush=True)
'''

art_cell = '''
sh(f'python cdg_external.py score --out {OUT_X} --checkpoints {RUNS} --only e10_resnet50')
sh(f'python cdg_external.py report --out {OUT_X} --title "ArtiFact faces: E10 baselines vs E2"')
'''

corr_cell = '''
def eval_cmd(cond, f):
    corrupt = '' if cond == 'none' else f"--test-corrupt '{cond}'"
    return (f'python cdg_train.py --data {DATA} --splits splits_matched --fold {f} --seed 1 {RES} --epochs 0 '
            f'--init {RUNS}/e10_resnet50_{f}_s1/best.pth {corrupt} --out {RUNS}/e10c_resnet50_{tag(cond)}_{f}_s1')
jobs = [(f'e10c_resnet50_{tag(c)}_{f}_s1', eval_cmd(c, f)) for c in CONDITIONS[1:] for f in 'AB'
        if not os.path.exists(f'{RUNS}/e10c_resnet50_{tag(c)}_{f}_s1/test_metrics.csv')]
for i in range(0, len(jobs), NGPU):
    run_parallel([(n, c, k % NGPU) for k, (n, c) in enumerate(jobs[i:i + NGPU])])
print(f'corruption tests done | elapsed {(time.time() - T0) / 3600:.2f} h')
'''

summary = '''
import re as _re
OUT = '/kaggle/working/e10_results'; os.makedirs(OUT, exist_ok=True)
UNSEEN = {'A': 'DFF:sd15_text2img vs DFF:real_matched', 'B': 'DFF:sd_inpainting vs DFF:real_matched'}
KEEP = ['CelebDF:celebdf_swap vs CelebDF:real', 'HardFake:hardfake_gan vs HardFake:real', 'DFF:insightface vs DFF:real_matched']
def collect(pattern):
    rows = []
    for mpath in sorted(glob.glob(f'{RUNS}/*/test_metrics.csv')):
        m = _re.match(pattern, os.path.basename(os.path.dirname(mpath)))
        if m:
            t = pd.read_csv(mpath); t = t.assign(**m.groupdict()); rows.append(t)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(columns=['comparison', 'fold', 'method', 'seed', 'AUC', 'cond'])
    allm = pd.concat(rows, ignore_index=True)
    main = allm[[c == UNSEEN[f] for c, f in zip(allm.comparison, allm.fold)]].assign(comparison='unseen DFF generator')
    flag = allm[allm.comparison == 'DFF:real_matched flagged as fake'].assign(AUC=lambda d: d.FPR)
    return allm, pd.concat([main, allm[allm.comparison.isin(KEEP)], flag])
allc, tab = collect(r'^e10_(?P<method>univfd|resnet50)_(?P<fold>[AB])_s(?P<seed>\\d)$')
allc.to_csv(f'{OUT}/E10_all_metrics.csv', index=False)
s1 = tab.groupby(['comparison', 'fold', 'method']).agg(n_seeds=('AUC', 'count'), AUC_mean=('AUC', 'mean'),
                                                     AUC_sd=('AUC', 'std')).round(4).reset_index()
s1.to_csv(f'{OUT}/E10_summary.csv', index=False)
display(s1.pivot_table(index='comparison', columns=['method', 'fold'], values='AUC_mean').round(3))
allx, tabx = collect(r'^e10c_(?P<method>univfd|resnet50)_(?P<cond>.+)_(?P<fold>[AB])_s(?P<seed>\\d)$')
tabx['condition'] = tabx.cond.map({tag(c): c for c in CONDITIONS})
cl = tab[tab.seed == '1'].assign(condition='none')          # clean row = the seed-1 model, as in the corrupted runs
s2 = pd.concat([cl, tabx]).groupby(['comparison', 'fold', 'method', 'condition']).AUC.mean().round(4).reset_index()
s2.to_csv(f'{OUT}/E10_corrupt_summary.csv', index=False)
display(s2[s2.comparison == 'unseen DFF generator'].pivot_table(index='condition', columns=['method', 'fold'], values='AUC').reindex(CONDITIONS).round(3))
for f in glob.glob(f'{OUT_X}/*.csv') + glob.glob(f'{OUT_X}/*.md'):
    if not f.endswith('crops.csv'):
        shutil.copy(f, OUT)
for d in glob.glob(f'{RUNS}/e10_*'):
    for f in ('summary.json', 'history.csv', 'config.json'):
        if os.path.exists(f'{d}/{f}'):
            os.makedirs(f'{OUT}/runs/{os.path.basename(d)}', exist_ok=True); shutil.copy(f'{d}/{f}', f'{OUT}/runs/{os.path.basename(d)}/')
print('saved:', sorted(os.listdir(OUT)), f'| total {(time.time() - T0) / 3600:.2f} h')
'''

cleanup = '''
for d in ('/kaggle/working/_data', '/kaggle/working/_x', '/kaggle/working/_univfd', OUT_X):
    shutil.rmtree(d, ignore_errors=True)
for p in glob.glob(f'{RUNS}/*/*.pth') + glob.glob(f'{RUNS}/*/*.npy'):
    os.remove(p)
print('cleaned; output kept small (results, figures, logs)')
'''

cells = [
    md('''# CDG-DFD Phase 2, E10: published baselines under the leakage-controlled protocol (Reviewer 2, Comments 4 and 8)

Two published detectors, re-run on exactly the same matched split, test sets and 14 ArtiFact generators as E1b, E2 and E6:

1. **UnivFD** (Ojha, Li and Lee, CVPR 2023): frozen CLIP ViT-L/14 (OpenAI weights) + a linear classifier on the
   normalised embedding, fitted on the fold's training split; regularisation and threshold chosen on validation.
2. **CNNDetection** (Wang et al., CVPR 2020): ImageNet ResNet-50 on RGB, trained with blur + JPEG augmentation;
   training data, balancing, schedule and checkpoint selection identical to E2. 2 folds x 2 seeds.

Then: ArtiFact scoring (E6 crops, no re-sampling), test-time corruptions, summary tables and figures comparing all
four methods (UnivFD, CNNDetection, RGB EfficientNet-B4 from E2, DCT + band SAM + DAT from E1b/E5).

**Inputs:** the **oct18**, **phase2-e2-ablation**, **phase2-e5-robustness** and **phase2-e6-rgb-followup** notebook outputs.
**Settings:** GPU **T4 x2**, Internet on (CLIP weights are downloaded once).'''),
    code(setup),
    code('%%writefile cdg_phase1.py\n' + phase1),
    code('%%writefile cdg_train.py\n' + train),
    code('%%writefile cdg_external.py\n' + ext),
    code('%%writefile cdg_univfd.py\n' + uf),
    md('## 1. Matched split and leakage audit'),
    code(audit),
    md('## 2. UnivFD: CLIP features, linear probe, ArtiFact, corruptions'),
    code(univfd),
    md('## 3. CNNDetection (ResNet-50): smoke test, then 2 folds x 2 seeds'),
    code(smoke),
    code(train_cell),
    md('## 4. ArtiFact scoring and report (all methods)'),
    code(art_cell),
    md('## 5. Corrupted test images, CNNDetection (seed 1)'),
    code(corr_cell),
    md('## 6. Summary'),
    code(summary),
    md('## 7. Figures (saved to `e10_results/figs` as PNG and PDF)'),
    code(plots),
    md('## 8. Clean up large intermediate files'),
    code(cleanup),
]
nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}}
json.dump(nb, open('phase2_E10_baselines.ipynb', 'w'), indent=1)
print('wrote phase2_E10_baselines.ipynb')
