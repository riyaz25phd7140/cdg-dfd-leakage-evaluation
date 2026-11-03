# Figures for the paper and as a record of the runs (saved to e6_results/figs, shown below)
import matplotlib
import matplotlib.pyplot as plt
FIG = '/kaggle/working/e6_results/figs'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'rgb': '#4C72B0', 'dct': '#DD8452', 'dctse': '#55A868', 'band': '#C44E52', 'band+DAT': '#8172B3'}

def input_files(want):
    # finds files named in `want` under /kaggle/input (also inside notebook-output zips), skipping image datasets
    found = {}
    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
        if 'metadata.csv' in files or len(files) > 5000 or root.count(os.sep) > 8 or 'crops' in root:
            dirs[:] = []; continue
        for f in files:
            if f in want and f not in found:
                found[f] = pd.read_csv(os.path.join(root, f))
            elif f.endswith('.zip') and 'oct18' not in root and os.path.getsize(os.path.join(root, f)) < 4e9:
                try:
                    with zipfile.ZipFile(os.path.join(root, f)) as zf:
                        for n in zf.namelist():
                            b = os.path.basename(n)
                            if b in want and b not in found:
                                found[b] = pd.read_csv(zf.open(n))
                except zipfile.BadZipFile:
                    pass
    return found

prev = input_files({'E2_summary.csv', 'E5_summary.csv'})
e2, e5 = prev.get('E2_summary.csv'), prev.get('E5_summary.csv')
print('E2 summary:', e2 is not None, '| E5 summary:', e5 is not None)
def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    plt.show(); print('saved', f'{FIG}/{name}.png')

# 1. E2 ablation: AUC per test set and configuration (mean of 2 seeds, error bar = sd); E1b full model from E5 clean runs
try:
    tests = [('unseen DFF generator', 'Held-out SD generator'), ('HardFake:hardfake_gan vs HardFake:real', 'HardFake (GAN)'),
             ('CelebDF:celebdf_swap vs CelebDF:real', 'Celeb-DF'), ('DFF:insightface vs DFF:real_matched', 'DFF face swap')]
    cfgs = [('rgb', 'RGB'), ('dct', 'DCT'), ('dctse', 'DCT + SE'), ('band', 'DCT + band SAM'), ('band+DAT', 'DCT + band SAM + DAT (E1b)')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, fold in zip(axes, 'AB'):
        w = 0.16
        for j, (c, lab) in enumerate(cfgs):
            m, s = [], []
            for t, _ in tests:
                if c == 'band+DAT':
                    r = e5[(e5.comparison == t) & (e5.fold == fold) & (e5.condition == 'none')] if e5 is not None else pd.DataFrame()
                    m.append(r['mean'].iloc[0] if len(r) else np.nan); s.append(r['sd'].iloc[0] if len(r) else 0)
                else:
                    r = e2[(e2.comparison == t) & (e2.fold == fold) & (e2.config == c)]
                    m.append(r.AUC_mean.iloc[0] if len(r) else np.nan); s.append(r.AUC_sd.iloc[0] if len(r) else 0)
            x = np.arange(len(tests)) + (j - 2) * w
            ax.bar(x, m, w, yerr=s, capsize=2, color=COL[c], label=lab)
        ax.set_xticks(np.arange(len(tests))); ax.set_xticklabels([n for _, n in tests])
        ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_ylim(0.4, 1.0); ax.set_title(f'Fold {fold}')
    axes[0].set_ylabel('Test AUC')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('E2 ablation under the leakage-controlled protocol (matched reals)')
    save(fig, 'fig_E2_ablation')
except Exception as ex:  # noqa: BLE001
    print('figure 1 skipped:', ex)

# 2. Robustness: held-out SD generator AUC under corruption, RGB (E6) vs DCT + band SAM + DAT (E5)
try:
    rob = pd.read_csv(f'{OUT}/E6_corrupt_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, fold in zip(axes, 'AB'):
        r6 = rob[(rob.comparison == 'unseen DFF generator') & (rob.fold == fold) & (rob.config == 'rgb')].set_index('condition').reindex(CONDITIONS)
        ax.errorbar(range(len(CONDITIONS)), r6['mean'], yerr=r6['sd'], marker='o', color=COL['rgb'], label='RGB (E2 checkpoints)')
        if e5 is not None:
            r5 = e5[(e5.comparison == 'unseen DFF generator') & (e5.fold == fold)].set_index('condition').reindex(CONDITIONS)
            ax.errorbar(range(len(CONDITIONS)), r5['mean'], yerr=r5['sd'], marker='s', color=COL['band+DAT'], label='DCT + band SAM + DAT (E1b)')
        ax.set_xticks(range(len(CONDITIONS))); ax.set_xticklabels(CONDITIONS, rotation=35, ha='right')
        ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_ylim(0.3, 1.0); ax.set_title(f'Fold {fold}: held-out SD generator')
    axes[0].set_ylabel('Test AUC'); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Robustness to test-time degradation (mean +/- sd over seeds)')
    save(fig, 'fig_robustness_rgb_vs_dct')
except Exception as ex:  # noqa: BLE001
    print('figure 2 skipped:', ex)

# 3. ArtiFact: AUC per unseen generator, RGB vs DCT + band SAM (files as distributed)
try:
    pr = pd.read_csv(f'{OUT_X}/report_per_run.csv')
    pr = pr[(pr.view == 'asis') & pr.AUC.notna() & pr.sam.isin(['e2_rgb', 'e2_band'])]
    g = pr.groupby(['fake_set', 'sam']).AUC.agg(['mean', 'std']).unstack('sam')
    g = g.sort_values(('mean', 'e2_rgb'))
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(g) + 1.5))
    y = np.arange(len(g))
    for k, (s, lab, c) in enumerate([('e2_band', 'DCT + band SAM', COL['band']), ('e2_rgb', 'RGB', COL['rgb'])]):
        ax.barh(y + (k - 0.5) * 0.38, g[('mean', s)], 0.38, xerr=g[('std', s)].fillna(0), color=c, label=lab, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([s.replace('AF_', '') for s in g.index])
    ax.axvline(0.5, color='grey', lw=0.8, ls='--'); ax.set_xlim(0, 1); ax.set_xlabel('AUC vs ArtiFact real faces')
    ax.legend(frameon=False, loc='lower right'); ax.set_title('Unseen generators (ArtiFact): mean over 2 folds x 2 seeds')
    save(fig, 'fig_artifact_rgb_vs_dct')
    g.round(4).to_csv(f'{OUT}/E6_artifact_per_generator.csv')
except Exception as ex:  # noqa: BLE001
    print('figure 3 skipped:', ex)

# 4. GPU throughput (E9)
try:
    l64 = lat[lat.batch == 64].set_index('config')
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(l64.index, l64.images_per_s, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(l64)])
    for i, (v, mem) in enumerate(zip(l64.images_per_s, l64.peak_mem_MB)):
        ax.text(v, i, f'  {v:.0f} img/s, {mem} MB', va='center', fontsize=8)
    ax.set_xlabel('Images per second (T4, fp16, batch 64)'); ax.set_title('Inference cost per configuration')
    ax.set_xlim(0, l64.images_per_s.max() * 1.35)
    save(fig, 'fig_E9_throughput')
except Exception as ex:  # noqa: BLE001
    print('figure 4 skipped:', ex)
print('figures:', sorted(os.listdir(FIG)))
