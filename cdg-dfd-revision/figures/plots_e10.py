# Figures comparing the four methods (saved to e10_results/figs as PNG and PDF, shown below)
import matplotlib
import matplotlib.pyplot as plt
FIG = f'{OUT}/figs'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
M = [('univfd', 'UnivFD (CLIP ViT-L/14, CVPR 2023)', '#8172B3'), ('resnet50', 'CNNDetection (ResNet-50, CVPR 2020)', '#DD8452'),
     ('rgb', 'RGB EfficientNet-B4 (E2)', '#4C72B0'), ('band', 'DCT + band SAM, ours (E2)', '#C44E52')]

def input_files(want):
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

prev = input_files({'E2_summary.csv', 'E5_summary.csv', 'E6_corrupt_summary.csv'})
e2, e5, e6 = prev.get('E2_summary.csv'), prev.get('E5_summary.csv'), prev.get('E6_corrupt_summary.csv')
print('inputs found:', {k: v is not None for k, v in [('E2', e2), ('E5', e5), ('E6', e6)]})
e10 = pd.read_csv(f'{OUT}/E10_summary.csv')
e10c = pd.read_csv(f'{OUT}/E10_corrupt_summary.csv')

def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    plt.show(); print('saved', f'{FIG}/{name}.png')

def auc(method, test, fold):
    if method in ('univfd', 'resnet50'):
        r = e10[(e10.method == method) & (e10.comparison == test) & (e10.fold == fold)]
        return (r.AUC_mean.iloc[0], r.AUC_sd.fillna(0).iloc[0]) if len(r) else (np.nan, 0)
    r = e2[(e2.config == method) & (e2.comparison == test) & (e2.fold == fold)] if e2 is not None else pd.DataFrame()
    return (r.AUC_mean.iloc[0], r.AUC_sd.iloc[0]) if len(r) else (np.nan, 0)

# 1. Same test sets, four methods
try:
    tests = [('unseen DFF generator', 'Held-out SD generator'), ('HardFake:hardfake_gan vs HardFake:real', 'HardFake (GAN)'),
             ('CelebDF:celebdf_swap vs CelebDF:real', 'Celeb-DF*'), ('DFF:insightface vs DFF:real_matched', 'DFF face swap')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, fold in zip(axes, 'AB'):
        w = 0.2
        for j, (m, lab, col) in enumerate(M):
            v = [auc(m, t, fold) for t, _ in tests]
            ax.bar(np.arange(len(tests)) + (j - 1.5) * w, [x[0] for x in v], w, yerr=[x[1] for x in v], capsize=2, color=col, label=lab)
        ax.set_xticks(range(len(tests))); ax.set_xticklabels([n for _, n in tests])
        ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_ylim(0.3, 1.0); ax.set_title(f'Fold {fold}')
    axes[0].set_ylabel('Test AUC')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('Published baselines vs ours, leakage-controlled protocol (*Celeb-DF: video-disjoint, identities overlap)')
    save(fig, 'fig_E10_methods')
except Exception as ex:  # noqa: BLE001
    print('figure 1 skipped:', ex)

# 2. ArtiFact: AUC per unseen generator, four methods
try:
    pr = pd.read_csv(f'{OUT_X}/report_per_run.csv')
    pr = pr[(pr.view == 'asis') & pr.AUC.notna()]
    key = {'e10_univfd': 'univfd', 'e10_resnet50': 'resnet50', 'e2_rgb': 'rgb', 'e2_band': 'band'}
    pr['m'] = pr.sam.map(key)
    g = pr.dropna(subset=['m']).groupby(['fake_set', 'm']).AUC.mean().unstack('m')
    g = g.sort_values('univfd' if 'univfd' in g else g.columns[0])
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(g) + 1.6))
    for j, (m, lab, col) in enumerate(M):
        if m in g:
            ax.scatter(g[m], np.arange(len(g)) + (j - 1.5) * 0.15, color=col, label=lab, s=28, zorder=3)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([s.replace('AF_', '') for s in g.index])
    ax.axvline(0.5, color='grey', lw=0.8, ls='--'); ax.set_xlim(0, 1); ax.set_xlabel('AUC vs ArtiFact real faces')
    ax.grid(axis='y', color='#eeeeee'); ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('14 unseen generators (ArtiFact), mean over folds and seeds')
    save(fig, 'fig_E10_artifact')
    g.round(4).to_csv(f'{OUT}/E10_artifact_per_generator.csv')
    print(g.mean().round(3).to_dict())
except Exception as ex:  # noqa: BLE001
    print('figure 2 skipped:', ex)

# 3. Robustness: held-out SD generator AUC under corruption, four methods
try:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, fold in zip(axes, 'AB'):
        x = range(len(CONDITIONS))
        for m, lab, col in M:
            if m in ('univfd', 'resnet50'):
                r = e10c[(e10c.method == m) & (e10c.comparison == 'unseen DFF generator') & (e10c.fold == fold)].set_index('condition').AUC
            elif m == 'rgb' and e6 is not None:
                r = e6[(e6.config == 'rgb') & (e6.comparison == 'unseen DFF generator') & (e6.fold == fold)].set_index('condition')['mean']
            elif m == 'band' and e5 is not None:
                r = e5[(e5.comparison == 'unseen DFF generator') & (e5.fold == fold)].set_index('condition')['mean']
                lab = 'DCT + band SAM + DAT, ours (E1b)'
            else:
                continue
            ax.plot(x, r.reindex(CONDITIONS).values, marker='o', color=col, label=lab)
        ax.set_xticks(list(x)); ax.set_xticklabels(CONDITIONS, rotation=35, ha='right')
        ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_ylim(0.3, 1.0); ax.set_title(f'Fold {fold}: held-out SD generator')
    axes[0].set_ylabel('Test AUC'); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Robustness to test-time degradation (UnivFD, CNNDetection: seed 1; RGB: 2 seeds; ours: 3 seeds)')
    save(fig, 'fig_E10_robustness')
except Exception as ex:  # noqa: BLE001
    print('figure 3 skipped:', ex)
print('figures:', sorted(os.listdir(FIG)))
