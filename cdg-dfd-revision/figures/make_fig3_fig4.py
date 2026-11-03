"""Redraws Figures 3 and 4 with publication labels from the E10 summary numbers (Kaggle run phase2_E10_baselines)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, sys
OUT = sys.argv[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
M = [('UnivFD', '#8172B3', 'o'), ('CNNDetection', '#DD8452', 's'), ('RGB EfficientNet-B4', '#4C72B0', 'D'), ('Frequency detector (ours)', '#C44E52', '^')]
# Figure 3: AUC per ArtiFact generator, mean over folds and seeds (order: UnivFD, CNNDetection, RGB, ours)
G = [('StarGAN', .89, .33, .38, .44), ('Stable Diffusion faces', .86, .70, .74, .61), ('GANformer', .81, .59, .62, .60),
     ('ProjectedGAN', .77, .57, .55, .45), ('SFHQ', .74, .72, .65, .59), ('FaceSynthetics', .73, .34, .44, .42),
     ('Taming Transformer', .70, .58, .60, .54), ('StyleGAN3', .69, .54, .56, .46), ('StyleGAN2', .68, .70, .71, .59),
     ('Diffusion-GAN', .66, .58, .59, .56), ('StyleGAN1', .65, .64, .62, .55), ('CIPS', .63, .70, .60, .46),
     ('Denoising-diffusion GAN', .61, .47, .43, .47), ('LaMa', .56, .46, .44, .41)]
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(G))
off = [-0.24, -0.08, 0.08, 0.24]
for k, (lab, c, mk) in enumerate(M):
    ax.scatter(x + off[k], [g[1 + k] for g in G], color=c, marker=mk, s=30, label=lab, zorder=3, edgecolor='white', linewidth=0.5)
for xx in x[1:]: ax.axvline(xx - 0.5, color='#ececec', lw=0.6, zorder=0)
ax.axhline(0.5, color='grey', lw=0.9, ls='--')
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in G], rotation=35, ha='right'); ax.set_ylim(0.25, 1.0); ax.set_xlim(-0.6, len(G) - 0.4)
ax.set_ylabel('AUC against ArtiFact real faces')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_artifact.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{OUT}/fig3_artifact.pdf', bbox_inches='tight')
# Figure 4: held-out diffusion generator under degradation (folds A and B)
C = ['Clean', 'JPEG q30', 'Blur r=1', 'Noise σ=10', '0.5× + JPEG 70']
R = {'A': [(.933, .811, .917, .831, .860), (.987, .963, .981, .944, .970), (.989, .865, .956, .401, .904), (.913, .760, .495, .645, .631)],
     'B': [(.909, .839, .856, .847, .874), (.884, .840, .874, .758, .848), (.901, .796, .885, .473, .829), (.783, .638, .539, .512, .528)]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, f in zip(axes, 'AB'):
    for (lab, c, mk), v in zip(M, R[f]):
        ax.plot(range(len(C)), v, marker=mk, color=c, label=lab, lw=1.6, ms=5)
    ax.axhline(0.5, color='grey', lw=0.9, ls='--'); ax.set_ylim(0.35, 1.0)
    ax.set_xticks(range(len(C))); ax.set_xticklabels(C, rotation=25, ha='right')
    ax.set_title(f'Fold {f}: held-out SD {"1.5 text-to-image" if f == "A" else "inpainting"}', fontsize=9)
axes[0].set_ylabel('Test AUC')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_robustness.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{OUT}/fig4_robustness.pdf', bbox_inches='tight')
print('ok')
