"""Figure 1: CDG-DFD architecture (vector). Shapes and parameter counts match cdg_train.py."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
INK, MUTED, LINE = '#1F2933', '#5B6670', '#8A949E'
C_IN, C_DCT, C_SAM, C_BB, C_HEAD, C_DOM = '#EEF2F6', '#E3EEF9', '#FBE9DD', '#E6F1E8', '#EFE8F6', '#F4F4F4'

fig = plt.figure(figsize=(13.5, 6.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 135); ax.set_ylim(0, 62); ax.axis('off')


def box(x, y, w, h, fc, title, lines=(), dashed=False, bold=True, tsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc,
                                ec=LINE if not dashed else MUTED, lw=1.2, ls='--' if dashed else '-'))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=tsize, color=INK,
            weight='bold' if bold else 'normal')
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.2 - 3.3 * i, l, ha='center', va='top', fontsize=8.2, color=MUTED)


def arrow(x0, y0, x1, y1, dashed=False, color=INK, text=None, tx=0, ty=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11, lw=1.3, color=color,
                                 ls='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, ha='center', va='bottom', fontsize=7.8, color=MUTED)


# ---------------- main pipeline (top row) ----------------
Y, H = 36, 20
# input face crop: a simple face glyph
box(1, Y, 15, H, C_IN, 'Face crop', ['MTCNN, 224×224', 'RGB, X ∈ ℝ³ˣ²²⁴ˣ²²⁴'])

box(20, Y, 18, H, C_DCT, '2-D DCT + signed log', ['per channel: D = M X Mᵀ', 'D̃ = sign(D)·log(1+|D|)', '0 learnable params'])
# small spectrum glyph with radial rings
for r, a in zip(np.linspace(1, 4.6, 6), np.linspace(0.9, 0.25, 6)):
    ax.add_patch(plt.matplotlib.patches.Wedge((25.5, Y + 1.6), r, 0, 90, width=0.62, fc='#4C72B0', alpha=a, ec='none'))

box(42, Y, 22, H, C_SAM, 'Band spectral attention', ['16 radial bands × 3 channels', 'energy e ∈ ℝ⁴⁸ → MLP → w ∈ (0, 2)', 'D̂ = w ⊙ D̃', '3,152 params'])

box(68, Y, 20, H, C_BB, 'EfficientNet-B4', ['ImageNet init, no head', 'features 1792×7×7', '17.5 M params'])

box(92, Y, 14, H, C_IN, 'GAP + dropout', ['f ∈ ℝ¹⁷⁹²', 'p = 0.5'])

box(110, Y + 6, 23, 14, C_HEAD, 'Real / fake classifier', ['1792 → 256 → 2', 'loss: CE(ŷ, y)'])

for x0, x1 in [(16.4, 19.6), (38.4, 41.6), (64.4, 67.6), (88.4, 91.6)]:
    arrow(x0, Y + H / 2, x1, Y + H / 2)
arrow(106.4, Y + 13, 109.6, Y + 13)

# ---------------- training-only domain branch ----------------
box(110, 13, 23, 15, C_DOM, 'Domain classifier', ['1792 → 256 → 2', 'loss: λ·CE(d̂, d), λ = 0.1', 'training only'], dashed=True)
ax.add_patch(FancyBboxPatch((96.2, 17.2), 8.6, 6.6, boxstyle='round,pad=0.2,rounding_size=1', fc='white', ec=MUTED, lw=1.1, ls='--'))
ax.text(100.5, 21.9, 'GRL', ha='center', va='center', fontsize=9, weight='bold', color=INK)
ax.text(100.5, 18.7, '×(−α)', ha='center', va='center', fontsize=8, color=MUTED)
arrow(99, Y - 0.4, 99, 24.2, dashed=True, color=MUTED)
arrow(105.2, 20.5, 109.6, 20.5, dashed=True, color=MUTED)
ax.text(100.5, 14.8, 'reversed gradient pushes f\ntowards domain invariance', ha='center', va='center', fontsize=7.8,
        color=MUTED, style='italic')

# ---------------- band attention detail (bottom-left inset) ----------------
ix, iy, iw, ih = 1, 1.5, 87, 27
ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle='round,pad=0.25,rounding_size=1.2', fc='white', ec=LINE, lw=1, ls=':'))
ax.text(ix + 1.5, iy + ih - 1.5, 'Inside the band spectral attention', ha='left', va='top', fontsize=9.5, weight='bold', color=INK)
arrow(53, Y - 0.4, 53, iy + ih + 0.4, dashed=True, color=LINE)

# (a) radial band map
K, N = 16, 64
u = np.arange(N)
r = np.sqrt(u[:, None] ** 2 + u[None, :] ** 2) / (np.sqrt(2) * (N - 1))
band = np.clip((r * K).astype(int), 0, K - 1)
ia = fig.add_axes([3/135, 5.5/62, 12/135, 15/62])
ia.imshow(band, cmap='Blues_r', origin='upper', interpolation='nearest')
ia.set_xticks([]); ia.set_yticks([])
for s in ia.spines.values(): s.set_color(LINE)
ax.text(9, 2.2, 'band index b(u,v): 16 rings,\n0 = DC/low → 15 = high', ha='center', va='bottom', fontsize=7.6, color=MUTED)

# steps
steps = [(28, 'e_c,k = mean |D̃_c| over ring k', '48 numbers per image'),
         (48.5, 'z = W₂ ReLU(W₁e + b₁) + b₂', 'W₂, b₂ = 0 at init → w = 1'),
         (69, 'D̂_c(u,v) = w_c,b(u,v) · D̃_c(u,v)', 'one gain per ring and channel')]
for i, (x, t1, t2) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x - 9, 10.5), 18, 9, boxstyle='round,pad=0.2,rounding_size=1', fc=C_SAM, ec=LINE, lw=1))
    ax.text(x, 16.6, t1, ha='center', va='center', fontsize=7.6, color=INK)
    ax.text(x, 13.0, t2, ha='center', va='center', fontsize=7.4, color=MUTED)
for a0, a1 in [(16.0, 18.6), (37.3, 39.2), (57.8, 59.7)]:
    arrow(a0, 15, a1, 15)
ax.text(69, 26.0, 'learned: low bands damped (R band 0 ≈ 0.94),\nmid bands 6–11 boosted (up to ≈ 1.05)',
        ha='center', va='top', fontsize=7.4, color=MUTED, style='italic')

fig.savefig('fig1_architecture.pdf', bbox_inches='tight')
fig.savefig('fig1_architecture.png', bbox_inches='tight', dpi=300)
fig.savefig('fig1_architecture.svg', bbox_inches='tight')
print('saved')
