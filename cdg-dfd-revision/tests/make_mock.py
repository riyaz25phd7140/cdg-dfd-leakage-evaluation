"""Builds a small mock of the real dataset layouts (with real faces, so MTCNN runs) for testing."""
import os, shutil, sys
import numpy as np, cv2
from PIL import Image, ImageOps, ImageEnhance
from skimage import data

root = sys.argv[1]
shutil.rmtree(root, ignore_errors=True)
rng = np.random.default_rng(0)
base = Image.fromarray(data.astronaut())


def variant(seed, noise_face=False):
    r = np.random.default_rng(seed)
    if noise_face:  # no face: MTCNN should fail
        return Image.fromarray(r.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    im = base.copy()
    if r.random() < 0.5:
        im = ImageOps.mirror(im)
    im = im.rotate(r.uniform(-12, 12), resample=Image.BICUBIC)
    im = ImageEnhance.Color(im).enhance(r.uniform(0.2, 1.8))
    im = ImageEnhance.Brightness(im).enhance(r.uniform(0.6, 1.4))
    s = r.uniform(0.55, 1.0)
    im = im.resize((int(512 * s), int(512 * s)))
    canvas = Image.new('RGB', (640, 560), tuple(int(x) for x in r.integers(0, 255, 3)))
    canvas.paste(im, (int(r.integers(0, 640 - im.size[0])), int(r.integers(0, 560 - im.size[1]))))
    arr = np.asarray(canvas).astype(np.float32) + r.normal(0, r.uniform(2, 20), (560, 640, 3))
    # coarse random blotches so dHash differs between variants
    for _ in range(6):
        x, y = r.integers(0, 600), r.integers(0, 520)
        arr[y:y + 40, x:x + 40] = r.integers(0, 255, 3)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


seed = 0
def save(im, path, fmt='JPEG', q=90):
    global seed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == 'JPEG':
        im.save(path, quality=q)
    else:
        im.save(path)

# HardFake: 20 real JPEG q75, 24 fake JPEG q75, one no-face fake
for i in range(20):
    save(variant(1000 + i), f'{root}/hardfake/real/real_{i:03d}.jpg', q=75)
for i in range(24):
    save(variant(2000 + i, noise_face=(i == 0)), f'{root}/hardfake/fake/fake_{i:03d}.jpg', q=75)

# DFF: 30 identities, 1-3 images each; fakes mirror wiki paths (text2img as PNG to trigger format warning)
names = []
for pid in range(30):
    for k in range(1 + pid % 3):
        names.append((f'{pid % 4:02d}', f'{10000000 + pid}_19{60 + k}-01-0{k + 1}_2010'))
for j, (folder, stem) in enumerate(names):
    # like the real DFF: reals vary in size/quality, every fake is 512x512 JPEG q75
    save(variant(3000 + j, noise_face=(j == 5)), f'{root}/dff/wiki/wiki/{folder}/{stem}.jpg', q=int(rng.integers(80, 96)))
    save(variant(4000 + j).resize((512, 512)), f'{root}/dff/text2img/text2img/{folder}/{stem}.jpg', q=75)
    save(variant(5000 + j).resize((512, 512)), f'{root}/dff/inpainting/inpainting/{folder}/{stem}.jpg', q=75)
    if j < 20:
        save(variant(6000 + j).resize((512, 512)), f'{root}/dff/insight/insight/{folder}/{stem}.jpg', q=75)
# exact duplicate file inside wiki (different name, same bytes)
shutil.copy(f'{root}/dff/wiki/wiki/{names[0][0]}/{names[0][1]}.jpg', f'{root}/dff/wiki/wiki/03/dupcopy_x.jpg')
# near duplicate: re-encode an inpainting image at lower quality under another identity
Image.open(f'{root}/dff/inpainting/inpainting/{names[3][0]}/{names[3][1]}.jpg').save(
    f'{root}/dff/inpainting/inpainting/03/10009999_1999-01-01_2010.jpg', quality=74)

# CelebDF: real ids 0..5, YouTube-real 3 videos, fakes id_a_id_b
def write_video(path, seed0, n=24):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 12, (640, 560))
    for f in range(n):
        vw.write(cv2.cvtColor(np.asarray(variant(seed0 + f)), cv2.COLOR_RGB2BGR))
    vw.release()

vids = []
for i in range(6):
    vids.append(f'Celeb-real/id{i}_0000.mp4')
for i in range(3):
    vids.append(f'YouTube-real/{i:05d}.mp4')
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 3)]:
    vids.append(f'Celeb-synthesis/id{a}_id{b}_0000.mp4')
for n, v in enumerate(vids):
    write_video(f'{root}/celebdf/{v}', 9000 + 100 * n)
test = ['Celeb-real/id5_0000.mp4', 'YouTube-real/00002.mp4', 'Celeb-synthesis/id5_id0_0000.mp4',
        'Celeb-synthesis/id4_id5_0000.mp4']
with open(f'{root}/celebdf/List_of_testing_videos.txt', 'w') as f:
    for v in test:
        f.write(f'{0 if "synthesis" in v else 1} {v}\n')

# external test-only set
for i in range(6):
    save(variant(7000 + i), f'{root}/ext_df40/fake/img_{i}.png', fmt='PNG')
print('mock written to', root)
