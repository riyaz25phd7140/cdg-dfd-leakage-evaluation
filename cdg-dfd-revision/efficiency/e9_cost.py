"""E9: parameters, FLOPs and CPU latency per stage of the Detector (counts are hardware independent)."""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
from torch.utils.flop_counter import FlopCounterMode
import cdg_train as T

torch.set_num_threads(4)
def flops(mod, *inp):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad(): mod(*inp)
    return fc.get_total_flops()

def lat(fn, n=10):
    with torch.no_grad():
        for _ in range(2): fn()
        t = time.perf_counter()
        for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

rows = []
cfgs = {'rgb': dict(spectral='none', input_mode='rgb'),
        'dct': dict(spectral='none', input_mode='dct'),
        'dct+se': dict(spectral='none', input_mode='dct', channel_attn='se'),
        'dct+band (full)': dict(spectral='band', input_mode='dct')}
x = torch.randn(1, 3, 224, 224)
for name, kw in cfgs.items():
    m = T.Detector(pretrained=False, **kw).eval()
    p_all = sum(p.numel() for p in m.parameters())
    p_inf = p_all - sum(p.numel() for p in m.domain_classifier.parameters())
    f = flops(m, x)
    l1 = lat(lambda: m(x))
    xb = torch.randn(8, 3, 224, 224)
    l8 = lat(lambda: m(xb), n=3) / 8
    rows.append(dict(config=name, params_total=p_all, params_inference=p_inf,
                     gflops=round(f / 2e9, 3), cpu_ms_b1=round(l1, 1), cpu_ms_per_img_b8=round(l8, 1)))
    print(rows[-1], flush=True)

# per-stage for the full model
m = T.Detector(pretrained=False).eval()
d = m.dct(x); b, _ = m.band_attention(d)
stage = {'DCT': (m.dct, (x,)), 'band attention': (m.band_attention, (d,)),
         'backbone': (m.backbone, (b.contiguous(memory_format=torch.channels_last),))}
for k, (mod, inp) in stage.items():
    rows.append(dict(config='stage: ' + k, params_total=sum(p.numel() for p in mod.parameters()),
                     gflops=round(flops(mod, *inp) / 2e9, 4), cpu_ms_b1=round(lat(lambda: mod(*inp)), 2)))
    print(rows[-1], flush=True)
se = T.SpectralAttention(1792).eval(); fm = torch.randn(1, 1792, 7, 7)
rows.append(dict(config='stage: SE channel attention', params_total=sum(p.numel() for p in se.parameters()),
                 gflops=round(flops(se, fm) / 2e9, 5), cpu_ms_b1=round(lat(lambda: se(fm)), 3)))
print(rows[-1])
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'E9_cpu.json'), 'w'), indent=1)
