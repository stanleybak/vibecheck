"""vgg16-7 spec15 CASCADED patch refinement (M5 probe 2): refine relu
pre-act edges in order, INTERSECT into inter, and reforward with the
zono sym-budget forward carrying the refined bounds as CLAMPS before
each next edge -- downstream unstable counts collapse before their
(expensive) refinement. Ends with the output crown on the refined
intermediates. ab's target: root > 0 (ab: +5.33 in 1104s, no BaB)."""
import sys
import time

import torch

sys.path.insert(0, '/home/ubuntu/vc2/src')
from vibecheck2.core.graph import load  # noqa: E402
from vibecheck2.core import backward, forward  # noqa: E402
from vibecheck2.core.patches import patch_refine  # noqa: E402
from vibecheck2.frontend.vnnlib_loader import load_vnnlib  # noqa: E402

net = load('/home/ubuntu/vgg/vgg16-7.onnx')
sp = load_vnnlib('/home/ubuntu/vgg/spec15_tiger_beetle.vnnlib')
dev = 'cuda'
lo = torch.tensor(sp.x_lo, dtype=torch.float32, device=dev).reshape(1, -1)
hi = torch.tensor(sp.x_hi, dtype=torch.float32, device=dev).reshape(1, -1)
t0 = time.time()

relus = [nm for nm in net.order
         if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']
clamps = {}


def reforward():
    """interval reforward with the refined pre-act clamps intersected --
    seconds, and every downstream edge inherits the tightening."""
    st = backward.fwd.interval(net, lo, hi, return_state=True,
                               range_clamps=clamps)
    return backward._inter_from_state(net, lambda e: st[e])


import os
CKPT = '/home/ubuntu/gapwork/vgg_clamps.pt'
if os.path.exists(CKPT):
    clamps.update({k: (a.to(dev), b.to(dev)) for k, (a, b)
                   in torch.load(CKPT).items()})
    print(f'loaded {len(clamps)} clamped edges from checkpoint', flush=True)
inter = reforward()
print(f'[{time.time()-t0:6.1f}s] initial interval reforward', flush=True)
REFINE_CUTOFF = -1.0   # ab tolerates 744k unstable in the deep layers;
                        # refine only while inside this budget, then crown
for nm in relus:
    if nm in clamps:
        print(f'[{time.time()-t0:6.1f}s] {nm[:28]:30s} checkpointed, skip',
              flush=True)
        continue
    if time.time() - t0 > REFINE_CUTOFF:
        print(f'[{time.time()-t0:6.1f}s] refine cutoff; -> final crown',
              flush=True)
        break
    op = net.ops[nm]
    e = op.inputs[0]
    l0, h0 = inter[nm]
    u0 = int(((l0 < 0) & (h0 > 0)).sum())
    if u0 == 0:
        print(f'[{time.time()-t0:6.1f}s] {nm[:28]:30s} stable, skip',
              flush=True)
        continue
    eop = net.ops[e]
    if eop.kind != 'linmap' or not hasattr(eop.lm, 'out_shape'):
        print(f'[{time.time()-t0:6.1f}s] {nm[:28]:30s} SKIP (non-conv '
              f'pre-act: tree relu, bounds come from the conv cascade)',
              flush=True)
        continue
    try:
        t1 = time.time()
        lb, ub = patch_refine(net, e, lo, hi, inter)
        l1 = torch.maximum(l0, lb)
        h1 = torch.minimum(h0, ub)
        h1 = torch.maximum(h1, l1)
        clamps[nm] = (l1, h1)
        torch.cuda.empty_cache()
        torch.save({k: (a.cpu(), b.cpu()) for k, (a, b) in clamps.items()},
                   CKPT)
        inter = reforward()
        u1 = int(((l1 < 0) & (h1 > 0)).sum())
        print(f'[{time.time()-t0:6.1f}s] {nm[:28]:30s} unstable '
              f'{u0:8d} -> {u1:6d} ({time.time()-t1:.1f}s)', flush=True)
    except NotImplementedError as ex:
        print(f'[{time.time()-t0:6.1f}s] {nm[:28]:30s} SKIP ({ex})',
              flush=True)


# ab's dynamic-forward analogue for the DEEP unrefined layers: the
# interval reforward left them at 1e6 magnitudes and the final crown
# read -11.2M through their chords. sym-budget rad-zono (top-K input
# dims symbolic = ab's forward.max_dim=100) with the refined clamps
# gives sane-magnitude deep bounds; the crown consumes those.
t1 = time.time()
try:
    # GPU ground the allocator at 21.9GB for 10+ min; the earlier
    # scoping measured this exact pass at ~30s on CPU (124GB). Run it
    # there and move the merged bounds back.
    torch.set_num_threads(24)   # probe-only: the import pins BLAS to 1
    lo_c, hi_c = lo.cpu(), hi.cpu()
    cl_c = {k: (a.cpu(), b.cpu()) for k, (a, b) in clamps.items()}
    _l, _h, zst = forward.zono(net, lo_c, hi_c, return_state=True,
                               box_remainder='all', sym_budget=96,
                               clamp_bounds=cl_c)
    zi = {k: tuple(t.to(dev) for t in zst[k].bounds()) for k in zst}
    del zst
    torch.cuda.empty_cache()
    for k in inter:
        li, hi_ = inter[k]
        lz, hz = zi[k]
        inter[k] = (torch.maximum(li, lz), torch.minimum(hi_, hz))
    print(f'[{time.time()-t0:6.1f}s] sym-budget zono reforward merged '
          f'({time.time()-t1:.1f}s)', flush=True)
except Exception as ex:
    print(f'zono reforward failed: {type(ex).__name__}: {ex}', flush=True)

# COST PROBE: one q-chunk of dense identity crown at relu8 (ab computes
# every deep intermediate exactly this way; extrapolate chunk time)
relu8 = 'vgg0_relu8_fwd'
if relu8 in [nm for nm in net.order]:
    e8 = net.ops[relu8].inputs[0]
    n8 = net.ops[e8].n
    t1 = time.time()
    Wc = torch.zeros(4096, n8, device=dev)
    Wc[torch.arange(4096), torch.arange(4096)] = 1.0
    out = backward.crown(net, lo, hi, Wc.unsqueeze(0), inter, start=e8)
    dt = time.time() - t1
    print(f'[{time.time()-t0:6.1f}s] relu8 dense-crown chunk 4096/{n8}: '
          f'{dt:.1f}s -> full edge ~{dt * (2 * n8 / 4096):.0f}s '
          f'(x2 for ub)', flush=True)

# final output crown with the refined intermediates
from vibecheck2.frontend.spec import VNNSpec  # noqa: E402
qrows = sp.as_linear_queries(net.n_out)
import numpy as np  # noqa: E402
W = torch.tensor(np.stack([np.asarray(w).ravel() for _, w, _ in qrows]),
                 dtype=torch.float32, device=dev)
b = torch.tensor([float(bb) for _, _, bb in qrows], device=dev)
t1 = time.time()
lbq = backward.crown(net, lo, hi, W.unsqueeze(0), inter)
print(f'[{time.time()-t0:6.1f}s] FINAL crown lbq+bias ='
      f' {[round(float(v), 3) for v in (lbq[0] + b).tolist()]}'
      f' (crown {time.time()-t1:.1f}s; >0 everywhere = verified)',
      flush=True)
