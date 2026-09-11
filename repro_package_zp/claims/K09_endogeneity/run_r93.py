"""K09 run_r93.py — fixedWS release (撤钉 falsification gate).

Package port of repro/r93_fixedws_release.py (M2.5, 2026-09-02).
Verbatim except: common/ imports, repro paths via REPRO_DIR (the
prefix gate reads the ARCHIVED r92 pkl -- certifying this rerun's
phase-1 directly against the archive), claim-dir output, SMOKE env
R93_SMOKE -> K09R93_SMOKE.

Design per (seed, d in {2000, 5000}), CAP=14000: phase 1 (1..d)
fixedWS-eq training verbatim R92 (streams identical: fresh_model,
pi_0 stream 7000+seed, one randint per step -> prefix bitwise vs
archived r92 fixedWS); phase 2 (d+1..CAP) surgery removed, plain
native training on the SAME batch chain.

Claim (ledger K9, part 2): release yields a reset-like long-delay
trajectory (escape 8.5-10.6k in archive) -- no blocking scar, no
transferable shortcut.  Preregistered §6db; verdict §6dd(+dd-note).
"""
import hashlib
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task
from common import gates as cgates
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)
SMOKE = bool(os.environ.get('K09R93_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
EVAL_EVERY = 25
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else (0, 1, 2)
DS = (40,) if SMOKE else (2000, 5000)
P = task.GROUP['p']

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K09-r93] SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')
print(f'[K09-r93] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} d={DS} '
      f'CAP={CAP} BATCH={BATCH}', flush=True)

R = ZPReadouts(DEV)
zp = cgates.ZPIndexing()
N_PAIRS, N_TRAIN = R.N_PAIRS, R.N_TRAIN
IDX, Y_ARR, S_ARR, DIAG, ORB_REPS = (zp.IDX, zp.Y_ARR, zp.S_ARR,
                                     zp.DIAG, zp.ORB_REPS)
plain_forward_attn = R.plain_forward_attn


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


def val_probe(seed):
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    rngv = np.random.default_rng(555)
    vb = rngv.permutation(perm[N_TRAIN:])[:1024]
    return (torch.from_numpy(all_a[vb]).long().to(DEV),
            torch.from_numpy(all_b[vb]).long().to(DEV),
            torch.from_numpy(all_y[vb]).long().to(DEV),
            np.asarray(vb, dtype=np.int64))


def surgery_forward(m, a, b, src_a, src_b):
    """Second-forward in-graph surgery (r87/r92 verbatim)."""
    B = a.shape[0]
    H, dh = m.n_heads, m.d_head
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    h = m.ln1(x)
    q = m.Wq(h).view(B, 3, H, dh).transpose(1, 2)
    k = m.Wk(h).view(B, 3, H, dh).transpose(1, 2)
    v = m.Wv(h).view(B, 3, H, dh).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    attn_main = scores.softmax(dim=-1)
    eq_s = torch.full_like(src_a, task.N_EQ)
    tok_s = torch.stack([src_a, src_b, eq_s], dim=1)
    x_s = m.emb(tok_s) + m.pos[None, :, :]
    h_s = m.ln1(x_s)
    q_s = m.Wq(h_s).view(B, 3, H, dh).transpose(1, 2)
    k_s = m.Wk(h_s).view(B, 3, H, dh).transpose(1, 2)
    scores_s = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(dh)
    a2_src = scores_s.softmax(dim=-1)[:, :, 2, :]
    attn = torch.cat([attn_main[:, :, :2, :], a2_src.unsqueeze(2)],
                     dim=2)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, 3, m.d_model)
    x = x + m.Wo(out)
    h2 = m.ln2(x)
    mm = m.mlp2(F.gelu(m.mlp1(h2)))
    x = x + mm
    last = x[:, 2, :]
    return m.Wu(m.ln_f(last))


def classify_train(tr_mask):
    full = {c: [] for c in range(P)}
    half = {c: [] for c in range(P)}
    diag_T = DIAG[tr_mask[DIAG]]
    for rep in ORB_REPS:
        i, j = int(rep), int(S_ARR[rep])
        mi, mj = bool(tr_mask[i]), bool(tr_mask[j])
        c = int(Y_ARR[rep])
        if mi and mj:
            full[c].append(rep)
        elif mi:
            half[c].append(i)
        elif mj:
            half[c].append(j)
    U = set(int(x) for x in diag_T)
    for c in range(P):
        if len(half[c]) == 1:
            U.add(half[c][0])
    return full, half, U


def sample_pi(seed, draw, full, half):
    if draw == 0:
        rng = np.random.default_rng(7000 + seed)
    else:
        rng = np.random.default_rng(np.random.SeedSequence(
            [7500 + seed, draw]))
    pi = IDX.copy()
    for c in range(P):
        fo = full[c]
        if len(fo) >= 2:
            k = len(fo)
            permk = rng.permutation(k)
            tgt = np.roll(permk, 1)
            orient = rng.integers(0, 2, size=k)
            for m in range(k):
                i = int(fo[permk[m]])
                j = int(fo[tgt[m]])
                if orient[m] == 0:
                    pi[i], pi[S_ARR[i]] = j, S_ARR[j]
                else:
                    pi[i], pi[S_ARR[i]] = S_ARR[j], j
        elif len(fo) == 1:
            i = int(fo[0])
            pi[i], pi[S_ARR[i]] = int(S_ARR[i]), i
        hh = half[c]
        if len(hh) >= 2:
            kk = len(hh)
            permh = rng.permutation(kk)
            pi[np.asarray(hh)[permh]] = np.asarray(hh)[
                np.roll(permh, 1)]
    return pi


def _strat_stats(correct, self_in, swap_in, diag):
    m1 = ~self_in & swap_in & ~diag
    m2 = ~self_in & ~swap_in & ~diag
    m3 = ~self_in & diag
    return {'train_acc': float(correct[self_in].float().mean()),
            'va_grid': float(correct[~self_in].float().mean()),
            'm1': float(correct[m1].float().mean()),
            'm2': float(correct[m2].float().mean()),
            'm3': float(correct[m3].float().mean())}


def run_release(seed, d):
    t0 = time.time()
    va_a, va_b, va_y, va_ids = val_probe(seed)
    va = (va_a, va_b, va_y)
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    id_train = perm[:N_TRAIN]                 # UNSORTED
    tr_a = torch.from_numpy(all_a[id_train]).long()
    tr_b = torch.from_numpy(all_b[id_train]).long()
    tr_y = torch.from_numpy(all_y[id_train]).long()
    tr = np.sort(id_train)
    tr_mask = np.zeros(N_PAIRS, dtype=bool)
    tr_mask[tr] = True
    full, half, U = classify_train(tr_mask)
    pi0 = sample_pi(seed, 0, full, half)
    src = np.full(N_PAIRS, -1, dtype=np.int64)
    src[tr] = pi0[tr]
    all_a_dev = torch.from_numpy(all_a).long().to(DEV)
    all_b_dev = torch.from_numpy(all_b).long().to(DEV)

    m = fresh_model(seed)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    rec = {'t': [], 'va_plain': [], 'cross95': None,
           'cross95_post': None, 'strat': {}}
    ga = torch.arange(P).repeat_interleave(P)
    gb = torch.arange(P).repeat(P)
    gy = (ga + gb) % P
    gidx = torch.arange(P * P)
    grid_ids = ((gidx % P) * P + (gidx // P)).numpy()
    ga_dev = ga.long().to(DEV)
    gb_dev = gb.long().to(DEV)
    tr_mask_g = torch.from_numpy(tr_mask[grid_ids]).bool()
    tr_mask_swap_g = torch.from_numpy(tr_mask).bool()
    diag_grid = (ga == gb)

    for step in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (BATCH,))
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        m.train()
        opt.zero_grad()
        if step <= d:
            tid = id_train[idx.numpy()]
            img_ids = src[tid]
            assert tr_mask[img_ids].all(), 'G-LEAK FAIL'
            img_t = torch.from_numpy(img_ids).long().to(DEV)
            logits = surgery_forward(m, a, b, all_a_dev[img_t],
                                     all_b_dev[img_t])
        else:
            logits, _ = plain_forward_attn(m, a, b)
        LOSS(logits, y).backward()
        if step <= 20:
            m.pos.grad[2].zero_()
        opt.step()
        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                m.eval()
                lp = float((m(va[0], va[1]).argmax(-1) == va[2])
                           .float().mean())
                rec['t'].append(step)
                rec['va_plain'].append(lp)
                if lp > 0.95 and rec['cross95'] is None:
                    rec['cross95'] = step
                if lp > 0.95 and step > d and \
                        rec['cross95_post'] is None:
                    rec['cross95_post'] = step
                m.train()
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                m.eval()
                lg, _ = plain_forward_attn(m, ga_dev, gb_dev)
                rec['strat'][step] = _strat_stats(
                    lg.argmax(-1).cpu() == gy, tr_mask_g,
                    tr_mask_swap_g, diag_grid)
                m.train()
        if step % 1000 == 0:
            last = rec['strat'][sorted(rec['strat'].keys())[-1]]
            print(f"  [s{seed} d{d} t={step}] "
                  f"va={rec['va_plain'][-1]:.2f} "
                  f"m1={last['m1']:.2f}", flush=True)

    rec['escape'] = (rec['cross95_post'] - d
                     if rec['cross95_post'] is not None else None)
    print(f"[s{seed} d{d} done in {time.time() - t0:.0f}s] "
          f"T95_post={rec['cross95_post']} "
          f"escape={rec['escape']}", flush=True)
    return rec


def prefix_gate(out):
    """Prefix (t <= d) va_plain must bitwise-match ARCHIVED r92
    fixedWS arm."""
    with open(os.path.join(REPRO_DIR,
                           'r92_native_ingraph_ws_results.pkl'),
              'rb') as f:
        r92 = pickle.load(f)
    ok_all = True
    for seed in SEEDS:
        ref = r92[seed]['arms']['fixedWS']
        for d in DS:
            new = out[seed][d]
            n = len([t for t in ref['t'] if t <= d])
            same_t = new['t'][:n] == ref['t'][:n]
            dv = max((abs(x - y) for x, y in
                      zip(new['va_plain'][:n],
                          ref['va_plain'][:n])), default=0.0)
            ok = same_t and dv == 0.0
            print(f"prefix-gate seed{seed} d{d}: bitwise={ok} "
                  f"(n={n} maxdv={dv:.2e})", flush=True)
            ok_all &= ok
    return ok_all


def main():
    out = {seed: {} for seed in SEEDS}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_r93.pkl')
    for seed in SEEDS:
        for d in DS:
            out[seed][d] = run_release(seed, d)
            with open(out_path, 'wb') as f:
                pickle.dump(out, f)

    print('\n=== escape table ===', flush=True)
    escapes = []
    for seed in SEEDS:
        for d in DS:
            e = out[seed][d]['escape']
            escapes.append(e)
            m1_end = out[seed][d]['strat'][
                max(out[seed][d]['strat'].keys())]['m1']
            print(f"  seed{seed} d{d}: T95_post="
                  f"{out[seed][d]['cross95_post']} escape={e} "
                  f"plain_m1_final={m1_end:.3f}", flush=True)
    if all(e is not None and e <= 4000 for e in escapes):
        print("VERDICT: CONTINUOUS DEPENDENCE, NO PERMANENT SCAR -- "
              "fixedWS branch closes", flush=True)
    elif sum(1 for e in escapes
             if e is None or e > 6000) >= 2:
        print("VERDICT: SCAR -- crutch explanation insufficient",
              flush=True)
    else:
        print("VERDICT: middle band -- register per cell", flush=True)
    if not SMOKE:
        g = prefix_gate(out)
        print('PREFIX GATES', 'PASS' if g else 'FAIL', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results_r93.pkl', flush=True)


if __name__ == '__main__':
    main()
