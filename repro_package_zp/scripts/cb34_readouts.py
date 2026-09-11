"""scripts/cb34_readouts.py -- CB3 (lookup M2y) + CB4 (Fourier ladder)
under wd_1111.  Both are zero-training readouts on a mature wd_1111
checkpoint (the natS arm at its final saved step from Stage W1).

CB3 (M2y): global additive decomposition of the MLP write, then four
  frozen-forward arms: NATIVE / DELTA-ONLY / SAME-Y / CENTROID.
  Gate: DELTA-ONLY >> SAME-Y/CENTROID => pair-specific lookup holds.

CB4 (Fourier L1-L4): train-defined class-centroid field -> DFT
  spectral concentration (L1), top-K reconstruction (L2), 2D diagonal
  factorization (L3), KEEP-K/DROP-K causal filter (L4).

Usage:
  python scripts/cb34_readouts.py --bridge-dir DIR --out DIR \
      --seeds 0,1,2 --mode cb3 [--device cuda]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from common import task
from common.model import D57Model

task.set_group('zp', 113)
P = 113
T_N_EQ = 113


def load_wd1111(bridge_dir, seed, tag_suffix='final', dev='cpu'):
    d = os.path.join(bridge_dir, 'bridge_wd1111_A')
    fps = [f for f in os.listdir(d)
           if f.startswith(f'seed{seed}_') and tag_suffix in f]
    assert len(fps) >= 1, f'{d}: {fps}'
    fp = sorted(fps)[-1]  # take the last one (final > snapshot)
    ck = torch.load(os.path.join(d, fp), map_location='cpu',
                    weights_only=False)
    m = D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                 eq_alpha=0.0)
    m.load_state_dict(ck['model'])
    return m.to(dev).eval()


def full_grid_states(m, dev):
    ids = np.arange(P * P)
    x2_all, m_all = [], []
    for s in range(0, P * P, 4096):
        a = torch.from_numpy(ids[s:s + 4096] % P).long().to(dev)
        b = torch.from_numpy(ids[s:s + 4096] // P).long().to(dev)
        with torch.no_grad():
            eq = torch.full_like(a, T_N_EQ)
            tok = torch.stack([a, b, eq], dim=1)
            x = m.emb(tok) + m.pos[None, :, :]
            h = m.ln1(x)
            H, dh = m.n_heads, m.d_head
            q = m.Wq(h).view(-1, 3, H, dh).transpose(1, 2)
            k = m.Wk(h).view(-1, 3, H, dh).transpose(1, 2)
            v = m.Wv(h).view(-1, 3, H, dh).transpose(1, 2)
            attn = ((q @ k.transpose(-1, -2)) / (dh ** 0.5))\
                .softmax(dim=-1)
            o = (attn @ v).transpose(1, 2).contiguous()\
                .view(-1, 3, H * dh)
            x = x + m.Wo(o)
            x2 = x[:, 2, :]
            mo = m.mlp2(torch.nn.functional.gelu(
                m.mlp1(m.ln2(x)[:, 2, :])))
        x2_all.append(x2.cpu().numpy())
        m_all.append(mo.cpu().numpy())
    return (np.concatenate(x2_all).astype(np.float64),
            np.concatenate(m_all).astype(np.float64))


def additive_global(Mnp):
    n = len(Mnp)
    ua, ia = np.unique(np.arange(P)[None, :].repeat(P, 0).ravel(),
                       return_inverse=True)
    ub, ib = np.unique(np.arange(P)[:, None].repeat(P, 1).ravel(),
                       return_inverse=True)
    X = np.zeros((n, 1 + len(ua) + len(ub)))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + ia] = 1.0
    X[np.arange(n), 1 + len(ua) + ib] = 1.0
    coef, *_ = np.linalg.lstsq(X, Mnp, rcond=None)
    return X @ coef


def read_w(m, x2_t, w_t, dev):
    with torch.no_grad():
        return m.Wu(m.ln_f(x2_t + w_t))


def strat_acc(logits, y_all, masks):
    pred = logits.argmax(-1).cpu().numpy() == ALL_Y
    tr, m1m, m2m = masks
    return {'train_acc': float(pred[tr].mean()),
            'm1': float(pred[m1m].mean()) if m1m.any() else float('nan'),
            'm2': float(pred[m2m].mean()) if m2m.any() else float('nan'),
            'top1': float(pred.mean())}


def run_cb3(m, dev, seed):
    ALL_Y = task.group_mul_np(
        np.arange(P)[None, :].repeat(P, 0).ravel(),
        np.arange(P)[:, None].repeat(P, 1).ravel())
    x2_np, m_np = full_grid_states(m, dev)
    m_add = additive_global(m_np)
    gam = m_np - m_add
    yv = ALL_Y

    gbar = np.zeros((P, gam.shape[1]))
    np.add.at(gbar, yv, gam)
    cnt = np.bincount(yv, minlength=P)
    gbar /= cnt[:, None]
    gbar_of = gbar[yv]
    delta = gam - gbar_of

    # same-y derangement within m1 stratum
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(P * P)
    n_tr = int(round(P * P * 0.30))
    tr_set = set(perm[:n_tr].tolist())
    tr = np.zeros(P * P, dtype=bool)
    tr[perm[:n_tr]] = True
    diag = (np.arange(P * P) % P) == (np.arange(P * P) // P)
    swap_of = (np.arange(P * P) % P) * P + np.arange(P * P) // P
    swap_in = tr[swap_of] & ~diag
    m1m = ~tr & swap_in
    m2m = ~tr & ~swap_in & ~diag

    rng2 = np.random.default_rng(8800 + seed)
    sig = np.arange(P * P)
    for y in range(P):
        grp = np.where(m1m & (yv == y))[0]
        if len(grp) >= 2:
            off = int(rng2.integers(1, len(grp)))
            sig[grp] = np.roll(grp, off)
    del_sy = delta[sig]

    x2_t = torch.from_numpy(x2_np.astype(np.float32)).to(dev)
    add_t = torch.from_numpy(m_add.astype(np.float32)).to(dev)
    gam_t = torch.from_numpy(gam.astype(np.float32)).to(dev)
    gbar_t = torch.from_numpy(gbar_of.astype(np.float32)).to(dev)
    del_t = torch.from_numpy(delta.astype(np.float32)).to(dev)
    sy_t = torch.from_numpy(del_sy.astype(np.float32)).to(dev)
    y_all = torch.from_numpy(ALL_Y).long().to(dev)
    masks = (tr, m1m, m2m)

    arms = {
        'NATIVE': read_w(m, x2_t, add_t + gam_t, dev),
        'REMOVE': read_w(m, x2_t, add_t, dev),
        'DELTA-ONLY': read_w(m, x2_t, add_t + del_t, dev),
        'SAME-Y': read_w(m, x2_t, add_t + sy_t, dev),
        'CENTROID': read_w(m, x2_t, add_t + gbar_t, dev),
    }
    rec = {}
    for name, lg in arms.items():
        pred = lg.argmax(-1).cpu().numpy() == ALL_Y
        rec[name] = {
            'm1': float(pred[m1m].mean()) if m1m.any() else float('nan'),
            'm2': float(pred[m2m].mean()) if m2m.any() else float('nan'),
            'train_acc': float(pred[tr].mean())}
    d_only = rec['DELTA-ONLY']['m1']
    c_max = max(rec['SAME-Y']['m1'], rec['CENTROID']['m1'])
    rec['verdict'] = ('PAIR-SPECIFIC' if d_only > 0.7 and c_max < 0.5
                      else 'CLASS-SHARED' if c_max > 0.7 and d_only < 0.3
                      else 'MIXED')
    return rec


def run_cb4(m, dev, seed, train_mask):
    ALL_Y = task.group_mul_np(
        np.arange(P)[None, :].repeat(P, 0).ravel(),
        np.arange(P)[:, None].repeat(P, 1).ravel())
    yv = ALL_Y
    x2_np, m_np = full_grid_states(m, dev)
    m_add = additive_global(m_np)
    gam = m_np - m_add
    gbar = np.zeros((P, gam.shape[1]))
    tr_ids = np.where(train_mask)[0]
    np.add.at(gbar, yv[tr_ids], gam[tr_ids])
    cnt = np.bincount(yv[tr_ids], minlength=P)
    gbar /= np.maximum(cnt, 1)[:, None]
    gbar_pair = gbar[yv]

    # L1: DFT over class axis
    F = np.fft.fft(gbar, axis=0) / P
    ek = np.linalg.norm(F, axis=1)
    e_pair = np.zeros(P // 2 + 1)
    e_pair[0] = ek[0]
    for k in range(1, P // 2 + 1):
        e_pair[k] = ek[k] + (ek[P - k] if k != P - k else 0.0)
    order = np.argsort(e_pair)[::-1]
    cum = np.cumsum(e_pair[order]) / (e_pair.sum() + 1e-30)
    n50 = int(np.searchsorted(cum, 0.50)) + 1

    # L2: reconstruction K=8
    K = 8
    keep = np.zeros(P, dtype=bool)
    for i in range(min(K, len(order))):
        k = order[i]
        keep[k] = True
        if k != 0 and k != P - k:
            keep[P - k] = True
    Fk = F.copy()
    Fk[~keep] = 0
    gbK = np.fft.ifft(Fk, axis=0).real * P
    gbK_pair = gbK[yv]

    x2_t = torch.from_numpy(x2_np.astype(np.float32)).to(dev)
    add_t = torch.from_numpy(m_add.astype(np.float32)).to(dev)
    gfull_t = torch.from_numpy(gbar_pair.astype(np.float32)).to(dev)
    gk_t = torch.from_numpy(gbK_pair.astype(np.float32)).to(dev)
    y_all = torch.from_numpy(ALL_Y).long().to(dev)

    def meas(w_t):
        with torch.no_grad():
            lg = m.Wu(m.ln_f(x2_t + w_t))
        pred = lg.argmax(-1).cpu().numpy() == ALL_Y
        return float(pred.mean())

    rec_l2 = {'full_centroid': meas(add_t + gfull_t),
              'K8_reconstruction': meas(add_t + gk_t),
              'no_interaction': meas(add_t)}

    # L3: 2D diagonal factorization on Gamma
    gam2 = gam.reshape(P, P, -1)
    fa = gam2.mean(axis=1, keepdims=True)
    fb = gam2.mean(axis=0, keepdims=True)
    fg = gam2.mean(axis=(0, 1), keepdims=True)
    Fperp = gam2 - fa - fb + fg
    Fhat = np.fft.fft2(Fperp, axes=(0, 1)) / (P * P)
    diag_e = tot_e = 0.0
    for k in range(P):
        for l in range(P):
            if k == 0 or l == 0:
                continue
            e = float(np.linalg.norm(Fhat[k, l]) ** 2)
            tot_e += e
            if k == l or k == (P - l) % P:
                diag_e += e
    rec_l3 = {'diag_share': diag_e / (tot_e + 1e-30)}

    return {'L1_n50': n50, 'L2': rec_l2, 'L3_diag_share': rec_l3}


ALL_Y = None  # set in main()


def main():
    global ALL_Y
    ap = argparse.ArgumentParser()
    ap.add_argument('--bridge-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--mode', required=True, choices=['cb3', 'cb4'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    dev = torch.device(args.device if torch.cuda.is_available()
                       else 'cpu')
    ALL_Y = task.group_mul_np(
        np.arange(P)[None, :].repeat(P, 0).ravel(),
        np.arange(P)[:, None].repeat(P, 1).ravel())

    out = {}
    for seed in seeds:
        m = load_wd1111(args.bridge_dir, seed, dev=dev)
        if args.mode == 'cb3':
            out[seed] = run_cb3(m, dev, seed)
        else:
            # need train mask for CB4 train-defined centroid
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            perm = rng.permutation(P * P)
            tr = np.zeros(P * P, dtype=bool)
            tr[perm[:int(round(P * P * 0.30))]] = True
            out[seed] = run_cb4(m, dev, seed, tr)
        print(f'  seed{seed} {args.mode}: '
              f'{json.dumps(out[seed], default=str)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, f'{args.mode}_results.json')
    with open(fp, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'{fp} written', flush=True)


if __name__ == '__main__':
    main()
