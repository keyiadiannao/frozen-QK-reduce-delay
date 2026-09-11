"""claims/M11_wr_factorial/run.py — S2 secondary replication
(R136b / M11 port for the 10-seed confirmatory).

Protocol source: repro/r136b_kw_read_factorial.py (§6i AC/AD),
verbatim except the registered deviations (see M10 run.py docstring;
same four).  Judge per REPRO_PLAN §10.2a (frozen 2026-09-05):

  S2 gate: I_match = A_LL - A_PL - A_LP + A_PP > 0 in >= 7/10 seeds
  (raw A_ij; RMS-matched control is reported in full but is NOT a
  gate).  P/L offsets per §10.2a; K from Stage 4 frozen_k.

Cell definition (r136b verbatim): A_ij = median novel margin of
reader-state j under (reader's own additive baseline + source i's
pure K-component gamK).  RMS-matched pass rescales both sources to
the smaller of the two K-RMS values.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task                                    # noqa: E402
from common.model import D57Model                          # noqa: E402
from common.readouts import ZPReadouts                     # noqa: E402

DEV = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
torch.set_num_threads(4)

task.set_group('zp', 113)
R = ZPReadouts(DEV)
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
N_PAIRS = R.N_PAIRS
P = 113
T_N_EQ = 113


def k_mask(seed, K):
    idx = np.indices((P, P))
    on = np.zeros((P, P), dtype=bool)
    for k in K:
        on |= (idx[0] == k) & (idx[1] == k)
        on |= (idx[0] == k) & (idx[1] == (P - k) % P)
        on |= (idx[0] == (P - k) % P) & (idx[1] == k)
        on |= (idx[0] == (P - k) % P) & (idx[1] == (P - k) % P)
    unary = (idx[0] == 0) | (idx[1] == 0)
    return on & ~unary


def strata_masks(seed):
    rng2 = np.random.default_rng(np.random.SeedSequence(seed))
    pm = rng2.permutation(N_PAIRS)
    train = np.zeros(N_PAIRS, bool)
    train[np.sort(pm[:R.N_TRAIN])] = True
    diag = ALL_A == ALL_B
    swap_of = ALL_A * P + ALL_B
    swap_in = train[swap_of] & ~diag
    return train, ~train & swap_in, ~train & ~swap_in & ~diag


def full_parts(m):
    ids = np.arange(N_PAIRS)
    m_all, x2_all = [], []
    CH = 4096
    for s in range(0, N_PAIRS, CH):
        a = torch.from_numpy(ALL_A[ids[s:s + CH]]).long().to(DEV)
        b = torch.from_numpy(ALL_B[ids[s:s + CH]]).long().to(DEV)
        n = len(a)
        with torch.no_grad():
            eq = torch.full_like(a, T_N_EQ)
            tok = torch.stack([a, b, eq], dim=1)
            x = m.emb(tok) + m.pos[None, :, :]
            h = m.ln1(x)
            H, dh = m.n_heads, m.d_head
            q = m.Wq(h).view(n, 3, H, dh).transpose(1, 2)
            k = m.Wk(h).view(n, 3, H, dh).transpose(1, 2)
            v = m.Wv(h).view(n, 3, H, dh).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / (dh ** 0.5))\
                .softmax(dim=-1)
            o = (att @ v).transpose(1, 2).contiguous()\
                .view(n, 3, H * dh)
            x = x + m.Wo(o)
            x2 = x[:, 2, :]
            mo = m.mlp2(torch.nn.functional.gelu(
                m.mlp1(m.ln2(x)[:, 2, :])))
        m_all.append(mo.cpu().numpy())
        x2_all.append(x2.cpu().numpy())
    return (np.concatenate(m_all).astype(np.float64),
            np.concatenate(x2_all).astype(np.float64))


def additive_of(F):
    ua, ia = np.unique(ALL_A, return_inverse=True)
    ub, ib = np.unique(ALL_B, return_inverse=True)
    X = np.zeros((N_PAIRS, 1 + len(ua) + len(ub)))
    X[:, 0] = 1.0
    X[np.arange(N_PAIRS), 1 + ia] = 1.0
    X[np.arange(N_PAIRS), 1 + len(ua) + ib] = 1.0
    coef, *_ = np.linalg.lstsq(X, F, rcond=None)
    return X @ coef, F - X @ coef


def k_component(gam, seed, K):
    mask = k_mask(seed, K)
    F2 = np.fft.fft2(gam.reshape(P, P, -1), axes=(0, 1))
    keep = np.zeros((P, P), dtype=bool)
    keep[mask] = True
    Fk = np.where(keep[:, :, None], F2, 0)
    return np.fft.ifft2(Fk, axes=(0, 1)).real.reshape(N_PAIRS, -1)


def median_margin(m, x2, w, ids):
    x2_t = torch.from_numpy(x2[ids].astype(np.float32)).to(DEV)
    w_t = torch.from_numpy(w[ids].astype(np.float32)).to(DEV)
    y = torch.from_numpy(ALL_Y[ids]).long().to(DEV)
    with torch.no_grad():
        lg = m.Wu(m.ln_f(x2_t + w_t)).cpu().numpy()
    lgy = lg[np.arange(len(ids)), ALL_Y[ids]]
    lgo = lg.copy()
    lgo[np.arange(len(ids)), ALL_Y[ids]] = -1e9
    M = lgy - lgo.max(axis=1)
    return float(np.median(M)), float(M.mean())


def nearest(rows, target):
    best = min(rows, key=lambda r: abs(r['t'] - target))
    assert abs(best['t'] - target) <= 200, \
        f'no ckpt within 200 of t={target}'
    return best['t']


def run_seed(seed, ckpt_dir, frozen_k_dir):
    with open(os.path.join(os.path.dirname(ckpt_dir),
                           'replay_summary.json')) as f:
        cross = json.load(f)[str(seed)]['cross']
    with open(os.path.join(frozen_k_dir, f'seed{seed}.json')) as f:
        K = json.load(f)['K']
    avail = sorted(int(os.path.basename(f).split('_t')[1]
                       .replace('.pt', ''))
                   for f in __import__('glob').glob(
                       os.path.join(ckpt_dir, f's{seed}_t*.pt')))
    pre = [t for t in avail if t < cross]
    # P/L per §10.2a: nearest grid point to the frozen offsets
    p_t = min(pre, key=lambda t: abs(t - (cross - 6000)))
    assert abs(p_t - (cross - 6000)) <= 200
    l_t = min(pre, key=lambda t: abs(t - (cross - 1000)))
    assert abs(l_t - (cross - 1000)) <= 200

    print(f'--- seed{seed}: P=t{p_t}, L=t{l_t} (cross={cross}) ---',
          flush=True)
    models = {}
    for k, t in (('P', p_t), ('L', l_t)):
        ck = torch.load(os.path.join(ckpt_dir, f's{seed}_t{t}.pt'),
                        map_location=DEV, weights_only=False)
        m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        m.load_state_dict(ck['model'])
        m.eval()
        models[k] = m
    parts = {k: full_parts(m) for k, m in models.items()}
    kw = {}
    for k in ('P', 'L'):
        m_np, _ = parts[k]
        _, gam = additive_of(m_np)
        gK = k_component(gam, seed, K)
        kw[k] = {'gamK': gK,
                 'rms': float(np.sqrt((gK ** 2).mean()))}
    _, m1m, m2m = strata_masks(seed)
    nov_ids = np.where(m2m)[0]
    swp_ids = np.where(m1m)[0]

    def cell(wi, rj, scale=1.0):
        mR, x2R = parts[rj]
        _, gam_r = additive_of(parts[rj][0])
        gamK_r = k_component(gam_r, seed, K)
        m_negK_r = parts[rj][0] - gamK_r
        src = kw[wi]['gamK'] * scale
        res = {}
        for tag, ids in (('nov', nov_ids), ('swp', swp_ids)):
            raw, _ = median_margin(models[rj], x2R,
                                   m_negK_r + src, ids)
            res[tag] = raw
        return res

    A = {(wi, rj): cell(wi, rj)
         for wi in ('P', 'L') for rj in ('P', 'L')}
    rms_min = min(kw['P']['rms'], kw['L']['rms'])
    A_norm = {(wi, rj): cell(wi, rj, scale=rms_min / kw[wi]['rms'])
              for wi in ('P', 'L') for rj in ('P', 'L')}
    for wi in ('P', 'L'):
        for rj in ('P', 'L'):
            print(f'    W{wi}->R{rj}: nov={A[(wi, rj)]["nov"]:+.3f} '
                  f'swp={A[(wi, rj)]["swp"]:+.3f} '
                  f'| norm nov={A_norm[(wi, rj)]["nov"]:+.3f}',
                  flush=True)
    I = (A[('L', 'L')]['nov'] - A[('P', 'L')]['nov']
         - A[('L', 'P')]['nov'] + A[('P', 'P')]['nov'])
    print(f'  seed{seed}: I_match = {I:+.3f}', flush=True)
    out = {'A': {f'{a}{b}': v for (a, b), v in A.items()},
           'A_norm': {f'{a}{b}': v for (a, b), v in A_norm.items()},
           'rms': {k: kw[k]['rms'] for k in kw},
           'states': {'P': p_t, 'L': l_t},
           'cross': cross, 'K': K, 'I_match': float(I)}
    del models, parts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--frozen-k-dir', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    for seed in [int(s) for s in args.seeds.split(',')]:
        out[seed] = run_seed(seed, args.ckpt_dir, args.frozen_k_dir)
    I_vals = [out[s]['I_match'] for s in out]
    n_pass = sum(i > 0 for i in I_vals)
    print('\n=== S2 verdict (REPRO_PLAN §10.2a frozen) ===', flush=True)
    print(f'I_match: mean={np.mean(I_vals):+.3f} '
          f'SD={np.std(I_vals):.3f} '
          f'(per-seed {[round(v, 2) for v in I_vals]})')
    print(f'positive seeds: {n_pass}/{len(I_vals)}')
    print('S2 PASS' if n_pass >= 7 else 'S2 BELOW GATE', flush=True)
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'results.pkl')
    with open(fp, 'wb') as f:
        pickle.dump({'seeds': out,
                     'verdict': {'I_mean': float(np.mean(I_vals)),
                                 'I_sd': float(np.std(I_vals)),
                                 'I_per_seed': I_vals,
                                 'positive': int(n_pass),
                                 'n_seeds': len(I_vals)}}, f)
    print(f'saved results.pkl ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
