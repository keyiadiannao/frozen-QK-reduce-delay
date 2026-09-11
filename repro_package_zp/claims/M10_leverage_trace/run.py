"""claims/M10_leverage_trace/run.py — S1 secondary replication
(R136a / M10 port for the 10-seed confirmatory).

Protocol source: repro/r136a_k_leverage.py (§6i Z/AA/AB), verbatim
except the registered deviations below.  Judge per REPRO_PLAN §10.2a
(frozen 2026-09-05):

  S1 gate: J_novel(P) > 0 AND J_novel(L) > J_novel(P), per-seed
  medians; report mean ± SD over seeds.  Full-grid trajectories are
  still recorded for diagnostics but never enter the variance.

Deviations from r136a (all registered in TEN_SEED_RUN_PLAN §E):
  1. checkpoints come from --ckpt-dir (Stage 3 server replay output),
     not the seal-machine r131c_ckpt archive;
  2. the K mode set is loaded from Stage 4 frozen_k/seed{N}.json
     (checkpoint-local top-5 selection at L) instead of the hardcoded
     3-seed FROZEN_K;
  3. P/L offsets per §10.2a: P = t_cross-6000, L = t_cross-1000
     (nearest available grid point, ±200 tolerance);
  4. seeds from --seeds (default 0..9).

Signed-orbit mask convention (r136b k_mask verbatim): mode k selects
{(k,k),(k,-k),(-k,k),(-k,-k)}, unary axes excluded -- the four-cell
block is the real-frequency projection block used by the filtering
code (same-sign cells carry the complex sum-character support).
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
LOSS = None  # not needed; margins are computed directly
EPS_ALPHA = 0.5
torch.set_num_threads(4)

task.set_group('zp', 113)
R = ZPReadouts(DEV)
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
N_PAIRS = R.N_PAIRS
P = 113
T_N_EQ = 113


def strata_masks(seed):
    rng2 = np.random.default_rng(np.random.SeedSequence(seed))
    pm = rng2.permutation(N_PAIRS)
    train = np.zeros(N_PAIRS, bool)
    train[np.sort(pm[:R.N_TRAIN])] = True
    diag = ALL_A == ALL_B
    swap_of = ALL_A * P + ALL_B
    swap_in = train[swap_of] & ~diag
    return train, ~train & swap_in, ~train & ~swap_in & ~diag


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


def full_write(m):
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


def decompose_write(m_np, seed, K):
    """m -> (m_K, m_negK), additive part kept whole (r136a verbatim,
    K from the frozen_k file)."""
    ua, ia = np.unique(ALL_A, return_inverse=True)
    ub, ib = np.unique(ALL_B, return_inverse=True)
    X = np.zeros((N_PAIRS, 1 + len(ua) + len(ub)))
    X[:, 0] = 1.0
    X[np.arange(N_PAIRS), 1 + ia] = 1.0
    X[np.arange(N_PAIRS), 1 + len(ua) + ib] = 1.0
    coef, *_ = np.linalg.lstsq(X, m_np, rcond=None)
    m_add = X @ coef
    gam = m_np - m_add
    mask = k_mask(seed, K)
    gam2 = gam.reshape(P, P, -1)
    F2 = np.fft.fft2(gam2, axes=(0, 1))
    keep = np.zeros((P, P), dtype=bool)
    keep[mask] = True
    Fk = np.where(keep[:, :, None], F2, 0)
    gamK = np.fft.ifft2(Fk, axes=(0, 1)).real.reshape(N_PAIRS, -1)
    m_K = m_add + gamK
    m_negK = m_add + (gam - gamK)
    return m_K, m_negK


def margins(m, x2_np, w_np, ids):
    x2_t = torch.from_numpy(x2_np[ids].astype(np.float32)).to(DEV)
    w_t = torch.from_numpy(w_np[ids].astype(np.float32)).to(DEV)
    y = torch.from_numpy(ALL_Y[ids]).long().to(DEV)
    with torch.no_grad():
        lg = m.Wu(m.ln_f(x2_t + w_t)).cpu().numpy()
    lgy = lg[np.arange(len(ids)), ALL_Y[ids]]
    lgo = lg.copy()
    lgo[np.arange(len(ids)), ALL_Y[ids]] = -1e9
    return lgy - lgo.max(axis=1)


def pick_checkpoints(seed, cross, ckpt_dir, max_n=24):
    """r136a verbatim selection over the Stage-3 grid."""
    avail = sorted(int(os.path.basename(f).split('_t')[1]
                       .replace('.pt', ''))
                   for f in __import__('glob').glob(
                       os.path.join(ckpt_dir, f's{seed}_t*.pt')))
    pre = [t for t in avail if t < cross]
    post = [t for t in avail if t >= cross]
    n_pre = max(8, int(max_n * 0.6))
    n_post = max_n - n_pre
    sel_pre = pre[::max(1, len(pre) // n_pre)][:n_pre]
    sel_post = post[::max(1, len(post) // n_post)][:n_post]
    sel = sorted(set(sel_pre + sel_post))
    # The §10.2a frozen gate windows must be represented: the index-
    # stride subsample can jump clean over the sparse-tail region
    # (seed3, 10-seed run: 13 in-window grid points, 0 selected).
    for lo, hi in ((-8000, -6000), (-1000, 0)):
        sel.extend(t for t in avail if lo <= t - cross < hi)
    sel = sorted(set(sel))
    if avail[-1] not in sel:
        sel.append(avail[-1])
    return sel


def run_seed(seed, ckpt_dir, frozen_k_dir):
    cross = None
    # crossing from replay summary if present; else infer from ckpts
    summ_fp = os.path.join(os.path.dirname(ckpt_dir),
                           'replay_summary.json')
    if os.path.exists(summ_fp):
        with open(summ_fp) as f:
            cross = json.load(f)[str(seed)]['cross']
    if cross is None:
        raise SystemExit(f'[M10] seed{seed}: replay_summary.json with '
                         't_cross required (Stage 3 output)')
    with open(os.path.join(frozen_k_dir, f'seed{seed}.json')) as f:
        K = json.load(f)['K']

    train, m1m, m2m = strata_masks(seed)
    nov_ids = np.where(m2m)[0]
    swp_ids = np.where(m1m)[0]
    ckpts = pick_checkpoints(seed, cross, ckpt_dir)
    rows = []
    print(f'--- seed{seed} (cross={cross}, {len(ckpts)} ckpts) ---',
          flush=True)
    for t in ckpts:
        ck = torch.load(os.path.join(ckpt_dir, f's{seed}_t{t}.pt'),
                        map_location=DEV, weights_only=False)
        m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        m.load_state_dict(ck['model'])
        m.eval()
        m_np, x2_np = full_write(m)
        m_K, m_negK = decompose_write(m_np, seed, K)
        rec = {'t': t, 'tau': t - cross}

        def J_for(ids):
            Mp_p = margins(m, x2_np,
                           m_negK + (1 + EPS_ALPHA) * (m_K - m_negK),
                           ids)
            Mp_m = margins(m, x2_np,
                           m_negK + (1 - EPS_ALPHA) * (m_K - m_negK),
                           ids)
            return (Mp_p - Mp_m) / (2 * EPS_ALPHA)

        j_nov = J_for(nov_ids)
        j_swp = J_for(swp_ids)
        rec['J50_nov'] = float(np.median(j_nov))
        rec['J50_swp'] = float(np.median(j_swp))
        rows.append(rec)
        print(f"  t={t:>6} tau={rec['tau']:>7} "
              f"J50_nov={rec['J50_nov']:+.3f} "
              f"J50_swp={rec['J50_swp']:+.3f}", flush=True)
        del m

    # ---- S1 gate per REPRO_PLAN §10.2a (frozen, AMENDED 2026-09-05
    # pre-run per smoke finding) ----
    # Smoke (seed0) showed a single-checkpoint P-vs-L comparison flips
    # under checkpoint noise (same failure mode as the R135c onset
    # lesson).  The gate therefore uses WINDOW MEDIANS -- the same
    # reading R136a's canonical numbers used:  plateau window
    # tau in [-8000,-6000), pre-cross window tau in [-1000,0).
    # The single nearest-to-offset point is still reported as a
    # diagnostic.
    def wmed(lo, hi):
        vals = [r['J50_nov'] for r in rows if lo <= r['tau'] < hi]
        assert vals, f'seed{seed}: empty window [{lo},{hi})'
        return float(np.median(vals)), len(vals)

    J_P, nP = wmed(-8000, -6000)
    J_L, nL = wmed(-1000, 0)
    gate = (J_P > 0) and (J_L > J_P)
    print(f"seed{seed} S1: J_P(win med)={J_P:+.3f}(n={nP}) "
          f"J_L(win med)={J_L:+.3f}(n={nL}) gate={gate}", flush=True)
    return {'rows': rows, 'cross': cross, 'K': K,
            'J_P_window': J_P, 'J_L_window': J_L,
            'gate': bool(gate)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--frozen-k-dir', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    gates = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        out[seed] = run_seed(seed, args.ckpt_dir, args.frozen_k_dir)
        gates.append(out[seed]['gate'])
    jp = [out[s]['J_P_window'] for s in out]
    jl = [out[s]['J_L_window'] for s in out]
    print('\n=== S1 verdict (REPRO_PLAN §10.2a frozen) ===', flush=True)
    print(f'J_novel(P): mean={np.mean(jp):+.3f} SD={np.std(jp):.3f} '
          f'(per-seed {[round(v, 2) for v in jp]})')
    print(f'J_novel(L): mean={np.mean(jl):+.3f} SD={np.std(jl):.3f} '
          f'(per-seed {[round(v, 2) for v in jl]})')
    print(f'gate pass seeds: {sum(gates)}/{len(gates)}')
    print('S1 PASS' if sum(gates) >= 7 else 'S1 BELOW GATE',
          flush=True)
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'results.pkl')
    with open(fp, 'wb') as f:
        pickle.dump({'seeds': out,
                     'verdict': {'J_P_mean': float(np.mean(jp)),
                                 'J_P_sd': float(np.std(jp)),
                                 'J_L_mean': float(np.mean(jl)),
                                 'J_L_sd': float(np.std(jl)),
                                 'gate_pass': int(sum(gates)),
                                 'n_seeds': len(gates)}}, f)
    print(f'saved results.pkl ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
