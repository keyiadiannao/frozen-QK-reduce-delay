"""K04 run.py — Delta-M row-2 functional dissection (Q1 content, observational).

Package port of repro/r102_dm_dissect.py (M2.5, 2026-09-02).  Code is
verbatim except:
  * repro's train_d57_abeq (T) -> common.task / common.model /
    common.states;
  * output pkl written to this claim directory (never to repro/);
  * SMOKE env renamed R102_SMOKE -> K04_SMOKE.

Claim (ledger K4): the carrier content is an operand-level scalar scoring
functional f(a)=v.h0(a) with self-position suppression, strictly
separable (pair residual == 0), role-symmetric, seed-specific code.
Preregistered in EXTENSION_SUMMARY §6du; verdict §6dv; wording §6dw.
"""
import os
import sys
import math
import pickle
import time

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task, states
from common.model import D57Model

task.set_group('zp', 113)

SMOKE = bool(os.environ.get('K04_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K04] K04_SMOKE=1 but CUDA visible -- abort')
print(f'[K04] DEV={DEV} SMOKE={SMOKE}', flush=True)

P = task.GROUP['p']
N_PAIRS = task.N_PAIRS
N_TRAIN = int(round(N_PAIRS * 0.30))
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPS = 1e-8

ALL_A, ALL_B, ALL_Y = task.build_tables()
assert ALL_A.shape[0] == N_PAIRS


def fresh_init(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq')


def load_s_model(seed):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    m.load_state_dict(dict(sd['model']))
    return m


@torch.no_grad()
def body_h(m):
    """h = ln1(emb + pos) for all pairs -- independent of Wq/Wk."""
    tid = torch.arange(N_PAIRS, dtype=torch.long, device=DEV)
    a, b = tid % P, tid // P
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    outs = []
    for s in range(0, N_PAIRS, 4096):
        x = m.emb(tok[s:s + 4096]) + m.pos[None, :, :]
        outs.append(m.ln1(x))
    return torch.cat(outs, 0)          # (N_PAIRS, 3, d_model)


@torch.no_grad()
def row2_scores(m, H):
    """Row-2 pre-softmax scores per head: (N_PAIRS, n_heads, 3)."""
    H = H.to(DEV)
    out = []
    for s in range(0, N_PAIRS, 4096):
        hh = H[s:s + 4096]
        q = m.Wq(hh).view(-1, 3, m.n_heads, m.d_head)
        k = m.Wk(hh).view(-1, 3, m.n_heads, m.d_head)
        q2 = q[:, 2, :, :]                       # (n, H, dh)
        sc = torch.einsum('nhd,njhd->nhj', q2, k)
        out.append((sc / math.sqrt(m.d_head)).cpu())
    return torch.cat(out, 0)


def grid_by_pair(vals):
    """vals (N_PAIRS,) -> grid g[b, a]."""
    g = np.full((P, P), np.nan, dtype=np.float64)
    g[ALL_B, ALL_A] = vals
    assert not np.isnan(g).any()
    return g


def sep_stats(vals, axis):
    """grid g[b,a]. axis=0: test dependence on a only (remove mean
    over b); axis=1: test dependence on b only (remove mean over a).
    Returns (max residual, function vector)."""
    g = grid_by_pair(vals)
    if axis == 0:
        f = g.mean(axis=0)                        # f(a)
        resid = g - f[None, :]
    else:
        f = g.mean(axis=1)                        # f(b)
        resid = g - f[:, None]
    return float(np.abs(resid).max()), f


def main():
    results = {}
    t_start = time.time()
    for seed in SEEDS:
        mS = load_s_model(seed).to(DEV)
        m0 = fresh_init(seed).to(DEV)
        pos_norm_S = float(mS.pos.detach().norm())
        pos_norm_0 = float(m0.pos.detach().norm())

        H = body_h(mS)
        pd = mS.pos.detach()[0]           # (3, d_model)
        pos_diffs = [float((pd[0] - pd[1]).norm()),
                     float((pd[0] - pd[2]).norm()),
                     float((pd[1] - pd[2]).norm())]
        tid = np.arange(N_PAIRS)
        h0v = H[:P, 0, :]                  # tid=a -> (a, 0): h0 per a
        h1v = H[::P, 1, :]                 # tid = b*P -> (0, b): h1 per b
        op_diff = float(max(float((h0v[x] - h1v[x]).norm())
                            for x in range(P)))
        scS = row2_scores(mS, H)          # (N, H, 3)
        sc0 = row2_scores(m0, H)
        dL = (scS - sc0).numpy()          # (N, H, 3)

        nh = mS.n_heads
        per_head = []
        for h in range(nh):
            res0, f0 = sep_stats(dL[:, h, 0], 0)  # expect f(a) only
            res1, f1 = sep_stats(dL[:, h, 1], 1)  # expect f(b) only
            g2 = grid_by_pair(dL[:, h, 2])
            res2 = float(np.abs(g2 - g2.mean()).max())   # expect const
            amp = {'f0_std': float(f0.std()), 'f1_std': float(f1.std()),
                   'dL2_abs': float(abs(dL[:, h, 2].mean())),
                   'dL2_std': float(dL[:, h, 2].std())}
            pS = torch.softmax(scS[:, h, :], dim=-1).numpy()
            p0 = torch.softmax(sc0[:, h, :], dim=-1).numpy()
            dp = np.abs(pS - p0)                   # (N, 3)
            mean_dp = dp.mean(axis=0)              # per position j
            g0 = grid_by_pair(sc0[:, h, 0])
            g0_f = g0.mean(axis=0)                 # function of a (Q0)
            g1 = grid_by_pair(sc0[:, h, 1])
            g1_f = g1.mean(axis=1)                 # function of b (Q0)
            c_d = float(np.corrcoef(f0, f1)[0, 1])
            c_b = float(np.corrcoef(g0_f, g1_f)[0, 1])
            WqS = mS.Wq.weight.detach().cpu()
            WkS = mS.Wk.weight.detach().cpu()
            Wq0 = m0.Wq.weight.detach().cpu()
            Wk0 = m0.Wk.weight.detach().cpu()
            sl = slice(h * mS.d_head, (h + 1) * mS.d_head)
            dM = (WqS[:, sl] @ WkS[:, sl].T -
                  Wq0[:, sl] @ Wk0[:, sl].T).double()      # (dm, dm)
            h2 = H[0, 2, :].double().cpu()
            gain_q = float((dM @ h2).norm() / h2.norm())
            rng = np.random.default_rng(1000 + seed)
            R = rng.standard_normal((256, dM.shape[0]))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            gains = np.linalg.norm(R @ dM.numpy(), axis=1)
            enr = gain_q / float(gains.mean())
            per_head.append({
                'res0': res0, 'res1': res1, 'res2': res2, 'amp': amp,
                'mean_dp': mean_dp.tolist(),
                'corr_d_swap': c_d, 'corr_q0_baseline': c_b,
                'enrichment': enr, 'gain_query': gain_q,
                'rand_gain_mean': float(gains.mean()),
                'f0': f0.tolist(), 'f1': f1.tolist(),
                'dL2': float(dL[:, h, 2].mean())})
        results[seed] = {
            'pos_norm_S': pos_norm_S, 'pos_norm_0': pos_norm_0,
            'pos_diffs': pos_diffs, 'operand_rep_diff': op_diff,
            'heads': per_head, 'dL': dL,
            'scS': scS.numpy(), 'sc0': sc0.numpy()}
        print(f'seed{seed}: pos_norm S={pos_norm_S:.4f} '
              f'init={pos_norm_0:.4f} '
              f'pos_diffs(0-1,0-2,1-2)={["%.4f" % d for d in pos_diffs]} '
              f'operand_rep_diff={op_diff:.4f}', flush=True)
        for h, ph in enumerate(per_head):
            print(f'  h{h}: res0={ph["res0"]:.2e} res1={ph["res1"]:.2e} '
                  f'res2={ph["res2"]:.2e} | f0_std={ph["amp"]["f0_std"]:.4f} '
                  f'f1_std={ph["amp"]["f1_std"]:.4f} '
                  f'dL2={ph["dL2"]:+.4f}', flush=True)
            print(f'      mean|dp| per j: '
                  f'{["%.4f" % x for x in ph["mean_dp"]]} | '
                  f'corr_swap(dL)={ph["corr_d_swap"]:+.3f} '
                  f'baseline(Q0)={ph["corr_q0_baseline"]:+.3f} | '
                  f'enrich={ph["enrichment"]:.2f}x', flush=True)

    out_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(results, f)
    print(f'\n[K04] done in {time.time() - t_start:.1f}s -> {out_name}',
          flush=True)


if __name__ == '__main__':
    main()
