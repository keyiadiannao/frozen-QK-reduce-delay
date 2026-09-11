"""K05 run.py — formation accounting (C2'): how the real QK updates of
steps 0:200 accumulate into the final operand code.

Package port of repro/r105_formation_accounting.py (M2.5, 2026-09-02).
Verbatim except common/ imports (ZPReadouts for split+chain), claim-dir
output, SMOKE env R105_SMOKE -> K05_SMOKE.

Self-contained: trains a fresh repro-native run per seed (plain, fqk0,
wd_0011); no bridge checkpoint dependency.  Bit-exactness rests on the
GPU-deterministic kernels established in M2.

Claim (ledger K5): MODEL-A — early (~40 step) direction selection
(C_40 = 0.85-0.91) followed by selective QK-gradient amplification;
QK updates dominate (body_share 0.001); role-antisymmetry never
dominant.  Preregistered §6ec; verdict §6ed.
"""
import os
import sys
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)

SMOKE = bool(os.environ.get('K05_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K05] K05_SMOKE=1 but CUDA visible -- abort')
print(f'[K05] DEV={DEV} SMOKE={SMOKE}', flush=True)
_m0 = D57Model(pos_mode='zeros', arch='abeq')
print(f'[K05] d_head={_m0.d_head} (scale={math.sqrt(_m0.d_head):.3f})',
      flush=True)
del _m0

P = task.GROUP['p']
N_PAIRS = task.N_PAIRS
N_TRAIN = int(round(N_PAIRS * 0.30))
BATCH = 512
STEPS = 20 if SMOKE else 200
SEEDS = (0,) if SMOKE else (0, 1, 2)
LOSS = nn.CrossEntropyLoss()

ALL_A, ALL_B, ALL_Y = task.build_tables()
R = ZPReadouts(DEV)


def functional_terms(m, H2, H0, H1):
    """Row-2 functional on operand values for the current Wq/Wk and
    cached body representations. Returns s0, s1 (H,113), c (H,)."""
    with torch.no_grad():
        q2 = (m.Wq.weight.detach() @ H2).view(m.n_heads, m.d_head)
        Wk = m.Wk.weight.detach().view(m.n_heads, m.d_head, -1)
        k0 = torch.einsum('hkd,gd->hgk', Wk, H0)     # (H,113,dh)
        k1 = torch.einsum('hkd,gd->hgk', Wk, H1)
        k2 = torch.einsum('hkd,d->hk', Wk, H2)
        scale = math.sqrt(m.d_head)   # match model convention & R102
        s0 = torch.einsum('hk,hgk->hg', q2, k0) / scale
        s1 = torch.einsum('hk,hgk->hg', q2, k1) / scale
        c = torch.einsum('hk,hk->h', q2, k2) / scale
    return s0.cpu().numpy(), s1.cpu().numpy(), c.cpu().numpy()


def body_repr(m):
    """ln1 representations for h2 and the 113 operand values at
    positions 0 and 1 (no bias in Wq/Wk; ln1 per token)."""
    with torch.no_grad():
        g = torch.arange(P, dtype=torch.long, device=DEV)
        tok_op = torch.stack([g, g, g], dim=1)       # operand value g
        x = m.emb(tok_op) + m.pos[None, :, :]
        hh = m.ln1(x)                                 # (P,3,d)
        H0 = hh[:, 0, :]                              # ln1(e_g+p0)
        H1 = hh[:, 1, :]
        tok2 = torch.tensor([[0, 0, task.N_EQ]], device=DEV)
        x2 = m.emb(tok2) + m.pos[None, :, :]
        H2 = m.ln1(x2)[0, 2, :]
    return H2, H0, H1


def main():
    out = {}
    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        opt, _, _ = make_optimizer(m, 'wd_0011')
        id_train, chain = R.batch_chain(seed, STEPS)

        S0 = np.zeros((STEPS + 1, m.n_heads, P))
        S1 = np.zeros_like(S0)
        C = np.zeros((STEPS + 1, m.n_heads))
        S0q = np.zeros_like(S0)
        S1q = np.zeros_like(S1)
        Cq = np.zeros_like(C)
        losses = []

        H2, H0, H1 = body_repr(m)
        s0, s1, cc = functional_terms(m, H2, H0, H1)
        S0[0], S1[0], C[0] = s0, s1, cc
        S0q[0], S1q[0], Cq[0] = s0, s1, cc
        for step in range(1, STEPS + 1):
            pairs = id_train[chain[step - 1]]
            a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
            b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
            y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
            loss = LOSS(m(a, b), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            # QK-attributed increment: post-step Wq/Wk on the CACHED body
            s0q, s1q, cq = functional_terms(m, H2, H0, H1)
            S0q[step], S1q[step], Cq[step] = s0q, s1q, cq
            # full trajectory (new body)
            H2, H0, H1 = body_repr(m)
            s0, s1, cc = functional_terms(m, H2, H0, H1)
            S0[step], S1[step], C[step] = s0, s1, cc

        # increments (QK-attributed)
        U = ((S0q[1:] - S0q[:-1]) + (S1q[1:] - S1q[:-1])) / 2   # (T,H,113)
        Rr = ((S0q[1:] - S0q[:-1]) - (S1q[1:] - S1q[:-1])) / 2
        DC = Cq[1:] - Cq[:-1]
        # total symmetric displacement (full trajectory)
        D = ((S0[-1] + S1[-1]) - (S0[0] + S1[0])) / 2           # (H,113)
        D_qk = U.sum(axis=0)
        body_share = float(np.linalg.norm(D - D_qk) /
                           (np.linalg.norm(D) + 1e-30))

        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-30 or nb < 1e-30:
                return 0.0
            return float((a @ b) / (na * nb))

        A_t = np.array([[cos(U[t, h], D[h]) for h in range(m.n_heads)]
                        for t in range(STEPS)])                 # (T,H)
        Csum = np.cumsum(U, axis=0)
        C_t = np.array([[cos(Csum[t, h], D[h]) for h in range(m.n_heads)]
                        for t in range(STEPS)])

        # phenotype match at t=200: Delta-functional vs step 0
        d0 = S0[-1] - S0[0]                                    # (H,113)
        d1 = S1[-1] - S1[0]
        f_std = float(np.concatenate([d0, d1]).std())
        self_mean = float((C[-1] - C[0]).mean())
        corr_role = float(np.mean([np.corrcoef(d0[h], d1[h])[0, 1]
                                   for h in range(m.n_heads)]))
        pheno = {'f_std': f_std, 'self_mean': self_mean,
                 'corr_role': corr_role,
                 'ok': 8 <= f_std <= 16 and -40 <= self_mean <= -15
                 and corr_role >= 0.99}

        C40 = float(C_t[39].mean()) if STEPS >= 40 else float('nan')
        C_last = float(C_t[-1].mean())
        late_min = float(C_t[39:].mean(axis=1).min()) if STEPS >= 40 \
            else float('nan')
        if SMOKE:
            verdict = 'SMOKE'
        elif C40 >= 0.7 and late_min >= 0.3:
            verdict = 'MODEL-A (early selection + amplification)'
        elif C40 <= 0.3 and C_last >= 0.7:
            verdict = 'MODEL-B (late lock-in)'
        else:
            verdict = 'GRADED/MIXED'
        out[seed] = {'U': U, 'R': Rr, 'DC': DC, 'D': D, 'D_qk': D_qk,
                     'A_t': A_t, 'C_t': C_t, 'C40': C40,
                     'C_last': C_last, 'late_min': late_min,
                     'body_share': body_share, 'pheno': pheno,
                     'verdict': verdict, 'S0': S0, 'S1': S1, 'Cself': C,
                     'losses': losses}
        print(f'seed{seed}: pheno f_std={f_std:.2f} '
              f'self_mean={self_mean:+.2f} corr_role={corr_role:.4f} '
              f'ok={pheno["ok"]}', flush=True)
        print(f'  body_share(QK vs drift)={body_share:.3f} | '
              f'C40={C40:+.3f} C_last={C_last:+.3f} '
              f'late_min={late_min:+.3f} -> {verdict}', flush=True)
        wins = [(a, b) for a, b in ((1, 20), (21, 50), (51, 100),
                                    (101, 200)) if b <= STEPS]
        for a, b in wins:
            ru = (np.linalg.norm(Rr[a - 1:b], axis=2).mean() /
                  (np.linalg.norm(U[a - 1:b], axis=2).mean() + 1e-30))
            print(f'  A_t mean [{a:3d},{b:3d}]: '
                  f'{A_t[a - 1:b].mean():+.3f} | |r|/|u|={ru:.3f}',
                  flush=True)

    print(f'\n[K05] done in {time.time() - t0:.1f}s', flush=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'steps': STEPS, 'arms': out},
                    f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
