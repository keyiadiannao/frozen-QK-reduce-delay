"""scripts/select_k_modes.py — Stage 4 of the 10-seed server plan.

Per-seed K-mode selection at the L checkpoint (REPRO_PLAN §10.2a):
write field at P = t_cross-6000... no -- K is selected from the L
state's write (the pre-cross selective-growth point), top-5 diagonal
modes by R134b canonical diag_measures on the additive-orthogonal
residual, unary axes excluded.

Verbatim math from repro/r134b_2d_factorization.py (interface_fields
+ diag_measures); single deviation: only the 'm' (MLP write)
interface is needed, and only at the L checkpoint.

Output: <out>/frozen_k/seed{N}.json
        {'seed': N, 't_L': int, 'K': [k1..k5],
         'diag_share_L': float, 'top8': [[k, share], ...]}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task                                    # noqa: E402
from common.model import D57Model                          # noqa: E402

DEV = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
TOPN = 8
NK = 5

task.set_group('zp', 113)
_ALL = task.build_tables()
ALL_A, ALL_B = _ALL[0], _ALL[1]
N_PAIRS = len(ALL_A)
Pn = task.GROUP['p']
CH = 4096


def write_field(m):
    """Full-grid MLP write m (verbatim r134b interface_fields, 'm'
    branch only)."""
    ids = np.arange(N_PAIRS)
    out = []
    for s in range(0, N_PAIRS, CH):
        a = torch.from_numpy(ALL_A[ids[s:s + CH]]).long().to(DEV)
        b = torch.from_numpy(ALL_B[ids[s:s + CH]]).long().to(DEV)
        n = len(a)
        with torch.no_grad():
            eq = torch.full_like(a, task.N_EQ)
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
            mo = m.mlp2(torch.nn.functional.gelu(
                m.mlp1(m.ln2(x)[:, 2, :])))
        out.append(mo.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def diag_measures(m_np):
    """R134b canonical: additive-orthogonal residual -> 2D DFT ->
    per-|k| diagonal energy; top modes over k in 1..P-1 (unary axes
    excluded).  Verbatim math."""
    P = Pn
    fa = m_np.mean(axis=1, keepdims=True)
    fb = m_np.mean(axis=0, keepdims=True)
    fg = m_np.mean(axis=(0, 1), keepdims=True)
    Fperp = m_np - fa - fb + fg
    Fhat = np.fft.fft2(Fperp, axes=(0, 1)) / (P * P)
    ek_kk = np.zeros(P)
    tot = 0.0
    diag = 0.0
    for k in range(P):
        for l in range(P):
            if k == 0 or l == 0:
                continue
            e = float(np.linalg.norm(Fhat[k, l]) ** 2)
            tot += e
            if k == l or k == (P - l) % P:
                diag += e
                kk = min(k, (P - k) % P)
                ek_kk[kk] += e
    share = diag / (tot + 1e-30)
    order = np.argsort(ek_kk)[::-1]
    top = [(int(k), float(ek_kk[k] / (diag + 1e-30)))
           for k in order[:TOPN] if ek_kk[k] > 0]
    return share, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', required=True)
    ap.add_argument('--replay-summary', required=True,
                    help='Stage 3 replay_summary.json')
    ap.add_argument('--ckpt-dir', required=True,
                    help='dir with r131c_ckpt/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'frozen_k'), exist_ok=True)
    with open(args.replay_summary) as f:
        replay = json.load(f)
    for seed in [int(s) for s in args.seeds.split(',')]:
        cross = replay[str(seed)]['cross']
        t_L = cross - 1000
        ck = torch.load(os.path.join(args.ckpt_dir, 'r131c_ckpt',
                                     f's{seed}_t{t_L}.pt'),
                        map_location='cpu', weights_only=False)
        m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        m.load_state_dict(ck['model'])
        m.eval()
        m_np = write_field(m)
        share, top = diag_measures(m_np.reshape(Pn, Pn, -1))
        K = [k for k, _ in top[:NK]]
        rec = {'seed': seed, 't_L': t_L, 'K': K, 'diag_share_L': share,
               'top8': top}
        with open(os.path.join(args.out, 'frozen_k',
                               f'seed{seed}.json'), 'w') as f:
            json.dump(rec, f, indent=2)
        print(f's{seed}: t_L={t_L} K={K} diag_share={share:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
