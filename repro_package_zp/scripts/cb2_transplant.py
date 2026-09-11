"""scripts/cb2_transplant.py -- CB2: R97-style 2x2 carrier transplant
under wd_1111.  Takes wd_1111 bridge t=200 snapshots, builds four
arms (body x QK), runs to 4000-step cap with QK frozen.

Usage:
  python scripts/cb2_transplant.py --bridge-dir DIR --out DIR \
      --seeds 0,1,2 [--device cuda]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from common import task
from common.model import D57Model, make_optimizer

task.set_group('zp', 113)
LOSS = nn.CrossEntropyLoss()
P = 113
QK = ('Wq.weight', 'Wk.weight')
# CAP2 4000 (R97 wd_0011 calibration) censored ALL cells under
# wd_1111 -- including the native controls (CB1: natF crosses ~4000,
# natS 8000-12000), a floor effect, not a carrier result.  Scaled to
# the full bridge horizon before any cell outcome was observed.
CAP2 = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bridge-dir', required=True,
                    help='parent dir containing bridge_wd1111_A/ and C/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    dev = torch.device(args.device if torch.cuda.is_available()
                       else 'cpu')

    def load_snap(family, seed, which='body'):
        # battery arms: 'S'/'F' map to the trained bridge families
        # A (natS, fqk0: QK free from step 0) / C (natF, fqk200: QK
        # frozen for the FIRST 200 steps, then free) -- CB1's mapping.
        # R97/R125 semantics, restored: BOTH host and QK source come
        # from the t=200 full snapshots.  At t=200 the arms differ
        # exactly as the design requires -- A's QK has 200 steps of
        # training (Q_S, the native configuration), C's QK is the
        # bitwise initial construction (Q_0) -- so the four cells are
        # (S,Q_S) (F,Q_0) (S,Q_0) (F,Q_S) as preregistered.  An
        # earlier revision sourced QK from the _final checkpoints,
        # which silently redefined Q_0 as a trained block; reverted.
        fam = {'S': 'A', 'F': 'C',
               'natS': 'A', 'natF': 'C'}[family]
        d = os.path.join(args.bridge_dir, f'bridge_wd1111_{fam}')
        fps = [f for f in os.listdir(d) if 'full0000200' in f
               and f.startswith(f'seed{seed}_')]
        assert len(fps) == 1, f'{d}: {fps}'
        return torch.load(os.path.join(d, fps[0]),
                          map_location='cpu', weights_only=False)

    def val_probe(seed):
        all_a = np.arange(P)[None, :].repeat(P, 0).ravel()
        all_b = np.arange(P)[:, None].repeat(P, 1).ravel()
        all_y = task.group_mul_np(all_a, all_b)
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        perm = rng.permutation(P * P)
        n_tr = int(round(P * P * 0.30))
        rngv = np.random.default_rng(555)
        vb = rngv.permutation(perm[n_tr:])[:1024]
        return (torch.from_numpy(all_a[vb]).long().to(dev),
                torch.from_numpy(all_b[vb]).long().to(dev),
                torch.from_numpy(all_y[vb]).long().to(dev))

    results = {}
    for seed in seeds:
        va = val_probe(seed)
        arms = [('S', 'natS'), ('F', 'natF'), ('S', 'natF'),
                ('F', 'natS')]
        cell_name = {('S', 'natS'): 'SQS', ('F', 'natF'): 'FQ0',
                     ('S', 'natF'): 'SQ0', ('F', 'natS'): 'FQS'}
        rec = {}
        for body, qsrc in arms:
            torch.manual_seed(seed)
            m = D57Model(pos_mode='zeros', arch='abeq',
                         ln_eps=1e-5, eq_alpha=0.0).to(dev)
            body_sd = load_snap(body, seed, 'body')
            q_sd = load_snap(qsrc, seed, 'qk')
            state = dict(body_sd['model'])
            for n in QK:
                state[n] = q_sd['model'][n].clone()
            m.load_state_dict(state)
            opt, _, _ = make_optimizer(m, 'wd_1111')
            opt.load_state_dict(body_sd['opt'])

            # rebuild data
            all_a = np.arange(P)[None, :].repeat(P, 0).ravel()
            all_b = np.arange(P)[:, None].repeat(P, 1).ravel()
            all_y = task.group_mul_np(all_a, all_b)
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            perm = rng.permutation(P * P)
            n_tr = int(round(P * P * 0.30))
            tr_a = torch.from_numpy(all_a[perm[:n_tr]]).long()
            tr_b = torch.from_numpy(all_b[perm[:n_tr]]).long()
            tr_y = torch.from_numpy(all_y[perm[:n_tr]]).long()

            cross = None
            m.train()
            va_hist = []
            for step in range(1, CAP2 + 1):
                idx = torch.randint(tr_a.shape[0], (512,))
                a, b, y = (tr_a[idx].to(dev), tr_b[idx].to(dev),
                           tr_y[idx].to(dev))
                opt.zero_grad()
                LOSS(m(a, b), y).backward()
                m.Wq.weight.grad = None
                m.Wk.weight.grad = None
                opt.step()
                if step % 25 == 0:
                    m.eval()
                    with torch.no_grad():
                        acc = float(
                            (m(va[0], va[1]).argmax(-1) == va[2])
                            .float().mean())
                    va_hist.append((step, round(acc, 4)))
                    if acc > 0.9 and cross is None:
                        cross = step
                        break
                    m.train()
            rec[cell_name[(body, qsrc)]] = cross
            rec[cell_name[(body, qsrc)] + '_traj'] = va_hist
            print(f'  s{seed} {cell_name[(body, qsrc)]}: cross={cross} '
                  f'last_va={va_hist[-1][1] if va_hist else None}',
                  flush=True)

        # Carrier reading (R125-style): installing the native t=200
        # configuration must impose/hold the SLOW fate -- crossing
        # late or not at all within CAP2 -- while installing Q_0 must
        # yield the fast fate.  A cell alive at CAP2 is a hold, not a
        # failure: the old label logic treated None as no-signal,
        # which was only meaningful under the short R125 cap.
        def late_or_held(x, ref_early):
            if x is None:
                return True
            return ref_early is not None and x > ref_early + 1000

        def fast(x):
            return x is not None

        s_side = (late_or_held(rec['SQS'], rec['SQ0'])
                  and fast(rec['SQ0']))
        f_side = (late_or_held(rec['FQS'], rec['FQ0'])
                  and fast(rec['FQ0']))
        rec['verdict'] = ('CARRIER-HOLDS' if s_side and f_side
                          else 'NOT-CARRIER' if not s_side and not f_side
                          else 'MIXED')
        results[seed] = rec
        print(f'  seed{seed} verdict: {rec["verdict"]}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'cb2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('cb2_results.json written', flush=True)


if __name__ == '__main__':
    main()
