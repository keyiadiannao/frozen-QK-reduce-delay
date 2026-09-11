"""scripts/replay_native.py — Stage 3 of the 10-seed server plan.

Native-trajectory replay with checkpoint saving, port of
repro/r131c_replay_offline.py phase1 (verbatim protocol) with three
registered deviations for the 10-seed run:

  1. r113.build_qk(seed) is replaced by LOADING the Stage-2 file
     qk_lineage/seed{N}_F_QS.pt (weights materialized once, hashed);
  2. the bridge anchor is loaded via common.states.bridge_ckpt_path
     from --repro-dir (same as r113.ckpt did);
  3. two FORCED save points per seed: t_cross-6000 and t_cross-1000
     (REPRO_PLAN §10.2a P/L offsets; saved only if the regular cadence
     did not already cover them within ±200 steps).  These are extra
     saves, not a cadence change.

Cadence (verbatim r131c): every-25-step RNG-free va probe;
save if t<=6000 and t%200==0; elif va>0.60 every step; elif va>0.30
and t%50==0; else t%200==0; plus all 25-step grid points up to
cross+500.  Cross = va>0.9; stop at cross+1000 (CAP 20000).

Output: <out>/r131c_ckpt/s{seed}_t{T}.pt  + replay_summary.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task                                    # noqa: E402
from common.model import D57Model, make_optimizer          # noqa: E402
from common.readouts import ZPReadouts                     # noqa: E402
from common.states import bridge_ckpt_path                 # noqa: E402

DEV = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
LOSS = nn.CrossEntropyLoss()
CAP = 20000

task.set_group('zp', 113)
R = ZPReadouts(DEV)
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
N_PAIRS = R.N_PAIRS
P = 113
torch.set_num_threads(4)


def va_probe(m, va):
    with torch.no_grad():
        return float((m(va[0], va[1]).argmax(-1) ==
                      va[2]).float().mean())


def phase1_replay(seed, repro_dir, qk_dir, out_dir):
    """Pure training replay, checkpoint saving only (r131c verbatim
    except the registered deviations in the module docstring)."""
    ckpt_dir = os.path.join(out_dir, 'r131c_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    # bridge anchor + QK lineage file (r113mod.ckpt + build_qk replaced)
    sd = torch.load(bridge_ckpt_path(repro_dir, 'A', seed, None,
                                     'full0000200'),
                    map_location='cpu', weights_only=False)
    qk = torch.load(os.path.join(qk_dir, 'qk_lineage',
                                 f'seed{seed}_F_QS.pt'),
                    map_location='cpu', weights_only=False)
    id_train, chain = R.batch_chain(seed, CAP)
    va = R.val_probe(seed)
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    body = dict(sd['model'])
    body['Wq.weight'] = qk['Wq'].to(DEV).clone()
    body['Wk.weight'] = qk['Wk'].to(DEV).clone()
    m.load_state_dict(body)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()

    t0 = time.time()
    cross = None
    saved = []
    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None
        m.Wk.weight.grad = None
        opt.step()
        t = 200 + step
        if step % 25 == 0:
            acc = va_probe(m, va)
            if cross is None and acc > 0.9:
                cross = t
                print(f'  s{seed} CROSS at t={cross}', flush=True)
            # save policy (r131c verbatim)
            if t <= 6000:
                do_save = (t % 200 == 0)
            elif acc > 0.60:
                do_save = True
            elif acc > 0.30:
                do_save = (t % 50 == 0)
            else:
                do_save = (t % 200 == 0)
            if do_save or (cross is not None and
                           t <= cross + 500 and t % 25 == 0):
                fp = os.path.join(ckpt_dir, f's{seed}_t{t}.pt')
                torch.save({'model': m.state_dict(),
                            't': t, 'va': acc}, fp)
                saved.append(t)
            if t % 1000 == 0:
                print(f'  s{seed} t={t} va={acc:.3f} '
                      f'({time.time() - t0:.0f}s)', flush=True)
        if cross is not None and t >= cross + 1000:
            break
    m.eval()

    # forced P/L save points (10-seed patch; only if the grid missed
    # them by more than 200 steps)
    forced = []
    if cross is not None:
        for target in (cross - 6000, cross - 1000):
            if target < 200:
                continue
            near = [s for s in saved if abs(s - target) <= 200]
            if not near:
                # rewind is expensive; instead the forced points should
                # be hit during the run -- re-walk the chain up to
                # target.  The walk is deterministic from the anchor,
                # so a fresh in-memory replay to `target` is verbatim.
                m2, opt2, _ = _replay_to(repro_dir, qk_dir, seed,
                                         id_train, chain, target)
                acc2 = va_probe(m2, va)
                fp = os.path.join(ckpt_dir, f's{seed}_t{target}.pt')
                torch.save({'model': m2.state_dict(),
                            't': target, 'va': acc2,
                            'forced': True}, fp)
                saved.append(target)
                forced.append(target)
                del m2, opt2
                torch.cuda.empty_cache() if DEV == 'cuda' else None
    saved = sorted(set(saved))
    print(f'  s{seed} replay done: cross={cross} saved={len(saved)} '
          f'forced={forced} ({time.time() - t0:.0f}s)', flush=True)
    return {'seed': seed, 'cross': cross, 'saved': saved,
            'forced_saves': forced}


def _replay_to(repro_dir, qk_dir, seed, id_train, chain, target):
    """Fresh in-memory verbatim replay from the bridge anchor to
    `target` (used only for forced save points the cadence missed)."""
    sd = torch.load(bridge_ckpt_path(repro_dir, 'A', seed, None,
                                     'full0000200'),
                    map_location='cpu', weights_only=False)
    qk = torch.load(os.path.join(qk_dir, 'qk_lineage',
                                 f'seed{seed}_F_QS.pt'),
                    map_location='cpu', weights_only=False)
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    body = dict(sd['model'])
    body['Wq.weight'] = qk['Wq'].to(DEV).clone()
    body['Wk.weight'] = qk['Wk'].to(DEV).clone()
    m.load_state_dict(body)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    for step in range(1, target - 200 + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None
        m.Wk.weight.grad = None
        opt.step()
    return m, opt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', required=True)
    ap.add_argument('--qk-dir', required=True,
                    help='dir containing qk_lineage/ (Stage 2 output)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    summary = {}
    for seed in [int(s) for s in args.seeds.split(',')]:
        summary[seed] = phase1_replay(seed, args.repro_dir, args.qk_dir,
                                      args.out)
        # hash saved ckpts
        ckpt_dir = os.path.join(args.out, 'r131c_ckpt')
        h = {}
        for t in summary[seed]['saved']:
            fp = os.path.join(ckpt_dir, f's{seed}_t{t}.pt')
            d = hashlib.sha256()
            with open(fp, 'rb') as f:
                d.update(f.read())
            h[t] = d.hexdigest()
        summary[seed]['sha256'] = h
        with open(os.path.join(args.out, 'replay_summary.json'),
                  'w') as f:
            json.dump(summary, f, indent=2)
    print('replay complete', flush=True)


if __name__ == '__main__':
    main()
