"""Base training entry for wd_1111 bridge checkpoints.

Identical protocol to base/train_base.py except make_optimizer uses
wd_1111 (attention weights additionally decayed).  Produces bridge
checkpoints compatible with states.bridge_ckpt_path().

Run:
  python scripts/train_wd1111_bridge.py \
      --family A --seed 0 --steps 30000 --out /path/bridge_wd1111/A
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
from common.model import D57Model, make_optimizer, accuracy

FAMILY_MAP = {'A': {'arch': 'abeq', 'fqk': 0},
              'C': {'arch': 'abeq', 'fqk': 200},
              'B': {'arch': 'abeq_noeqemb', 'fqk': 0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=list(FAMILY_MAP))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    spec = FAMILY_MAP[args.family]
    task.set_group('zp', 113)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n = task.N_ELEM
    all_a = np.arange(n)[None, :].repeat(n, 0).ravel()
    all_b = np.arange(n)[:, None].repeat(n, 1).ravel()
    all_y = task.group_mul_np(all_a, all_b)
    ss = np.random.SeedSequence(args.seed)
    rng = np.random.default_rng(ss)
    perm = rng.permutation(task.N_PAIRS)
    n_train = int(round(task.N_PAIRS * 0.30))
    tr_a = torch.from_numpy(all_a[perm[:n_train]]).long()
    tr_b = torch.from_numpy(all_b[perm[:n_train]]).long()
    tr_y = torch.from_numpy(all_y[perm[:n_train]]).long()
    va_a = torch.from_numpy(all_a[perm[n_train:]]).long()
    va_b = torch.from_numpy(all_b[perm[n_train:]]).long()
    va_y = torch.from_numpy(all_y[perm[n_train:]]).long()

    torch.manual_seed(args.seed)
    m = D57Model(pos_mode='zeros', arch=spec['arch'],
                 ln_eps=1e-5, eq_alpha=0.0).to(dev)
    opt, _, _ = make_optimizer(m, 'wd_1111')

    loss_fn = nn.CrossEntropyLoss()
    history = []
    first_cross = None
    tag = (f"seed{args.seed}_{spec['arch']}_a0.0_eps1e-05_"
           f"fqk{spec['fqk']}_fwv0_fmlp0_fp20_zeros_wd_1111")
    os.makedirs(args.out, exist_ok=True)
    snap200 = None

    for step in range(1, args.steps + 1):
        m.train()
        idx = torch.randint(tr_a.shape[0], (512,))
        a, b, y = tr_a[idx].to(dev), tr_b[idx].to(dev), tr_y[idx].to(dev)
        opt.zero_grad()
        loss_fn(m(a, b), y).backward()
        if spec['fqk'] and step <= spec['fqk']:
            m.Wq.weight.grad = None
            m.Wk.weight.grad = None
        if step <= 20:
            m.pos.grad[2].zero_()
        opt.step()

        if step == 200:
            snap200 = ({k: v.clone() for k, v in m.state_dict().items()},
                       opt.state_dict())
            torch.save({'step': 200, 'model': snap200[0],
                        'opt': snap200[1], 'val_acc': 0.0},
                       os.path.join(args.out, f'{tag}_full0000200.pt'))

        # intermediate snapshots every 2000 steps for CB3/CB4 sampling
        if spec['fqk'] == 0 and step % 2000 == 0 and step >= 200:
            torch.save({'step': step, 'model': m.state_dict(),
                        'opt': opt.state_dict(), 'val_acc': 0.0},
                       os.path.join(args.out,
                                    f'{tag}_snap{step:07d}.pt'))

        if step % args.eval_every == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                correct = 0
                for s in range(0, len(va_a), 4096):
                    lg = m(va_a[s:s+4096].to(dev),
                           va_b[s:s+4096].to(dev))
                    correct += (lg.argmax(-1).cpu() ==
                                va_y[s:s+4096]).sum().item()
                va_acc = correct / len(va_a)
            history.append({'step': step, 'val_acc': va_acc})
            if first_cross is None and va_acc > 0.9:
                first_cross = step
            m.train()

    torch.save({'step': args.steps, 'model': m.state_dict(),
                'opt': opt.state_dict(),
                'val_acc': history[-1]['val_acc'] if history else 0},
               os.path.join(args.out, f'{tag}_final.pt'))
    summary = {'seed': args.seed, 'family': args.family,
               'wd_arm': 'wd_1111', 'first_cross': first_cross,
               'final_val': history[-1]['val_acc'] if history else 0,
               'history': history}
    with open(os.path.join(args.out, f'{tag}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[wd1111-bridge] {args.family} seed{args.seed}: '
          f'cross={first_cross} final_va='
          f'{history[-1]["val_acc"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
