"""scripts/run_wd1111_battery.py -- WD1111 Core Battery driver.

Runs four experiments under the attention-inclusive weight-decay
regime (wd_1111) to verify that the paper's primary claims are not
artifacts of selective decay.  Designed to run AFTER (or alongside)
the 10-seed wd_0011 confirmatory.

  CB1  fate contrast: natS vs natF from scratch (wd_1111)
  CB2  carrier transplant: R97 2x2 on wd_1111-born t=200 snapshots
  CB3  lookup M2y: REMOVE/DELTA/SAME-Y/CENTROID on mature wd_1111
  CB4  Fourier ladder L1-L4 on wd_1111 pre/post-crossing checkpoints

Usage:
  python scripts/run_wd1111_battery.py \
      --out /path/wd1111_battery [--seeds 0,1,2] [--stage N]

Output structure:
  {out}/bridge_wd1111_{A,C}/     bridge checkpoints
  {out}/cb1_fate/                CB1 trajectories + crossings
  {out}/cb2_transplant/          CB2 four-cell results
  {out}/cb3_lookup/              CB3 M2y four-arm results
  {out}/cb4_fourier/             CB4 ladder results
  {out}/wd1111_battery_summary.json
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from common import task
from common.model import D57Model, make_optimizer

task.set_group('zp', 113)

LOSS = nn.CrossEntropyLoss()
P = 113
SEEDS = (0, 1, 2)


def sh(cmd):
    print(f'$ {cmd}', flush=True)
    import subprocess
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f'FAILED: {cmd}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--stage', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    out = args.out
    os.makedirs(out, exist_ok=True)
    S = args.stage

    def want(n):
        return S in (0, n)

    PY = sys.executable
    bridge_script = os.path.join(HERE, 'train_wd1111_bridge.py')

    # ---- W1: bridge checkpoints (natS + natF per seed) ------------
    if want(1):
        for seed in seeds:
            for fam in ('A', 'C'):
                bdir = os.path.join(out, f'bridge_wd1111_{fam}')
                marker = os.path.join(
                    bdir, f'seed{seed}_*_summary.json')
                import glob
                if glob.glob(marker):
                    print(f'[W1] skip {fam} seed{seed}')
                    continue
                sh(f'"{PY}" "{bridge_script}" --family {fam} '
                   f'--seed {seed} --steps 30000 --out "{bdir}" '
                   f'--device {args.device}')

    # ---- CB1: fate contrast ------------------------------------------
    if want(2):
        cb1_dir = os.path.join(out, 'cb1_fate')
        os.makedirs(cb1_dir, exist_ok=True)
        results = {}
        for seed in seeds:
            tag_a = f'seed{seed}_abeq_a0.0_eps1e-05_fqk0_fwv0_fmlp0_fp20_zeros_wd_1111'
            tag_c = f'seed{seed}_abeq_a0.0_eps1e-05_fqk200_fwv0_fmlp0_fp20_zeros_wd_1111'
            sa = json.load(open(os.path.join(
                out, f'bridge_wd1111_A', f'{tag_a}_summary.json')))
            sc = json.load(open(os.path.join(
                out, f'bridge_wd1111_C', f'{tag_c}_summary.json')))
            results[seed] = {'natS_cross': sa.get('first_cross'),
                             'natF_cross': sc.get('first_cross')}
        with open(os.path.join(cb1_dir, 'cb1_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print('[CB1] fate contrast results:', json.dumps(results,
                                                          indent=2))

    # ---- CB2: carrier transplant (needs more code, run separately) --
    if want(3):
        cb2_dir = os.path.join(out, 'cb2_transplant')
        os.makedirs(cb2_dir, exist_ok=True)
        sh(f'"{PY}" "{os.path.join(HERE, "cb2_transplant.py")}" '
           f'--bridge-dir "{out}" --out "{cb2_dir}" '
           f'--seeds {args.seeds} --device {args.device}')

    # ---- CB3: lookup M2y -------------------------------------------
    if want(4):
        cb3_dir = os.path.join(out, 'cb3_lookup')
        os.makedirs(cb3_dir, exist_ok=True)
        sh(f'"{PY}" "{os.path.join(HERE, "cb34_readouts.py")}" '
           f'--bridge-dir "{out}" --out "{cb3_dir}" '
           f'--seeds {args.seeds} --mode cb3 --device {args.device}')

    # ---- CB4: Fourier ladder ----------------------------------------
    if want(5):
        cb4_dir = os.path.join(out, 'cb4_fourier')
        os.makedirs(cb4_dir, exist_ok=True)
        sh(f'"{PY}" "{os.path.join(HERE, "cb34_readouts.py")}" '
           f'--bridge-dir "{out}" --out "{cb4_dir}" '
           f'--seeds {args.seeds} --mode cb4 --device {args.device}')

    # ---- summary -----------------------------------------------------
    if want(6):
        summary_path = os.path.join(out, 'wd1111_battery_summary.json')
        # already written by individual stages if they implement it
        print(f'[summary] check {summary_path}')


if __name__ == '__main__':
    main()
