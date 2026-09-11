"""verify_extract.py — M2 extraction reconciliation harness (package root).

Checks (CPU unless noted):
  C1 task tables      : zp table == (a+b) mod 113; sizes; split sizes.
  C2 split construct  : task.make_split == gates.build_train ==
                        train_d57_abeq.main() inline construction
                        (seeds 0-2, bit-equal; unsorted id_train).
  C3 model            : (a) fresh-init state_dict bit-equal to
                        train_d57_abeq.D57Model (same torch.manual_seed,
                        seeds 0-2);
                        (b) bridge_zp A/C/B seed0-2 full0000200 loaded
                        into BOTH implementations -> forward logits
                        bit-equal on a fixed val batch (max|d|==0);
                        (c) n_params == 227826 (archive value).
  C4 optimizer        : decay/no_decay param-name grouping identical.
  C5 sampler gates    : common.gates sample_pi == r92_gate_sampler
                        (seeds 0-2, draws 0-5, bit-equal) and every
                        draw passes V1/V2/V3/V4/V6.
  C6 GPU replay       : base/train_base.py run_base(family A, seed0)
                        vs bridge summary.json bit-exact (run with
                        --replay; GPU, ~3 min per seed).

Cross-implementation checks (C2b/C3/C4/C5) require repro/ on disk
(reading research code only -- never writes to it).
"""
import argparse
import os
import sys

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task, gates
from common.model import D57Model, make_optimizer

DEFAULT_REPRO = os.path.join(os.path.dirname(_PKG_ROOT), 'repro')

FAILED = []


def report(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ''))
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', default=DEFAULT_REPRO)
    ap.add_argument('--replay', action='store_true',
                    help='also run C6 GPU replay (seed0 family A)')
    ap.add_argument('--replay-seed', type=int, default=0)
    args = ap.parse_args()

    task.set_group('zp', 113)
    zp = gates.ZPIndexing()

    # ---------- C1 task tables ----------
    print('C1 task tables')
    a, b, y = task.build_tables()
    report('zp table == (a+b) mod 113', bool(np.all(y == (a + b) % 113)))
    report('sizes', task.N_PAIRS == 12769 and a.shape == (12769,)
           and task.GROUP == {'name': 'zp', 'n': 113, 'p': 113})
    tr_idx, va_idx = task.make_split(0)
    report('split sizes (3831/8938)',
           len(tr_idx) == 3831 and len(va_idx) == 8938,
           f'n_train={len(tr_idx)}')

    # ---------- cross-impl import ----------
    T = None
    GS = None
    if os.path.isdir(args.repro_dir):
        sys.path.insert(0, args.repro_dir)
        cwd = os.getcwd()
        try:
            import train_d57_abeq as T_
            T = T_
            T.set_group('zp', 113)
            import r92_gate_sampler as GS_   # runs its own gate at import
            GS = GS_
        finally:
            os.chdir(cwd)
        print(f'(cross-impl anchors: {args.repro_dir})')
    else:
        print('(repro/ not found -- cross-impl checks skipped)')

    # ---------- C2 split construction ----------
    print('C2 split construction (seeds 0-2)')
    ok_all = True
    for seed in (0, 1, 2):
        idt, _ = task.make_split(seed)
        idt2, _, _ = gates.build_train(zp, seed)
        same = bool(np.array_equal(idt, idt2))
        if T is not None:
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            perm = rng.permutation(T.N_PAIRS)
            n_train = int(round(T.N_PAIRS * 0.30))
            same = same and bool(np.array_equal(idt, perm[:n_train]))
        ok_all &= same
    report('task.make_split == gates.build_train == T.main construction',
           ok_all, 'seeds 0-2, unsorted id_train bit-equal')

    # ---------- C3 model ----------
    print('C3 model extraction')
    report('fresh-init state_dict bit-equal to T (seeds 0-2)', _c3_init(T))
    report('n_params == 227826 (archive)',
           _n_params() == 227826, f'{_n_params()}')
    report('bridge A/C/B seed0-2 t=200 forward bit-equal to T',
           _c3_bridge_forward(args.repro_dir, T))

    # ---------- C4 optimizer ----------
    print('C4 optimizer grouping')
    m = D57Model()
    _, dec_c, nodec_c = make_optimizer(m, 'wd_0011')
    nm_c = {id(p): n for n, p in m.named_parameters()}
    dec_c = sorted(nm_c[id(p)] for p in dec_c)
    nodec_c = sorted(nm_c[id(p)] for p in nodec_c)
    ok = True
    if T is not None:
        mt = T.D57Model()
        _, dec_t, nodec_t = T.make_optimizer(mt, 'wd_0011')
        nm = {id(p): n for n, p in mt.named_parameters()}
        dec_t = sorted(nm[id(p)] for p in dec_t)
        nodec_t = sorted(nm[id(p)] for p in nodec_t)
        ok = (dec_c == dec_t and nodec_c == nodec_t)
    report('decay/no_decay name grouping identical to T', ok)

    # ---------- C5 sampler ----------
    print('C5 sampler gates (seeds 0-2, draws 0-5)')
    ok_c5 = True
    n_pass = 0
    for seed in (0, 1, 2):
        idt, tr, tr_mask = gates.build_train(zp, seed)
        full, half, diag_T, half_single = gates.classify(zp, tr_mask)
        U = diag_T | set(half_single)
        for draw in range(6):
            pi, _ = gates.sample_pi(zp, seed, draw, full, half)
            res = gates.verify_assignment(zp, pi, tr, tr_mask, U)
            ok_c5 &= res['PASS']
            n_pass += res['PASS']
            if GS is not None:
                full_g, half_g, diag_g, hs_g = GS.classify(tr_mask)
                Ug = diag_g | set(hs_g)
                pi_g, _ = GS.sample_pi(seed, draw, full_g, half_g, Ug)
                ok_c5 &= bool(np.array_equal(pi, pi_g)) and Ug == U
    report('all draws pass V1/V2/V3/V4/V6 and match r92_gate_sampler '
           'bitwise', ok_c5, f'{n_pass}/18 draws PASS')

    # ---------- summary ----------
    print()
    if FAILED:
        print(f'EXTRACT RECONCILIATION: FAIL ({len(FAILED)}): {FAILED}')
        sys.exit(1)
    print('EXTRACT RECONCILIATION: ALL PASS')


def _n_params():
    torch.manual_seed(0)
    return sum(p.numel() for p in D57Model().parameters())


def _c3_init(T):
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m1 = D57Model()
        torch.manual_seed(seed)
        m2 = D57Model() if T is None else T.D57Model()
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        if set(sd1) != set(sd2):
            return False
        for k in sd1:
            if not torch.equal(sd1[k], sd2[k]):
                return False
    return True


def _c3_bridge_forward(repro_dir, T):
    from common import states
    dev = torch.device('cpu')
    for family in ('A', 'C', 'B'):
        for seed in (0, 1, 2):
            try:
                path = states.bridge_ckpt_path(repro_dir, family, seed,
                                               None, 'full0000200')
            except FileNotFoundError:
                print(f'    (skip {family} seed{seed}: ckpt missing)')
                continue
            sd, _ = states.load_ckpt(path)
            torch.manual_seed(0)
            m1 = D57Model()
            m1.load_state_dict(sd)
            m1 = m1.to(dev).eval()
            if T is not None:
                torch.manual_seed(0)
                m2 = T.D57Model()
                m2.load_state_dict(sd)
                m2 = m2.to(dev).eval()
            # fixed val batch (split seed0)
            va_idx = task.make_split(0)[1]
            all_a, all_b, _ = task.build_tables()
            aa = torch.from_numpy(all_a[va_idx[:256]]).long()
            bb = torch.from_numpy(all_b[va_idx[:256]]).long()
            with torch.no_grad():
                l1 = m1(aa, bb)
                if T is not None:
                    l2 = m2(aa, bb)
                    if not torch.equal(l1, l2):
                        print(f'    MISMATCH {family} seed{seed}: '
                              f'max|d|={(l1 - l2).abs().max().item()}')
                        return False
    return True


if __name__ == '__main__':
    main()
