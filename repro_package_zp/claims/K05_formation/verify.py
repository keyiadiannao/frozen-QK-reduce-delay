"""K05 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K05_formation/results.pkl against
repro/r105_formation_accounting_results.pkl: per seed, all arrays
(U/R/DC/D/D_qk/A_t/C_t/S0/S1/Cself — float64), scalars (C40/C_last/
late_min/body_share), pheno dict, verdict string, and losses list.
Fresh 200-step GPU training with explicit batch chains; determinism
established in M2.
"""
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

MISMATCH = []


def cmp_scalar(tag, a, b):
    same = (a == b) or (isinstance(a, float) and isinstance(b, float)
                        and math.isnan(a) and math.isnan(b))
    if not same:
        MISMATCH.append(f'{tag}: mine={a!r} ref={b!r}')


def cmp_arr(tag, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        MISMATCH.append(f'{tag}: shape {a.shape} vs {b.shape}')
    elif not np.array_equal(a, b):
        MISMATCH.append(f'{tag}: differs (max|d|='
                        f'{np.abs(a - b).max()})')


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r105_formation_accounting_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if mine['seeds'] != ref['seeds'] or mine['steps'] != ref['steps']:
        MISMATCH.append('meta seeds/steps differ')
    for seed in ref['arms']:
        r = ref['arms'][seed]
        m = mine['arms'][seed]
        for arr in ('U', 'R', 'DC', 'D', 'D_qk', 'A_t', 'C_t',
                    'S0', 'S1', 'Cself'):
            cmp_arr(f'seed{seed}.{arr}', m[arr], r[arr])
        for key in ('C40', 'C_last', 'late_min', 'body_share'):
            cmp_scalar(f'seed{seed}.{key}', m[key], r[key])
        for key in ('f_std', 'self_mean', 'corr_role', 'ok'):
            cmp_scalar(f'seed{seed}.pheno.{key}', m['pheno'][key],
                       r['pheno'][key])
        if m['verdict'] != r['verdict']:
            MISMATCH.append(f'seed{seed}.verdict: {m["verdict"]} '
                            f'vs {r["verdict"]}')
        cmp_arr(f'seed{seed}.losses', np.array(m['losses']),
                np.array(r['losses']))
    if MISMATCH:
        print(f'[K05 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K05 verify] PASS (bit-exact vs archived pkl, seeds 0-2)')


if __name__ == '__main__':
    main()
