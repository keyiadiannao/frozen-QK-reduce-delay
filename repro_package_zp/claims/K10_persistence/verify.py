"""K10 verify.py — reconcile package r84 run vs archived repro pkl.

Bit-exact comparison of claims/K10_persistence/results_r84.pkl
against repro/r84_persistence_missing_cell_results.pkl, per seed:
gate0, g3 (incl. constant), g3_steps_fail, bank checks/hashes/
y_same, res_hashes/res_y_same, overlap, strat (all arms), and per-arm
t/va_surg/va_plain/cross95_surg/cross90_plain/w2/inj_stat (+ swapres
bank records, swapfix pi, bank).
"""
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
    if a != b:
        MISMATCH.append(f'{tag}: mine={a!r} ref={b!r}')


def cmp_list(tag, a, b):
    if len(a) != len(b):
        MISMATCH.append(f'{tag}: length {len(a)} vs {len(b)}')
        return
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, dict):
            for f in y:
                if x[f] != y[f]:
                    MISMATCH.append(f'{tag}[{i}].{f}: '
                                    f'{x[f]!r} vs {y[f]!r}')
        elif x != y:
            MISMATCH.append(f'{tag}[{i}]: mine={x!r} ref={y!r}')


def cmp_arr(tag, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or \
            not np.array_equal(a, b):
        MISMATCH.append(f'{tag}: differs')


def cmp_strat(tag, m, r):
    if sorted(m, key=str) != sorted(r, key=str):
        MISMATCH.append(f'{tag}: steps differ')
        return
    for t in r:
        for arm in r[t]:
            for k in r[t][arm]:
                cmp_scalar(f'{tag}[{t}].{arm}.{k}',
                           m[t][arm][k], r[t][arm][k])


def main():
    with open(os.path.join(REPRO_DIR,
                           'r84_persistence_missing_cell_results.pkl'),
              'rb') as f:
        ref = pickle.load(f)
    with open(os.path.join(_HERE, 'results_r84.pkl'), 'rb') as f:
        mine = pickle.load(f)
    for seed in ref:
        r = ref[seed]
        m = mine[seed]
        cmp_scalar(f'seed{seed}.gate0', m['gate0'], r['gate0'])
        for k in ('fixed_points', 'bijective', 'equivariant',
                  'no_swap_self', 'y_same_frac', 'constant'):
            cmp_scalar(f'seed{seed}.g3.{k}', m['g3'][k], r['g3'][k])
        cmp_scalar(f'seed{seed}.g3_steps_fail', m['g3_steps_fail'],
                   r['g3_steps_fail'])
        cmp_list(f'seed{seed}.bank_checks', m['bank_checks'],
                 r['bank_checks'])
        cmp_list(f'seed{seed}.bank_hashes', m['bank_hashes'],
                 r['bank_hashes'])
        cmp_list(f'seed{seed}.bank_y_same', m['bank_y_same'],
                 r['bank_y_same'])
        cmp_list(f'seed{seed}.res_hashes', m['res_hashes'],
                 r['res_hashes'])
        cmp_list(f'seed{seed}.res_y_same', m['res_y_same'],
                 r['res_y_same'])
        for k in r['overlap']:
            if isinstance(r['overlap'][k], dict):
                for kk in r['overlap'][k]:
                    cmp_scalar(f'seed{seed}.overlap.{k}.{kk}',
                               m['overlap'][k][kk],
                               r['overlap'][k][kk])
            else:
                cmp_scalar(f'seed{seed}.overlap.{k}',
                           m['overlap'][k], r['overlap'][k])
        cmp_strat(f'seed{seed}.strat', m['strat'], r['strat'])
        for arm in r['arms']:
            ra = r['arms'][arm]
            ma = mine[seed]['arms'][arm]
            for key in ('t', 'va_surg', 'va_plain', 'inj_stat'):
                cmp_list(f'seed{seed}.{arm}.{key}', ma[key], ra[key])
            for key in ('cross95_surg', 'cross90_plain'):
                cmp_scalar(f'seed{seed}.{arm}.{key}', ma[key],
                           ra[key])
            for t in ra['w2']:
                cmp_arr(f'seed{seed}.{arm}.w2[{t}]', ma['w2'][t],
                        ra['w2'][t])
            for key in ('va_surg_bank_draws', 'va_surg_curpi',
                        'bank_sd'):
                if key in ra:
                    cmp_list(f'seed{seed}.{arm}.{key}', ma[key],
                             ra[key])
        cmp_arr(f'seed{seed}.pi_fix', m['pi_fix'], r['pi_fix'])
        for i in range(len(r['bank'])):
            cmp_arr(f'seed{seed}.bank[{i}]', m['bank'][i],
                    r['bank'][i])
    if MISMATCH:
        print(f'[K10 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K10 verify] PASS (bit-exact vs archived pkl: gates, '
          'samplers, bank, curves, w2, inj_stat, strat)')


if __name__ == '__main__':
    main()
