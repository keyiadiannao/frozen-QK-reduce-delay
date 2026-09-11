"""K10 verify_r86.py — reconcile package r86 run vs archived repro pkl.

Bit-exact comparison of claims/K10_persistence/results_r86.pkl
against repro/r86_lifetime_dose_results.pkl, per seed: gate0, g3
(incl. constant), strat, and per-arm t/va_surg/va_plain/cross95_surg/
cross90_plain/w2/inj_stat + swapL* pi_audit/va_curpi/bank_sd +
swapfix pi.
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
        elif isinstance(x, tuple):
            if tuple(x) != tuple(y):
                MISMATCH.append(f'{tag}[{i}]: {x!r} vs {y!r}')
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
                           'r86_lifetime_dose_results.pkl'),
              'rb') as f:
        ref = pickle.load(f)
    with open(os.path.join(_HERE, 'results_r86.pkl'), 'rb') as f:
        mine = pickle.load(f)
    for seed in ref:
        r = ref[seed]
        m = mine[seed]
        cmp_scalar(f'seed{seed}.gate0', m['gate0'], r['gate0'])
        for k in ('fixed_points', 'bijective', 'equivariant',
                  'no_swap_self', 'y_same_frac', 'constant'):
            cmp_scalar(f'seed{seed}.g3.{k}', m['g3'][k], r['g3'][k])
        cmp_strat(f'seed{seed}.strat', m['strat'], r['strat'])
        for arm in r['arms']:
            ra = r['arms'][arm]
            ma = mine[seed]['arms'][arm]
            for key in ('t', 'va_surg', 'va_plain', 'inj_stat',
                        'va_curpi', 'bank_sd'):
                if key in ra:
                    cmp_list(f'seed{seed}.{arm}.{key}', ma[key],
                             ra[key])
            for key in ('cross95_surg', 'cross90_plain'):
                cmp_scalar(f'seed{seed}.{arm}.{key}', ma[key],
                           ra[key])
            for t in ra['w2']:
                cmp_arr(f'seed{seed}.{arm}.w2[{t}]', ma['w2'][t],
                        ra['w2'][t])
            if 'pi_audit' in ra:
                pa_r = ra['pi_audit']
                pa_m = ma['pi_audit']
                for k in ('L', 'n_draws', 'g3_fail', 'y_same_mean'):
                    cmp_scalar(f'seed{seed}.{arm}.pi_audit.{k}',
                               pa_m[k], pa_r[k])
                cmp_list(f'seed{seed}.{arm}.pi_audit.hash_head',
                         pa_m['hash_head'], pa_r['hash_head'])
                for k in ('all', 'diag', 'off'):
                    cmp_scalar(f'seed{seed}.{arm}.pi_audit.'
                               f'overlap.{k}',
                               pa_m['overlap'][k],
                               pa_r['overlap'][k])
            if 'pi' in ra:
                cmp_arr(f'seed{seed}.{arm}.pi', ma['pi'], ra['pi'])
    if MISMATCH:
        print(f'[K10-r86 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K10-r86 verify] PASS (bit-exact vs archived pkl: 6 arms '
          'x curves + pi_audit + strat)')


if __name__ == '__main__':
    main()
