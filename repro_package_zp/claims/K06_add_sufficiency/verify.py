"""K06 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K06_add_sufficiency/results.pkl against
repro/r104_native_addremove_results.pkl: gates (d_add/d_rem) and per
seed x arm (00/SS/ADD/REMOVE) readout lists must match exactly.
"""
import math
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

MISMATCH = []


def cmp_list(tag, a, b):
    if len(a) != len(b):
        MISMATCH.append(f'{tag}: length {len(a)} vs {len(b)}')
        return
    for i, (x, y) in enumerate(zip(a, b)):
        same = (x == y) or (isinstance(x, float) and isinstance(y, float)
                            and math.isnan(x) and math.isnan(y))
        if not same:
            MISMATCH.append(f'{tag}[{i}]: mine={x!r} ref={y!r}')


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r104_native_addremove_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    for meta in ('seeds', 'steps', 'eval_dl'):
        if mine[meta] != ref[meta]:
            MISMATCH.append(f'{meta}: {mine[meta]} vs {ref[meta]}')
    for seed in ref['gates']:
        for key in ('d_add', 'd_rem'):
            cmp_list(f'gates[{seed}].{key}',
                     [mine['gates'][seed][key]], [ref['gates'][seed][key]])
        if mine['gates'][seed]['ok'] is not True:
            MISMATCH.append(f'gates[{seed}]: ok False')
    for seed in ref['arms']:
        for arm in ref['arms'][seed]:
            r = ref['arms'][seed][arm]
            m = mine['arms'][seed][arm]
            for key in ('t', 'I', 'rho_m', 'dm', 'D', 'va', 'va_t',
                        'loss', 'm_med', 'm_mean'):
                cmp_list(f'seed{seed}.{arm}.{key}', m[key], r[key])
            if m['frozen_ok'] is not True:
                MISMATCH.append(f'seed{seed}.{arm}: frozen_ok False')
    if MISMATCH:
        print(f'[K06 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K06 verify] PASS (bit-exact vs archived pkl; '
          '3 seeds x 4 arms + gates)')


if __name__ == '__main__':
    main()
