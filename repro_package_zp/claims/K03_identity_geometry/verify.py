"""K03 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K03_identity_geometry/results.pkl
against repro/r85_revisit_identity_results.pkl, per seed:
gate0 (G0'' bitwise), t_div, branch, and per-arm t/val_acc lists,
cross90/cross95, last-revisit records (all fields), fixed-lag records,
and snapshot code tables (c8/c6 float32 arrays, bitwise).
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
        if x != y:
            MISMATCH.append(f'{tag}[{i}]: mine={x!r} ref={y!r}')


def cmp_rec_list(tag, a, b, fields):
    if len(a) != len(b):
        MISMATCH.append(f'{tag}: length {len(a)} vs {len(b)}')
        return
    for i, (x, y) in enumerate(zip(a, b)):
        for f in fields:
            if x[f] != y[f]:
                MISMATCH.append(f'{tag}[{i}].{f}: mine={x[f]!r} '
                                f'ref={y[f]!r}')


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r85_revisit_identity_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if sorted(mine, key=str) != sorted(ref, key=str):
        print(f'[K03 verify] seed keys differ: '
              f'{sorted(mine, key=str)} vs {sorted(ref, key=str)}')
        sys.exit(1)
    for seed in ref:
        r = ref[seed]
        m = mine[seed]
        for name in ('natS', 'natF'):
            cmp_scalar(f'seed{seed}.gate0[{name}]', m['gate0'][name],
                       r['gate0'][name])
        cmp_scalar(f'seed{seed}.t_div', m['t_div'], r['t_div'])
        cmp_scalar(f'seed{seed}.branch', m['branch'], r['branch'])
        for arm in ('natS', 'natF'):
            rm = ref[seed]['arms'][arm]
            mm = mine[seed]['arms'][arm]
            cmp_list(f'seed{seed}.{arm}.t', mm['t'], rm['t'])
            cmp_list(f'seed{seed}.{arm}.val_acc', mm['val_acc'],
                     rm['val_acc'])
            for key in ('cross90', 'cross95'):
                cmp_scalar(f'seed{seed}.{arm}.{key}', mm[key], rm[key])
            cmp_rec_list(f'seed{seed}.{arm}.last', mm['last'],
                         rm['last'],
                         ('t', 'P8', 'chance', 'mrr8', 'P6', 'mrr6',
                          'Praw', 'P8_diag', 'P8_off', 'n_classes'))
            if sorted(mm['fixedlag'], key=str) != \
                    sorted(rm['fixedlag'], key=str):
                MISMATCH.append(f'seed{seed}.{arm}.fixedlag: keys differ')
            else:
                for k in rm['fixedlag']:
                    for f in ('t', 'P8', 'chance', 'mrr8', 'P6'):
                        cmp_scalar(f'seed{seed}.{arm}.fixedlag[{k}].{f}',
                                   mm['fixedlag'][k][f],
                                   rm['fixedlag'][k][f])
            if sorted(mm['snaps']) != sorted(rm['snaps']):
                MISMATCH.append(f'seed{seed}.{arm}.snaps: steps differ')
            else:
                for s in rm['snaps']:
                    for arr in ('c8', 'c6'):
                        a = np.asarray(mm['snaps'][s][arr])
                        b = np.asarray(rm['snaps'][s][arr])
                        if not np.array_equal(a, b):
                            d = float(np.abs(a - b).max())
                            MISMATCH.append(
                                f'seed{seed}.{arm}.snaps[{s}].{arr} '
                                f'differs (max|d|={d})')
    if MISMATCH:
        print(f'[K03 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K03 verify] PASS (bit-exact vs archived pkl: curves, '
          'retrieval records, snapshot codes)')


if __name__ == '__main__':
    main()
