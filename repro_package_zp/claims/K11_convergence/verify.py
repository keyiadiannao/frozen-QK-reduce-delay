"""K11 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K11_convergence/results.pkl against
repro/r98_pretransition_assay_results.pkl, per key (S0/F0/Z0/...):
control (t/va_plain/cross95/cross90/strat/Rpre), codes_T0 (float16
bitwise), branch (t/va/cross/strat/Rctrl/Rpre/pulse_end/va_at_end/
g3_fail).
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


def cmp_dict_of_scalars(tag, a, b):
    if sorted(a, key=str) != sorted(b, key=str):
        MISMATCH.append(f'{tag}: keys differ')
        return
    for k in b:
        cmp_scalar(f'{tag}[{k}]', a[k], b[k])


def cmp_codes(tag, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or \
            not np.array_equal(a, b):
        MISMATCH.append(f'{tag}: differs')


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r98_pretransition_assay_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if sorted(mine) != sorted(ref):
        print(f'[K11 verify] keys differ: {sorted(mine)} vs '
              f'{sorted(ref)}')
        sys.exit(1)
    for key in ref:
        r = ref[key]
        m = mine[key]
        c_ref, c_mine = r['control'], m['control']
        cmp_list(f'{key}.control.t', c_mine['t'], c_ref['t'])
        cmp_list(f'{key}.control.va_plain', c_mine['va_plain'],
                 c_ref['va_plain'])
        for k in ('cross95', 'cross90'):
            cmp_scalar(f'{key}.control.{k}', c_mine[k], c_ref[k])
        for t in c_ref['strat']:
            for kk in c_ref['strat'][t]:
                cmp_scalar(f'{key}.control.strat[{t}].{kk}',
                           c_mine['strat'][t][kk],
                           c_ref['strat'][t][kk])
        cmp_dict_of_scalars(f'{key}.control.Rpre',
                            c_mine['Rpre'], c_ref['Rpre'])
        cmp_codes(f'{key}.codes_T0', m['codes_T0'], r['codes_T0'])
        rb = r['branch']
        mb = m['branch']
        cmp_list(f'{key}.branch.t', mb['t'], rb['t'])
        cmp_list(f'{key}.branch.va_plain', mb['va_plain'],
                 rb['va_plain'])
        for k in ('cross95', 'cross90', 'pulse_end', 'va_at_end',
                  'g3_fail'):
            cmp_scalar(f'{key}.branch.{k}', mb[k], rb[k])
        for t in rb.get('strat', {}):
            for kk in rb['strat'][t]:
                cmp_scalar(f'{key}.branch.strat[{t}].{kk}',
                           mb['strat'][t][kk], rb['strat'][t][kk])
        cmp_dict_of_scalars(f'{key}.branch.Rctrl',
                            mb['Rctrl'], rb['Rctrl'])
        cmp_dict_of_scalars(f'{key}.branch.Rpre',
                            mb['Rpre'], rb['Rpre'])
    if MISMATCH:
        print(f'[K11 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K11 verify] PASS (bit-exact vs archived pkl: control '
          'curves, strat, Rctrl/Rpre, codes_T0)')


if __name__ == '__main__':
    main()
