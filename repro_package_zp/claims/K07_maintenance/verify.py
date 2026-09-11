"""K07 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K07_maintenance/results.pkl against
repro/r103_frozen_carrier_retention_results.pkl: per seed x arm, the
I/rho_m/dm/m_med/m_mean/D/va/va_t/loss lists must match exactly.
Training is 500 real-gradient steps on explicit numpy batch chains with
GPU-deterministic kernels; the QS/Q0 call order (hence the q0_cache RNG
semantics) is preserved by run.py.
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
                             'r103_frozen_carrier_retention_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    for meta in ('seeds', 'steps', 'eval_dl'):
        if mine[meta] != ref[meta]:
            MISMATCH.append(f'{meta}: {mine[meta]} vs {ref[meta]}')
    for seed in ref['arms']:
        for qsrc in ref['arms'][seed]:
            r = ref['arms'][seed][qsrc]
            m = mine['arms'][seed][qsrc]
            for key in ('t', 'I', 'rho_m', 'dm', 'm_med', 'm_mean',
                        'D', 'va', 'va_t', 'loss'):
                cmp_list(f'seed{seed}.{qsrc}.{key}', m[key], r[key])
            if m['frozen_ok'] is not True:
                MISMATCH.append(f'seed{seed}.{qsrc}: frozen_ok False')
    if MISMATCH:
        print(f'[K07 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K07 verify] PASS (bit-exact vs archived pkl; 3 seeds x 2 arms)')


if __name__ == '__main__':
    main()
