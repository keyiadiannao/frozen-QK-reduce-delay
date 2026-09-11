"""K14 verify.py — bit-exact reconciliation against the archived run.

Compares this claim's results.pkl with
repro/r107_code_amplitude_results.pkl (the run that produced §6eh).
Every gate value, crossing step, validation curve, stratification
bucket and the frozen-QK flag must match exactly.
"""
import os
import sys
import pickle

import numpy as np

_CLAIM = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_CLAIM))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

ARCH = os.path.join(REPRO_DIR, 'r107_code_amplitude_results.pkl')
MINE = os.path.join(_CLAIM, 'results.pkl')

FAILS = []
NCHK = [0]


def cmp_scalar(tag, a, b):
    NCHK[0] += 1
    if a is None or b is None:
        ok = (a is None) and (b is None)
    else:
        ok = (a == b)
    if not ok:
        FAILS.append(f'{tag}: mine={a} archive={b}')


def cmp_list(tag, a, b):
    NCHK[0] += 1
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=0):
        FAILS.append(f'{tag}: shape {a.shape} vs {b.shape}, '
                     f'maxdiff={np.abs(a - b).max() if a.shape == b.shape else "n/a"}')


def main():
    with open(ARCH, 'rb') as f:
        arch = pickle.load(f)
    with open(MINE, 'rb') as f:
        mine = pickle.load(f)

    cmp_scalar('cap', mine['cap'], arch['cap'])
    cmp_list('lams', mine['lams'], arch['lams'])

    for seed in arch['seeds']:
        # gates[seed] is keyed by ARM NAME
        for armname, g in arch['gates'][seed].items():
            cmp_scalar(f'gate s{seed}.{armname}',
                       mine['gates'][seed][armname], g)
        for arm, ra in arch['arms'][seed].items():
            rm = mine['arms'][seed][arm]
            cmp_scalar(f's{seed}.{arm}.cross', rm['cross'], ra['cross'])
            cmp_scalar(f's{seed}.{arm}.frozen_ok',
                       rm['frozen_ok'], ra['frozen_ok'])
            cmp_list(f's{seed}.{arm}.va_t', rm['va_t'], ra['va_t'])
            cmp_list(f's{seed}.{arm}.va', rm['va'], ra['va'])
            cmp_scalar(f's{seed}.{arm}.n_strat',
                       len(rm['strat']), len(ra['strat']))
            for t in sorted(ra['strat']):
                for k in ('train_acc', 'm1', 'm2', 'm3'):
                    cmp_scalar(f's{seed}.{arm}.strat[{t}].{k}',
                               rm['strat'][t][k], ra['strat'][t][k])

    print(f'[K14] checks = {NCHK[0]}, failures = {len(FAILS)}')
    for f in FAILS[:20]:
        print('  FAIL', f)
    print('K14 VERIFY: ' + ('PASS' if not FAILS else 'FAIL'))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
