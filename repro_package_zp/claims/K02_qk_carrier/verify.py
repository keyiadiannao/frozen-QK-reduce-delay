"""K02 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K02_qk_carrier/results.pkl against
repro/r97_state_factorial_results.pkl: per arm-key ({name}-{seed}),
t/val_acc/loss lists, crossed step, and the acute readout dict
(loss/va/logit_d_vs_SQs/A2_cos_vs_SQs).  4000-step training on the
global-RNG chain (R51 replay construction); determinism established
in M2.
"""
import os
import pickle
import sys

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


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r97_state_factorial_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if sorted(mine) != sorted(ref):
        print(f'[K02 verify] arm keys differ: {sorted(mine)} vs '
              f'{sorted(ref)}')
        sys.exit(1)
    for key in ref:
        r = ref[key]
        m = mine[key]
        cmp_scalar(f'{key}.crossed', m['crossed'], r['crossed'])
        cmp_list(f'{key}.t', m['t'], r['t'])
        cmp_list(f'{key}.val_acc', m['val_acc'], r['val_acc'])
        cmp_list(f'{key}.loss', m['loss'], r['loss'])
        for ak in ('loss', 'va', 'logit_d_vs_SQs', 'A2_cos_vs_SQs'):
            cmp_scalar(f'{key}.acute.{ak}', m['acute'][ak],
                       r['acute'][ak])
    if MISMATCH:
        print(f'[K02 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K02 verify] PASS (bit-exact vs archived pkl; '
          '3 seeds x 4 arms + acute readouts)')


if __name__ == '__main__':
    main()
