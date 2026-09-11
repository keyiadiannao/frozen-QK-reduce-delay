"""K01 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K01_fate_lock/results.pkl against
repro/amp_necess2x2_results.pkl (keys "F-train-0" etc.): t/val_acc/
loss lists, gnorm dicts (per snap step, per param), gvalc/gvalqk
lists, crossed.  4000-step replay on the global-RNG chain; determinism
established in M2.
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
    arch_path = os.path.join(REPRO_DIR, 'amp_necess2x2_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if sorted(mine) != sorted(ref):
        print(f'[K01 verify] keys differ: {sorted(mine)} vs {sorted(ref)}')
        sys.exit(1)
    for key in ref:
        r = ref[key]
        m = mine[key]
        cmp_scalar(f'{key}.crossed', m['crossed'], r['crossed'])
        cmp_list(f'{key}.t', m['t'], r['t'])
        cmp_list(f'{key}.val_acc', m['val_acc'], r['val_acc'])
        cmp_list(f'{key}.loss', m['loss'], r['loss'])
        cmp_list(f'{key}.gvalc', m['gvalc'], r['gvalc'])
        cmp_list(f'{key}.gvalqk', m['gvalqk'], r['gvalqk'])
        if sorted(m['gnorm']) != sorted(r['gnorm']):
            MISMATCH.append(f'{key}.gnorm: snap steps differ')
        else:
            for t in r['gnorm']:
                gm, gr = m['gnorm'][t], r['gnorm'][t]
                if sorted(gm) != sorted(gr):
                    MISMATCH.append(f'{key}.gnorm[{t}]: param set differs')
                    continue
                for n in gr:
                    cmp_scalar(f'{key}.gnorm[{t}].{n}', gm[n], gr[n])
    if MISMATCH:
        print(f'[K01 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K01 verify] PASS (bit-exact vs archived pkl; '
          '2 states x 2 modes x 3 seeds incl. full grad-norm probes)')


if __name__ == '__main__':
    main()
