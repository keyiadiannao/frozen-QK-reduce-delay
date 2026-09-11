"""K04 verify.py — reconcile package run vs archived repro pkl.

Compares claims/K04_scoring_functional/results.pkl (produced by run.py
on GPU) bit-exactly against repro/r102_dm_dissect_results.pkl:
  * dL / scS / sc0 arrays (float32, bitwise via np.array_equal);
  * per-head scalars (res0/res1/res2, amp, correlations, enrichment);
  * f0/f1 function vectors (float64 lists, bitwise).
Bit-equality is expected: forward-only pipeline, deterministic kernels,
identical RNG streams (np default_rng(1000+seed)).
"""
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

MISMATCH = []


def close_enough_tag(a, b):
    return ''


def cmp_scalars(tag, mine, ref):
    if mine != ref:
        MISMATCH.append(f'{tag}: mine={mine!r} ref={ref!r}')


def main():
    arch_path = os.path.join(REPRO_DIR, 'r102_dm_dissect_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    if sorted(mine) != sorted(ref):
        print(f'[K04 verify] seed sets differ: {sorted(mine)} vs {sorted(ref)}')
        sys.exit(1)
    for seed in ref:
        r, m = ref[seed], mine[seed]
        for key in ('pos_norm_S', 'pos_norm_0'):
            cmp_scalars(f'seed{seed}.{key}', m[key], r[key])
        for i, (a, b) in enumerate(zip(m['pos_diffs'], r['pos_diffs'])):
            cmp_scalars(f'seed{seed}.pos_diffs[{i}]', a, b)
        cmp_scalars(f'seed{seed}.operand_rep_diff',
                    m['operand_rep_diff'], r['operand_rep_diff'])
        for arr in ('dL', 'scS', 'sc0'):
            if not (m[arr].shape == r[arr].shape
                    and m[arr].dtype == r[arr].dtype
                    and (m[arr] == r[arr]).all()):
                import numpy as np
                d = float(np.abs(m[arr] - r[arr]).max())
                MISMATCH.append(f'seed{seed}.{arr} differs (max|d|={d})')
        for h, (ph, rh) in enumerate(zip(m['heads'], r['heads'])):
            for key in ('res0', 'res1', 'res2', 'corr_d_swap',
                        'corr_q0_baseline', 'enrichment', 'gain_query',
                        'rand_gain_mean', 'dL2'):
                cmp_scalars(f'seed{seed}.h{h}.{key}', ph[key], rh[key])
            for key, sub in ph['amp'].items():
                cmp_scalars(f'seed{seed}.h{h}.amp.{key}',
                            sub, rh['amp'][key])
            for j, (a, b) in enumerate(zip(ph['mean_dp'], rh['mean_dp'])):
                cmp_scalars(f'seed{seed}.h{h}.mean_dp[{j}]', a, b)
            for vec in ('f0', 'f1'):
                a = ph[vec]
                b = rh[vec]
                if len(a) != len(b) or any(x != y for x, y in zip(a, b)):
                    MISMATCH.append(f'seed{seed}.h{h}.{vec} differs')
    if MISMATCH:
        print(f'[K04 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K04 verify] PASS (bit-exact vs archived pkl, seeds 0-2, 4 heads)')


if __name__ == '__main__':
    main()
