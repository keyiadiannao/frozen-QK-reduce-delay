"""K09 verify.py — reconcile package runs vs archived repro pkls.

Covers both halves of the claim:
  * results_r92.pkl vs repro/r92_native_ingraph_ws_results.pkl:
    gate (d_logit/worst_cos/worst_rel), U_size, pi0_hash, per-arm
    t/va_plain/cross95/cross90/train_surg (+ va_surg for permBS,
    g3_fail/n_draws/hash_head for the WS arms), strat_plain;
  * results_r93.pkl vs repro/r93_fixedws_release_results.pkl:
    per seed x d: t/va_plain/cross95/cross95_post/strat/escape.
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
        if isinstance(x, dict):
            for f in y:
                if x[f] != y[f]:
                    MISMATCH.append(f'{tag}[{i}].{f}: '
                                    f'{x[f]!r} vs {y[f]!r}')
        elif x != y:
            MISMATCH.append(f'{tag}[{i}]: mine={x!r} ref={y!r}')


def cmp_arm(tag, m, r):
    for key in ('t', 'va_plain', 'train_surg'):
        cmp_list(f'{tag}.{key}', m[key], r[key])
    for key in ('cross95', 'cross90'):
        cmp_scalar(f'{tag}.{key}', m[key], r[key])
    for key in ('g3_fail', 'n_draws'):
        if key in r:
            cmp_scalar(f'{tag}.{key}', m[key], r[key])
    if 'hash_head' in r:
        cmp_list(f'{tag}.hash_head', m['hash_head'],
                 r['hash_head'])
    if 'va_surg' in r:
        cmp_list(f'{tag}.va_surg', m['va_surg'], r['va_surg'])


def cmp_strat_plain(tag, m, r):
    if sorted(m, key=str) != sorted(r, key=str):
        MISMATCH.append(f'{tag}: steps differ')
        return
    for t in r:
        for arm in r[t]:
            for k in r[t][arm]:
                cmp_scalar(f'{tag}[{t}].{arm}.{k}',
                           m[t][arm][k], r[t][arm][k])


def main():
    # ---- r92 ----
    with open(os.path.join(REPRO_DIR,
                           'r92_native_ingraph_ws_results.pkl'),
              'rb') as f:
        ref = pickle.load(f)
    with open(os.path.join(_HERE, 'results_r92.pkl'), 'rb') as f:
        mine = pickle.load(f)
    for seed in ref:
        r, m = ref[seed], mine[seed]
        for k in ('d_logit', 'worst_cos', 'worst_rel'):
            cmp_scalar(f'seed{seed}.gate.{k}', m['gate'][k],
                       r['gate'][k])
        cmp_scalar(f'seed{seed}.U_size', m['U_size'], r['U_size'])
        cmp_scalar(f'seed{seed}.pi0_hash', m['pi0_hash'],
                   r['pi0_hash'])
        for arm in r['arms']:
            cmp_arm(f'seed{seed}.{arm}', m['arms'][arm],
                    r['arms'][arm])
        cmp_strat_plain(f'seed{seed}.strat_plain',
                        m['strat_plain'], r['strat_plain'])

    # ---- r93 ----
    with open(os.path.join(REPRO_DIR,
                           'r93_fixedws_release_results.pkl'),
              'rb') as f:
        ref = pickle.load(f)
    with open(os.path.join(_HERE, 'results_r93.pkl'), 'rb') as f:
        mine = pickle.load(f)
    for seed in ref:
        for d in ref[seed]:
            r = ref[seed][d]
            m = mine[seed][d]
            for key in ('t', 'va_plain'):
                cmp_list(f'r93 seed{seed} d{d}.{key}', m[key],
                         r[key])
            for key in ('cross95', 'cross95_post', 'escape'):
                cmp_scalar(f'r93 seed{seed} d{d}.{key}', m[key],
                           r[key])
            for t in r['strat']:
                for k in r['strat'][t]:
                    cmp_scalar(f'r93 seed{seed} d{d}.strat[{t}].{k}',
                               m['strat'][t][k], r['strat'][t][k])

    if MISMATCH:
        print(f'[K09 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K09 verify] PASS (bit-exact: r92 4 arms x 3 seeds incl. '
          'strat_plain + r93 6 release runs)')


if __name__ == '__main__':
    main()
