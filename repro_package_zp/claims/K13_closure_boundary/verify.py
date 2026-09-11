"""K13 verify.py — reconcile package run vs archived repro pkl.

Bit-exact comparison of claims/K13_closure_boundary/results.pkl
against repro/r106_closure_regime_results.pkl, per seed:
  gates (d_add); arms 00/SS/ADD and their -P50 branches:
  codes dict (int t -> 12769x12 float32, bitwise), va_t/va, strat
  dicts, cross, samp_fail, frozen_ok, R_ctrl, restored, disrupted;
  fork snapshots (model state + deep-copied opt state) compared
  structurally tensor-by-tensor.
"""
import os
import pickle
import sys

import torch

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


def cmp_state_dicts(tag, a, b):
    if sorted(a) != sorted(b):
        MISMATCH.append(f'{tag}: keys differ')
        return
    for k in a:
        if not torch.equal(a[k], b[k]):
            MISMATCH.append(f'{tag}.{k}: tensor differs')


def cmp_opt_state(tag, a, b):
    if sorted(a['state'], key=str) != sorted(b['state'], key=str):
        MISMATCH.append(f'{tag}: state keys differ')
        return
    for k in b['state']:
        sa, sb = a['state'][k], b['state'][k]
        for kk in sb:
            va_, vb_ = sa.get(kk), sb[kk]
            if torch.is_tensor(vb_):
                if va_ is None or not torch.equal(va_, vb_):
                    MISMATCH.append(f'{tag}.state[{k}].{kk}: differs')
            else:
                if va_ != vb_:
                    MISMATCH.append(f'{tag}.state[{k}].{kk}: '
                                    f'{va_!r} vs {vb_!r}')
    if a['param_groups'] != b['param_groups']:
        MISMATCH.append(f'{tag}.param_groups differ')


def cmp_fork(tag, mf, rf):
    cmp_state_dicts(f'{tag}.model', mf[0], rf[0])
    cmp_opt_state(f'{tag}.opt', mf[1], rf[1])


def main():
    arch_path = os.path.join(REPRO_DIR,
                             'r106_closure_regime_results.pkl')
    mine_path = os.path.join(_HERE, 'results.pkl')
    with open(arch_path, 'rb') as f:
        ref = pickle.load(f)
    with open(mine_path, 'rb') as f:
        mine = pickle.load(f)
    for meta in ('seeds', 'cap'):
        if mine[meta] != ref[meta]:
            MISMATCH.append(f'{meta} differ')
    for seed in ref['gates']:
        cmp_scalar(f'gates[{seed}].d_add',
                   mine['gates'][seed]['d_add'],
                   ref['gates'][seed]['d_add'])
        if mine['gates'][seed]['ok'] is not True:
            MISMATCH.append(f'gates[{seed}]: ok False')
    for seed in ref['arms']:
        for arm in ref['arms'][seed]:
            r = ref['arms'][seed][arm]
            m = mine['arms'][seed][arm]
            if sorted(m['codes'], key=str) != \
                    sorted(r['codes'], key=str):
                MISMATCH.append(f'seed{seed}.{arm}: code steps differ')
                continue
            for t in r['codes']:
                a = m['codes'][t]
                b = r['codes'][t]
                if not (a.shape == b.shape and (a == b).all()):
                    MISMATCH.append(
                        f'seed{seed}.{arm}.codes[{t}] differs')
            cmp_list(f'seed{seed}.{arm}.va_t', m['va_t'], r['va_t'])
            cmp_list(f'seed{seed}.{arm}.va', m['va'], r['va'])
            for t in r.get('strat', {}):
                for k in r['strat'][t]:
                    cmp_scalar(f'seed{seed}.{arm}.strat[{t}].{k}',
                               m['strat'][t][k], r['strat'][t][k])
            for key in ('cross', 'samp_fail', 'frozen_ok', 'restored',
                        'disrupted', 'm_mean0'):
                if key in r or key in m:
                    cmp_scalar(f'seed{seed}.{arm}.{key}', m.get(key),
                               r.get(key))
            if 'R_ctrl' in r:
                for t in r['R_ctrl']:
                    cmp_scalar(f'seed{seed}.{arm}.R_ctrl[{t}]',
                               m['R_ctrl'][t], r['R_ctrl'][t])
            if 'fork' in r and r['fork'] is not None:
                cmp_fork(f'seed{seed}.{arm}.fork', m['fork'], r['fork'])
    if MISMATCH:
        print(f'[K13 verify] FAIL ({len(MISMATCH)} mismatches)')
        for x in MISMATCH[:10]:
            print('  ' + x)
        sys.exit(1)
    print('[K13 verify] PASS (bit-exact vs archived pkl: codes, va, '
          'strat, R_ctrl, forks)')


if __name__ == '__main__':
    main()
