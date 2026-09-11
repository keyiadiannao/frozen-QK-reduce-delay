"""K15 verify.py — bit-exact reconciliation vs the archived R108 pkl.

Compares the package run (claims/K15_plastic_release/results.pkl) against
repro/r108_plastic_qk_bridge_results.pkl on:
  * install gates (all arms, all seeds)
  * per arm/seed: lam_eff + lam_v trajectories, va trajectory, I(t) + t_I,
    crossing step, strat dicts, G-RESET first-step norms, mode tag
  * run metadata (cap, lams, every, code_every, strat_at, seeds)
Bit-exact (==) on floats; no tolerance.
"""
import os
import pickle
import sys

_CLAIM = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_CLAIM))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

ARCH = os.path.join(REPRO_DIR, 'r108_plastic_qk_bridge_results.pkl')
MINE = os.path.join(_CLAIM, 'results.pkl')

ORDER = ('SS_nat', 'SS_rst', 'L000_rst', 'L050_rst', 'L100_rst',
         'SS_fr', 'L000_fr', 'L100_fr')

a = pickle.load(open(ARCH, 'rb'))
m = pickle.load(open(MINE, 'rb'))

fails = []
n = 0


def chk(tag, x, y):
    global n
    n += 1
    if x != y:
        fails.append(f'{tag}: archived={x!r} mine={y!r}')


chk('meta.seeds', a['seeds'], m['seeds'])
chk('meta.cap', a['cap'], m['cap'])
chk('meta.lams', a['lams'], m['lams'])
chk('meta.every', a['every'], m['every'])
chk('meta.code_every', a['code_every'], m['code_every'])
chk('meta.strat_at', a['strat_at'], m['strat_at'])

for seed in a['seeds']:
    for arm, g in a['gates'][seed].items():
        chk(f'gate s{seed}.{arm}', g, m['gates'][seed][arm])
    for arm in ORDER:
        ra, rm = a['arms'][seed][arm], m['arms'][seed][arm]
        chk(f's{seed}.{arm}.name', ra['name'], rm['name'])
        chk(f's{seed}.{arm}.mode', ra['mode'], rm['mode'])
        chk(f's{seed}.{arm}.t', ra['t'], rm['t'])
        chk(f's{seed}.{arm}.lam_eff', ra['lam_eff'], rm['lam_eff'])
        chk(f's{seed}.{arm}.lam_v', ra['lam_v'], rm['lam_v'])
        chk(f's{seed}.{arm}.va', ra['va'], rm['va'])
        chk(f's{seed}.{arm}.t_I', ra['t_I'], rm['t_I'])
        chk(f's{seed}.{arm}.I', ra['I'], rm['I'])
        chk(f's{seed}.{arm}.cross', ra['cross'], rm['cross'])
        for t, st in ra['strat'].items():
            chk(f's{seed}.{arm}.strat[{t}]', st, rm['strat'][t])
        chk(f's{seed}.{arm}.reset_q', ra['reset_q'], rm['reset_q'])
        chk(f's{seed}.{arm}.reset_k', ra['reset_k'], rm['reset_k'])

print(f'K15 verify: {n} checks, {len(fails)} failures')
for f in fails[:20]:
    print(' ', f)
print('VERDICT:', 'PASS' if not fails else 'FAIL')
sys.exit(0 if not fails else 1)
