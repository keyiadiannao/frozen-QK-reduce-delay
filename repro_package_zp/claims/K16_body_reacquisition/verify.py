"""K16 verify.py — bit-exact reconciliation vs the archived R108b pkl."""
import os
import pickle
import sys

_CLAIM = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_CLAIM))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

a = pickle.load(open(os.path.join(
    REPRO_DIR, 'r108b_body_reacquisition_results.pkl'), 'rb'))
m = pickle.load(open(os.path.join(_CLAIM, 'results.pkl'), 'rb'))

fails, n = [], 0


def chk(tag, x, y):
    global n
    n += 1
    if x != y:
        fails.append(f'{tag}: archived={x!r} mine={y!r}')


chk('meta.seeds', a['seeds'], m['seeds'])
chk('meta.cap', a['cap'], m['cap'])

for seed in a['seeds']:
    ra, rm = a['arms'][seed], m['arms'][seed]
    chk(f's{seed}.gate', ra['gate'], rm['gate'])
    chk(f's{seed}.t', ra['t'], rm['t'])
    chk(f's{seed}.va', ra['va'], rm['va'])
    chk(f's{seed}.t_I', ra['t_I'], rm['t_I'])
    chk(f's{seed}.I', ra['I'], rm['I'])
    chk(f's{seed}.cross', ra['cross'], rm['cross'])
    chk(f's{seed}.lam', ra['lam'], rm['lam'])
    for t, st in ra['strat'].items():
        chk(f's{seed}.strat[{t}]', st, rm['strat'][t])

print(f'K16 verify: {n} checks, {len(fails)} failures')
for f in fails[:20]:
    print(' ', f)
print('VERDICT:', 'PASS' if not fails else 'FAIL')
sys.exit(0 if not fails else 1)
