"""K07 run.py — frozen-carrier retention assay.

Package port of repro/r103_frozen_carrier_retention.py (M2.5,
2026-09-02).  Verbatim except: shared readouts live in
common/readouts.py (ZPReadouts; bit-exact relocation, re-verified),
common/ imports, claim-dir output, SMOKE env R103_SMOKE -> K07_SMOKE.

RNG-SEMANTICS WARNING (do not "fix"): in build_arm, for the Q0 arm the
fresh QK model m0 is constructed AFTER m (torch.manual_seed(seed) at
function entry, then two D57Model constructions) -- so q0_cache holds
the SECOND construction's QK.  This call-order-dependent definition is
what the archive used; changing it breaks reconciliation.

Claim (ledger K7): frozen Q_S makes downstream real-gradient dynamics
identity-maintaining (I(500): QS 0.965-0.970 vs Q0 0.658-0.686).
Preregistered §6dt; verdict §6dx; margin amendment §6dw.
"""
import os
import sys
import pickle
import time

import torch
import torch.nn as nn

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task, states
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)

SMOKE = bool(os.environ.get('K07_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K07] K07_SMOKE=1 but CUDA visible -- abort')
print(f'[K07] DEV={DEV} SMOKE={SMOKE}', flush=True)

STEPS = 8 if SMOKE else 500
EVAL_DL = (0, 4, 8) if SMOKE else (0, 25, 100, 250, 500)
SEEDS = (0,) if SMOKE else tuple(
    int(s) for s in os.environ.get('CLAIM_SEEDS', '0,1,2').split(','))
LOSS = nn.CrossEntropyLoss()

R = ZPReadouts(DEV)
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y


def build_arm(qsrc, seed, q0_cache):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    body = dict(sd['model'])
    if qsrc == 'Q0':
        if seed not in q0_cache:
            m0 = D57Model(pos_mode='zeros',
                          arch='abeq')            # SECOND construction
            q0_cache[seed] = (m0.Wq.weight.detach().cpu().clone(),
                              m0.Wk.weight.detach().cpu().clone())
        body['Wq.weight'] = q0_cache[seed][0]
        body['Wk.weight'] = q0_cache[seed][1]
    m.load_state_dict(body)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    return m, opt, dict(sd['model'])


def run_arm(qsrc, seed, chain, id_train, q0_cache):
    va = R.val_probe(seed)
    m, opt, body0 = build_arm(qsrc, seed, q0_cache)
    wq0 = m.Wq.weight.detach().clone()
    wk0 = m.Wk.weight.detach().clone()
    rec = {'t': [200], 'I': [], 'rho_m': [], 'dm': [],
           'm_med': [], 'm_mean': [], 'D': [], 'va': [],
           'va_t': [200], 'loss': []}
    c_ref = R.own_codes(m)
    rec['I'].append(1.0)
    mm0 = R.margin_self(c_ref, c_ref)
    rec['m_med'].append(mm0[0])
    rec['m_mean'].append(mm0[1])
    rec['rho_m'].append(1.0)
    rec['dm'].append(0.0)
    rec['D'].append(R.retrieve_self(c_ref, c_ref))
    rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                            va[2]).float().mean()))
    for step in range(1, STEPS + 1):
        idx = chain[step - 1]
        pairs = id_train[idx]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None            # QK frozen throughout
        m.Wk.weight.grad = None
        opt.step()
        rec['loss'].append(float(loss.detach()))
        t = 200 + step
        if step in EVAL_DL:
            c_t = R.own_codes(m)
            rec['t'].append(t)
            rec['I'].append(R.retrieve_self(c_t, c_ref))
            mm = R.margin_self(c_t, c_ref)
            rec['m_med'].append(mm[0])
            rec['m_mean'].append(mm[1])
            rec['rho_m'].append(mm[1] / mm0[1] if abs(mm0[1]) > 1e-9
                                else float('nan'))
            rec['dm'].append(mm[1] - mm0[1])
            rec['D'].append(R.retrieve_self(c_t, c_t))
        if step % 25 == 0 or step == STEPS:
            m.eval()
            with torch.no_grad():
                v = float((m(va[0], va[1]).argmax(-1) ==
                           va[2]).float().mean())
            m.train()
            rec['va_t'].append(t)
            rec['va'].append(v)
    frozen_ok = bool(torch.equal(m.Wq.weight.detach(), wq0) and
                     torch.equal(m.Wk.weight.detach(), wk0))
    rec['frozen_ok'] = frozen_ok
    return rec


def main():
    out = {}
    ok_all = True
    t0 = time.time()
    q0_cache = {}
    chains = {seed: R.batch_chain(seed, STEPS) for seed in SEEDS}
    for seed in SEEDS:
        out[seed] = {}
        id_train, chain = chains[seed]
        for qsrc in ('QS', 'Q0'):
            rec = run_arm(qsrc, seed, chain, id_train, q0_cache)
            out[seed][qsrc] = rec
            ok_all &= rec['frozen_ok']
            print(f'seed{seed} {qsrc}: I={["%.3f" % x for x in rec["I"]]} '
                  f'rho_m={["%.3f" % x for x in rec["rho_m"]]} '
                  f'va_last={rec["va"][-1]:.3f} '
                  f'frozen={rec["frozen_ok"]}', flush=True)

    print('\n=== verdict (locked §6dt + §6dw margin amendment) ===',
          flush=True)
    if not SMOKE:
        dIs, drhos = [], []
        for seed in SEEDS:
            dI = out[seed]['QS']['I'][-1] - out[seed]['Q0']['I'][-1]
            dr = (out[seed]['QS']['rho_m'][-1] -
                  out[seed]['Q0']['rho_m'][-1])
            dIs.append(dI)
            drhos.append(dr)
            print(f'  seed{seed}: dI(500)={dI:+.3f} '
                  f'drho_m(500)={dr:+.3f}', flush=True)
        n_maint = sum(d >= 0.15 for d in dIs)
        n_nodiff = sum(abs(d) < 0.05 for d in dIs)
        n_inv = sum(d <= -0.15 for d in dIs)
        rho_pos = sum(r > 0 for r in drhos) >= 2
        if n_maint >= 2 and rho_pos:
            print('VERDICT: MAINTAINED -- frozen Q_S carrier makes '
                  'downstream true-gradient dynamics identity-maintaining',
                  flush=True)
        elif n_inv >= 2:
            print('VERDICT: INVERTED -- QK->identity-reinforcement '
                  'explanation dies', flush=True)
        elif n_nodiff >= 2:
            print('VERDICT: NO-DIFF -- QK fate carrier does not act via '
                  'cross-time identity retention', flush=True)
        else:
            print('VERDICT: MIXED -- register per cell', flush=True)
    print(f'HARNESS (frozen gates): {"PASS" if ok_all else "FAIL"}',
          flush=True)
    print(f'[K07] done in {time.time() - t0:.1f}s', flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'steps': STEPS,
                     'eval_dl': EVAL_DL, 'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
