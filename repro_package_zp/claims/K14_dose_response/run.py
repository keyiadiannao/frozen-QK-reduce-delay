"""K14 run.py — dose-response of the delay to the installed code.

Package port of repro/r107_code_amplitude.py (§6eh, 2026-09-03).
Core logic is a verbatim copy of the archived script; only the
import shim, REPRO_DIR resolution, SMOKE env var and the output
path (claim dir) differ.

Motivation (user's argument, 2026-09-03): if the identity-lookup routing
mechanism is what slows generalization, then it cannot be a passive
placeholder --- it must give a DOSE RESPONSE. We therefore interpolate
the whole functional (operand code + self-suppression) continuously and
ask whether the crossing time tracks the amplitude.

Design (extends the R104 / K06 machinery, verbatim):
  * body (non-QK) from the S_200 checkpoint; QK frozen throughout;
  * row-2 attention depends only on v = Wk^T q2 (per head), so the
    functional is installed by a rank-1 edit on Wk:
        Wk_lam = Wk_0 + lam * q0.outer(vS - v0) / (q0 @ q0)
    which gives EXACTLY  v_lam = v_0 + lam (v_S - v_0)
    => operand score  f_lam(g) = f_0(g) + lam (f_S(g) - f_0(g))
    => self term      v_lam.h2 = self_0 + lam * dL22
    so lam interpolates operand code AND self-suppression together.
  * arms: L000(=00), L025, L050, L075, L100(=ADD), SS (natural control),
    RND (random operand code: same per-head norm as (vS - v0) but
    orthogonalised against h2, so the SELF TERM IS UNCHANGED).
    RND separates "any injective operand code suffices" from "the
    self-suppression (or the specific structure) is required".

Primary readout: crossing step = first t with plain val acc > 0.9
within the cap. Secondary: stratification (m1, m2) at checkpoints.

PREREGISTERED DECISION RULE (locked before running):
  DOSE-RESPONSE   : T_cross non-decreasing in lam, AND
                    (T_cross(1.0) censored while T_cross(0) crosses,
                     OR T_cross(1.0) >= 2 * T_cross(0))
                    -> the installed code causally lengthens the delay.
  NO-DOSE         : every lam > 0 crosses at ~= T_cross(0)
                    -> the delay is not set by code amplitude
                       (escape has its own clock).
  THRESHOLD-LIKE  : all lam >= some value fail equally
                    -> register as a threshold; do NOT claim a gradient.

RNG SEMANTICS WARNING (K06 lineage): Q0 weights come from the SECOND
D57Model construction after torch.manual_seed(seed) -- call-order
dependent, do not "fix".
"""
import os
import sys
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

_CLAIM = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_CLAIM))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task, states
from common import gates as cgates
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)

_zp = cgates.ZPIndexing()
N_PAIRS = _zp.N_PAIRS
A_ARR, B_ARR, Y_ARR, S_ARR = _zp.A_ARR, _zp.B_ARR, _zp.Y_ARR, _zp.S_ARR

SMOKE = bool(os.environ.get('K14_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K14] K14_SMOKE=1 but CUDA visible -- abort')

CAP = 40 if SMOKE else 12000
SEEDS = (0,) if SMOKE else tuple(
    int(s) for s in os.environ.get('CLAIM_SEEDS', '0,1,2').split(','))
STRAT_AT = (20, 40) if SMOKE else (4000, 8000, 12000)
LOSS = nn.CrossEntropyLoss()
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)

print(f'[K14] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)

R = ZPReadouts(DEV)
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y


def load_models(seed):
    torch.manual_seed(seed)
    mS = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    mS.load_state_dict(dict(sd['model']))
    m0 = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    return mS, m0, sd


def model_with(seed, wq, wk):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    body = dict(sd['model'])
    body['Wq.weight'] = wq.clone().to(DEV)
    body['Wk.weight'] = wk.clone().to(DEV)
    m.load_state_dict(body)
    return m, sd


def build_arms(seed):
    """Returns dict name -> (Wq, Wk, v_target) with v_target the intended
    per-head v (for the installation gate)."""
    mS, m0, sd = load_models(seed)
    with torch.no_grad():
        tok = torch.tensor([[0, 0, task.N_EQ]], device=DEV)
        x = mS.emb(tok) + mS.pos[None, :, :]
        h2 = mS.ln1(x)[0, 2, :].detach()            # (d,)
        qhat0 = m0.Wq.weight.detach() @ h2          # (H*dh,)
        q2S = mS.Wq.weight.detach() @ h2
    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()
    Hh, dh = mS.n_heads, mS.d_head

    # per-head: v0_h = Wk0_h^T q0_h,  vS_h = WkS_h^T qS_h
    v0, vS, dv = [], [], []
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        Wk0_h, WkS_h = Wk0[sl, :], WkS[sl, :]
        q0_h, qS_h = qhat0[sl], q2S[sl]
        v0.append(Wk0_h.T @ q0_h)
        vS.append(WkS_h.T @ qS_h)

    # random direction: same per-head norm as dv, orthogonal to h2
    rng = np.random.default_rng(1234 + seed)
    h2n = h2 / float(h2 @ h2) ** 0.5
    v_rand, d_par, d_perp = [], [], []
    for h in range(Hh):
        dv_h = vS[h] - v0[h]
        nrm = float(dv_h @ dv_h) ** 0.5
        # orthogonal decomposition of the change w.r.t. h2:
        #   d_par  -> carries the ENTIRE self-term change
        #   d_perp -> carries ZERO self-term change
        par_h = float(dv_h @ h2n) * h2n
        d_par.append(par_h)
        d_perp.append(dv_h - par_h)
        g = torch.from_numpy(rng.standard_normal(
            dv_h.shape)).to(device=DEV, dtype=dv_h.dtype)
        g = g - float(g @ h2n) * h2n              # orthogonal to h2
        g = g / (float(g @ g) ** 0.5 + 1e-30) * nrm
        v_rand.append(g)
        dv.append(dv_h)

    def _install(base_wk, delta):
        """rank-1 edit installing `delta` (list of per-head dv)."""
        Wk_e = base_wk.clone()
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            q0_h = qhat0[sl]
            Wk_e[sl, :] += q0_h.outer(delta[h]) / float(q0_h @ q0_h)
        return Wk_e

    arms = {}
    for lam in LAMS:
        Wk_lam = Wk0.clone()
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            q0_h = qhat0[sl]
            Wk_lam[sl, :] += lam * q0_h.outer(dv[h]) / float(q0_h @ q0_h)
        arms['L%03d' % int(round(lam * 100))] = (
            Wq0, Wk_lam,
            [v0[h] + lam * dv[h] for h in range(Hh)])
    # random-code arm: replace dv by the random direction (lam = 1)
    Wk_rnd = Wk0.clone()
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        q0_h = qhat0[sl]
        Wk_rnd[sl, :] += q0_h.outer(v_rand[h]) / float(q0_h @ q0_h)
    arms['RND'] = (Wq0, Wk_rnd,
                   [v0[h] + v_rand[h] for h in range(Hh)])
    # decomposition arms: SELF = h2-aligned part (full self-term change)
    #                     OPER = h2-orthogonal part (zero self-term change)
    arms['SELF'] = (Wq0, _install(Wk0, d_par),
                    [v0[h] + d_par[h] for h in range(Hh)])
    arms['OPER'] = (Wq0, _install(Wk0, d_perp),
                    [v0[h] + d_perp[h] for h in range(Hh)])
    arms['SS'] = (WqS, WkS, [vS[h] for h in range(Hh)])

    # ---- gate: installed v must equal the intended v exactly ----------
    gates = {}
    for name, (wq, wk, v_tgt) in arms.items():
        mtmp, _ = model_with(seed, wq, wk)
        with torch.no_grad():
            q2 = mtmp.Wq.weight.detach() @ h2
            worst = 0.0
            for h in range(Hh):
                sl = slice(h * dh, (h + 1) * dh)
                v_act = mtmp.Wk.weight.detach()[sl, :].T @ q2[sl]
                worst = max(worst, float((v_act - v_tgt[h]).abs().max()))
        gates[name] = worst
    print(f'  seed{seed} install-gate max|v-v_target|: '
          + ' '.join(f'{k}={v:.1e}' for k, v in gates.items()), flush=True)
    return arms, gates


def strat_plain(m, tr_mask):
    """R72/K13-semantics buckets on the full grid (own forward).

    m1 = val, non-diagonal pairs whose SWAP TWIN was in train;
    m2 = val, non-diagonal pairs whose swap twin was held out;
    m3 = val, diagonal pairs (a == b).
    """
    was_training = m.training
    m.eval()
    correct = np.zeros(N_PAIRS, dtype=bool)
    with torch.no_grad():
        for s in range(0, N_PAIRS, 4096):
            a = torch.from_numpy(ALL_A[s:s + 4096]).long().to(DEV)
            b = torch.from_numpy(ALL_B[s:s + 4096]).long().to(DEV)
            lg = m(a, b)
            correct[s:s + 4096] = (lg.argmax(-1).cpu().numpy() ==
                                   Y_ARR[s:s + 4096])
    if was_training:
        m.train()
    val = ~tr_mask
    nondiag_val = val & (A_ARR != B_ARR)
    partner_in = tr_mask[S_ARR]
    m1 = nondiag_val & partner_in
    m2 = nondiag_val & ~partner_in
    m3 = val & (A_ARR == B_ARR)
    return {'train_acc': float(correct[tr_mask].mean()),
            'm1': float(correct[m1].mean()),
            'm2': float(correct[m2].mean()),
            'm3': float(correct[m3].mean())}


def run_arm(name, wq_w, wk_w, seed, chain, id_train, va, tr_mask):
    m, sd = model_with(seed, wq_w, wk_w)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    wq_ref = m.Wq.weight.detach().clone()
    wk_ref = m.Wk.weight.detach().clone()
    rec = {'name': name, 'va_t': [200], 'va': [], 'cross': None,
           'strat': {}, 'loss': []}
    m.eval()
    with torch.no_grad():
        v0acc = float((m(va[0], va[1]).argmax(-1) == va[2]).float().mean())
    m.train()
    rec['va'].append(v0acc)
    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None                    # QK frozen
        m.Wk.weight.grad = None
        opt.step()
        t = 200 + step
        if step % 25 == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                acc = float((m(va[0], va[1]).argmax(-1) ==
                             va[2]).float().mean())
            m.train()
            rec['va_t'].append(t)
            rec['va'].append(acc)
            if rec['cross'] is None and acc > 0.9:
                rec['cross'] = t
        if step in STRAT_AT:
            rec['strat'][t] = strat_plain(m, tr_mask)
    rec['frozen_ok'] = bool(
        torch.equal(m.Wq.weight.detach(), wq_ref) and
        torch.equal(m.Wk.weight.detach(), wk_ref))
    return rec


def main():
    t0 = time.time()
    out = {}
    gates = {}
    ok_all = True
    chains = {s: R.batch_chain(s, CAP) for s in SEEDS}
    tr_masks = {}
    for seed in SEEDS:
        id_train, _c = chains[seed]
        tr = np.sort(id_train)
        tm = np.zeros(N_PAIRS, dtype=bool)
        tm[tr] = True
        tr_masks[seed] = tm
    for seed in SEEDS:
        arms, g = build_arms(seed)
        gates[seed] = g
        ok_all &= all(v <= 1e-4 for v in g.values())
        out[seed] = {}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        for name, (wq, wk, _vt) in arms.items():
            rec = run_arm(name, wq, wk, seed, chain, id_train, va,
                          tr_masks[seed])
            out[seed][name] = rec
            ok_all &= rec['frozen_ok']
            print(f'  seed{seed} {name}: cross={rec["cross"]} '
                  f'va_end={rec["va"][-1]:.3f} frozen={rec["frozen_ok"]}',
                  flush=True)

    ARMS_SHOW = ('L000', 'L025', 'L050', 'L075', 'L100',
                 'SS', 'SELF', 'OPER', 'RND')
    print('\n=== dose-response table (crossing step vs lambda) ===',
          flush=True)
    print('%-6s ' % 'seed'
          + ' '.join('%8s' % a for a in ARMS_SHOW), flush=True)
    rows = {}
    for seed in SEEDS:
        cells = [out[seed][nm]['cross'] for nm in ARMS_SHOW]
        rows[seed] = cells
        print('%-6d ' % seed
              + ' '.join('%8s' % ('cens' if c is None else c)
                         for c in cells), flush=True)

    if not SMOKE:
        print('\n=== verdict (preregistered) ===', flush=True)
        t_zero = [out[s]['L000']['cross'] for s in SEEDS]
        t_one = [out[s]['L100']['cross'] for s in SEEDS]
        t_mid = [out[s]['L050']['cross'] for s in SEEDS]
        crossed0 = [t for t in t_zero if t is not None]
        if len(crossed0) >= 2:
            mono = True
            for seed in SEEDS:
                seq = [out[seed][nm]['cross'] for nm in
                       ('L000', 'L025', 'L050', 'L075', 'L100')]
                vals = [c if c is not None else 10 ** 9 for c in seq]
                mono &= all(vals[i] <= vals[i + 1]
                            for i in range(len(vals) - 1))
            cens1 = sum(1 for t in t_one if t is None)
            ratio_ok = any(
                (t is None) or (tz is not None and t >= 2 * tz)
                for t, tz in zip(t_one, t_zero))
            if mono and (cens1 >= 2 or ratio_ok):
                print('VERDICT: DOSE-RESPONSE -- the installed routing '
                      'code causally lengthens the delay.', flush=True)
            elif all((t is not None and tz is not None and
                      abs(t - tz) <= 0.25 * max(t, tz))
                     for t, tz in zip(t_one, t_zero) if t is not None):
                print('VERDICT: NO-DOSE -- crossing time is independent '
                      'of code amplitude (escape has its own clock).',
                      flush=True)
            else:
                print('VERDICT: THRESHOLD-LIKE / MIXED -- register per '
                      'cell; do not claim a gradient.', flush=True)
        else:
            print('VERDICT: UNDECIDED -- L000 baseline did not cross in '
                  '>=2 seeds.', flush=True)
        # ---- decomposition: self-suppression vs operand code ----------
        def _cens(c):
            return 'cens' if c is None else c
        for nm, desc in (('SELF', 'h2-aligned: FULL self-term change'),
                         ('OPER', 'h2-orth: ZERO self-term change'),
                         ('RND', 'random code, self term unchanged')):
            vals = [out[s][nm]['cross'] for s in SEEDS]
            print('  %-5s (%s): %s'
                  % (nm, desc, [_cens(c) for c in vals]), flush=True)
        t_self = [out[s]['SELF']['cross'] for s in SEEDS]
        t_oper = [out[s]['OPER']['cross'] for s in SEEDS]
        slow = lambda v: (v is None)
        n_self_slow = sum(1 for c in t_self if slow(c))
        n_oper_slow = sum(1 for c in t_oper if slow(c))
        if n_self_slow >= 2 and n_oper_slow == 0:
            print('  -> SELF-suppression alone reproduces the delay; the '
                  'operand code is not required.', flush=True)
        elif n_oper_slow >= 2 and n_self_slow == 0:
            print('  -> the operand/identity code alone reproduces the '
                  'delay; self-suppression is not required.', flush=True)
        else:
            print('  -> decomposition inconclusive / both or neither '
                  'suffice; register per cell.', flush=True)

    print(f'HARNESS: {"PASS" if ok_all else "FAIL"}', flush=True)
    print(f'[K14] done in {time.time() - t0:.1f}s', flush=True)
    with open(os.path.join(_CLAIM, 'results.pkl'), 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'lams': LAMS,
                     'gates': gates, 'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
