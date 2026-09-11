"""K15 run.py — plastic-QK bridge: does the installed amplitude survive release?

Package port of repro/r108_plastic_qk_bridge.py (§6ei, 2026-09-03).
Core logic is a VERBATIM copy of the archived script; only the
import shim, REPRO_DIR resolution, SMOKE env var and the output
path (claim dir) differ.

Question (user, 2026-09-03; design after the round-4 audit):
    R107/K14 showed a graded dose-response of the delay to the installed
    operand code, but with the query--key block FROZEN.  The frozen block
    is what makes the intervention clean: parameter value and optimizer
    dynamics are decoupled.  Release the block and a new confound appears
    --- the t=200 checkpoint carries the NATIVE Adam moments, so an arm
    whose W_Q/W_K have been replaced by the lambda-edit values inherits
    moments that correspond to neither endpoint.  Any ladder run naively
    could equally be read as a dose-response of Adam-state mismatch.

This experiment asks only the narrow question:

    WHEN THE QUERY--KEY BLOCK IS ALLOWED TO ADAPT AGAIN, DOES THE
    INSTALLED AMPLITUDE CONTINUE TO PROGRAM THE FUTURE TRAJECTORY,
    OR DOES PLASTICITY WASH IT OUT?

It is NOT a re-run of the 9-arm frozen ladder, and it deliberately does
NOT chase the native escape (that is the separate d3 question).

Design
------
Body (everything except W_Q, W_K) is always the natural S state at
t=200, loaded from bridge_zp_A.  W_Q is always the INIT W_Q (as in
K14/R107, so that v = W_K^T q_2 with q_2 = W_Q0 h_2), except in the two
natural arms.  W_K carries the rank-1 edit v_lambda = v_0 + lambda
(v_S - v_0) per head, installed exactly as in K14 (installation gate
reused verbatim).

Two optimiser regimes, because the archived t=200 branches LOAD the
native Adam state (`opt.load_state_dict(sd['opt'])`) and are therefore
already committed to it:

  (P) 'rst'  native body Adam history PRESERVED, W_Q/W_K moments ZEROED
             -> no hybrid state anywhere: body continues faithfully, the
                released block starts with no moment history.
             arms: SS_rst, L000_rst, L050_rst, L100_rst
  (N) 'nat'  native state, untouched                     -> ground-truth
             continuation; the reference the (P) arms are read against.
             arms: SS_nat
  (F) 'fr'   fresh AdamW, no history at all              -> robustness:
             shows the ordering does not depend on option (P) vs (N).
             arms: SS_fr, L000_fr, L100_fr

Primary readout (not crossing): the amplitude of the row-2 score field
projected on the learned direction,

    s_t(g)      = v(t) . h_g(t),      v(t) = W_K(t)^T W_Q(t) h_2(t)
    ds_t(g)     = (v(t) - v_0(t)) . h_g(t)
    ds_ref(g)   = (v_S - v_0) . h_g(200)          [locked at t=200]
    lambda_eff  = <ds_t, ds_ref> / ||ds_ref||^2

which equals lambda exactly at t=200 by construction (the row-2 score
difference is (v_lambda - v_0).h_g = lambda dv.h_g).  A v-space
projection is reported alongside for compatibility with K14.

Secondary: identity maintenance I(dt) (own-snapshot self-retrieval),
stratification m1/m2, plain validation accuracy, crossing step.

G-RESET gate: zeroed moments make the first Adam step atypical
(mhat/(sqrt(vhat)+eps) -> 0.43 sign(g) at step 200, i.e. ~0.43*lr).
It is IDENTICAL across arms, so it cannot confound the lambda ordering,
but the first 50 W_Q/W_K update norms are recorded as a sanity check.

PREREGISTERED DECISION RULE (locked before running)
---------------------------------------------------
With T = 6000 and lambda in {0, .5, 1} in the (P) regime:

  PERSIST    lambda_eff trajectories remain ordered by lambda at t=T
             (spread >= 0.2 between L000 and L100 in >=2/3 seeds) AND
             the downstream phenotype is ordered (I(T) and m2(T) monotone
             in lambda in >=2/3 seeds) AND L000 crosses while L100 does
             not, in >=2/3 seeds.
             -> the early write continues to program the trajectory when
                the carrier stays plastic.
  WASHOUT    all lambda_eff curves converge (max pairwise spread < 0.1
             by t <= 2000, sustained) AND crossing/m2 indistinguishable.
             -> plasticity renormalises the installed field; the frozen
                result is a property of the intervention, not of the
                native mechanism.
  MIXED      anything else -> register per cell, claim nothing.

  RESET-COST (reported, not a verdict): SS_rst vs SS_nat.  If these
  diverge materially, the moment reset is itself load-bearing and the
  (P) ladder is read with that caveat.

RNG SEMANTICS WARNING (K06/K14 lineage): Q_0 weights come from the SECOND
D57Model construction after torch.manual_seed(seed); batch chains are
SeedSequence([9000+seed]) and are identical across arms by construction.
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

SMOKE = bool(os.environ.get('K15_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K15] K15_SMOKE=1 but CUDA visible -- abort')

CAP = 40 if SMOKE else 6000
SEEDS = (0,) if SMOKE else (0, 1, 2)
EVERY = 5 if SMOKE else 25          # lambda_eff / va cadence
CODE_EVERY = 10 if SMOKE else 50    # identity cadence
STRAT_AT = (20, 40) if SMOKE else (2000, 4000, 6000)
RESET_LOG = 10 if SMOKE else 50     # G-RESET window
LOSS = nn.CrossEntropyLoss()
LAMS = (0.0, 0.5, 1.0)

print(f'[K15] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)

R = ZPReadouts(DEV)
P = R.P
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y


# ------------------------------------------------------------------ setup
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


@torch.no_grad()
def h2_of(m):
    """LN1(emb['='] + pos[2]) -> (d,), read from the CURRENT model."""
    eq = torch.tensor([task.N_EQ], device=DEV)
    return m.ln1(m.emb(eq) + m.pos[2])[0]


@torch.no_grad()
def operand_h(m, slot):
    """LN1(emb[g] + pos[slot]) for every operand g -> (P, d)."""
    g = torch.arange(P, device=DEV)
    return m.ln1(m.emb(g) + m.pos[slot])


@torch.no_grad()
def per_head_v(m, Wq0, Wk0, Hh, dh):
    """Per-head v_h(t) = W_K,h^T q_2,h(t) and the init-QK counterpart,
    both with the CURRENT embeddings (K14 semantics: v_h in R^d, one
    vector per head -- NOT a concatenation)."""
    h2 = h2_of(m)
    Wq, Wk = m.Wq.weight.detach(), m.Wk.weight.detach()
    q, q0 = Wq @ h2, Wq0 @ h2
    vh, v0h = [], []
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        vh.append(Wk[sl, :].T @ q[sl])
        v0h.append(Wk0[sl, :].T @ q0[sl])
    return vh, v0h


@torch.no_grad()
def ds_field(m, Wq0, Wk0, Hh, dh):
    """Current score-difference field ds_t, shape (2*Hh, P).

    ds^{(h,slot)}(g) = (v_h(t) - v0_h(t)) . h_slot(g, t),
    with v0_h(t) = W_K0,h^T (W_Q0 h_2(t))_h  (init QK, current embeddings).
    """
    vh, v0h = per_head_v(m, Wq0, Wk0, Hh, dh)
    outs = []
    for slot in (0, 1):
        G = operand_h(m, slot)
        for h in range(Hh):
            outs.append(G @ (vh[h] - v0h[h]))
    return torch.stack(outs)


@torch.no_grad()
def lam_v(m, Wq0, Wk0, dv, Hh, dh):
    """v-space projection (K14-compatible secondary readout)."""
    vh, v0h = per_head_v(m, Wq0, Wk0, Hh, dh)
    num, den = 0.0, 0.0
    for h in range(Hh):
        d = vh[h] - v0h[h]
        num += float(d @ dv[h])
        den += float(dv[h] @ dv[h])
    return num / (den + 1e-30)


def build_arms(seed):
    """name -> (Wq, Wk, mode); mode in {'rst', 'nat', 'fr'}."""
    mS, m0, sd = load_models(seed)
    Hh, dh = mS.n_heads, mS.d_head
    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()

    with torch.no_grad():
        h2 = h2_of(mS)
        qhat0 = Wq0 @ h2
        q2S = WqS @ h2
        v0 = [Wk0[h * dh:(h + 1) * dh, :].T @ qhat0[h * dh:(h + 1) * dh]
              for h in range(Hh)]
        vS = [WkS[h * dh:(h + 1) * dh, :].T @ q2S[h * dh:(h + 1) * dh]
              for h in range(Hh)]
        dv = [vS[h] - v0[h] for h in range(Hh)]
        # locked reference field, t=200 embeddings
        refs = []
        for slot in (0, 1):
            G = operand_h(mS, slot)
            for h in range(Hh):
                refs.append(G @ dv[h])
        ref = torch.stack(refs)                      # (2Hh, P)
        # init-QK v at t=200 embeddings, for the v-projection denominator
        v0_at200 = [Wk0[h * dh:(h + 1) * dh, :].T @ qhat0[h * dh:(h + 1) * dh]
                    for h in range(Hh)]

    def _install(base_wk, lam):
        Wk_e = base_wk.clone()
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            q0_h = qhat0[sl]
            Wk_e[sl, :] += lam * q0_h.outer(dv[h]) / float(q0_h @ q0_h)
        return Wk_e

    arms = {}
    for lam in LAMS:
        nm = 'L%03d' % int(round(lam * 100))
        arms[nm + '_rst'] = (Wq0, _install(Wk0, lam), 'rst')
    arms['SS_rst'] = (WqS, WkS, 'rst')
    arms['SS_nat'] = (WqS, WkS, 'nat')
    for lam in (0.0, 1.0):
        nm = 'L%03d' % int(round(lam * 100))
        arms[nm + '_fr'] = (Wq0, _install(Wk0, lam), 'fr')
    arms['SS_fr'] = (WqS, WkS, 'fr')

    # ---- installation gate (verbatim K14 semantics) --------------------
    gates = {}
    for name, (wq, wk, _mode) in arms.items():
        mtmp, _ = model_with(seed, wq, wk)
        with torch.no_grad():
            q2 = mtmp.Wq.weight.detach() @ h2
            worst = 0.0
            for h in range(Hh):
                sl = slice(h * dh, (h + 1) * dh)
                v_act = mtmp.Wk.weight.detach()[sl, :].T @ q2[sl]
                v_tgt = v0_at200[h] + (vS[h] - v0_at200[h]) * (
                    1.0 if name.startswith('SS') else
                    int(name[1:4]) / 100.0)
                worst = max(worst, float((v_act - v_tgt).abs().max()))
        gates[name] = worst
    print(f'  seed{seed} install-gate: '
          + ' '.join(f'{k}={v:.1e}' for k, v in gates.items()), flush=True)
    return arms, gates, (ref, dv, Wq0, Wk0, Hh, dh)


# ------------------------------------------------------------- readouts
def strat_plain(m, tr_mask):
    """K13/R72 semantics: m1 = val pair whose swap twin was in train,
    m2 = val pair whose swap twin was held out."""
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
    return {'train_acc': float(correct[tr_mask].mean()),
            'm1': float(correct[nondiag_val & partner_in].mean()),
            'm2': float(correct[nondiag_val & ~partner_in].mean())}


def run_arm(name, wq_w, wk_w, mode, seed, chain, id_train, va, tr_mask,
            aux):
    ref, dv, Wq0, Wk0, Hh, dh = aux
    m, sd = model_with(seed, wq_w, wk_w)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    if mode in ('rst', 'nat'):
        opt.load_state_dict(sd['opt'])
        if mode == 'rst':
            for p in (m.Wq.weight, m.Wk.weight):
                st = opt.state.get(p)
                if st:
                    if 'exp_avg' in st:
                        st['exp_avg'].zero_()
                    if 'exp_avg_sq' in st:
                        st['exp_avg_sq'].zero_()

    rec = {'name': name, 'mode': mode, 't': [], 'lam_eff': [],
           'lam_v': [], 'va': [], 'I': [], 't_I': [], 'cross': None,
           'strat': {}, 'reset_q': [], 'reset_k': []}

    m.eval()
    with torch.no_grad():
        code0 = R.own_codes(m)
        ds0 = ds_field(m, Wq0, Wk0, Hh, dh)
        rec['lam_eff'].append(float((ds0 * ref).sum() /
                                    (ref * ref).sum()))
        rec['lam_v'].append(lam_v(m, Wq0, Wk0, dv, Hh, dh))
        rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                                va[2]).float().mean()))
    rec['t'].append(200)
    rec['t_I'].append(200)
    rec['I'].append(R.retrieve_self(code0, code0))
    m.train()

    wq_prev = m.Wq.weight.detach().clone()
    wk_prev = m.Wk.weight.detach().clone()

    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()                       # QK TRAINABLE (the whole point)

        if step <= RESET_LOG:
            with torch.no_grad():
                rec['reset_q'].append(float(
                    (m.Wq.weight.detach() - wq_prev).norm()))
                rec['reset_k'].append(float(
                    (m.Wk.weight.detach() - wk_prev).norm()))
        if step == RESET_LOG:
            wq_prev, wk_prev = None, None

        t = 200 + step
        if step % EVERY == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                ds = ds_field(m, Wq0, Wk0, Hh, dh)
                rec['lam_eff'].append(float((ds * ref).sum() /
                                            (ref * ref).sum()))
                rec['lam_v'].append(lam_v(m, Wq0, Wk0, dv, Hh, dh))
                acc = float((m(va[0], va[1]).argmax(-1) ==
                             va[2]).float().mean())
            m.train()
            rec['t'].append(t)
            rec['va'].append(acc)
            if rec['cross'] is None and acc > 0.9:
                rec['cross'] = t
        if step % CODE_EVERY == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                rec['I'].append(R.retrieve_self(R.own_codes(m), code0))
            m.train()
            rec['t_I'].append(t)
        if step in STRAT_AT:
            rec['strat'][t] = strat_plain(m, tr_mask)
    return rec


def main():
    t0 = time.time()
    out, gates, ok_all = {}, {}, True
    chains = {s: R.batch_chain(s, CAP) for s in SEEDS}
    tr_masks = {}
    for seed in SEEDS:
        id_train, _c = chains[seed]
        tm = np.zeros(N_PAIRS, dtype=bool)
        tm[np.sort(id_train)] = True
        tr_masks[seed] = tm

    for seed in SEEDS:
        arms, g, aux = build_arms(seed)
        gates[seed] = g
        ok_all &= all(v <= 1e-4 for v in g.values())
        out[seed] = {}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        for name in ('SS_nat', 'SS_rst', 'L000_rst', 'L050_rst',
                     'L100_rst', 'SS_fr', 'L000_fr', 'L100_fr'):
            wq, wk, mode = arms[name]
            rec = run_arm(name, wq, wk, mode, seed, chain, id_train, va,
                          tr_masks[seed], aux)
            out[seed][name] = rec
            print(f'  seed{seed} {name:9s} '
                  f'lam_eff0={rec["lam_eff"][0]:+.3f} '
                  f'lam_effT={rec["lam_eff"][-1]:+.3f} '
                  f'I_end={rec["I"][-1]:.3f} '
                  f'cross={rec["cross"]} va_end={rec["va"][-1]:.3f}',
                  flush=True)

    ORDER = ('SS_nat', 'SS_rst', 'L000_rst', 'L050_rst', 'L100_rst',
             'SS_fr', 'L000_fr', 'L100_fr')
    print('\n=== lambda_eff trajectory (score-field projection) ===',
          flush=True)
    print('%-6s %-9s ' % ('seed', 'arm')
          + ' '.join('%7s' % t for t in (200, 1000, 2000, 4000, 6000)),
          flush=True)
    for seed in SEEDS:
        for nm in ORDER:
            rec = out[seed][nm]
            row = []
            for tt in (200, 1000, 2000, 4000, 6000):
                if tt > 200 + CAP:
                    row.append('   --  ')
                    continue
                i = min(range(len(rec['t'])),
                        key=lambda k: abs(rec['t'][k] - tt))
                row.append('%+7.3f' % rec['lam_eff'][i])
            print('%-6d %-9s ' % (seed, nm) + ' '.join(row), flush=True)
        print(flush=True)

    # descriptive only (NOT part of the preregistered rule): does a
    # released block grow a code of its own from whatever it started with?
    print('=== descriptive: lambda_eff range over the run ===', flush=True)
    print('%-6s %-9s %8s %8s %8s' %
          ('seed', 'arm', 'start', 'max', 'mean'), flush=True)
    for seed in SEEDS:
        for nm in ORDER:
            le = out[seed][nm]['lam_eff']
            print('%-6d %-9s %+8.3f %+8.3f %+8.3f' % (
                seed, nm, le[0], max(le), sum(le) / len(le)), flush=True)
        print(flush=True)

    print('=== identity I(dt) and endpoint phenotype ===', flush=True)
    print('%-6s %-9s %7s %7s %7s %7s %7s %7s %8s' %
          ('seed', 'arm', 'I(1000)', 'I(3000)', 'I(T)', 'm1(T)', 'm2(T)',
           'va(T)', 'cross'), flush=True)
    for seed in SEEDS:
        for nm in ORDER:
            rec = out[seed][nm]

            def _at(arr, tarr, tt):
                i = min(range(len(tarr)), key=lambda k: abs(tarr[k] - tt))
                return arr[i]
            ts = sorted(rec['strat'])
            m2 = rec['strat'][ts[-1]]['m2'] if ts else float('nan')
            m1 = rec['strat'][ts[-1]]['m1'] if ts else float('nan')
            print('%-6d %-9s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %8s' % (
                seed, nm, _at(rec['I'], rec['t_I'], 1000),
                _at(rec['I'], rec['t_I'], 3000), rec['I'][-1],
                m1, m2, rec['va'][-1],
                'cens' if rec['cross'] is None else rec['cross']),
                flush=True)
        print(flush=True)

    if not SMOKE:
        print('=== verdict (preregistered) ===', flush=True)
        # --- spread of lambda_eff between L000 and L100 at T, per seed --
        spread, i_ord, cross_ok = [], 0, 0
        for seed in SEEDS:
            a = out[seed]['L000_rst']['lam_eff'][-1]
            c = out[seed]['L100_rst']['lam_eff'][-1]
            spread.append(c - a)
        for seed in SEEDS:
            seq = [out[seed][nm]['lam_eff'][-1]
                   for nm in ('L000_rst', 'L050_rst', 'L100_rst')]
            if seq[0] <= seq[1] <= seq[2]:
                i_ord += 1
            if (out[seed]['L000_rst']['cross'] is not None
                    and out[seed]['L100_rst']['cross'] is None):
                cross_ok += 1
        n_big = sum(1 for s in spread if s >= 0.2)
        print('  lam_eff spread L100-L000 at T: '
              + ' '.join('%+.3f' % s for s in spread)
              + f'   (>=0.2 in {n_big}/3 seeds)', flush=True)
        print(f'  lam_eff monotone in lambda: {i_ord}/3 seeds', flush=True)
        print(f'  L000 crosses & L100 does not: {cross_ok}/3 seeds',
              flush=True)
        # --- washout test: convergence of the three curves -------------
        conv = 0
        for seed in SEEDS:
            ts = out[seed]['L000_rst']['t']
            vals = []
            for tt in out[seed]['L000_rst']['t']:
                if tt > 2000:
                    break
                i = ts.index(tt)
                vals.append(max(out[seed][nm]['lam_eff'][i]
                                for nm in ('L000_rst', 'L050_rst',
                                           'L100_rst'))
                            - min(out[seed][nm]['lam_eff'][i]
                                  for nm in ('L000_rst', 'L050_rst',
                                             'L100_rst')))
            if vals and vals[-1] < 0.1:
                conv += 1
        print(f'  curves converged (spread<0.1 by t=2000): {conv}/3 seeds',
              flush=True)

        if n_big >= 2 and i_ord >= 2 and cross_ok >= 2:
            print('VERDICT: PERSIST -- the installed amplitude continues '
                  'to program the trajectory with the carrier plastic.',
                  flush=True)
        elif conv >= 2:
            print('VERDICT: WASHOUT -- plasticity renormalises the '
                  'installed field; the frozen dose-response is a '
                  'property of the intervention.', flush=True)
        else:
            print('VERDICT: MIXED -- register per cell; claim nothing '
                  'beyond the recorded trajectories.', flush=True)

        # --- reset cost (reported, not a verdict) ----------------------
        print('  reset cost (SS_rst vs SS_nat):', flush=True)
        for seed in SEEDS:
            a = out[seed]['SS_nat']
            b = out[seed]['SS_rst']
            print('    seed%d  lam_effT %.3f vs %.3f | I(T) %.3f vs %.3f'
                  % (seed, a['lam_eff'][-1], b['lam_eff'][-1],
                     a['I'][-1], b['I'][-1]), flush=True)
        # --- G-RESET gate ---------------------------------------------
        print('  G-RESET (first-step QK update norm, mode=rst):', flush=True)
        for seed in SEEDS:
            r = out[seed]['L100_rst']
            print('    seed%d  |dWq|1=%.2e |dWk|1=%.2e  |dWq|50=%.2e '
                  '|dWk|50=%.2e' % (seed, r['reset_q'][0], r['reset_k'][0],
                                    r['reset_q'][-1], r['reset_k'][-1]),
                  flush=True)

    print(f'HARNESS: {"PASS" if ok_all else "FAIL"}', flush=True)
    print(f'[K15] done in {time.time() - t0:.1f}s', flush=True)
    with open(os.path.join(_CLAIM, 'results.pkl'), 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'lams': LAMS,
                     'every': EVERY, 'code_every': CODE_EVERY,
                     'strat_at': STRAT_AT, 'gates': gates, 'arms': out},
                    f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
