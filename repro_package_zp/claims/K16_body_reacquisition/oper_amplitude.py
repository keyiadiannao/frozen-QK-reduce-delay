"""R108c — orientation-free operand-field amplitude diagnostic.

Round-9 audit (GPT, 2026-09-03): K16's lambda_eff is a projection on the
S-LEARNED direction, so "F body does not regrow" only means "does not
regrow THAT direction".  R107's RND arm showed a random same-scale field
also delays, so the causal object is "a strong operand-discriminating
scoring field", not a specific seed vector.  This run adds the
orientation-free readout to the SAME two released arms:

    u_h(g)     = (f_L,h(g) + f_R,h(g)) / 2          (role-mean unary score)
    A_oper(t)  = ( sum_h Var_g u_h^{(t)}(g) )^{1/2}  (centered per head)

plus role symmetry corr(f_L, f_R) and the mean self score.  Arms:
L000_S_rst (S body, Q_0, released -- K15) and L000_F_rst (F body, its
own init QK, released -- K16).  Same harness/batch chains as R108/R108b;
the va/cross trajectories must reproduce the archived runs bit-exactly
(consistency anchor), so the new readouts ride on identical dynamics.

Natural-scale reference: A_oper of the S@200 model itself (the learned
code's amplitude) is recorded once per seed.

PREREGISTERED INTERPRETATION RULE (locked before running):
  CODE-REACQ-SPLIT   F A_oper stays near its own t=200 level (max growth
                     factor < 2) while S A_oper grows (factor > 3 toward
                     the S@200 natural scale)
                     -> "body-dependent carrier reacquisition" holds at
                        the orientation-free (field) level.
  FIELD-BOTH-GROW    F A_oper grows comparably to S
                     -> a strong operand field regrows under both bodies;
                        the fate split is downstream of the field, and
                        "reacquisition of the code" must be dropped in
                        favour of "re-entry into the slow trajectory".
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REPRO = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_REPRO), 'repro_package_zp')
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
REPRO_DIR = _REPRO

import r108_plastic_qk_bridge as base          # noqa: E402  (harness reuse)
from common import states                       # noqa: E402
from common.model import D57Model, make_optimizer  # noqa: E402

task, R = base.task, base.R
N_PAIRS, LOSS, DEV = base.N_PAIRS, base.LOSS, base.DEV
ALL_A, ALL_B, ALL_Y = base.ALL_A, base.ALL_B, base.ALL_Y
A_ARR, B_ARR, S_ARR, Y_ARR = base.A_ARR, base.B_ARR, base.S_ARR, base.Y_ARR

SMOKE = bool(os.environ.get('R108C_SMOKE'))
CAP = 40 if SMOKE else 6000
SEEDS = (0,) if SMOKE else (0, 1, 2)
EVERY = 5 if SMOKE else 25
STRAT_AT = (20, 40) if SMOKE else (2000, 4000, 6000)

print(f'[R108c] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)


def model_body(seed, fam):
    """S@200 (fam='A') or F@200 (fam='C') state, verbatim."""
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, fam, seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    m.load_state_dict(dict(sd['model']))
    return m, sd


@torch.no_grad()
def oper_readout(m, Hh, dh):
    """(A_oper, role_corr, self_mean) -- orientation-free diagnostics.

    f_slot,h(g) = q_2,h . k_slot,h(g)  with CURRENT Wq/Wk/embeddings;
    u_h = role mean; A_oper = sqrt(sum_h Var_g u_h) after centering.
    """
    h2 = base.h2_of(m)
    Wq, Wk = m.Wq.weight.detach(), m.Wk.weight.detach()
    q = Wq @ h2                                   # (H*dh,)
    fs = []
    for slot in (0, 1):
        G = base.operand_h(m, slot)               # (P, d)
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            fs.append(G @ (Wk[sl, :].T @ q[sl]))
    Fv = torch.stack(fs)                          # (2H, P)
    H = Hh
    fL, fR = Fv[:H], Fv[H:]
    u = 0.5 * (fL + fR)
    u = u - u.mean(dim=1, keepdim=True)
    a_oper = float(torch.sqrt((u ** 2).mean(dim=1)).sum())
    cs = []
    for h in range(H):
        x = fL[h] - fL[h].mean()
        y = fR[h] - fR[h].mean()
        cs.append(float((x @ y) / (x.norm() * y.norm() + 1e-30)))
    self_mean = float((Fv[:H] * 0).sum())         # placeholder, set below
    # self score: q_2 . k_2 per head (mean over heads)
    k2 = Wk @ h2
    sl_scores = []
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        sl_scores.append(float(q[sl] @ k2[sl]))
    self_mean = sum(sl_scores) / Hh
    return a_oper, sum(cs) / H, self_mean


def strat_plain(m, tr_mask):
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


def run_arm(name, m0_state, sd, seed, chain, id_train, va, tr_mask,
            Hh, dh, wq_wk=None):
    """wq_wk: optional (Wq, Wk) override -- the S arm passes Q_0 here
    (K14 convention, 2nd construction); the F arm keeps its own QK."""
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    m.load_state_dict(dict(m0_state))
    if wq_wk is not None:
        wq, wk = wq_wk
        m.Wq.weight.data.copy_(wq.to(DEV))
        m.Wk.weight.data.copy_(wk.to(DEV))
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])     # native body history (rst regime)

    rec = {'name': name, 't': [], 'va': [], 'a_oper': [], 'role': [],
           'self': [], 'cross': None, 'strat': {}}
    m.eval()
    with torch.no_grad():
        a, c, sm = oper_readout(m, Hh, dh)
        rec['a_oper'].append(a)
        rec['role'].append(c)
        rec['self'].append(sm)
        rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                                va[2]).float().mean()))
    rec['t'].append(200)
    m.train()
    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a_ = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b_ = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a_, b_), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        t = 200 + step
        if step % EVERY == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                a, c, sm = oper_readout(m, Hh, dh)
                acc = float((m(va[0], va[1]).argmax(-1) ==
                             va[2]).float().mean())
            m.train()
            rec['t'].append(t)
            rec['a_oper'].append(a)
            rec['role'].append(c)
            rec['self'].append(sm)
            rec['va'].append(acc)
            if rec['cross'] is None and acc > 0.9:
                rec['cross'] = t
        if step in STRAT_AT:
            rec['strat'][t] = strat_plain(m, tr_mask)
    return rec


def main():
    t0 = time.time()
    out = {}
    chains = {s: R.batch_chain(s, CAP) for s in SEEDS}
    tr_masks = {}
    for seed in SEEDS:
        id_train, _c = chains[seed]
        tm = np.zeros(N_PAIRS, dtype=bool)
        tm[np.sort(id_train)] = True
        tr_masks[seed] = tm

    for seed in SEEDS:
        # natural-scale reference: A_oper of the S@200 model itself
        mS, _m0, _sd = base.load_models(seed)
        Hh, dh = mS.n_heads, mS.d_head
        mS.eval()
        with torch.no_grad():
            nat, nat_role, nat_self = oper_readout(mS, Hh, dh)
        out[seed] = {'natural_S200': {'a_oper': nat, 'role': nat_role,
                                      'self': nat_self}}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        # Q_0 override for the S arm (K14 convention: 2nd construction
        # after manual_seed, exactly base.load_models's m0)
        _mS, mQ0, _sd0 = base.load_models(seed)
        q0 = (mQ0.Wq.weight.detach().clone(),
              mQ0.Wk.weight.detach().clone())
        for name, fam, override in (('L000_S_rst', 'A', q0),
                                    ('L000_F_rst', 'C', None)):
            m_st, sd = model_body(seed, fam)
            rec = run_arm(name, dict(m_st.state_dict()), sd, seed, chain,
                          id_train, va, tr_masks[seed], Hh, dh,
                          wq_wk=override)
            out[seed][name] = rec
            print(f'  seed{seed} {name:10s} cross={rec["cross"]} '
                  f'va_end={rec["va"][-1]:.3f} '
                  f'A_oper: {rec["a_oper"][0]:.3f} -> {rec["a_oper"][-1]:.3f} '
                  f'(max {max(rec["a_oper"]):.3f}; S@200 natural {nat:.3f}) '
                  f'role {rec["role"][0]:.4f} -> {rec["role"][-1]:.4f} '
                  f'self {rec["self"][0]:+.2f} -> {rec["self"][-1]:+.2f}',
                  flush=True)

    print('\n=== A_oper trajectories (orientation-free) ===', flush=True)
    print('%-6s %-10s ' % ('seed', 'arm')
          + ' '.join('%7s' % t for t in (200, 1000, 2000, 4000, 6000))
          + ' %8s' % 'S200nat', flush=True)
    for seed in SEEDS:
        nat = out[seed]['natural_S200']['a_oper']
        for nm in ('L000_S_rst', 'L000_F_rst'):
            rec = out[seed][nm]
            row = []
            for tt in (200, 1000, 2000, 4000, 6000):
                i = min(range(len(rec['t'])),
                        key=lambda k: abs(rec['t'][k] - tt))
                row.append('%7.3f' % rec['a_oper'][i])
            print('%-6d %-10s ' % (seed, nm) + ' '.join(row)
                  + ' %8.3f' % nat, flush=True)
        print(flush=True)

    if not SMOKE:
        print('=== verdict (preregistered) ===', flush=True)
        s_split, both = 0, 0
        for seed in SEEDS:
            sA = out[seed]['L000_S_rst']['a_oper']
            fA = out[seed]['L000_F_rst']['a_oper']
            s_fac = max(sA) / max(sA[0], 1e-30)
            f_fac = max(fA) / max(fA[0], 1e-30)
            print(f'  seed{seed}: S growth x{s_fac:.2f}, '
                  f'F growth x{f_fac:.2f}', flush=True)
            if s_fac > 3 and f_fac < 2:
                s_split += 1
            elif f_fac > 2:
                both += 1
        print(f'  CODE-REACQ-SPLIT cells: {s_split}/3; '
              f'FIELD-BOTH-GROW cells: {both}/3', flush=True)
        if s_split >= 2:
            print('VERDICT: CODE-REACQ-SPLIT -- the S body regenerates the '
                  'operand-scoring field (orientation-free); the F body '
                  'does not.  Body-dependent carrier reacquisition holds '
                  'at the field level.', flush=True)
        elif both >= 2:
            print('VERDICT: FIELD-BOTH-GROW -- a strong operand field '
                  'regrows under both bodies; the fate split is downstream '
                  'of the field.', flush=True)
        else:
            print('VERDICT: UNDECIDED.', flush=True)

    print(f'[R108c] done in {time.time() - t0:.1f}s', flush=True)
    import pickle
    with open(os.path.join(
            _REPRO, 'r108c_oper_amplitude_results.pkl'), 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
