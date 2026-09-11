"""R110 — coarse body localization: where does the re-entry competence
live in the non-QK state?

Round-11 (GPT + user, 2026-09-03).  Known so far:
  * K15/K16: with QK released, the S body re-enters the slow trajectory
    while the F body does not (2/3, 1 divergent);
  * R108c: BOTH bodies regrow an operand field (F reaches ~1/4 of the
    natural scale) -- field amplitude is not the controlling variable;
  * R109: the raw early gradient field shows no directional bias
    (NO-SELECT) -- the co-adaptation is not a checkpoint-local
    gradient arrow.

So the open question is the original one: what did the first 200 steps
of QK learning leave in the non-QK WEIGHTS that makes the released
system re-enter the slow trajectory?  R110 localizes it coarsely.

Functional groups (D57Model parameter names):
  R (representation):  emb.weight, pos, ln1.weight, ln1.bias
  V (value/output):    Wv.weight, Wo.weight
  D (downstream):      ln2.*, mlp1.*, mlp2.*, ln_f.*, Wu.weight

Arms -- ALL with Q_0 (2nd construction), QK released (trainable),
FRESH AdamW (no moment history anywhere; moments already known to be
unnecessary), same batch chain, CAP 6000:

  baselines   B_S = (N_S, Q_0)          expect: cens 3/3 (R108 L000_fr)
              B_F = (N_F, Q_0)          new control
  transplants F-base + S block:  N_F+R_S, N_F+V_S, N_F+D_S
              S-base + F block:  N_S+R_F, N_S+V_F, N_S+D_F

Primary readout: crossing + m1/m2.  Secondary: A_oper(t) (orientation-
free field amplitude; NOT a mechanism criterion), role corr.

PREREGISTERED DECISION RULE (locked before running)
---------------------------------------------------
For each block X, against the same-side fresh baselines:
  LOCALIZES-X      (N_F+X_S) censored in >=2/3 seeds AND
                   (N_S+X_F) crosses in >=2/3 seeds
                   -> re-entry competence largely localizes to X.
  ONE-DIRECTIONAL  only one of the two directions shows the flip in
                   >=2/3 seeds
                   -> X contributes; competence not a clean carrier.
  DISTRIBUTED      no single block flips in either direction
                   -> competence is distributed; STOP (no pair search
                      beyond at most one preregistered second stage).

Consistency anchor: B_S must reproduce R108's L000_fr (cens 3/3,
va_end 0.348/0.407/0.314) -- same construction, same chain.
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

SMOKE = bool(os.environ.get('R110_SMOKE'))
CAP = 40 if SMOKE else 6000
SEEDS = (0,) if SMOKE else (0, 1, 2)
EVERY = 5 if SMOKE else 25
STRAT_AT = (20, 40) if SMOKE else (2000, 4000, 6000)

GROUPS = {
    'R': ['emb.weight', 'pos', 'ln1.weight', 'ln1.bias'],
    'V': ['Wv.weight', 'Wo.weight'],
    'D': ['ln2.weight', 'ln2.bias', 'mlp1.weight', 'mlp1.bias',
          'mlp2.weight', 'mlp2.bias', 'ln_f.weight', 'ln_f.bias',
          'Wu.weight'],
}

print(f'[R110] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)


def model_body(seed, fam):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, fam, seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    m.load_state_dict(dict(sd['model']))
    return m, sd


@torch.no_grad()
def oper_readout(m, Hh, dh):
    """(A_oper, role_corr, self_mean) -- orientation-free diagnostics
    (verbatim semantics of r108c.oper_readout)."""
    h2 = base.h2_of(m)
    Wq, Wk = m.Wq.weight.detach(), m.Wk.weight.detach()
    q = Wq @ h2
    fs = []
    for slot in (0, 1):
        G = base.operand_h(m, slot)
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            fs.append(G @ (Wk[sl, :].T @ q[sl]))
    Fv = torch.stack(fs)
    fL, fR = Fv[:Hh], Fv[Hh:]
    u = 0.5 * (fL + fR)
    u = u - u.mean(dim=1, keepdim=True)
    a_oper = float(torch.sqrt((u ** 2).mean(dim=1)).sum())
    cs = []
    for h in range(Hh):
        x = fL[h] - fL[h].mean()
        y = fR[h] - fR[h].mean()
        cs.append(float((x @ y) / (x.norm() * y.norm() + 1e-30)))
    k2 = Wk @ h2
    sm = sum(float(q[h * dh:(h + 1) * dh] @ k2[h * dh:(h + 1) * dh])
             for h in range(Hh)) / Hh
    return a_oper, sum(cs) / Hh, sm


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


def run_arm(name, base_state, wq0, wk0, seed, chain, id_train, va,
            tr_mask, Hh, dh):
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    st = dict(base_state)
    st['Wq.weight'] = wq0.clone().to(DEV)
    st['Wk.weight'] = wk0.clone().to(DEV)
    m.load_state_dict(st)
    opt, _, _ = make_optimizer(m, 'wd_0011')      # FRESH -- no history

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
        opt.step()                                 # QK trainable
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
    out, gates_ok = {}, True
    chains = {s: R.batch_chain(s, CAP) for s in SEEDS}
    tr_masks = {}
    for seed in SEEDS:
        id_train, _c = chains[seed]
        tm = np.zeros(N_PAIRS, dtype=bool)
        tm[np.sort(id_train)] = True
        tr_masks[seed] = tm

    for seed in SEEDS:
        mS, m0, _sdS = base.load_models(seed)      # m0 = Q_0 (2nd constr)
        wq0 = m0.Wq.weight.detach().clone()
        wk0 = m0.Wk.weight.detach().clone()
        mF, _sdF = model_body(seed, 'C')
        Hh, dh = mS.n_heads, mS.d_head
        mS.eval()
        with torch.no_grad():
            nat, _r, _s = oper_readout(mS, Hh, dh)
        out[seed] = {'natural_S200_aoper': nat}

        states = {'S': dict(mS.state_dict()), 'F': dict(mF.state_dict())}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]

        arms = [('B_S', 'S', None), ('B_F', 'F', None)]
        for blk in ('R', 'V', 'D'):
            arms.append((f'F+{blk}_S', 'F', ('S', blk)))
            arms.append((f'S+{blk}_F', 'S', ('F', blk)))

        for name, base_fam, tx in arms:
            st = dict(states[base_fam])
            if tx is not None:
                donor_fam, blk = tx
                for k in GROUPS[blk]:
                    st[k] = states[donor_fam][k].clone()
            # sanity: transplanted blocks really equal the donor
            if tx is not None:
                donor_fam, blk = tx
                worst = max(float((st[k] - states[donor_fam][k]).abs().max())
                            for k in GROUPS[blk])
                base_worst = max(
                    float((st[k] - states[base_fam][k]).abs().max())
                    for k in GROUPS[blk] if k in st)
                gates_ok &= (worst == 0.0)
            rec = run_arm(name, st, wq0, wk0, seed, chain, id_train, va,
                          tr_masks[seed], Hh, dh)
            out[seed][name] = rec
            print(f'  seed{seed} {name:8s} cross={str(rec["cross"]):5s} '
                  f'va_end={rec["va"][-1]:.3f} '
                  f'A_oper {rec["a_oper"][0]:7.2f} -> max '
                  f'{max(rec["a_oper"]):7.2f} (nat {nat:7.2f})',
                  flush=True)

    ARMS = ('B_S', 'B_F', 'F+R_S', 'F+V_S', 'F+D_S',
            'S+R_F', 'S+V_F', 'S+D_F')
    print('\n=== crossing / m2(T) table ===', flush=True)
    print('%-6s %-8s %7s %7s %7s %9s' %
          ('seed', 'arm', 'cross', '', '', 'm2(T)'), flush=True)
    for seed in SEEDS:
        for nm in ARMS:
            rec = out[seed][nm]
            ts = sorted(rec['strat'])
            m2 = rec['strat'][ts[-1]]['m2'] if ts else float('nan')
            print('%-6d %-8s %7s %9.3f' % (
                seed, nm,
                'cens' if rec['cross'] is None else rec['cross'], m2),
                flush=True)
        print(flush=True)

    if not SMOKE:
        print('=== verdict (preregistered) ===', flush=True)
        for blk in ('R', 'V', 'D'):
            f_cens = sum(1 for s in SEEDS
                         if out[s][f'F+{blk}_S']['cross'] is None)
            s_cross = sum(1 for s in SEEDS
                          if out[s][f'S+{blk}_F']['cross'] is not None)
            b_f_cross = sum(1 for s in SEEDS
                            if out[s]['B_F']['cross'] is not None)
            print(f'  {blk}: F+{blk}_S cens {f_cens}/3 '
                  f'(B_F crossed {b_f_cross}/3); '
                  f'S+{blk}_F crossed {s_cross}/3 (B_S cens 3/3)',
                  flush=True)
            if f_cens >= 2 and s_cross >= 2:
                print(f'    -> LOCALIZES to {blk} (both directions flip)',
                      flush=True)
            elif f_cens >= 2 or s_cross >= 2:
                print(f'    -> ONE-DIRECTIONAL contribution from {blk}',
                      flush=True)
            else:
                print(f'    -> {blk} does not flip either direction',
                      flush=True)

    print(f'[R110] done in {time.time() - t0:.1f}s', flush=True)
    import pickle
    with open(os.path.join(
            _REPRO, 'r110_body_localization_results.pkl'), 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'arms': out,
                     'groups': GROUPS}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
