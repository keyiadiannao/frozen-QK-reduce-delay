"""K16 run.py — the released-family split: (N_F, Q_0) released.

Package port of repro/r108b_body_reacquisition.py (§6ej, 2026-09-03;
updated with the lambda_eff diagnostic in the round-7 audit).
Core logic is a VERBATIM copy of the archived script; only the
import shim, REPRO_DIR, SMOKE env var and the output path differ,
and the r108-harness aliases are replaced by direct common/ imports.

Round-5 audit (GPT + user challenge, 2026-09-03).  R108 showed

    (N_S, Q_0, QK released at 200)  ->  slow  (censored at 6000, 3/3)

while the archived fqk200 bridge has

    (N_F, Q_0, QK released at 200)  ->  fast  (cross ~2000)

but in a DIFFERENT harness (native batch chain, eval cadence 2000).
Before drawing any conclusion, run the F counterpart through the R108
harness so the two arms differ ONLY in the body:

    body   = F@200 (bridge_zp_C full0000200; its W_Q,W_K ARE the init
             values, because fqk200 froze the block for 200 steps)
    QK     = released at t=200 (trainable)
    optim  = native F@200 Adam state; the QK moments are already zero
             (the block received no updates while frozen), so there is
             no hybrid state and no zeroing decision to make

PREREGISTERED DECISION RULE (locked before running):
  SPLIT   the F arm crosses within 6000 in >=2/3 seeds
          (the S counterpart L000_rst is censored 3/3 in R108)
          -> the body does not carry the frozen fate, but it controls
             whether a released block can re-acquire the carrier
             ("carrier != reacquisition competence").
  NOSPLIT the F arm is censored in >=2/3 seeds
          -> any released Q_0 rebuilds the slow organization; the
             archived fqk200 fast crossing then needs a harness-lineage
             explanation and the reacquisition story dies.

Secondary (descriptive only): routing-identity drift I(dt) against the
arm's own t=200 snapshot, stratification m1/m2, and -- added in the
round-7 audit (2026-09-03) -- the score-field amplitude projection
lambda_eff(t) on the S-locked reference direction (base.build_arms aux,
current embeddings).  This separates two reacquisition targets that
must not be conflated:

  does the F body regrow an operand-scoring FIELD aligned with the
  direction the S run learned?  (lambda_eff)

  does the F body re-enter the slow trajectory?                (cross)

Four outcomes, all informative: F regrows nothing (body-dependent
carrier reacquisition confirmed); F regrows a field but stays fast
(field reacquisition is not sufficient under a different body);
F regrows and goes slow (would contradict SPLIT -- would need
harness-lineage investigation); F diverges (registered per seed).
"""
import os
import sys
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

from common import states, task            # noqa: E402
from common import gates as cgates         # noqa: E402
from common.model import D57Model, make_optimizer  # noqa: E402
from common.readouts import ZPReadouts      # noqa: E402

SMOKE = bool(os.environ.get('K16_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K16] K16_SMOKE=1 but CUDA visible -- abort')

R = ZPReadouts(DEV)
N_PAIRS = R.N_PAIRS
LOSS = nn.CrossEntropyLoss()
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
_zp = cgates.ZPIndexing()
A_ARR, B_ARR, S_ARR, Y_ARR = (_zp.A_ARR, _zp.B_ARR, _zp.S_ARR,
                              _zp.Y_ARR)
CAP = 40 if SMOKE else 6000
SEEDS = (0,) if SMOKE else (0, 1, 2)
EVERY = 5 if SMOKE else 25
CODE_EVERY = 10 if SMOKE else 50
STRAT_AT = (20, 40) if SMOKE else (2000, 4000, 6000)

print(f'[K16] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)


def model_F(seed):
    """The F@200 state, verbatim from the bridge (QK = its init values)."""
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'C', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    m.load_state_dict(dict(sd['model']))
    # sanity: the frozen block really is at its init values
    torch.manual_seed(seed)
    m_init = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    dq = float((m.Wq.weight - m_init.Wq.weight).abs().max())
    dk = float((m.Wk.weight - m_init.Wk.weight).abs().max())
    assert max(dq, dk) < 1e-8, f'F@200 QK not at init: {dq:.2e}/{dk:.2e}'
    return m, sd, max(dq, dk)


def _s_ref(seed):
    """The S-locked reference field (ref, dv, Wq0, Wk0, Hh, dh),
    built exactly as base.build_arms does -- via the same S@200
    checkpoint and init-QK semantics."""
    mS, m0, sd = _load_S(seed)
    Hh, dh = mS.n_heads, mS.d_head
    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()
    with torch.no_grad():
        h2 = _h2_of(mS)
        qhat0 = Wq0 @ h2
        q2S = WqS @ h2
        v0 = [Wk0[h * dh:(h + 1) * dh, :].T @ qhat0[h * dh:(h + 1) * dh]
              for h in range(Hh)]
        vS = [WkS[h * dh:(h + 1) * dh, :].T @ q2S[h * dh:(h + 1) * dh]
              for h in range(Hh)]
        dv = [vS[h] - v0[h] for h in range(Hh)]
        refs = []
        for slot in (0, 1):
            G = _operand_h(mS, slot)
            for h in range(Hh):
                refs.append(G @ dv[h])
    return torch.stack(refs), dv, Wq0, Wk0, Hh, dh


@torch.no_grad()
def _h2_of(m):
    eq = torch.tensor([task.N_EQ], device=DEV)
    return m.ln1(m.emb(eq) + m.pos[2])[0]


@torch.no_grad()
def _operand_h(m, slot):
    g = torch.arange(R.P, device=DEV)
    return m.ln1(m.emb(g) + m.pos[slot])


def _load_S(seed):
    torch.manual_seed(seed)
    mS = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    mS.load_state_dict(dict(sd['model']))
    m0 = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    return mS, m0, sd


@torch.no_grad()
def _ds_field(m, Wq0, Wk0, Hh, dh):
    """Per-(head,slot) score-difference field on the CURRENT
    embeddings (verbatim semantics of r108.ds_field)."""
    h2 = _h2_of(m)
    Wq, Wk = m.Wq.weight.detach(), m.Wk.weight.detach()
    q, q0 = Wq @ h2, Wq0 @ h2
    outs = []
    for slot in (0, 1):
        G = _operand_h(m, slot)
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            outs.append(G @ (Wk[sl, :].T @ q[sl]
                             - Wk0[sl, :].T @ q0[sl]))
    return torch.stack(outs)


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


def run_arm(seed, chain, id_train, va, tr_mask, aux):
    ref, _dv, Wq0, Wk0, Hh, dh = aux
    m, sd, gate = model_F(seed)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])       # native F history; QK moments ~0

    rec = {'gate': gate, 't': [], 'va': [], 't_I': [], 'I': [],
           'lam': [], 'cross': None, 'strat': {}}
    m.eval()
    with torch.no_grad():
        code0 = R.own_codes(m)
        ds0 = _ds_field(m, Wq0, Wk0, Hh, dh)
        rec['lam'].append(float((ds0 * ref).sum() /
                                (ref * ref).sum()))
        rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                                va[2]).float().mean()))
    rec['t'].append(200)
    rec['t_I'].append(200)
    rec['I'].append(R.retrieve_self(code0, code0))
    m.train()

    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()                        # QK TRAINABLE -- the whole point
        t = 200 + step
        if step % EVERY == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                ds = _ds_field(m, Wq0, Wk0, Hh, dh)
                rec['lam'].append(float((ds * ref).sum() /
                                        (ref * ref).sum()))
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
    out = {}
    chains = {s: R.batch_chain(s, CAP) for s in SEEDS}
    tr_masks = {}
    for seed in SEEDS:
        id_train, _c = chains[seed]
        tm = np.zeros(N_PAIRS, dtype=bool)
        tm[np.sort(id_train)] = True
        tr_masks[seed] = tm
    for seed in SEEDS:
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        aux = _s_ref(seed)
        out[seed] = run_arm(seed, chain, id_train, va, tr_masks[seed],
                            aux)
        r = out[seed]
        ts = sorted(r['strat'])
        last = r['strat'][ts[-1]] if ts else {}
        lam = r['lam']
        print(f'  seed{seed} (N_F,Q_0 released): gate={r["gate"]:.1e} '
              f'cross={r["cross"]} va_end={r["va"][-1]:.3f} '
              f'I_end={r["I"][-1]:.3f} '
              f'm1={last.get("m1", float("nan")):.3f} '
              f'm2={last.get("m2", float("nan")):.3f} '
              f'lam0={lam[0]:+.3f} lamT={lam[-1]:+.3f} '
              f'lam_mean={sum(lam) / len(lam):+.3f}', flush=True)

    print('\n=== verdict (preregistered) ===', flush=True)
    crosses = [out[s]['cross'] for s in SEEDS]
    n_cross = sum(1 for c in crosses if c is not None)
    print('  F-arm crossings:', ['cens' if c is None else c
                                  for c in crosses], flush=True)
    print('  S counterpart (R108 L000_rst): censored 3/3', flush=True)
    if n_cross >= 2:
        print('VERDICT: SPLIT -- the body does not carry the frozen fate, '
              'but it controls whether a released block re-acquires the '
              'carrier.', flush=True)
    elif len(crosses) - n_cross >= 2:
        print('VERDICT: NOSPLIT -- any released Q_0 rebuilds the slow '
              'organization; the archived fqk200 fast crossing needs a '
              'harness-lineage explanation.', flush=True)
    else:
        print('VERDICT: UNDECIDED.', flush=True)

    print(f'[K16] done in {time.time() - t0:.1f}s', flush=True)
    import pickle
    with open(os.path.join(_CLAIM, 'results.pkl'), 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
