"""R109 — is the reacquisition a gradient-field effect?  Measure
nabla_{QK} L projected on the learned direction, S body vs F body.

Round-10 (user question, 2026-09-03): "body 是 S@200，可你这样不是有
可能是 N_S 所诱导导致适配的 QK 吗？"  Yes -- the released QK is
N_S-induced, and that is the claim.  But "induced" is so far an
interpretation; this run measures it directly.

Design
------
Same two released arms as R108b/K16 (identical harness, batch chains,
optimizer regimes):

    L000_S : body N_S@200 (bridge A), QK := Q_0 (2nd construction),
             native body Adam history, QK moments zeroed, QK trainable
    L000_F : body N_F@200 (bridge C), QK = its own init values,
             native body Adam history (QK moments already zero),
             QK trainable

At EVERY training step, before the optimizer step, read the data
gradient of the loss w.r.t. W_Q and W_K (QK is trainable, so these are
the gradients the coupled system actually follows) and project them on
the direction the native S run learned:

    dWq = Wq_S - Wq_0,   dWk = Wk_S - Wk_0     (frozen at setup)
    lam_dot(t) = (<gWq, dWq> + <gWk, dWk>) / (||dWq||^2 + ||dWk||^2)

lam_dot is the instantaneous rate at which the gradient moves the
system along the learned direction (it integrates to the realised
lambda_v).  Also recorded: ||gWq||, ||gWk|| (total gradient scale) and
the batch loss, so that "selective direction" can be separated from
"larger gradients".

PREREGISTERED DECISION RULE (locked before running)
---------------------------------------------------
Early window = steps 1..500.  Statistic = mean lam_dot over the window.

  GRAD-SELECTIVE   mean lam_dot(N_S) > 0 AND mean lam_dot(N_S) >
                   mean lam_dot(N_F) in >= 2/3 seeds
                   -> the N_S body's gradient field selectively
                      amplifies the learned direction: the backprop
                      co-adaptation story gets mechanism-level evidence.
  NO-SELECT        neither, or F >= S
                   -> the induction difference (if any) lives elsewhere
                      (e.g. in how gradients integrate over time, or in
                      higher-order interactions); the interpretation
                      stays an interpretation.

Consistency anchors: the va/cross trajectories must reproduce the
archived R108b runs (same harness, same chains).
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

SMOKE = bool(os.environ.get('R109_SMOKE'))
CAP = 40 if SMOKE else 6000
SEEDS = (0,) if SMOKE else (0, 1, 2)
EARLY = 10 if SMOKE else 500
WINDOWS = ((1, 100), (100, 500), (500, 2000), (2000, 6000)) if not SMOKE \
    else ((1, 40),)

print(f'[R109] DEV={DEV} SMOKE={SMOKE} CAP={CAP} SEEDS={SEEDS}', flush=True)


def model_body(seed, fam):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, fam, seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    m.load_state_dict(dict(sd['model']))
    return m, sd


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
                                   ALL_Y[s:s + 4096])
    if was_training:
        m.train()
    val = ~tr_mask
    nondiag_val = val & (base.A_ARR != base.B_ARR)
    partner_in = tr_mask[base.S_ARR]
    return {'train_acc': float(correct[tr_mask].mean()),
            'm1': float(correct[nondiag_val & partner_in].mean()),
            'm2': float(correct[nondiag_val & ~partner_in].mean())}


def run_arm(name, fam, seed, chain, id_train, va, tr_mask, dWq, dWk,
            dnorm2, q0_pair):
    m, sd = model_body(seed, fam)
    if fam == 'A':
        # Q_0 override (K14 convention: 2nd construction)
        m.Wq.weight.data.copy_(q0_pair[0].to(DEV))
        m.Wk.weight.data.copy_(q0_pair[1].to(DEV))
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    if fam == 'A':                     # zero the QK moments (rst regime);
        for p in (m.Wq.weight, m.Wk.weight):   # F moments already ~0
            st = opt.state.get(p)
            if st:
                if 'exp_avg' in st:
                    st['exp_avg'].zero_()
                if 'exp_avg_sq' in st:
                    st['exp_avg_sq'].zero_()

    rec = {'name': name, 'lam_dot': [], 'gnorm_q': [], 'gnorm_k': [],
           'loss': [], 'va_t': [], 'va': [], 'cross': None, 'strat': {}}
    m.eval()
    with torch.no_grad():
        rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                                va[2]).float().mean()))
    rec['va_t'].append(200)
    m.train()

    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gq = m.Wq.weight.grad
        gk = m.Wk.weight.grad
        proj = (float((gq * dWq).sum()) + float((gk * dWk).sum()))
        rec['lam_dot'].append(proj / dnorm2)
        rec['gnorm_q'].append(float(gq.norm()))
        rec['gnorm_k'].append(float(gk.norm()))
        rec['loss'].append(float(loss))
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
        if step in (2000, 4000, 6000):
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
        # learned direction: Q_S - Q_0 in parameter space (frozen here)
        mS, m0, _sdS = base.load_models(seed)
        dWq = (mS.Wq.weight.detach() - m0.Wq.weight.detach()).clone()
        dWk = (mS.Wk.weight.detach() - m0.Wk.weight.detach()).clone()
        dnorm2 = float((dWq * dWq).sum() + (dWk * dWk).sum())
        q0_pair = (m0.Wq.weight.detach().clone(),
                   m0.Wk.weight.detach().clone())
        out[seed] = {'dnorm2': dnorm2}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        for name, fam in (('L000_S', 'A'), ('L000_F', 'C')):
            rec = run_arm(name, fam, seed, chain, id_train, va,
                          tr_masks[seed], dWq, dWk, dnorm2, q0_pair)
            out[seed][name] = rec
            ld = rec['lam_dot']
            print(f'  seed{seed} {name}: cross={rec["cross"]} '
                  f'va_end={rec["va"][-1]:.3f} '
                  f'lam_dot mean={sum(ld) / len(ld):+.4f} '
                  f'early(1-{EARLY})={sum(ld[:EARLY]) / EARLY:+.4f}',
                  flush=True)

    print('\n=== lam_dot by window (mean over steps) ===', flush=True)
    print('%-6s %-8s' % ('seed', 'arm')
          + ' '.join('%12s' % f'{a}-{b}' for a, b in WINDOWS), flush=True)
    means = {}
    for seed in SEEDS:
        for nm in ('L000_S', 'L000_F'):
            ld = out[seed][nm]['lam_dot']
            row = []
            w_means = []
            for a, b in WINDOWS:
                seg = ld[a - 1:b]
                w_means.append(sum(seg) / len(seg))
                row.append('%+12.4f' % w_means[-1])
            means[(seed, nm)] = w_means
            print('%-6d %-8s' % (seed, nm) + ' '.join(row), flush=True)
        print(flush=True)

    if not SMOKE:
        print('=== verdict (preregistered) ===', flush=True)
        early_idx = [i for i, (a, b) in enumerate(WINDOWS)
                     if a <= EARLY < b]
        n_sel = 0
        for seed in SEEDS:
            s_mean = sum(ld for ld in [sum(out[seed]['L000_S']['lam_dot'][:EARLY]) / EARLY])
            f_mean = sum(out[seed]['L000_F']['lam_dot'][:EARLY]) / EARLY
            s_mean = sum(out[seed]['L000_S']['lam_dot'][:EARLY]) / EARLY
            print(f'  seed{seed}: early mean lam_dot  '
                  f'S={s_mean:+.4f}  F={f_mean:+.4f}', flush=True)
            if s_mean > 0 and s_mean > f_mean:
                n_sel += 1
        print(f'  GRAD-SELECTIVE cells (S>0 and S>F, early window): '
              f'{n_sel}/3', flush=True)
        if n_sel >= 2:
            print('VERDICT: GRAD-SELECTIVE -- the N_S body gradient field '
                  'selectively amplifies the learned direction; the '
                  'backprop co-adaptation reading has mechanism-level '
                  'evidence.', flush=True)
        else:
            print('VERDICT: NO-SELECT -- the induction difference does '
                  'not show up in the raw early gradient field; the '
                  'backprop reading stays an interpretation.', flush=True)

    print(f'[R109] done in {time.time() - t0:.1f}s', flush=True)
    import pickle
    with open(os.path.join(_REPRO, 'r109_gradient_field_results.pkl'),
              'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'early': EARLY,
                     'windows': [list(w) for w in WINDOWS], 'arms': out},
                    f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
