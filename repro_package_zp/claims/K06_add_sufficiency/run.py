"""K06 run.py — native add/remove of the K04 row-2 scoring functional.

Package port of repro/r104_native_addremove.py (M2.5, 2026-09-02).
Verbatim except: shared readouts from common/readouts.py, common/
imports, claim-dir output, SMOKE env R104_SMOKE -> K06_SMOKE.

Math (verbatim docstring): with biases, the row-2 query is q2 = Wq h2 +
bq, and the term q2.bk is a per-row constant that softmax cancels -- so
the row-2 ATTENTION depends only on v = Wk^T q2 (per head, d-dim).
  ADD    : Wq=Wq_0, Wk' = Wk_0 + (v_S - v_0) q2_0^T / |q2_0|^2
           -> row-2 attention exactly equals SS.
  REMOVE : Wq=Wq_S, Wk''= Wk_S - (Wk_S^T q2_S - v_0) q2_S^T/|q2_S|^2
           -> row-2 attention exactly equals 00.
Gates: |ADD-SS|<=1e-5 and |REMOVE-00|<=1e-5 on the val set; QK bitwise
frozen through training.  Verdict (locked §6dz): SUFFICIENT-only in the
archive (ADD SS-like 3/3, ADD ~= SS pointwise; REMOVE overshoots below
the 00 baseline -- necessity confounded, registered §6ea).

RNG-SEMANTICS WARNING: Q0 arm weights come from the SECOND D57Model
construction after torch.manual_seed(seed) (load_models line order) --
call-order-dependent, do not "fix".

Claim (ledger K6): adding the functional to a baseline rebuilds the
identity-maintaining dynamics (ADD ~= SS pointwise, I(500) 0.961-0.969).
"""
import os
import sys
import pickle
import time

import numpy as np
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

SMOKE = bool(os.environ.get('K06_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K06] K06_SMOKE=1 but CUDA visible -- abort')
print(f'[K06] DEV={DEV} SMOKE={SMOKE}', flush=True)

STEPS = 8 if SMOKE else 500
EVAL_DL = (0, 4, 8) if SMOKE else (0, 25, 100, 250, 500)
SEEDS = (0,) if SMOKE else (0, 1, 2)
LOSS = nn.CrossEntropyLoss()

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
    return m


def row2_probs(m, a, b):
    with torch.no_grad():
        _, attn = R.plain_forward_attn(m, a, b)
    return attn[:, :, 2, :].cpu()          # (B, H, 3)


def build_arms(seed):
    """Returns dict name -> (Wq_w, Wk_w) weight tensors (cpu).
    Bias convention (R97/R103 lineage): ALL arms keep the S-ckpt
    biases; 'Q_0' = init weights only. Functionals therefore use the
    queries the arms will actually have:
      qhat_0 = Wq_0 h2 + bq_S   (00/ADD query)
      q2_S   = Wq_S h2 + bq_S   (SS/REMOVE query)
      v_0    = Wk_0^T qhat_0    v_S = Wk_S^T q2_S
    NOTE: Wq/Wk have bias=None in this architecture -- the score is
    exactly bilinear, GPT's original q = Wq^T h2 form is exact (the
    bias concern was unfounded; registered in the r104 source).
    """
    mS, m0, sd = load_models(seed)
    with torch.no_grad():
        tok = torch.tensor([[0, 0, task.N_EQ]], device=DEV)
        x = mS.emb(tok) + mS.pos[None, :, :]
        h2 = mS.ln1(x)[0, 2, :]                     # (d,)
        qhat0 = m0.Wq.weight.detach() @ h2          # (H*dh,)
        q2S = mS.Wq.weight.detach() @ h2
    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()
    Hh, dh = mS.n_heads, mS.d_head
    Wk_add = Wk0.clone()
    Wk_rem = WkS.clone()
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        Wk0_h, WkS_h = Wk0[sl, :], WkS[sl, :]       # (dh, d)
        q0_h, qS_h = qhat0[sl], q2S[sl]             # (dh,)
        v0_h = Wk0_h.T @ q0_h                       # (d,)
        vS_h = WkS_h.T @ qS_h
        dW_add = q0_h.outer(vS_h - v0_h) / float(q0_h @ q0_h)
        dW_rem = qS_h.outer(vS_h - v0_h) / float(qS_h @ qS_h)
        Wk_add[sl, :] += dW_add
        Wk_rem[sl, :] -= dW_rem
    arms = {
        '00': (Wq0, Wk0),
        'SS': (WqS, WkS),
        'ADD': (Wq0, Wk_add),
        'REMOVE': (WqS, Wk_rem),
    }
    # gate: row-2 attention probs on the val set
    va = R.val_probe(seed)
    probs = {}
    for k, (wq, wk) in arms.items():
        mtmp = model_with(seed, wq, wk)
        probs[k] = row2_probs(mtmp, va[0], va[1])
    d_add = float((probs['ADD'] - probs['SS']).abs().max())
    d_rem = float((probs['REMOVE'] - probs['00']).abs().max())
    gate = {'d_add': d_add, 'd_rem': d_rem,
            'ok': d_add <= 1e-5 and d_rem <= 1e-5}
    print(f'gate seed{seed}: |ADD-SS|={d_add:.2e} '
          f'|REMOVE-00|={d_rem:.2e}', flush=True)
    return arms, gate


def run_arm(name, wq_w, wk_w, seed, chain, id_train, va):
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    body = dict(sd['model'])
    body['Wq.weight'] = wq_w.clone()
    body['Wk.weight'] = wk_w.clone()
    m.load_state_dict(body)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    wq_ref = m.Wq.weight.detach().clone()
    wk_ref = m.Wk.weight.detach().clone()
    rec = {'t': [200], 'I': [], 'rho_m': [], 'dm': [], 'D': [],
           'va': [], 'va_t': [200], 'loss': []}
    c_ref = R.own_codes(m)
    mm0 = R.margin_self(c_ref, c_ref)
    rec['I'].append(1.0)
    rec['m_med'], rec['m_mean'] = [mm0[0]], [mm0[1]]
    rec['rho_m'].append(1.0)
    rec['dm'].append(0.0)
    rec['D'].append(R.retrieve_self(c_ref, c_ref))
    with torch.no_grad():
        rec['va'].append(float((m(va[0], va[1]).argmax(-1) ==
                                va[2]).float().mean()))
    for step in range(1, STEPS + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None                 # QK frozen
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
    frozen_ok = bool(torch.equal(m.Wq.weight.detach(), wq_ref) and
                     torch.equal(m.Wk.weight.detach(), wk_ref))
    rec['frozen_ok'] = frozen_ok
    return rec


def main():
    out = {}
    gates = {}
    ok_all = True
    t0 = time.time()
    chains = {seed: R.batch_chain(seed, STEPS) for seed in SEEDS}
    for seed in SEEDS:
        arms, gate = build_arms(seed)
        gates[seed] = gate
        ok_all &= gate['ok']
        out[seed] = {}
        va = R.val_probe(seed)
        id_train, chain = chains[seed]
        for name, (wq, wk) in arms.items():
            rec = run_arm(name, wq, wk, seed, chain, id_train, va)
            out[seed][name] = rec
            ok_all &= rec['frozen_ok']
            print(f'seed{seed} {name}: '
                  f'I={["%.3f" % x for x in rec["I"]]} '
                  f'rho_m={["%.3f" % x for x in rec["rho_m"]]} '
                  f'va_last={rec["va"][-1]:.3f} '
                  f'frozen={rec["frozen_ok"]}', flush=True)

    print('\n=== classification & verdict (locked §6dz) ===', flush=True)
    if not SMOKE:
        cls = {}
        for seed in SEEDS:
            iSS = out[seed]['SS']['I'][-1]
            i00 = out[seed]['00']['I'][-1]
            row = {}
            for nm in ('ADD', 'REMOVE'):
                iv = out[seed][nm]['I'][-1]
                if abs(iv - iSS) <= 0.05:
                    row[nm] = 'SS-like'
                elif abs(iv - i00) <= 0.05:
                    row[nm] = '00-like'
                else:
                    row[nm] = 'mid'
            row['dI_ADD'] = out[seed]['ADD']['I'][-1] - i00
            row['dI_REM'] = out[seed]['REMOVE']['I'][-1] - i00
            cls[seed] = row
            print(f'  seed{seed}: {row}', flush=True)
        add_ss = sum(cls[s]['ADD'] == 'SS-like' for s in SEEDS)
        add_00 = sum(cls[s]['ADD'] == '00-like' for s in SEEDS)
        rem_00 = sum(cls[s]['REMOVE'] == '00-like' for s in SEEDS)
        rem_ss = sum(cls[s]['REMOVE'] == 'SS-like' for s in SEEDS)
        if add_ss >= 2 and rem_00 >= 2:
            print('VERDICT: SUFFICIENT+NECESSARY -- the R102 row-2 '
                  'functional is the load-bearing carrier content',
                  flush=True)
        elif add_ss >= 2:
            print('VERDICT: SUFFICIENT-only -- adding the functional '
                  'reproduces maintenance; removal does not abolish',
                  flush=True)
        elif rem_00 >= 2:
            print('VERDICT: NECESSARY-only', flush=True)
        elif add_00 >= 2 and rem_ss >= 2:
            print('VERDICT: PARADOX -- functional adds in but its '
                  'removal does not hurt (overdetermination)',
                  flush=True)
        else:
            print('VERDICT: MIXED -- register per cell', flush=True)
    print(f'HARNESS: {"PASS" if ok_all else "FAIL"}', flush=True)
    print(f'[K06] done in {time.time() - t0:.1f}s', flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'steps': STEPS,
                     'eval_dl': EVAL_DL, 'gates': gates, 'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
