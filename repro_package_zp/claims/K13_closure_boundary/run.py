"""K13 run.py — frozen-carrier closure: does the K04 row-2 functional
(ADD arm) reconstruct the mature restorative slow regime?

Package port of repro/r106_closure_regime.py (M2.5, 2026-09-02).
Verbatim except: shared machinery from common/ (readouts/indexing/
sampler), bridge paths via common.states, claim-dir output, SMOKE env
R106_SMOKE -> K13_SMOKE.

Structure: arms 00/SS/ADD (K06 native construction), QK frozen
throughout, body = N_S@200, matched batch chain (SeedSequence
[9000+seed]), CAP=4000.  Battery: va every 25; plain strat m1/m2/m3 at
t in {1000,2000,3000,4000} (R72 semantics); own-codes at the shared
eval grid.  P50 restorativity branches fork at t_pulse=1000 (DEEP-copied
opt state -- the live-reference smoke bug is documented inline in the
source and preserved here), 50 steps of resampWS-eq in-graph surgery
(stream [9500+seed, step]), then plain to 4000; R_ctrl vs main-arm
control codes; disruption gate <=0.85 @1050, restoration = R_ctrl>=0.9
for 3 consecutive evals.

Claim (ledger K13): ADD rebuilds the structured-lookup slow regime
(m1~0.98-0.99, no crossing to 4k); restorativity component
non-diagnostic under frozen QK (assay boundary, not ADD failure).
Preregistered §6ef; verdict §6eg.
"""
import math
import os
import sys
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPRO_DIR = os.environ.get('REPRO_DIR', os.path.join(
    os.path.dirname(_PKG_ROOT), 'repro'))

from common import task, states
from common import gates as cgates
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)

SMOKE = bool(os.environ.get('K13_SMOKE'))
DEV = ('cuda' if torch.cuda.device_count() > 0 else 'cpu')
if SMOKE and DEV == 'cuda':
    raise SystemExit('[K13] K13_SMOKE=1 but CUDA visible -- abort')
print(f'[K13] DEV={DEV} SMOKE={SMOKE}', flush=True)

BATCH = 512
CAP = 60 if SMOKE else 4000           # training steps
T_PULSE = 230 if SMOKE else 1000     # absolute t
PLEN = 10 if SMOKE else 50
EVAL_T = (([200, 205, 210, 220, 230, 235, 240, 245, 250, 255,
            260]) if SMOKE
          else sorted({0, 25, 100, 250, 500, 700, 1000, 1050, 1075,
                       1100, 1150, 1200, 1500, 2000, 3000, 4000}))
EVAL_T = sorted(t for t in EVAL_T if t <= 200 + CAP)
STRAT_T = ([230, 260] if SMOKE else [1000, 2000, 3000, 4000])
SEEDS = (0,) if SMOKE else (0, 1, 2)
LOSS = nn.CrossEntropyLoss()

torch.set_num_threads(4)

R = ZPReadouts(DEV)
zp = cgates.ZPIndexing()
P, N_PAIRS, N_TRAIN = R.P, R.N_PAIRS, R.N_TRAIN
ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
DIAG, ORB_REPS, Y_ARR, S_ARR = zp.DIAG, zp.ORB_REPS, zp.Y_ARR, zp.S_ARR
A_ARR, B_ARR = zp.A_ARR, zp.B_ARR


def surgery_forward(m, a, b, src_a, src_b):
    B = a.shape[0]
    H, dh = m.n_heads, m.d_head
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    h = m.ln1(x)
    q = m.Wq(h).view(B, 3, H, dh).transpose(1, 2)
    k = m.Wk(h).view(B, 3, H, dh).transpose(1, 2)
    v = m.Wv(h).view(B, 3, H, dh).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    attn_main = scores.softmax(dim=-1)
    eq_s = torch.full_like(src_a, task.N_EQ)
    tok_s = torch.stack([src_a, src_b, eq_s], dim=1)
    x_s = m.emb(tok_s) + m.pos[None, :, :]
    h_s = m.ln1(x_s)
    q_s = m.Wq(h_s).view(B, 3, H, dh).transpose(1, 2)
    k_s = m.Wk(h_s).view(B, 3, H, dh).transpose(1, 2)
    scores_s = (q_s @ k_s.transpose(-1, -2)) / math.sqrt(dh)
    a2_src = scores_s.softmax(dim=-1)[:, :, 2, :]
    attn = torch.cat([attn_main[:, :, :2, :], a2_src.unsqueeze(2)],
                     dim=2)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, 3, m.d_model)
    x = x + m.Wo(out)
    h2 = m.ln2(x)
    mm = m.mlp2(F.gelu(m.mlp1(h2)))
    x = x + mm
    last = x[:, 2, :]
    return m.Wu(m.ln_f(last))


def classify_train(tr_mask):
    """r106 classify_train (U = diag_T | half-singletons)."""
    full = {c: [] for c in range(P)}
    half = {c: [] for c in range(P)}
    diag_T = DIAG[tr_mask[DIAG]]
    for rep in ORB_REPS:
        i, j = int(rep), int(S_ARR[rep])
        mi, mj = bool(tr_mask[i]), bool(tr_mask[j])
        c = int(Y_ARR[rep])
        if mi and mj:
            full[c].append(rep)
        elif mi:
            half[c].append(i)
        elif mj:
            half[c].append(j)
    U = set(int(x) for x in diag_T)
    for c in range(P):
        if len(half[c]) == 1:
            U.add(half[c][0])
    return full, half, U


def sample_pi(seed, step, full, half):
    """r106 per-step stream (SeedSequence [9500+seed, step]) -- distinct
    from the K09 fixed/resamp family; verbatim."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [9500 + seed, step]))
    IDX = zp.IDX
    pi = IDX.copy()
    for c in range(P):
        fo = full[c]
        if len(fo) >= 2:
            k = len(fo)
            permk = rng.permutation(k)
            tgt = np.roll(permk, 1)
            orient = rng.integers(0, 2, size=k)
            for mm in range(k):
                i = int(fo[permk[mm]])
                j = int(fo[tgt[mm]])
                if orient[mm] == 0:
                    pi[i], pi[S_ARR[i]] = j, S_ARR[j]
                else:
                    pi[i], pi[S_ARR[i]] = S_ARR[j], j
        elif len(fo) == 1:
            i = int(fo[0])
            pi[i], pi[S_ARR[i]] = int(S_ARR[i]), i
        hh = half[c]
        if len(hh) >= 2:
            kk = len(hh)
            permh = rng.permutation(kk)
            pi[np.asarray(hh)[permh]] = np.asarray(hh)[
                np.roll(permh, 1)]
    return pi


def samp_checks(pi, tr, U):
    pt = pi[tr]
    v1 = bool(np.unique(pt).size == N_TRAIN)
    v2 = bool(np.all(Y_ARR[pt] == Y_ARR[tr]))
    fp = set(int(x) for x in tr[pt == tr])
    v3 = (fp == U)
    in_T = np.zeros(N_PAIRS, dtype=bool)
    in_T[tr] = True
    mk = in_T[S_ARR[tr]]
    v4 = bool(np.all(pi[S_ARR[tr[mk]]] == S_ARR[pi[tr[mk]]]))
    return v1 and v2 and v3 and v4


@torch.no_grad()
def strat_plain(m, tr_mask):
    """R72-semantics buckets on the full grid (own forward)."""
    correct = np.zeros(N_PAIRS, dtype=bool)
    for s in range(0, N_PAIRS, 4096):
        a = torch.from_numpy(ALL_A[s:s + 4096]).long().to(DEV)
        b = torch.from_numpy(ALL_B[s:s + 4096]).long().to(DEV)
        lg = m(a, b)
        correct[s:s + 4096] = (lg.argmax(-1).cpu().numpy() ==
                               ALL_Y[s:s + 4096])
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
    torch.manual_seed(seed)
    mS = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(states.bridge_ckpt_path(
        REPRO_DIR, 'A', seed, None, 'full0000200'),
        map_location='cpu', weights_only=False)
    mS.load_state_dict(dict(sd['model']))
    m0 = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    with torch.no_grad():
        tok = torch.tensor([[0, 0, task.N_EQ]], device=DEV)
        x = mS.emb(tok) + mS.pos[None, :, :]
        h2 = mS.ln1(x)[0, 2, :]
        qhat0 = m0.Wq.weight.detach() @ h2
        q2S = mS.Wq.weight.detach() @ h2
    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()
    Hh, dh = mS.n_heads, mS.d_head
    Wk_add = Wk0.clone()
    Wk_rem = WkS.clone()          # built for the gate only (unused)
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        Wk0_h, WkS_h = Wk0[sl, :], WkS[sl, :]
        q0_h, qS_h = qhat0[sl], q2S[sl]
        v0_h = Wk0_h.T @ q0_h
        vS_h = WkS_h.T @ qS_h
        Wk_add[sl, :] += q0_h.outer(vS_h - v0_h) / float(q0_h @ q0_h)
        Wk_rem[sl, :] -= qS_h.outer(vS_h - v0_h) / float(qS_h @ qS_h)
    arms = {'00': (Wq0, Wk0), 'SS': (WqS, WkS), 'ADD': (Wq0, Wk_add)}
    va = R.val_probe(seed)
    probs = {}
    for k, (wq, wk) in arms.items():
        mtmp, _ = model_with(seed, wq, wk)
        with torch.no_grad():
            _, attn = R.plain_forward_attn(mtmp, va[0], va[1])
        probs[k] = attn[:, :, 2, :].cpu()
    d_add = float((probs['ADD'] - probs['SS']).abs().max())
    gate = {'d_add': d_add, 'ok': d_add <= 1e-5}
    print(f'gate seed{seed}: |ADD-SS| row2={d_add:.2e}', flush=True)
    return arms, gate


def run_main(name, wq, wk, seed, chain, id_train, va, tr_mask):
    m, sd = model_with(seed, wq, wk)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    wq_ref = m.Wq.weight.detach().clone()
    wk_ref = m.Wk.weight.detach().clone()
    rec = {'codes': {}, 'va_t': [], 'va': [], 'strat': {},
           'cross': None, 'fork': None}
    c_ref = R.own_codes(m)
    rec['codes'][0] = c_ref
    rec['m_mean0'] = R.margin_self(c_ref, c_ref)[1]
    for step in range(1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        loss = LOSS(m(a, b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None
        m.Wk.weight.grad = None
        opt.step()
        t = 200 + step
        if t in EVAL_T:
            rec['codes'][t] = R.own_codes(m)
        if step % 25 == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                v = float((m(va[0], va[1]).argmax(-1) ==
                           va[2]).float().mean())
            m.train()
            rec['va_t'].append(t)
            rec['va'].append(v)
            if v > 0.9 and rec['cross'] is None:
                rec['cross'] = t
        if t in STRAT_T:
            m.eval()
            rec['strat'][t] = strat_plain(m, tr_mask)
            m.train()
        if t == T_PULSE:
            # DEEP-copy the optimizer state: opt.state_dict() returns
            # LIVE tensor references -- without cloning, the branch
            # would load the main arm's FINAL moments (R106 smoke bug)
            opt_sd = opt.state_dict()
            opt_copy = {'state': {k: {kk: (vv.clone() if
                                           torch.is_tensor(vv) else vv)
                                      for kk, vv in v.items()}
                                  for k, v in opt_sd['state'].items()},
                        'param_groups': opt_sd['param_groups']}
            rec['fork'] = (
                {k: v.detach().cpu().clone()
                 for k, v in m.state_dict().items()},
                opt_copy)
    frozen_ok = bool(torch.equal(m.Wq.weight.detach(), wq_ref) and
                     torch.equal(m.Wk.weight.detach(), wk_ref))
    rec['frozen_ok'] = frozen_ok
    return rec


def run_branch(name, fork, seed, chain, id_train, va, tr_mask):
    full, half, U = TRINFO[seed]['full'], TRINFO[seed]['half'], \
        TRINFO[seed]['U']
    m, sd = model_with(seed, *fork['weights'])
    m.load_state_dict({k: v.to(DEV) for k, v in
                       fork['state'].items()})
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(fork['opt'])
    wq_ref = m.Wq.weight.detach().clone()
    wk_ref = m.Wk.weight.detach().clone()
    rec = {'codes': {}, 'va_t': [], 'va': [], 'cross': None,
           'samp_fail': 0}
    STEP_PULSE = T_PULSE - 200            # branch-local step count
    for step in range(STEP_PULSE + 1, CAP + 1):
        pairs = id_train[chain[step - 1]]
        a = torch.from_numpy(ALL_A[pairs]).long().to(DEV)
        b = torch.from_numpy(ALL_B[pairs]).long().to(DEV)
        y = torch.from_numpy(ALL_Y[pairs]).long().to(DEV)
        if step <= STEP_PULSE + PLEN:
            pi = sample_pi(seed, 200 + step, full, half)
            if not samp_checks(pi, TRINFO[seed]['tr'], U):
                rec['samp_fail'] += 1
            src = pi[pairs]
            sa = torch.from_numpy(ALL_A[src]).long().to(DEV)
            sb = torch.from_numpy(ALL_B[src]).long().to(DEV)
            logits = surgery_forward(m, a, b, sa, sb)
        else:
            logits = m(a, b)
        loss = LOSS(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m.Wq.weight.grad = None
        m.Wk.weight.grad = None
        opt.step()
        t = 200 + step
        if t in EVAL_T:
            rec['codes'][t] = R.own_codes(m)
        if step % 25 == 0 or step == CAP:
            m.eval()
            with torch.no_grad():
                v = float((m(va[0], va[1]).argmax(-1) ==
                           va[2]).float().mean())
            m.train()
            rec['va_t'].append(t)
            rec['va'].append(v)
            if v > 0.9 and rec['cross'] is None:
                rec['cross'] = t
    frozen_ok = bool(torch.equal(m.Wq.weight.detach(), wq_ref) and
                     torch.equal(m.Wk.weight.detach(), wk_ref))
    rec['frozen_ok'] = frozen_ok
    return rec


TRINFO = {}
ID_TRAIN = {}
CHAINS = {}


def main():
    global TRINFO, ID_TRAIN, CHAINS
    for seed in SEEDS:
        id_train, chain = R.batch_chain(seed, CAP)
        ID_TRAIN[seed] = id_train
        CHAINS[seed] = chain
        tr = np.sort(id_train)
        tr_mask = np.zeros(N_PAIRS, dtype=bool)
        tr_mask[tr] = True
        fu, ha, U = classify_train(tr_mask)
        TRINFO[seed] = {'tr': tr, 'tr_mask': tr_mask,
                        'full': fu, 'half': ha, 'U': U}

    out = {}
    gates = {}
    ok_all = True
    t0 = time.time()
    for seed in SEEDS:
        arms, gate = build_arms(seed)
        gates[seed] = gate
        ok_all &= gate['ok']
        out[seed] = {}
        va = R.val_probe(seed)
        tr_mask = TRINFO[seed]['tr_mask']
        for name, (wq, wk) in arms.items():
            rec = run_main(name, wq, wk, seed, CHAINS[seed],
                           ID_TRAIN[seed], va, tr_mask)
            out[seed][name] = rec
            ok_all &= rec['frozen_ok']
            print(f'seed{seed} {name} main: cross={rec["cross"]} '
                  f'va_last={rec["va"][-1]:.3f} '
                  f'strat@4000={rec["strat"].get(4000)} '
                  f'frozen={rec["frozen_ok"]}', flush=True)
        # pulse branches
        for name in ('00', 'SS', 'ADD'):
            main = out[seed][name]
            fork = {'weights': arms[name], 'state': main['fork'][0],
                    'opt': main['fork'][1]}
            br = run_branch(name, fork, seed, CHAINS[seed],
                            ID_TRAIN[seed], va, tr_mask)
            out[seed][name + '-P50'] = br
            ok_all &= br['frozen_ok']
            rctrl = {t: R.retrieve_self(br['codes'][t],
                                        main['codes'][t])
                     for t in sorted(br['codes']) if t in main['codes']}
            br['R_ctrl'] = rctrl
            seq = [rctrl[t] for t in sorted(rctrl)
                   if t > T_PULSE + PLEN]
            restored = sum(1 for i in range(len(seq) - 2)
                           if all(x >= 0.9 for x in seq[i:i + 3])) > 0
            br['restored'] = restored
            br['disrupted'] = rctrl.get(T_PULSE + PLEN, 1.0) <= 0.85
            print(f'seed{seed} {name}-P50: '
                  f'R_ctrl@pulse_end='
                  f'{rctrl.get(T_PULSE + PLEN, float("nan")):.3f} '
                  f'restored={restored} frozen={br["frozen_ok"]}',
                  flush=True)

    print('\n=== closure verdict (locked §6ef) ===', flush=True)
    if not SMOKE:
        n_hit = 0
        for seed in SEEDS:
            add_m = out[seed]['ADD']
            ss_m = out[seed]['SS']
            zero_m = out[seed]['00']
            comp = {}
            comp['i_nocross'] = add_m['cross'] is None and \
                zero_m['cross'] is not None
            st = add_m['strat'].get(4000, {})
            comp['ii_lookup'] = (st.get('m1', 0) >= 0.9 and
                                 st.get('m2', 1) <= 0.1)
            iA = add_m['codes'][CAP] if CAP in add_m['codes'] else \
                add_m['codes'][max(add_m['codes'])]
            iS = ss_m['codes'][CAP] if CAP in ss_m['codes'] else \
                ss_m['codes'][max(ss_m['codes'])]
            comp['iii_identity'] = abs(
                R.retrieve_self(iA, add_m['codes'][0]) -
                R.retrieve_self(iS, ss_m['codes'][0])) <= 0.05
            add_br = out[seed]['ADD-P50']
            ss_br = out[seed]['SS-P50']
            comp['iv_restorative'] = bool(add_br['disrupted'] and
                                          add_br['restored'] and
                                          ss_br['disrupted'] and
                                          ss_br['restored'])
            hit = all(comp.values())
            n_hit += hit
            print(f'  seed{seed}: {comp} -> '
                  f'{"CLOSURE-HIT" if hit else "partial"}', flush=True)
        if n_hit >= 2:
            print('VERDICT: CLOSURE-HIT (>=2/3 seeds)', flush=True)
        else:
            print('VERDICT: PARTIAL / per-cell registration', flush=True)
    print(f'HARNESS: {"PASS" if ok_all else "FAIL"}', flush=True)
    print(f'[K13] done in {time.time() - t0:.1f}s', flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'seeds': list(SEEDS), 'cap': CAP, 'gates': gates,
                     'arms': out}, f)
    print('pkl written', flush=True)


if __name__ == '__main__':
    main()
