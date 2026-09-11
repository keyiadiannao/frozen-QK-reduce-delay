"""K08 run.py — perturb-release-recovery: restorative dynamical
stability of native (S) vs fqk200 (F) pre-grok routing organization.

Package port of repro/r94_perturb_recovery.py (M2.5, 2026-09-02).
Verbatim except: shared machinery from common/ (readouts/indexing),
repro paths via REPRO_DIR, claim-dir output, SMOKE env R94_SMOKE ->
K08_SMOKE.

Design per (condition, seed): control main run to CAP=14000 (S plain;
F Wq/Wk grad=None steps<=200), batch idx chain recorded; snapshot at
T0=700 (model + DEEP-copied opt state); branches P50/P200 replay the
same idx chain with a resampWS-eq in-graph pulse (per-step stream
[7500+seed, step], V1-V4 checks + G-LEAK guard), then plain to CAP.
R_ctrl/R_pre from CLR of the model's OWN row-2 plain attention
(float16 archived).  DISRUPTION GATE R_ctrl(pulse_end)<=0.85;
NO-TRANSITION GATE va@pulse_end<0.9; restoration = R_ctrl>=0.9 x3
consecutive.  Result tree cells A-E (locked §6de); prefix gates vs
amp_dirswap pkl.

Claim (ledger K8): S_slow restores after P50 (~150 steps back to
0.99-1.0); basin boundary (50,200] -- P200 permanently flips fate; F
never restores and self-destructs without perturbation.
"self-restoring" earned for P50 only.  Preregistered §6de; §6df.
"""
import copy
import math
import os
import pickle
import sys
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
SMOKE = bool(os.environ.get('K08_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
T0 = 40 if SMOKE else 700
PS = (10, 20) if SMOKE else (50, 200)
EVAL_EVERY = 25
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else (0, 1, 2)
P = task.GROUP['p']
EPS = 1e-8

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K08] K08_SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')

R = ZPReadouts(DEV)
zp = cgates.ZPIndexing()
N_PAIRS, N_TRAIN = R.N_PAIRS, R.N_TRAIN
IDX, A_ARR, B_ARR = zp.IDX, zp.A_ARR, zp.B_ARR
Y_ARR, S_ARR, DIAG, OFF = zp.Y_ARR, zp.S_ARR, zp.DIAG, zp.OFF
ORB_REPS = zp.ORB_REPS
CLS = R.CLS

if SMOKE:
    CKPTS = sorted(set(range(10, CAP + 1, 10)))
else:
    CKPTS = sorted(set(range(25, 1501, 25))
                   | set(range(1600, 3001, 100))
                   | set(range(3500, CAP + 1, 500)))
CKSET = set(CKPTS)

ALL_A, ALL_B, ALL_Y = R.ALL_A, R.ALL_B, R.ALL_Y
A_DEV = torch.from_numpy(ALL_A).long().to(DEV)
B_DEV = torch.from_numpy(ALL_B).long().to(DEV)
GA = torch.arange(P).repeat_interleave(P)
GB = torch.arange(P).repeat(P)
GY = (GA + GB) % P
GIDX = torch.arange(P * P)
GRID_IDS = ((GIDX % P) * P + (GIDX // P)).numpy()
GA_DEV = GA.long().to(DEV)
GB_DEV = GB.long().to(DEV)
print(f'[K08] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} T0={T0} P={PS} '
      f'CAP={CAP} ckpts={len(CKPTS)}', flush=True)


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


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
    """r94 per-step stream (SeedSequence [7500+seed, step]) -- the
    resampWS family shared with r92; verbatim."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [7500 + seed, step]))
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


def _strat_stats(correct, self_in, swap_in, diag):
    m1 = ~self_in & swap_in & ~diag
    m2 = ~self_in & ~swap_in & ~diag
    m3 = ~self_in & diag
    return {'train_acc': float(correct[self_in].float().mean()),
            'va_grid': float(correct[~self_in].float().mean()),
            'm1': float(correct[m1].float().mean()),
            'm2': float(correct[m2].float().mean()),
            'm3': float(correct[m3].float().mean())}


def run_main(cond, seed):
    t0 = time.time()
    fz = 200 if cond == 'F' else 0
    va = R.val_probe(seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    id_train = perm[:N_TRAIN]
    tr_a = torch.from_numpy(ALL_A[id_train]).long()
    tr_b = torch.from_numpy(ALL_B[id_train]).long()
    tr_y = torch.from_numpy(ALL_Y[id_train]).long()
    tr = np.sort(id_train)
    tr_mask = np.zeros(N_PAIRS, dtype=bool)
    tr_mask[tr] = True
    full, half, U = classify_train(tr_mask)

    m = fresh_model(seed)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    rec = {'t': [], 'va_plain': [], 'cross95': None,
           'cross90': None, 'strat': {}, 'codes': {}, 'Rpre': {}}
    idx_hist = np.zeros((CAP, BATCH), dtype=np.int64)
    tr_mask_g = torch.from_numpy(tr_mask[GRID_IDS]).bool()
    tr_mask_swap_g = torch.from_numpy(tr_mask).bool()
    diag_grid = (GA == GB)

    snap = None
    for step in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (BATCH,))
        idx_hist[step - 1] = idx.numpy()
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        m.train()
        opt.zero_grad()
        logits, _ = R.plain_forward_attn(m, a, b)
        LOSS(logits, y).backward()
        if fz and step <= fz:
            m.Wq.weight.grad = None
            m.Wk.weight.grad = None
        if step <= 20:
            m.pos.grad[2].zero_()
        opt.step()
        if step == T0:
            snap = ({k: v.clone() for k, v in
                     m.state_dict().items()},
                    copy.deepcopy(opt.state_dict()))
        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                m.eval()
                lp = float((m(va[0], va[1]).argmax(-1) == va[2])
                           .float().mean())
                rec['t'].append(step)
                rec['va_plain'].append(lp)
                if lp > 0.95 and rec['cross95'] is None:
                    rec['cross95'] = step
                if lp > 0.9 and rec['cross90'] is None:
                    rec['cross90'] = step
                m.train()
        if step in CKSET:
            with torch.no_grad():
                m.eval()
                cd = R.own_codes(m)
                rec['codes'][step] = cd.astype(np.float16)
                if T0 in rec['codes']:
                    rec['Rpre'][step] = R.retrieve_self(
                        cd, rec['codes'][T0].astype(np.float32))
                m.train()
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                m.eval()
                lg, _ = R.plain_forward_attn(m, GA_DEV, GB_DEV)
                rec['strat'][step] = _strat_stats(
                    lg.argmax(-1).cpu() == GY, tr_mask_g,
                    tr_mask_swap_g, diag_grid)
                m.train()
        if step % 2000 == 0:
            print(f"  [{cond}{seed} main t={step}] "
                  f"va={rec['va_plain'][-1]:.2f}", flush=True)

    rec['snap'] = snap
    rec['idx_hist'] = idx_hist
    rec['id_train'] = id_train
    rec['U'] = U
    rec['full'] = full
    rec['half'] = half
    print(f"[{cond}{seed} main done in {time.time() - t0:.0f}s] "
          f"cross95={rec['cross95']}", flush=True)
    return rec


def run_branch(cond, seed, Pp, main):
    t0 = time.time()
    U, full, half = main['U'], main['full'], main['half']
    id_train = main['id_train']
    tr = np.sort(id_train)
    va = R.val_probe(seed)
    tr_a = torch.from_numpy(ALL_A[id_train]).long()
    tr_b = torch.from_numpy(ALL_B[id_train]).long()
    tr_y = torch.from_numpy(ALL_Y[id_train]).long()
    tr_mask = np.zeros(N_PAIRS, dtype=bool)
    tr_mask[tr] = True

    m = fresh_model(seed)
    m.load_state_dict({k: v.clone() for k, v in
                       main['snap'][0].items()})
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(copy.deepcopy(main['snap'][1]))

    rec = {'t': [], 'va_plain': [], 'cross95': None,
           'cross90': None, 'strat': {}, 'Rctrl': {}, 'Rpre': {},
           'g3_fail': 0}
    end_pulse = T0 + Pp
    tr_mask_g = torch.from_numpy(tr_mask[GRID_IDS]).bool()
    tr_mask_swap_g = torch.from_numpy(tr_mask).bool()
    diag_grid = (GA == GB)

    for step in range(T0 + 1, CAP + 1):
        idx = torch.from_numpy(main['idx_hist'][step - 1]).long()
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        m.train()
        opt.zero_grad()
        if step <= end_pulse:
            pi = sample_pi(seed, step, full, half)
            if not samp_checks(pi, tr, U):
                rec['g3_fail'] += 1
            src = np.full(N_PAIRS, -1, dtype=np.int64)
            src[tr] = pi[tr]
            img = src[tr[idx.numpy()]]
            if (img < 0).any():
                raise SystemExit('[K08] G-LEAK FAIL: source '
                                 'outside T')
            img_t = torch.from_numpy(img).long().to(DEV)
            logits = surgery_forward(m, a, b, A_DEV[img_t],
                                     B_DEV[img_t])
        else:
            logits, _ = R.plain_forward_attn(m, a, b)
        LOSS(logits, y).backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                m.eval()
                lp = float((m(va[0], va[1]).argmax(-1) == va[2])
                           .float().mean())
                rec['t'].append(step)
                rec['va_plain'].append(lp)
                if lp > 0.95 and rec['cross95'] is None:
                    rec['cross95'] = step
                if lp > 0.9 and rec['cross90'] is None:
                    rec['cross90'] = step
                m.train()
        if step in CKSET:
            with torch.no_grad():
                m.eval()
                cd = R.own_codes(m)
                rec['Rctrl'][step] = R.retrieve_self(
                    cd, main['codes'][step].astype(np.float32))
                rec['Rpre'][step] = R.retrieve_self(
                    cd, main['codes'][T0].astype(np.float32))
                m.train()
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                m.eval()
                lg, _ = R.plain_forward_attn(m, GA_DEV, GB_DEV)
                rec['strat'][step] = _strat_stats(
                    lg.argmax(-1).cpu() == GY, tr_mask_g,
                    tr_mask_swap_g, diag_grid)
                m.train()
        if step % 2000 == 0:
            rc = rec['Rctrl'].get(step)
            print(f"  [{cond}{seed} P{Pp} t={step}] "
                  f"va={rec['va_plain'][-1]:.2f} Rctrl="
                  f"{rc if rc is None else round(rc, 3)}", flush=True)

    rec['pulse_end'] = end_pulse
    rec['va_at_end'] = rec['va_plain'][rec['t'].index(end_pulse)] \
        if end_pulse in rec['t'] else None
    print(f"[{cond}{seed} P{Pp} done in {time.time() - t0:.0f}s] "
          f"cross95={rec['cross95']} "
          f"Rctrl@end={rec['Rctrl'].get(end_pulse)}", flush=True)
    return rec


def prefix_gates(out):
    with open(os.path.join(REPRO_DIR, 'amp_dirswap_results.pkl'),
              'rb') as f:
        amp = pickle.load(f)
    ok_all = True
    for cond, arm in (('S', 'natS'), ('F', 'natF')):
        for seed in SEEDS:
            ref = amp[seed]['arms'][arm]
            if ref.get('crossed'):
                lim = min(ref['crossed'], 4000)
            else:
                lim = 4000
            new = out[f'{cond}{seed}']['control']
            pairs = [(t, v) for t, v in zip(ref['t'], ref['val_acc'])
                     if t <= lim]
            n = len(pairs)
            same_t = new['t'][:n] == [p[0] for p in pairs]
            dv = max((abs(x - y) for x, y in
                      zip(new['va_plain'][:n],
                          [p[1] for p in pairs])), default=0.0)
            ok = same_t and dv == 0.0
            print(f"prefix-gate {cond}{seed} (vs amp {arm}, "
                  f"t<={lim}): bitwise={ok} (n={n} "
                  f"maxdv={dv:.2e})", flush=True)
            ok_all &= ok
    return ok_all


def consecutive_high(d, lo, hi, thr=0.9, need=3):
    ts = sorted(t for t in d if lo < t <= hi)
    run = 0
    for t in ts:
        if d[t] >= thr:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


def main():
    out = {}
    for cond in ('S', 'F'):
        for seed in SEEDS:
            print(f'=== {cond}{seed} ===', flush=True)
            main = run_main(cond, seed)
            branches = {}
            for Pp in PS:
                branches[f'P{Pp}'] = run_branch(cond, seed, Pp, main)
            out[f'{cond}{seed}'] = {
                'control': {k: main[k] for k in
                            ('t', 'va_plain', 'cross95', 'cross90',
                             'strat', 'Rpre')},
                'codes_T0': main['codes'][T0],
                'branches': {f'P{Pp}': {k: branches[f'P{Pp}'][k]
                                        for k in
                                        ('t', 'va_plain', 'cross95',
                                         'cross90', 'strat', 'Rctrl',
                                         'Rpre', 'pulse_end',
                                         'va_at_end', 'g3_fail')}
                             for Pp in PS}}
            del main['codes'], main['idx_hist'], main['snap']
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(out, f)

    print('\n=== per-branch gates & restoration ===', flush=True)
    cls = {}
    for cond in ('S', 'F'):
        for seed in SEEDS:
            for Pp in PS:
                b = out[f'{cond}{seed}']['branches'][f'P{Pp}']
                r_end = b['Rctrl'].get(b['pulse_end'])
                disrupted = (r_end is not None and r_end <= 0.85)
                no_trans = (b['va_at_end'] is not None
                            and b['va_at_end'] < 0.9)
                hi = min(b['cross95'], CAP) if b['cross95'] else CAP
                restores = consecutive_high(b['Rctrl'],
                                            b['pulse_end'], hi)
                t95 = b['cross95']
                fate = ('fast' if (t95 is not None and t95 <= 5000)
                        else 'slow' if (t95 is None or t95 > 8000)
                        else 'intermediate')
                print(f"  {cond}{seed} P{Pp}: disrupted={disrupted} "
                      f"(Rctrl@end={r_end}) no_transition={no_trans} "
                      f"restores={restores} fate={fate} "
                      f"(T95={t95}) g3fail={b['g3_fail']}", flush=True)
                cls[(cond, seed, Pp)] = (disrupted, restores, fate,
                                         no_trans)

    def majority(vals):
        return sum(bool(v) for v in vals) > len(vals) / 2

    if not SMOKE:
        print('\n=== verdict (P200 primary, P50 replication) ===',
              flush=True)
        S_dis = majority([cls[('S', s, 200)][0] for s in SEEDS])
        F_dis = majority([cls[('F', s, 200)][0] for s in SEEDS])
        S_res = majority([cls[('S', s, 200)][1] for s in SEEDS])
        F_res = majority([cls[('F', s, 200)][1] for s in SEEDS])
        S_fast = majority([cls[('S', s, 200)][2] == 'fast'
                           for s in SEEDS])
        S_slow = majority([cls[('S', s, 200)][2] == 'slow'
                           for s in SEEDS])
        if not (S_dis or F_dis):
            print("CELL E: NON-DIAGNOSTIC -- perturbation failed "
                  "(own routing never disrupted); maintenance "
                  "hypothesis neither supported nor falsified",
                  flush=True)
        elif S_dis and S_res and not F_res and S_slow:
            print("CELL A: STRONGEST -- early-QK-conditioned "
                  "restorative organization (S restores, F does not, "
                  "S stays slow)", flush=True)
        elif S_dis and not S_res and S_fast:
            print("CELL B: causal link w/o self-restoration -- "
                  "disruption accelerates generalization", flush=True)
        elif S_dis and not S_res and S_slow:
            print("CELL C: restorative maintenance hypothesis DEAD -- "
                  "identity recovery not required for delay",
                  flush=True)
        elif S_res and F_res:
            print("CELL D: restoration is a generic optimization "
                  "property -- cannot explain early-QK freeze effect",
                  flush=True)
        else:
            print(f"HETEROGENEOUS/MIXED -- register per cell "
                  f"(S: dis={S_dis} res={S_res} fast={S_fast} "
                  f"slow={S_slow}; F: dis={F_dis} res={F_res})",
                  flush=True)
        g = prefix_gates(out)
        print('PREFIX GATES', 'PASS' if g else 'FAIL', flush=True)
        print('ALL GATES PASS' if g else 'GATE FAILURE', flush=True)
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results.pkl', flush=True)


if __name__ == '__main__':
    main()
