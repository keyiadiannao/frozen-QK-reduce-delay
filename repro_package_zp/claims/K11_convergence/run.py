"""K11 run.py — pre-transition restorative assay relocation: t0=400,
three conditions S/Z/F at the same absolute time.

Package port of repro/r98_pretransition_assay.py (M2.5, 2026-09-02).
Verbatim except: common/ imports where bit-identical (val_probe,
retrieval), repro paths via REPRO_DIR, claim-dir output, SMOKE env
R98_SMOKE -> K11_SMOKE.  The noeq (Z) forward variants stay local --
ZPReadouts.plain_forward_attn has no noeq parameter by design.

Design per (cond in {S,F,Z}, seed): control main run to CAP=14000
(S plain; F fqk200 grad-zero<=200; Z abeq_noeqemb), idx chain
recorded, code tables at the R94 schedule (float16); branch P50:
replay idx chain, steps 401..450 resampWS-eq in-graph pulse (stream
[7500+seed, step]), then plain to CAP.  Gates: S/F prefix vs
amp_dirswap (t<=400, bitwise); Z G0-noeq bitwise; Z bridge-200 soft
anchor (lineage-diff expected, registered).

Claim (ledger K11): Z and F converge on the same downstream deficit
(S-type restorative maintenance missing, CELL 2); S@400 already
restorative (early write).  Preregistered §6dl; verdict §6dm.
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
SMOKE = bool(os.environ.get('K11_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
T0 = 40 if SMOKE else 400
PP = 10 if SMOKE else 50
EVAL_EVERY = 25
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else (0, 1, 2)
P = task.GROUP['p']
EPS = 1e-8

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K11] K11_SMOKE=1 but CUDA visible -- abort')

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
print(f'[K11] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} T0={T0} P={PP} '
      f'CAP={CAP} ckpts={len(CKPTS)}', flush=True)


def fresh_model(cond, seed):
    arch = 'abeq_noeqemb' if cond == 'Z' else 'abeq'
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch=arch,
                    ln_eps=1e-5, eq_alpha=0.0).to(DEV)


def plain_forward_attn(m, a, b, noeq=False):
    """r98 local variant with the noeq branch (Z condition)."""
    B = a.shape[0]
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    if noeq:
        x = torch.cat([x[:, :2, :],
                       m.pos[None, 2, :].expand(B, 1, -1)], dim=1)
    h = m.ln1(x)
    q = m.Wq(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
    k = m.Wk(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
    v = m.Wv(h).view(B, 3, m.n_heads, m.d_head).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(m.d_head)
    attn = scores.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).contiguous().view(B, 3, m.d_model)
    x = x + m.Wo(out)
    h2 = m.ln2(x)
    mm = m.mlp2(F.gelu(m.mlp1(h2)))
    x = x + mm
    last = x[:, 2, :]
    return m.Wu(m.ln_f(last)), attn


def surgery_forward(m, a, b, src_a, src_b, noeq=False):
    B = a.shape[0]
    H, dh = m.n_heads, m.d_head
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    if noeq:
        x = torch.cat([x[:, :2, :],
                       m.pos[None, 2, :].expand(B, 1, -1)], dim=1)
    h = m.ln1(x)
    q = m.Wq(h).view(B, 3, H, dh).transpose(1, 2)
    k = m.Wk(h).view(B, 3, H, dh).transpose(1, 2)
    v = m.Wv(h).view(B, 3, H, dh).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    attn_main = scores.softmax(dim=-1)
    eq_s = torch.full_like(src_a, task.N_EQ)
    tok_s = torch.stack([src_a, src_b, eq_s], dim=1)
    x_s = m.emb(tok_s) + m.pos[None, :, :]
    if noeq:
        x_s = torch.cat([x_s[:, :2, :],
                         m.pos[None, 2, :].expand(B, 1, -1)], dim=1)
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
    """resampWS family stream ([7500+seed, step]) -- shared with
    r92/r94; verbatim."""
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


@torch.no_grad()
def own_codes(m, noeq):
    tid = torch.arange(N_PAIRS, dtype=torch.long, device=DEV)
    a, b = tid % P, tid // P
    outs = []
    for s in range(0, N_PAIRS, 4096):
        _, attn = plain_forward_attn(m, a[s:s + 4096],
                                     b[s:s + 4096], noeq=noeq)
        a2 = attn[:, :, 2, :].double().clamp_min(EPS)
        c = torch.log(a2)
        c = c - c.mean(dim=-1, keepdim=True)
        outs.append(c.reshape(c.shape[0], -1).float().cpu())
    return torch.cat(outs, 0).numpy()


def retrieve_self(query, ref):
    Q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-30)
    Rc = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-30)
    hits = 0
    for c in range(P):
        idx = CLS[c]
        sims = Q[idx] @ Rc[idx].T
        hits += int((sims.argmax(1) ==
                     np.arange(len(idx))).sum())
    return hits / N_PAIRS


def run_main(cond, seed):
    t0 = time.time()
    noeq = (cond == 'Z')
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

    m = fresh_model(cond, seed)
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
        logits, _ = plain_forward_attn(m, a, b, noeq=noeq)
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
                cd = own_codes(m, noeq)
                rec['codes'][step] = cd.astype(np.float16)
                if T0 in rec['codes']:
                    rec['Rpre'][step] = retrieve_self(
                        cd, rec['codes'][T0].astype(np.float32))
                m.train()
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                m.eval()
                lg, _ = plain_forward_attn(m, GA_DEV, GB_DEV,
                                           noeq=noeq)
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
    noeq = (cond == 'Z')
    U, full, half = main['U'], main['full'], main['half']
    id_train = main['id_train']
    tr = np.sort(id_train)
    va = R.val_probe(seed)
    tr_a = torch.from_numpy(ALL_A[id_train]).long()
    tr_b = torch.from_numpy(ALL_B[id_train]).long()
    tr_y = torch.from_numpy(ALL_Y[id_train]).long()
    tr_mask = np.zeros(N_PAIRS, dtype=bool)
    tr_mask[tr] = True

    m = fresh_model(cond, seed)
    m.load_state_dict({k: v.clone() for k, v in
                       main['snap'][0].items()})
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(copy.deepcopy(main['snap'][1]))

    rec = {'t': [], 'va_plain': [], 'cross95': None,
           'cross90': None, 'Rctrl': {}, 'Rpre': {}, 'g3_fail': 0}
    end_pulse = T0 + Pp

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
                raise SystemExit('[K11] G-LEAK FAIL')
            img_t = torch.from_numpy(img).long().to(DEV)
            logits = surgery_forward(m, a, b, A_DEV[img_t],
                                     B_DEV[img_t], noeq=noeq)
        else:
            logits, _ = plain_forward_attn(m, a, b, noeq=noeq)
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
                cd = own_codes(m, noeq)
                rec['Rctrl'][step] = retrieve_self(
                    cd, main['codes'][step].astype(np.float32))
                rec['Rpre'][step] = retrieve_self(
                    cd, main['codes'][T0].astype(np.float32))
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


def g0_noeq_gate():
    ggen = torch.Generator()
    ggen.manual_seed(888)
    a = torch.randint(0, P, (64,), generator=ggen).to(DEV)
    b = torch.randint(0, P, (64,), generator=ggen).to(DEV)
    m_ref = D57Model(pos_mode='zeros', arch='abeq_noeqemb',
                     ln_eps=1e-5, eq_alpha=0.0).to(DEV)
    m_ref.eval()
    m_our = fresh_model('Z', 0)
    m_our.load_state_dict(m_ref.state_dict())
    m_our.eval()
    with torch.no_grad():
        lg_ref = m_ref(a, b)
        lg_our, _ = plain_forward_attn(m_our, a, b, noeq=True)
    d = float((lg_ref - lg_our).abs().max())
    print(f"G0-noeq: max|dlogit|={d:.2e} -> "
          f"{'PASS' if d == 0.0 else 'FAIL'}", flush=True)
    return d == 0.0


def prefix_gates(out):
    with open(os.path.join(REPRO_DIR, 'amp_dirswap_results.pkl'),
              'rb') as f:
        amp = pickle.load(f)
    ok_all = True
    for cond, arm in (('S', 'natS'), ('F', 'natF')):
        for seed in SEEDS:
            ref = amp[seed]['arms'][arm]
            new = out[f'{cond}{seed}']['control']
            pairs = [(t, v) for t, v in zip(ref['t'], ref['val_acc'])
                     if t <= 400]
            n = len(pairs)
            same_t = new['t'][:n] == [p[0] for p in pairs]
            dv = max((abs(x - y) for x, y in
                      zip(new['va_plain'][:n],
                          [p[1] for p in pairs])), default=0.0)
            ok = same_t and dv == 0.0
            print(f"prefix-gate {cond}{seed} (vs amp {arm}, t<=400): "
                  f"bitwise={ok} (n={n} maxdv={dv:.2e})", flush=True)
            ok_all &= ok
    return ok_all


def bridge_soft_anchor(out):
    import glob
    for seed in SEEDS:
        fs = glob.glob(os.path.join(REPRO_DIR, 'bridge_zp_B',
                                    f'seed{seed}_*full0000200.pt'))
        d = torch.load(fs[0], map_location='cpu', weights_only=False)
        c = out[f'Z{seed}']['control']
        i = c['t'].index(200) if 200 in c['t'] else None
        if i is None:
            continue
        dva = abs(c['va_plain'][i] - float(d['val_acc']))
        tag = 'match' if dva == 0.0 else 'lineage-diff'
        print(f"bridge-anchor Z{seed}: va@200 ours="
              f"{c['va_plain'][i]:.4f} bridge="
              f"{float(d['val_acc']):.4f} ({tag})", flush=True)


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
    if not SMOKE:
        if not g0_noeq_gate():
            raise SystemExit('[K11] G0-noeq FAILED -- aborting')
    for cond in ('S', 'F', 'Z'):
        for seed in SEEDS:
            print(f'=== {cond}{seed} ===', flush=True)
            main = run_main(cond, seed)
            branch = run_branch(cond, seed, PP, main)
            out[f'{cond}{seed}'] = {
                'control': {k: main[k] for k in
                            ('t', 'va_plain', 'cross95', 'cross90',
                             'strat', 'Rpre')},
                'codes_T0': main['codes'][T0],
                'branch': {k: branch[k] for k in
                           ('t', 'va_plain', 'cross95', 'cross90',
                            'Rctrl', 'Rpre', 'pulse_end',
                            'va_at_end', 'g3_fail')}}
            del main['codes'], main['idx_hist'], main['snap']
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(out, f)

    print('\n=== per-condition readouts (t0=400, P50) ===', flush=True)
    res = {}
    for cond in ('S', 'F', 'Z'):
        for seed in SEEDS:
            c = out[f'{cond}{seed}']['control']
            b = out[f'{cond}{seed}']['branch']
            r_end = b['Rctrl'].get(b['pulse_end'])
            disrupted = (r_end is not None and r_end <= 0.85)
            hi = min(b['cross95'], CAP) if b['cross95'] else CAP
            restores = consecutive_high(b['Rctrl'], b['pulse_end'],
                                        hi)
            restores_pre = consecutive_high(b['Rctrl'],
                                            b['pulse_end'], 700)
            rpre500 = c['Rpre'].get(500)
            rpre700 = c['Rpre'].get(700)
            res[(cond, seed)] = (disrupted, restores, restores_pre,
                                 rpre500, rpre700, b['cross95'])
            print(f"  {cond}{seed}: Rctrl@450={r_end} "
                  f"disrupted={disrupted} restores={restores} "
                  f"(pre-transition win={restores_pre}) "
                  f"Rpre_d100={rpre500} Rpre_d300={rpre700} "
                  f"cross95={b['cross95']} g3fail={b['g3_fail']}",
                  flush=True)

    def majority(cond, idx):
        return sum(1 for s in SEEDS if res[(cond, s)][idx]) >= 2

    if not SMOKE:
        print('\n=== verdict (locked §6dl) ===', flush=True)
        z_dis = majority('Z', 0)
        z_res = majority('Z', 1)
        f_res = majority('F', 1)
        s_res = majority('S', 1)
        if z_dis and z_res and not f_res:
            print("CELL 1: restorative dynamics NOT sufficient for "
                  "slow -- Z restores (S-like) yet fast; Z/F proximal "
                  "dynamics differ; restorative property is not a "
                  "universal slow/fast order parameter", flush=True)
        elif z_dis and not z_res and not f_res:
            print("CELL 2: Z/F converge on the same downstream deficit "
                  "(S-type restorative maintenance missing) -- "
                  "destination-level support for one mechanism line",
                  flush=True)
        else:
            print(f"MIXED/HETEROGENEOUS: register per cell "
                  f"(S_res={s_res}, Z_dis={z_dis}, Z_res={z_res}, "
                  f"F_res={f_res})", flush=True)
        d2 = ("present by t=400 (early write)" if s_res
              else "matures 400->700")
        print(f"S@400 restorativity (delta2): restores={s_res} -> "
              f"{d2}", flush=True)
        g = prefix_gates(out)
        bridge_soft_anchor(out)
        print('PREFIX GATES', 'PASS' if g else 'FAIL', flush=True)
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results.pkl', flush=True)


if __name__ == '__main__':
    main()
