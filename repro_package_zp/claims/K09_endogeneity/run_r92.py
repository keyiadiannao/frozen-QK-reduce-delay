"""K09 run_r92.py — native in-graph fixed-vs-resampled persistence
test (crutch lock-in arm).

Package port of repro/r92_native_ingraph_ws.py (M2.5, 2026-09-02).
Verbatim except: common/ imports (indexing arrays via gates.ZPIndexing,
val_probe/plain_forward_attn from readouts -- G0''-equivalent gates
cover them), repro paths via REPRO_DIR, claim-dir output, SMOKE env
R92_SMOKE -> K09_SMOKE, and strat_plain promoted from module global to
run_seed local (identical contents per seed).

Arms: natS (plain anchor) / fixedWS (pi_0 forever, stream 7000+seed) /
resampWS (per-step redraw, SeedSequence [7500+seed, step]) / permBS
(R74 batch-level within-sum gather, gen 1000+seed -- machinery bridge,
not part of the causal comparison).  Gates: G-ID (pi=id double-forward
vs native), per-step sampler V1-V4 + sha256 hashes, G-LEAK, fixedWS
constancy, G-natS prefix bitwise vs r85 pkl.  Classification locked
§6cz (six cells A-F).

Claim (ledger K9, part 1): externally pinned persistence does not
rebuild native residence -- crutch lock-in (fixedWS surgery 1.000 /
own plain chance) while resampled fast.
"""
import hashlib
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

from common import task
from common import gates as cgates
from common.model import D57Model, make_optimizer
from common.readouts import ZPReadouts

task.set_group('zp', 113)
SMOKE = bool(os.environ.get('K09_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
EVAL_EVERY = 25
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else (0, 1, 2)
P = task.GROUP['p']
ARMS = ('natS', 'fixedWS', 'resampWS', 'permBS')

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K09] K09_SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')
print(f'[K09-r92] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} CAP={CAP} '
      f'BATCH={BATCH} threads={torch.get_num_threads()}', flush=True)

R = ZPReadouts(DEV)
zp = cgates.ZPIndexing()
N_PAIRS, N_TRAIN = R.N_PAIRS, R.N_TRAIN
IDX, A_ARR, B_ARR = zp.IDX, zp.A_ARR, zp.B_ARR
Y_ARR, S_ARR, DIAG, OFF = zp.Y_ARR, zp.S_ARR, zp.DIAG, zp.OFF
ORB_REPS = zp.ORB_REPS

plain_forward_attn = R.plain_forward_attn


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


def val_probe(seed):
    """r92 val_probe verbatim (4-tuple; 4th element kept for fidelity)."""
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    rngv = np.random.default_rng(555)
    vb = rngv.permutation(perm[N_TRAIN:])[:1024]
    return (torch.from_numpy(all_a[vb]).long().to(DEV),
            torch.from_numpy(all_b[vb]).long().to(DEV),
            torch.from_numpy(all_y[vb]).long().to(DEV),
            np.asarray(vb, dtype=np.int64))


def surgery_forward(m, a, b, src_a, src_b):
    """Second-forward in-graph surgery (r87/r92 verbatim)."""
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


def permbs_forward(m, a, b, rows2_perm):
    """R74 batch-level within-sum gather, verbatim semantics."""
    B = a.shape[0]
    H, dh, dm = m.n_heads, m.d_head, m.d_model
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    h = m.ln1(x)
    q_t = m.Wq(h).view(B, 3, H, dh).transpose(1, 2)
    k_t = m.Wk(h).view(B, 3, H, dh).transpose(1, 2)
    scores = (q_t @ k_t.transpose(-1, -2)) / math.sqrt(dh)
    attn = scores.softmax(-1)
    a2 = attn[:, :, 2, :][rows2_perm]
    attn = torch.cat([attn[:, :, :2, :], a2.unsqueeze(2)], dim=2)
    out = (attn @ m.Wv(h).view(B, 3, H, dh).transpose(1, 2))
    out = out.transpose(1, 2).contiguous().view(B, 3, dm)
    x_at = x + m.Wo(out)
    h2 = m.ln2(x_at)
    x_fin = x_at + m.mlp2(F.gelu(m.mlp1(h2)))
    logits = m.Wu(m.ln_f(x_fin[:, 2, :]))
    return logits, {'attn': attn.detach()}


def sum_perm(y_cpu, gen):
    """R74 verbatim: within-sum group permutation indices."""
    B = y_cpu.shape[0]
    perm = torch.empty(B, dtype=torch.long)
    for c in torch.unique(y_cpu):
        idx = (y_cpu == c).nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            perm[idx] = idx[torch.randperm(idx.numel(),
                                           generator=gen)]
        else:
            perm[idx] = idx
    return perm.to(DEV)


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


def sample_pi(seed, draw, full, half):
    """fixed arm: draw=0 via stream 7000+seed; resampled:
    SeedSequence([7500+seed, draw]).  Verbatim (r92 family)."""
    if draw == 0:
        rng = np.random.default_rng(7000 + seed)
    else:
        rng = np.random.default_rng(np.random.SeedSequence(
            [7500 + seed, draw]))
    pi = IDX.copy()
    for c in range(P):
        fo = full[c]
        if len(fo) >= 2:
            k = len(fo)
            permk = rng.permutation(k)
            tgt = np.roll(permk, 1)
            orient = rng.integers(0, 2, size=k)
            for m in range(k):
                i = int(fo[permk[m]])
                j = int(fo[tgt[m]])
                if orient[m] == 0:
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
    """V1 bijection on T / V2 within-sum / V3 fp==U / V4 equivariance."""
    pt = pi[tr]
    v1 = bool(np.unique(pt).size == N_TRAIN)
    v2 = bool(np.all(Y_ARR[pt] == Y_ARR[tr]))
    fp = set(int(x) for x in tr[pt == tr])
    v3 = (fp == U)
    in_T = np.zeros(N_PAIRS, dtype=bool)
    in_T[tr] = True
    mk = in_T[S_ARR[tr]]
    v4 = bool(np.all(pi[S_ARR[tr[mk]]] == S_ARR[pi[tr[mk]]]))
    return v1, v2, v3, v4


def pi_hash(pi):
    return hashlib.sha256(
        np.ascontiguousarray(pi).tobytes()).hexdigest()[:16]


def _strat_stats(correct, self_in, swap_in, diag):
    m1 = ~self_in & swap_in & ~diag
    m2 = ~self_in & ~swap_in & ~diag
    m3 = ~self_in & diag
    return {'train_acc': float(correct[self_in].float().mean()),
            'va_grid': float(correct[~self_in].float().mean()),
            'm1': float(correct[m1].float().mean()),
            'm2': float(correct[m2].float().mean()),
            'm3': float(correct[m3].float().mean())}


def in_T_all(ids, tr_mask):
    return bool(tr_mask[ids].all())


def run_seed(seed):
    t0 = time.time()
    va_a, va_b, va_y, va_ids = val_probe(seed)
    va = (va_a, va_b, va_y)
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    id_train = perm[:N_TRAIN]                 # UNSORTED (§6cz-prime)
    tr_a = torch.from_numpy(all_a[id_train]).long()
    tr_b = torch.from_numpy(all_b[id_train]).long()
    tr_y = torch.from_numpy(all_y[id_train]).long()
    tr = np.sort(id_train)
    tr_mask = np.zeros(N_PAIRS, dtype=bool)
    tr_mask[tr] = True
    full, half, U = classify_train(tr_mask)
    n_fp = len(U)
    print(f'seed{seed}: |T|={N_TRAIN} |U|={n_fp} '
          f'(diag={int(tr_mask[DIAG].sum())}, half_single='
          f'{n_fp - int(tr_mask[DIAG].sum())})', flush=True)

    # ---- pi_0 and per-arm source maps (table-id space) ----
    pi0 = sample_pi(seed, 0, full, half)
    v = samp_checks(pi0, tr, U)
    assert all(v), f'pi_0 failed sampler checks {v}'
    h0 = pi_hash(pi0)
    print(f'pi_0 checks: bij/wsum/fpU/eq = {v} hash={h0}',
          flush=True)
    src_fixed = np.full(N_PAIRS, -1, dtype=np.int64)
    src_fixed[tr] = pi0[tr]
    src_res = src_fixed.copy()      # step 1 shares pi_0 (r87 conv.)

    arms = {}
    for name in ARMS:
        m = fresh_model(seed)
        opt, _, _ = make_optimizer(m, 'wd_0011')
        arms[name] = {'m': m, 'opt': opt, 't': [], 'va_plain': [],
                      'cross95': None, 'cross90': None,
                      'train_surg': [], 'g3_fail': 0, 'n_draws': 0,
                      'hashes': []}
    genBS = torch.Generator().manual_seed(1000 + seed)  # r74 verbatim

    # fixed 1024 train subset for train_surg (no RNG at eval)
    sub = np.arange(1024)
    sub_a = tr_a[sub].to(DEV)
    sub_b = tr_b[sub].to(DEV)
    sub_y = tr_y[sub].to(DEV)
    sub_ids = id_train[sub]

    # grid masks (plain strat; r89 convention: grid_ids[k]=S(k))
    ga = torch.arange(P).repeat_interleave(P)
    gb = torch.arange(P).repeat(P)
    gy = (ga + gb) % P
    gidx = torch.arange(P * P)
    grid_ids = ((gidx % P) * P + (gidx // P)).numpy()
    ga_dev = ga.long().to(DEV)
    gb_dev = gb.long().to(DEV)
    tr_mask_g = torch.from_numpy(tr_mask[grid_ids]).bool()
    tr_mask_swap_g = torch.from_numpy(tr_mask).bool()
    diag_grid = (ga == gb)

    # ---- G-ID: pi=id double-forward vs native (dedicated gen) ----
    ggen = torch.Generator()
    ggen.manual_seed(777)
    ids0 = torch.randint(tr_a.shape[0], (BATCH,), generator=ggen)
    a0 = tr_a[ids0].to(DEV)
    b0 = tr_b[ids0].to(DEV)
    y0 = tr_y[ids0].to(DEV)
    mG = arms['fixedWS']['m']
    mG.train()
    arms['fixedWS']['opt'].zero_grad()
    lg_s = surgery_forward(mG, a0, b0, a0, b0)
    LOSS(lg_s, y0).backward()
    g_surg = {n: (p.grad.clone() if p.grad is not None else None)
              for n, p in mG.named_parameters()}
    arms['fixedWS']['opt'].zero_grad()
    with torch.no_grad():
        lg_nat = mG(a0, b0)
    d_logit = float((lg_nat - lg_s).detach().abs().max())
    arms['fixedWS']['opt'].zero_grad()
    LOSS(mG(a0, b0), y0).backward()
    worst_cos, worst_rel = 1.0, 0.0
    for n, p in mG.named_parameters():
        gs, gn = g_surg[n], p.grad
        if gs is None or gn is None:
            continue
        gs, gn = gs.flatten(), gn.flatten()
        cos = float(torch.dot(gs, gn) /
                    (gs.norm() * gn.norm() + 1e-30))
        rel = float((gs - gn).norm() / (gn.norm() + 1e-30))
        worst_cos = min(worst_cos, cos)
        worst_rel = max(worst_rel, rel)
    gid_ok = (d_logit <= 1e-6 and worst_cos >= 0.99999
              and worst_rel <= 1e-4)
    print(f"G-ID pi=id: d_logit={d_logit:.2e} worst_cos="
          f"{worst_cos:.8f} worst_rel={worst_rel:.2e} -> "
          f"{'PASS' if gid_ok else 'FAIL'}", flush=True)
    if not gid_ok:
        raise SystemExit('[K09] G-ID FAILED -- aborting')
    for name in ARMS:
        arms[name]['opt'].zero_grad()
        arms[name]['m'].train()

    all_a_dev = torch.from_numpy(all_a).long().to(DEV)
    all_b_dev = torch.from_numpy(all_b).long().to(DEV)
    va_ids_dev = torch.from_numpy(va_ids).long().to(DEV)

    strat_plain = {}
    for step in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (BATCH,))
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        tid = id_train[idx.numpy()]
        r2p = sum_perm(y.cpu(), genBS)     # permBS (r74 verbatim)
        for name in ARMS:
            A = arms[name]
            m, opt = A['m'], A['opt']
            m.train()
            opt.zero_grad()
            if name == 'natS':
                logits, _ = plain_forward_attn(m, a, b)
            elif name == 'permBS':
                logits, _ = permbs_forward(m, a, b, r2p)
            else:
                if name == 'resampWS' and step >= 2:
                    pit = sample_pi(seed, step, full, half)
                    vv = samp_checks(pit, tr, U)
                    if not all(vv):
                        A['g3_fail'] += 1
                    src_res[tr] = pit[tr]
                    A['hashes'].append(pi_hash(pit))
                    A['n_draws'] += 1
                img_ids = (src_fixed if name == 'fixedWS'
                           else src_res)[tid]
                if not in_T_all(img_ids, tr_mask):
                    raise SystemExit('[K09] G-LEAK FAIL: source '
                                     'outside T')
                img_t = torch.from_numpy(img_ids).long().to(DEV)
                logits = surgery_forward(m, a, b,
                                         all_a_dev[img_t],
                                         all_b_dev[img_t])
            LOSS(logits, y).backward()
            if step <= 20:
                m.pos.grad[2].zero_()
            opt.step()

        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                for name in ARMS:
                    A = arms[name]
                    m = A['m']
                    m.eval()
                    lp = float((m(va[0], va[1]).argmax(-1) == va[2])
                               .float().mean())
                    A['t'].append(step)
                    A['va_plain'].append(lp)
                    if lp > 0.95 and A['cross95'] is None:
                        A['cross95'] = step
                    if lp > 0.9 and A['cross90'] is None:
                        A['cross90'] = step
                    if name == 'permBS':
                        r2v = sum_perm(va[2].cpu(), genBS)
                        lgv, _ = permbs_forward(m, va[0], va[1], r2v)
                        vs = float((lgv.argmax(-1) == va[2])
                                   .float().mean())
                        A['va_surg'] = A.get('va_surg', []) + \
                            [{'t': step, 'P': vs}]
                    elif name != 'natS':
                        src = src_fixed if name == 'fixedWS' \
                            else src_res
                        si = torch.from_numpy(
                            src[sub_ids]).long().to(DEV)
                        lgt = surgery_forward(m, sub_a, sub_b,
                                              all_a_dev[si],
                                              all_b_dev[si])
                        A['train_surg'].append(
                            {'t': step,
                             'acc': float((lgt.argmax(-1) == sub_y)
                                          .float().mean())})
            for name in ARMS:
                arms[name]['m'].train()
        # plain strat every 500 (+50/200): PRIMARY structured readout
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                rec = {}
                for name in ARMS:
                    A = arms[name]
                    A['m'].eval()
                    lg, _ = plain_forward_attn(A['m'], ga_dev,
                                               gb_dev)
                    rec[name] = _strat_stats(
                        lg.argmax(-1).cpu() == gy, tr_mask_g,
                        tr_mask_swap_g, diag_grid)
                strat_plain[step] = rec
            for name in ARMS:
                arms[name]['m'].train()
        if step % 1000 == 0:
            last = strat_plain[sorted(strat_plain.keys())[-1]]
            line = []
            for name in ARMS:
                A = arms[name]
                m1s = last[name]['m1']
                line.append(f"{name}:{A['cross95'] or '-'}/"
                            f"{A['va_plain'][-1]:.2f}/m1={m1s:.2f}")
            print(f"  [seed{seed} t={step}] " + " ".join(line),
                  flush=True)

    # fixedWS constancy: pi_0 hash recompute (src_fixed untouched)
    assert pi_hash(pi0) == h0
    res = {'gate': {'d_logit': d_logit, 'worst_cos': worst_cos,
                    'worst_rel': worst_rel},
           'U_size': n_fp, 'pi0_hash': h0, 'arms': {},
           'strat_plain': strat_plain}
    for name in ARMS:
        A = arms[name]
        d = {'t': A['t'], 'va_plain': A['va_plain'],
             'cross95': A['cross95'], 'cross90': A['cross90'],
             'train_surg': A['train_surg']}
        if name == 'permBS':
            d['va_surg'] = A.get('va_surg', [])
        if name in ('fixedWS', 'resampWS'):
            d['g3_fail'] = A['g3_fail']
            d['n_draws'] = A['n_draws']
            d['hash_head'] = [h[:16] for h in A['hashes'][:5]]
        res['arms'][name] = d
    print(f"[seed{seed} done in {time.time() - t0:.0f}s]", flush=True)
    return res


def classify_arm(out, A, seed):
    """Locked §6cz classification (plain primary)."""
    ts = sorted(int(k) for k in out[seed]['strat_plain'])
    win = [t for t in ts if 10000 <= t <= 14000]
    m1_win = [out[seed]['strat_plain'][t][A]['m1'] for t in win]
    m1_last = [out[seed]['strat_plain'][t][A]['m1']
               for t in ts[-3:]]
    tsurg = A and out[seed]['arms'][A]['train_surg']
    acc_f = tsurg[-1]['acc'] if tsurg else None
    t95 = out[seed]['arms'][A]['cross95']
    if A == 'natS':
        return ('anchor', t95)
    if acc_f is not None and acc_f < 0.99:
        return ('D-pathology', t95)
    if t95 is not None and t95 <= 5000:
        return ('B-fast', t95)
    structured = (acc_f is not None and acc_f >= 0.99
                  and len(m1_win) > 0 and all(m >= 0.9 for m in m1_win)
                  and (t95 is None or t95 > 8000))
    if structured:
        return ('A-structured-slow', t95)
    if acc_f is not None and acc_f >= 0.99 and \
            (sum(m1_last) / len(m1_last)) < 0.5:
        return ('C-memorization', t95)
    return ('intermediate/partial', t95)


def main():
    out = {}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_r92.pkl')
    for seed in SEEDS:
        print(f'=== seed{seed} ===', flush=True)
        out[seed] = run_seed(seed)
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)

    print('\n=== summary ===', flush=True)
    all_pass = True
    cells = {}
    for seed in SEEDS:
        r = out[seed]
        gid = (r['gate']['d_logit'] <= 1e-6
               and r['gate']['worst_cos'] >= 0.99999
               and r['gate']['worst_rel'] <= 1e-4)
        all_pass &= gid
        assert r['arms']['fixedWS']['g3_fail'] == 0
        assert r['arms']['resampWS']['g3_fail'] == 0
        row = {n: classify_arm(out, n, seed) for n in ARMS}
        cells[seed] = (row['fixedWS'][0], row['resampWS'][0])
        ns = r['arms']['natS']
        if not SMOKE:
            all_pass &= (ns['cross95'] is not None
                         and 9000 <= ns['cross95'] <= 14000)
        print(f"seed{seed}: " + " ".join(
            f"{n}={row[n][0]}(T95={row[n][1]})" for n in ARMS)
            + f" | G-ID ok={gid} U={r['U_size']}", flush=True)
    if not SMOKE:
        with open(os.path.join(REPRO_DIR,
                               'r85_revisit_identity_results.pkl'),
                  'rb') as f:
            r85 = pickle.load(f)
        for seed in (0, 1):
            ours = out[seed]['arms']['natS']
            ref = r85[seed]['arms']['natS']
            n = len(ref['t'])
            same_t = ours['t'][:n] == ref['t']
            d = max((abs(x - y) for x, y in
                     zip(ours['va_plain'][:n], ref['val_acc'])),
                    default=0.0)
            ok = same_t and d == 0.0
            print(f"G-natS seed{seed}: bitwise={ok} (n={n} "
                  f"maxdva={d:.2e})", flush=True)
            all_pass &= ok
        if len(set(cells.values())) == 1:
            print(f"RESULT TREE CELL: {cells[0]}", flush=True)
        else:
            print(f"RESULT TREE: heterogeneous {cells}", flush=True)
    print('ALL GATES PASS' if all_pass else 'GATE FAILURE', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results_r92.pkl', flush=True)


if __name__ == '__main__':
    main()
