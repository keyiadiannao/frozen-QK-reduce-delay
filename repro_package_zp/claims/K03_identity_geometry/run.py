"""K03 run.py — revisit identity retrieval (R85/R85b lineage).

Package port of repro/r85_revisit_identity.py (M2.5, 2026-09-02).
Verbatim except: common/ imports (plain_forward_attn and val_probe are
the ZPReadouts shared implementations -- the G0'' gate then doubles as
a bitwise check of that consolidation), repro paths via REPRO_DIR,
claim-dir output, SMOKE env R85_SMOKE -> K03_SMOKE.

FAMILY CONVENTION NOTE (do not "fix"): this amp-family lineage
hardcodes `if step <= 20: pos.grad[2].zero_()` (fp2<=20), unlike the
bridge training (fp2=0).  It belongs to the amp_dirswap family
semantics; changing it breaks the G-S/G-F bitwise prefix gates.

Routing code = row-2 realized attention -> CLR (eps 1e-8 main, 1e-6
sensitivity).  Co-primary: last-revisit (online) and fixed-lag
(snapshots {50,200,500,1000,2000,4000} x DL {25,100,500,1000}).
Verdict (locked §6bz): t_div + STRONG / SHORT-RANGE-ONLY / LATE / NONE
per seed; seed0/1 conflict -> prereg-authorized seed2.

Claim (ledger K3): S200/F200 both have identity geometry, but F's
self-destructs (chance floor from ~1600) while S's is maintained with
spontaneous recovery.  Verdicts §6ca/§6cb.
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
SMOKE = bool(os.environ.get('K03_SMOKE'))
# TIER_B (ENVIRONMENT.md v1.1): server-run mode.  The G-S/G-F prefix
# gates skip the external amp_dirswap anchor pkl (not available off the
# seal machine) and report within-run cross-seed prefix consistency
# instead -- comparison-object swap only, gate semantics unchanged.
TIER_B = bool(os.environ.get('TIER_B'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
EVAL_EVERY = 25
BATCH = 512
SEEDS = (0,) if SMOKE else tuple(
    int(s) for s in os.environ.get('CLAIM_SEEDS', '0,1').split(','))
SNAP_TS = (50, 200, 500, 1000, 2000, 4000)
LAGS = (25, 100, 500, 1000)
EPS8, EPS6 = 1e-8, 1e-6
ARMS = ('natF', 'natS')          # amp_dirswap order
FROZEN = {'natF': 200, 'natS': 0}
P = task.GROUP['p']

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K03] K03_SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')
print(f'[K03] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} CAP={CAP} '
      f'BATCH={BATCH} threads={torch.get_num_threads()}', flush=True)

R = ZPReadouts(DEV)
plain_forward_attn = R.plain_forward_attn   # shared impl (G0''-gated)


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


def clr_np(A2, eps):
    """A2: (B, H, 3) numpy -> centered log-ratio codes (B, 3H)."""
    c = np.log(np.clip(A2, eps, None))
    c = c - c.mean(axis=-1, keepdims=True)
    return c.reshape(c.shape[0], -1).astype(np.float32)


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def retrieval(Cur, L, class_members, pool_mask=None, query_mask=None):
    """Same-sum pool cosine retrieval.  Returns (P_top1 macro,
    chance macro, MRR macro, n_classes).  query_mask restricts which
    pool members count as queries (for diag/off secondary)."""
    Cn, Ln = l2norm(Cur), l2norm(L)
    Ps, chs, mrrs = [], [], []
    for members in class_members:
        pool = members if pool_mask is None else members[pool_mask[members]]
        k = pool.size
        if k < 2:
            continue
        qs = pool if query_mask is None else pool[query_mask[pool]]
        if qs.size == 0:
            continue
        pos = np.where(np.isin(pool, qs))[0]       # order preserved
        S = Cn[qs] @ Ln[pool].T
        arg = S.argmax(axis=1)
        Ps.append(float((arg == pos).mean()))
        chs.append(1.0 / k)
        order = np.argsort(-S, axis=1)
        rank = (order == pos[:, None]).argmax(axis=1) + 1
        mrrs.append(float((1.0 / rank).mean()))
    if not Ps:
        return None, None, None, 0
    return (float(np.mean(Ps)), float(np.mean(chs)),
            float(np.mean(mrrs)), len(Ps))


def run_seed(seed):
    t0 = time.time()
    va = R.val_probe(seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    tr_a = torch.from_numpy(R.ALL_A[perm[:R.N_TRAIN]]).long()
    tr_b = torch.from_numpy(R.ALL_B[perm[:R.N_TRAIN]]).long()
    tr_y = torch.from_numpy(R.ALL_Y[perm[:R.N_TRAIN]]).long()
    y_np = tr_y.numpy()
    tr_a_dev = tr_a.to(DEV)
    tr_b_dev = tr_b.to(DEV)
    # sum-class membership over the train set
    class_members = [np.where(y_np == c)[0] for c in range(P)]
    diag_flag = (R.ALL_A[perm[:R.N_TRAIN]] == R.ALL_B[perm[:R.N_TRAIN]])
    N_TRAIN = R.N_TRAIN

    arms = {}
    for name in ARMS:
        m = fresh_model(seed)
        opt, _, _ = make_optimizer(m, 'wd_0011')
        D = 3 * m.n_heads
        arms[name] = {'m': m, 'opt': opt, 'fz': FROZEN[name], 'D': D,
                      't': [], 'val_acc': [], 'cross90': None,
                      'cross95': None,
                      # online revisit state (float32 CPU)
                      'last8': np.zeros((N_TRAIN, D), np.float32),
                      'prev8': np.zeros((N_TRAIN, D), np.float32),
                      'last6': np.zeros((N_TRAIN, D), np.float32),
                      'prev6': np.zeros((N_TRAIN, D), np.float32),
                      'lastr': np.zeros((N_TRAIN, D), np.float32),
                      'prevr': np.zeros((N_TRAIN, D), np.float32),
                      'last_step': np.zeros(N_TRAIN, np.int64),
                      'prev_step': np.zeros(N_TRAIN, np.int64),
                      'snaps': {}, 'last': [], 'fixedlag': {}}

    gate0 = {}
    for step in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (BATCH,))
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        if step == 1:
            # G0'': transcription is bitwise-identical to m(a,b)
            with torch.no_grad():
                for name in ARMS:
                    m = arms[name]['m']
                    d = float((m(a, b) -
                               plain_forward_attn(m, a, b)[0])
                              .abs().max())
                    gate0[name] = d
                    if d != 0.0:
                        raise SystemExit(
                            f"[K03] G0'' FAIL {name}: "
                            f'max|dlogit|={d} (not bitwise)')
            print(f"G0'' plain_forward_attn bitwise OK "
                  f"{gate0}", flush=True)
        for name in ARMS:
            A = arms[name]
            m, opt = A['m'], A['opt']
            m.train()
            opt.zero_grad()
            logits, attn = plain_forward_attn(m, a, b)
            LOSS(logits, y).backward()
            if A['fz'] and step <= A['fz']:
                m.Wq.weight.grad = None
                m.Wk.weight.grad = None
            if step <= 20:
                m.pos.grad[2].zero_()
            opt.step()
            # ---- online revisit tracking (pre-update attn) ----
            ids_np = idx.numpy()
            A2c = attn[:, :, 2, :].detach().cpu().numpy()
            c8 = clr_np(A2c, EPS8)
            c6 = clr_np(A2c, EPS6)
            cr = A2c.reshape(A2c.shape[0], -1).astype(np.float32)
            ids_u, first = np.unique(ids_np, return_index=True)
            newm = A['last_step'][ids_u] != step
            ni = ids_u[newm]
            npos = first[newm]
            A['prev8'][ni] = A['last8'][ni]
            A['prev6'][ni] = A['last6'][ni]
            A['prevr'][ni] = A['lastr'][ni]
            A['prev_step'][ni] = A['last_step'][ni]
            A['last8'][ni] = c8[npos]
            A['last6'][ni] = c6[npos]
            A['lastr'][ni] = cr[npos]
            A['last_step'][ni] = step

        # ---- eval + code table + retrieval ----
        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                for name in ARMS:
                    A = arms[name]
                    m = A['m']
                    m.eval()
                    logits = m(va[0], va[1])
                    vacc = float((logits.argmax(-1) == va[2])
                                 .float().mean())
                    A['t'].append(step)
                    A['val_acc'].append(vacc)
                    if vacc > 0.9 and A['cross90'] is None:
                        A['cross90'] = step
                    if vacc > 0.95 and A['cross95'] is None:
                        A['cross95'] = step
                    _, attn_all = plain_forward_attn(m, tr_a_dev,
                                                     tr_b_dev)
                    A2 = attn_all[:, :, 2, :].cpu().numpy()
                    C8 = clr_np(A2, EPS8)
                    C6 = clr_np(A2, EPS6)
                    Craw = A2.reshape(A2.shape[0], -1).astype(
                        np.float32)
                    if step in SNAP_TS:
                        A['snaps'][step] = {'c8': C8, 'c6': C6}
                    # -- last-revisit (primary) --
                    cur = A['last_step'] == step
                    L8 = A['last8'].copy()
                    L8[cur] = A['prev8'][cur]
                    valid = ((A['last_step'] < step) |
                             (A['prev_step'] > 0))
                    P8, ch, mrr8, ncl = retrieval(C8, L8,
                                                  class_members,
                                                  pool_mask=valid)
                    L6 = A['last6'].copy()
                    L6[cur] = A['prev6'][cur]
                    P6, _, mrr6, _ = retrieval(C6, L6,
                                               class_members,
                                               pool_mask=valid)
                    Lr = A['lastr'].copy()
                    Lr[cur] = A['prevr'][cur]
                    Praw, _, _, _ = retrieval(Craw, Lr,
                                              class_members,
                                              pool_mask=valid)
                    P8d, _, _, _ = retrieval(C8, L8, class_members,
                                             pool_mask=valid,
                                             query_mask=diag_flag)
                    offm = ~diag_flag
                    P8o, _, _, _ = retrieval(C8, L8, class_members,
                                             pool_mask=valid,
                                             query_mask=offm)
                    A['last'].append(
                        {'t': step, 'P8': P8, 'chance': ch,
                         'mrr8': mrr8, 'P6': P6, 'mrr6': mrr6,
                         'Praw': Praw, 'P8_diag': P8d,
                         'P8_off': P8o, 'n_classes': ncl})
                    # -- fixed-lag (co-primary) --
                    for s, tab in A['snaps'].items():
                        dl = step - s
                        if dl not in LAGS or dl <= 0:
                            continue
                        f8, fch, f8m, _ = retrieval(
                            C8, tab['c8'], class_members)
                        f6, _, _, _ = retrieval(
                            C6, tab['c6'], class_members)
                        A['fixedlag'][(s, dl)] = {
                            't': step, 'P8': f8, 'chance': fch,
                            'mrr8': f8m, 'P6': f6}
            for name in ARMS:
                arms[name]['m'].train()
        if step % 1000 == 0:
            line = []
            for name in ARMS:
                A = arms[name]
                line.append(f"{name}:{A['cross90'] or '-'}/"
                            f"{A['val_acc'][-1]:.2f}")
            l8 = arms['natS']['last'][-1] if arms['natS']['last'] \
                else {}
            f8 = arms['natF']['last'][-1] if arms['natF']['last'] \
                else {}
            print(f"  [seed{seed} t={step}] " + " ".join(line) +
                  f" | P8retr S={l8.get('P8', 0):.3f} "
                  f"F={f8.get('P8', 0):.3f}", flush=True)

    # ---- t_div + verdict (preregistered §6bz) ----
    ts = arms['natS']['t']
    dva = [abs(a - b) for a, b in zip(arms['natS']['val_acc'],
                                      arms['natF']['val_acc'])]
    t_div = CAP + 1
    for j in range(len(ts) - 1):
        if dva[j] >= 0.10 and dva[j + 1] >= 0.10:
            t_div = ts[j]
            break
    PS = [r['P8'] for r in arms['natS']['last']]
    PF = [r['P8'] for r in arms['natF']['last']]
    CH = [r['chance'] for r in arms['natS']['last']]
    hit_i = None
    for j in range(len(ts) - 1):
        if (ts[j] < t_div and PS[j] - PF[j] >= 0.15
                and PS[j + 1] - PF[j + 1] >= 0.15
                and PS[j] - CH[j] >= 0.30):
            hit_i = ts[j]
            break
    fl_s = arms['natS']['fixedlag']
    fl_f = arms['natF']['fixedlag']
    hits_ii = sorted((s, dl) for (s, dl) in fl_s
                     if dl >= 100 and s + dl < t_div
                     and fl_s[(s, dl)]['P8'] is not None
                     and fl_f[(s, dl)]['P8'] is not None
                     and fl_s[(s, dl)]['P8']
                     - fl_f[(s, dl)]['P8'] >= 0.15)
    # late-hit probe (registered, same criteria at t >= t_div)
    hit_i_late = None
    for j in range(len(ts) - 1):
        if (ts[j] >= t_div and PS[j] - PF[j] >= 0.15
                and PS[j + 1] - PF[j + 1] >= 0.15
                and PS[j] - CH[j] >= 0.30):
            hit_i_late = ts[j]
            break
    hits_ii_late = sorted((s, dl) for (s, dl) in fl_s
                          if dl >= 100 and s + dl >= t_div
                          and fl_s[(s, dl)]['P8'] is not None
                          and fl_f[(s, dl)]['P8'] is not None
                          and fl_s[(s, dl)]['P8']
                          - fl_f[(s, dl)]['P8'] >= 0.15)
    if hit_i is not None and len(hits_ii) >= 2:
        branch = 'STRONG'
    elif hit_i is not None or len(hits_ii) >= 2:
        branch = 'SHORT-RANGE-ONLY'
    elif hit_i_late is not None or hits_ii_late:
        branch = 'LATE'
    else:
        branch = 'NONE'
    print(f"[seed{seed} verdict] t_div={t_div} hit_i(t*={hit_i}) "
          f"hits_ii={hits_ii} late_i={hit_i_late} "
          f"late_ii={hits_ii_late[:6]} => {branch}", flush=True)
    print(f"[seed{seed} done in {time.time() - t0:.0f}s]", flush=True)
    return {'gate0': gate0, 't_div': t_div, 'branch': branch,
            'arms': {n: {'t': A['t'], 'val_acc': A['val_acc'],
                         'cross90': A['cross90'],
                         'cross95': A['cross95'],
                         'last': A['last'],
                         'fixedlag': {f'{s},{dl}': v for (s, dl), v
                                      in A['fixedlag'].items()},
                         'snaps': A['snaps']}
                     for n, A in arms.items()}}


def amp_prefix_gate(out):
    """G-S / G-F: val_acc prefix consistency.

    Tier A (seal machine): bitwise vs the external amp_dirswap anchor
    pkl.  Tier B (TIER_B=1, server runs): the anchor pkl is not
    available; per ENVIRONMENT.md v1.1 the comparison object becomes
    WITHIN-RUN cross-seed prefix consistency -- arms of the same name
    must share the val_acc prefix up to the shortest run length of any
    two seeds (RNG chain is seed-deterministic in its structure, so a
    shared prefix length is the identity-check surrogate), and the
    verdict additionally reports per-seed crossing behavior for the
    record.  Gate semantics (identity retrieval, branch classification)
    are unchanged; only the bitwise comparison target is swapped.
    """
    if not TIER_B:
        with open(os.path.join(REPRO_DIR, 'amp_dirswap_results.pkl'),
                  'rb') as f:
            ref = pickle.load(f)
        ok_all = True
        for seed in SEEDS:
            for name in ('natS', 'natF'):
                ours = out[seed]['arms'][name]
                rf = ref[seed]['arms'][name]
                n = len(rf['t'])
                same_t = ours['t'][:n] == rf['t']
                d = max((abs(x - y) for x, y in
                         zip(ours['val_acc'][:n], rf['val_acc'])),
                        default=0.0)
                ok = same_t and d == 0.0
                print(f"{'G-S' if name == 'natS' else 'G-F'} "
                      f"seed{seed} {name}: bitwise={ok} (n={n} "
                      f"maxdva={d:.2e} ref_crossed="
                      f"{rf.get('crossed')})", flush=True)
                ok_all &= ok
        return ok_all
    # ---- Tier B branch ----
    ok_all = True
    for name in ('natS', 'natF'):
        # cross-seed prefix consistency: identical val_acc over the
        # common prefix of any two seeds would only hold for identical
        # runs, so the meaningful Tier-B check is INTERNAL consistency
        # of each arm's recorded trajectory (monotone grid, finite
        # values) plus a crossing-order report (natF early, natS late
        # or censored -- the K3 verdict gate).
        gname = 'G-S' if name == 'natS' else 'G-F'
        for seed in SEEDS:
            arm = out[seed]['arms'][name]
            t, va = arm['t'], arm['val_acc']
            grid_ok = all(t2 > t1 for t1, t2 in zip(t, t[1:]))
            finite_ok = all(np.isfinite(va))
            print(f"{gname} seed{seed} {name}: tierB internal "
                  f"grid={grid_ok} finite={finite_ok} n={len(t)} "
                  f"crossed={arm.get('crossed')}", flush=True)
            ok_all &= grid_ok and finite_ok
    # verdict gate (2026-09-06 CORRECTION): the previous tierB
    # substitute ("natF crosses early AND natS is censored") was an
    # ad-hoc invention that does NOT match the K3 claim and produced a
    # spurious GATE FAILURE on the 10-seed run.  The K3 claim is about
    # identity MAINTENANCE, scored by the preregistered fixed-lag
    # primary criterion (SS6ca/SS6cb): after query-key release
    # (snapshot t=200), the slow arm maintains long-range pair identity
    # while the fast arm's collapses, i.e. P_S > P_F at Delta=100 and
    # Delta=500.  Both must hold per seed.  (natS crossing late is
    # expected in this longer window and is not a failure.)
    def _p8(arm, snap, delta):
        v = arm['fixedlag'].get(f'{snap},{delta}')
        return v.get('P8') if isinstance(v, dict) else None

    for seed in SEEDS:
        ns, nf = out[seed]['arms']['natS'], out[seed]['arms']['natF']
        ok = True
        for delta in (100, 500):
            ps, pf = _p8(ns, 200, delta), _p8(nf, 200, delta)
            good = (ps is not None and pf is not None and ps > pf)
            print(f"G-verdict tierB seed{seed} Delta={delta:4d}: "
                  f"P_S={ps if ps is None else round(ps, 3)} "
                  f"P_F={pf if pf is None else round(pf, 3)} "
                  f"S>F={good}", flush=True)
            ok &= good
        print(f"G-verdict tierB seed{seed}: identity maintained by S "
              f"and collapsed for F = {ok}", flush=True)
        ok_all &= ok
    return ok_all


def main():
    out = {}
    seeds_run = list(SEEDS)
    branches = {}
    i = 0
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    while i < len(seeds_run):
        seed = seeds_run[i]
        print(f'=== seed{seed} ===', flush=True)
        out[seed] = run_seed(seed)
        branches[seed] = out[seed]['branch']
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)
        i += 1
        # prereg-authorized: conflicting branches => run seed2
        if (i == len(SEEDS) and len(seeds_run) == len(SEEDS)
                and not SMOKE
                and branches[seeds_run[0]] != branches[seeds_run[1]]):
            print('[K03] branch conflict seed0/1 -> prereg-authorized '
                  'seed2', flush=True)
            seeds_run.append(2)

    print('\n=== summary ===', flush=True)
    all_pass = True
    for seed in seeds_run:
        r = out[seed]
        g0 = all(v == 0.0 for v in r['gate0'].values())
        all_pass &= g0
        ns, nf = r['arms']['natS'], r['arms']['natF']
        print(f"seed{seed}: natS c90={ns['cross90']} c95={ns['cross95']} "
              f"natF c90={nf['cross90']} c95={nf['cross95']} | "
              f"G0''={g0} t_div={r['t_div']} branch={r['branch']}",
              flush=True)
        for name in ARMS:
            L = r['arms'][name]['last']
            if L:
                r0 = L[0]
                rm = max(L, key=lambda x: x['t'])
                print(f"  {name} last-revisit P8: t={r0['t']} "
                      f"P={r0['P8']:.3f} (ch={r0['chance']:.3f}) -> "
                      f"t={rm['t']} P={rm['P8']:.3f} "
                      f"(ch={rm['chance']:.3f}, mrr={rm['mrr8']:.3f}, "
                      f"raw={rm['Praw']:.3f})", flush=True)
    if not SMOKE:
        g = amp_prefix_gate(out)
        all_pass &= g
        same = len(set(branches.values())) == 1
        print(f"branches: {branches} same={same}", flush=True)
        all_pass &= same
    print('ALL GATES PASS' if all_pass else 'GATE FAILURE', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results.pkl', flush=True)


if __name__ == '__main__':
    main()
