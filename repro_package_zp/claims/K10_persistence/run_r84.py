"""K10 run_r84.py — persistence missing cell: per-step resampled
swap-equivariant wrong routing vs fixed (R80 swapfix, rerun bitwise).

Package port of repro/r84_persistence_missing_cell.py (M2.5,
2026-09-02).  Verbatim except: common/ imports, repro paths via
REPRO_DIR (G2' continuity reads the ARCHIVED r80 pkl), claim-dir
output, SMOKE env R84_SMOKE -> K10_SMOKE.

Harness fact (documented §6bx, load-bearing): in the replay family
(swapfix/swapres) the surgery replaces row-2 realized attention with
the DETACHED donor A2 row; by the R62 architecture identity the
replaced scores are discarded, so Wq/Wk gradients are exactly zero and
QK stays at init (wd_0011 no-decay) as an EFFECT -- not an explicit
freeze.  swapfix vs swapres differ ONLY in pi_t == pi vs pi_t ~
P_equiv per step (dedicated stream 4000+seed).

Samplers: build_swap_pi (R80 loop, stream 3000+seed) for swapfix;
build_swap_pi_vec (vectorized, IDENTICAL RNG consumption order --
smoke-gated bitwise for seeds 0..3) for the fixed eval bank (K=8,
stream 5000+seed) and the per-step swapres draws.

Claim (ledger K10, part 1): within the equivariant family, removing
cross-revisit assignment persistence eliminates the long delay
(swapres T_cross^bank fast vs swapfix plateau to CAP).

Note: K10 also covers R86 lifetime dose (run_r86.py) -- ported
separately; this file is the r84 half.
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
from common.model import D57Model, make_optimizer

task.set_group('zp', 113)
SMOKE = bool(os.environ.get('K10_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
EVAL_EVERY = 25
EVERY = 5
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else (0, 1)
GATE_NATS = {0: 13825, 1: 9300}
BANK_K = 8

P = task.GROUP["p"]
N_PAIRS = task.N_PAIRS
N_TRAIN = int(round(N_PAIRS * 0.30))
N_PAIRS = task.N_PAIRS
IDX = np.arange(N_PAIRS, dtype=np.int64)
S_ARR = (IDX % P) * P + IDX // P
DIAG_IDX = IDX[S_ARR == IDX]
OFF_IDX = IDX[S_ARR != IDX]
REP_IDX = np.unique(np.minimum(OFF_IDX, S_ARR[OFF_IDX]))
assert DIAG_IDX.size == P and REP_IDX.size == OFF_IDX.size // 2
ARMS = ('natS', 'swapfix', 'swapres')
ROWS_OWN = {0: 'own', 1: 'own', 2: 'own'}
ROWS_REPLAY = {0: 'own', 1: 'own', 2: 'replay'}

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K10] K10_SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')
print(f'[K10-r84] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} CAP={CAP} '
      f'BATCH={BATCH} threads={torch.get_num_threads()}', flush=True)


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


def val_probe(seed):
    """Returns (a, b, y on DEV, pair-id tensor on CPU)."""
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    rngv = np.random.default_rng(555)
    vb = rngv.permutation(perm[N_TRAIN:])[:1024]
    return (torch.from_numpy(all_a[vb]).long().to(DEV),
            torch.from_numpy(all_b[vb]).long().to(DEV),
            torch.from_numpy(all_y[vb]).long().to(DEV),
            torch.from_numpy(np.asarray(vb)).long())


def routed_forward2(m, a, b, rows, Wq0, Wk0, donor_A2=None):
    """arch='abeq' forward with per-row attention surgery (R71/R72/R74
    verbatim)."""
    B = a.shape[0]
    H, dh, dm = m.n_heads, m.d_head, m.d_model
    eq = torch.full_like(a, task.N_EQ)
    tok = torch.stack([a, b, eq], dim=1)
    x = m.emb(tok) + m.pos[None, :, :]
    h = m.ln1(x)
    q_t = m.Wq(h).view(B, 3, H, dh).transpose(1, 2)
    k_t = m.Wk(h).view(B, 3, H, dh).transpose(1, 2)
    scores = (q_t @ k_t.transpose(-1, -2)) / math.sqrt(dh)
    for r in (0, 1, 2):
        if rows[r] != 'frozen':
            continue
        q_s = F.linear(h, Wq0).view(B, 3, H, dh).transpose(1, 2)
        k_s = F.linear(h, Wk0).view(B, 3, H, dh).transpose(1, 2)
        scores[:, :, r, :] = torch.einsum(
            'bhd,bhjd->bhj', q_s[:, :, r, :], k_s) / math.sqrt(dh)
    attn = scores.softmax(-1)
    if rows[2] == 'replay':
        attn = torch.cat([attn[:, :, :2, :], donor_A2.unsqueeze(2)],
                         dim=2)
    out = (attn @ m.Wv(h).view(B, 3, H, dh).transpose(1, 2))
    out = out.transpose(1, 2).contiguous().view(B, 3, dm)
    x_at = x + m.Wo(out)
    h2 = m.ln2(x_at)
    x_fin = x_at + m.mlp2(F.gelu(m.mlp1(h2)))
    logits = m.Wu(m.ln_f(x_fin[:, 2, :]))
    return logits, {'attn': attn.detach()}


def grid_masks():
    p = task.GROUP['p']
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    y = (a + b) % p
    return a, b, y


def _grid_y():
    return grid_masks()[2]


def _strat_stats(correct, self_in, swap_in, diag):
    """Correct swap semantics (R72; grid i = a*p+b, table id = b*p+a)."""
    m1 = ~self_in & swap_in & ~diag
    m2 = ~self_in & ~swap_in & ~diag
    m3 = ~self_in & diag
    return {'train_acc': float(correct[self_in].float().mean()),
            'va_grid': float(correct[~self_in].float().mean()),
            'm1': float(correct[m1].float().mean()),
            'm2': float(correct[m2].float().mean()),
            'm3': float(correct[m3].float().mean())}


@torch.no_grad()
def strat_probe(m, Wq0, Wk0, rows, donor_A2, self_in, swap_in, diag,
                ga, gb):
    lg, _ = routed_forward2(m, ga.long().to(DEV), gb.long().to(DEV),
                            rows, Wq0, Wk0, donor_A2=donor_A2)
    correct = (lg.argmax(-1).cpu() == _grid_y())
    return _strat_stats(correct, self_in, swap_in, diag)


def build_swap_pi(seed):
    """R80 constructor, VERBATIM (loop form; dedicated numpy RNG
    3000+seed, drawn once).  Used only by the swapfix arm."""
    idx = np.arange(N_PAIRS, dtype=np.int64)
    diag = DIAG_IDX
    reps = REP_IDX
    rng = np.random.default_rng(3000 + seed)
    pi = np.empty(N_PAIRS, dtype=np.int64)
    r = rng.permutation(diag)
    pi[r] = np.roll(r, 1)
    r = rng.permutation(reps)
    img = np.roll(r, 1)
    orient = rng.integers(0, 2, size=reps.size)
    for k in range(reps.size):
        i, j = int(r[k]), int(img[k])
        if orient[k] == 0:
            pi[i], pi[S_ARR[i]] = j, S_ARR[j]
        else:
            pi[i], pi[S_ARR[i]] = S_ARR[j], j
    ok_fp = int((pi == idx).sum())
    ok_bij = int(np.unique(pi).size) == N_PAIRS
    ok_equiv = bool((pi[S_ARR] == S_ARR[pi]).all())
    ok_no_swap_self = bool(np.all(pi[OFF_IDX] != S_ARR[OFF_IDX]))
    all_a, all_b, all_y = task.build_tables()
    y_same_frac = float((np.asarray(all_y)[pi] == np.asarray(all_y))
                        .mean())
    return pi, ok_fp, ok_bij, ok_equiv, ok_no_swap_self, y_same_frac


def build_swap_pi_vec(rng):
    """Vectorized form of build_swap_pi with IDENTICAL RNG consumption
    order => bitwise-equal draws for the same generator state
    (smoke-gated for seeds 0..3)."""
    pi = np.empty(N_PAIRS, dtype=np.int64)
    r = rng.permutation(DIAG_IDX)
    pi[r] = np.roll(r, 1)
    r = rng.permutation(REP_IDX)
    img = np.roll(r, 1)
    orient = rng.integers(0, 2, size=REP_IDX.size)
    img_sw = S_ARR[img]
    pi[r] = np.where(orient == 0, img, img_sw)
    pi[S_ARR[r]] = np.where(orient == 0, img_sw, img)
    return pi


def pi_checks(pi):
    """(fixed_points, bijective, equivariant, no_swap_self)."""
    fp = int((pi == IDX).sum())
    bij = bool(np.bincount(pi, minlength=N_PAIRS).max() == 1)
    eqv = bool((pi[S_ARR] == S_ARR[pi]).all())
    nos = bool(np.all(pi[OFF_IDX] != S_ARR[OFF_IDX]))
    return fp, bij, eqv, nos


def pi_hash(pi):
    return hashlib.sha256(pi.tobytes()).hexdigest()


# ---- smoke gate G3'': vectorized == loop constructor, bitwise ----
def sampler_equivalence_gate():
    ok_all = True
    for s in range(4):
        pi_loop, fp0, bij0, eq0, nos0, _ = build_swap_pi(s)
        pi_vec = build_swap_pi_vec(np.random.default_rng(3000 + s))
        ok = bool((pi_loop == pi_vec).all())
        fp, bij, eqv, nos = pi_checks(pi_vec)
        ok &= (fp == 0) and bij and eqv and nos
        print(f"G3'' sampler equivalence seed{s}: bitwise={ok} "
              f"checks=({fp},{bij},{eqv},{nos})", flush=True)
        ok_all &= ok
    return ok_all


def run_seed(seed):
    t0 = time.time()
    va_a, va_b, va_y, id_va = val_probe(seed)
    va = (va_a, va_b, va_y)
    all_a, all_b, all_y_np = task.build_tables()
    y_np = np.asarray(all_y_np)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    tr_set = set(perm[:N_TRAIN].tolist())
    tr_a = torch.from_numpy(all_a[perm[:N_TRAIN]]).long()
    tr_b = torch.from_numpy(all_b[perm[:N_TRAIN]]).long()
    tr_y = torch.from_numpy(all_y_np[perm[:N_TRAIN]]).long()
    tr_mask = torch.zeros(task.N_PAIRS, dtype=torch.bool)
    tr_mask[torch.tensor(sorted(tr_set))] = True
    id_train = torch.from_numpy(
        np.asarray(perm[:N_TRAIN], dtype=np.int64))

    ga, gb, gy = grid_masks()
    diag_grid = (torch.arange(P).repeat_interleave(P) ==
                 torch.arange(P).repeat(P))
    self_in_grid = tr_mask[gb * P + ga]
    swap_in_grid = tr_mask.clone()

    # grid-index -> table-id (transpose convention, R72 _strat_stats)
    gidx = torch.arange(P * P)
    table_ids_grid = ((gidx % P) * P + (gidx // P)).numpy()

    # ---- fixed equivariant pi (swapfix arm, R80 verbatim) ----
    pi_fix, ok_fp, ok_bij, ok_equiv, ok_noswap, y_same = \
        build_swap_pi(seed)
    pi_fix_dev = torch.from_numpy(pi_fix).long().to(DEV)
    g3 = {'fixed_points': ok_fp, 'bijective': ok_bij,
          'equivariant': ok_equiv, 'no_swap_self': ok_noswap,
          'y_same_frac': y_same}
    print(f"G4 swapfix pi checks: fixed_points={ok_fp} "
          f"bijective={ok_bij} equivariant={ok_equiv} "
          f"no_swap_self={ok_noswap} y_same_frac={y_same:.4f}",
          flush=True)

    # ---- fixed eval bank (K=8, stream 5000+seed, drawn once) ----
    bank_rng = np.random.default_rng(5000 + seed)
    bank = [build_swap_pi_vec(bank_rng) for _ in range(BANK_K)]
    bank_dev = [torch.from_numpy(b).long().to(DEV) for b in bank]
    bank_checks = [pi_checks(b) for b in bank]
    bank_hashes = [pi_hash(b) for b in bank]
    bank_y_same = [float((y_np[b] == y_np).mean()) for b in bank]
    bank_ok = all(fp == 0 and bij and eqv and nos
                  for fp, bij, eqv, nos in bank_checks)
    print(f"eval bank K={BANK_K}: all_checks_pass={bank_ok} "
          f"y_same range=[{min(bank_y_same):.4f},"
          f"{max(bank_y_same):.4f}]", flush=True)
    assert bank_ok, 'eval bank member failed equivariance checks'

    all_a_dev = torch.from_numpy(all_a).long().to(DEV)
    all_b_dev = torch.from_numpy(all_b).long().to(DEV)

    arms = {}
    for name in ARMS:
        m = fresh_model(seed)
        Wq0 = m.Wq.weight.detach().clone()
        Wk0 = m.Wk.weight.detach().clone()
        opt, _, _ = make_optimizer(m, 'wd_0011')
        arms[name] = {'m': m, 'opt': opt, 'Wq0': Wq0, 'Wk0': Wk0,
                      't': [], 'va_surg': [], 'va_plain': [],
                      'cross95_surg': None, 'cross90_plain': None,
                      'w2': {}, 'inj_stat': {}}
    # swapres-only records
    res_rec = {'va_surg_bank_draws': [], 'va_surg_curpi': [],
               'bank_sd': []}

    # per-step resample stream + audit
    rng_res = np.random.default_rng(4000 + seed)
    res_hashes = []
    res_y_same = []
    ov_all = ov_diag = ov_off = 0
    pi_prev = None
    g3_steps_fail = 0

    def donor_image_pi(mN, ids, pi_arr):
        """no_grad donor forward on the pi-image batch -> A2 rows."""
        img_ids = pi_arr[ids.to(DEV)]
        a_img = all_a_dev[img_ids]
        b_img = all_b_dev[img_ids]
        _, ex = routed_forward2(mN, a_img, b_img, ROWS_OWN,
                                arms['natS']['Wq0'],
                                arms['natS']['Wk0'])
        return ex['attn'][:, :, 2, :]

    gate0 = None
    strat = {}
    for step in range(1, CAP + 1):
        # ---- per-step equivariant assignment (swapres) ----
        pi_t = build_swap_pi_vec(rng_res)
        fp, bij, eqv, nos = pi_checks(pi_t)
        if not (fp == 0 and bij and eqv and nos):
            g3_steps_fail += 1
        res_hashes.append(pi_hash(pi_t))
        res_y_same.append(float((y_np[pi_t] == y_np).mean()))
        if pi_prev is not None:
            same = (pi_t == pi_prev)
            ov_all += int(same.sum())
            ov_diag += int(same[DIAG_IDX].sum())
            ov_off += int(same[OFF_IDX].sum())
        pi_prev = pi_t
        pi_t_dev = torch.from_numpy(pi_t).long().to(DEV)

        idx = torch.randint(tr_a.shape[0], (BATCH,))
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        ids = id_train[idx]
        mN = arms['natS']['m']
        with torch.no_grad():
            inj_fix = donor_image_pi(mN, ids, pi_fix_dev)
            inj_res = donor_image_pi(mN, ids, pi_t_dev)
        donor_A2 = None
        # ---- phase 1: forward + backward for all arms at theta_t ----
        for name in ARMS:
            A = arms[name]
            m, opt = A['m'], A['opt']
            m.train()
            opt.zero_grad()
            rows = ROWS_OWN
            if name == 'natS':
                logits, extras = routed_forward2(
                    m, a, b, ROWS_OWN, A['Wq0'], A['Wk0'])
                if step == 1:
                    gate0 = float((logits - m(a, b)).abs().max()
                                  .detach())
                donor_A2 = extras['attn'][:, :, 2, :].detach().clone()
            elif name == 'swapfix':
                logits, extras = routed_forward2(
                    m, a, b, ROWS_REPLAY, A['Wq0'], A['Wk0'],
                    donor_A2=inj_fix)
            elif name == 'swapres':
                logits, extras = routed_forward2(
                    m, a, b, ROWS_REPLAY, A['Wq0'], A['Wk0'],
                    donor_A2=inj_res)
            if step == 1 or step % 1000 == 0:
                inj = inj_fix if name == 'swapfix' else (
                    inj_res if name == 'swapres' else None)
                if inj is not None:
                    A['inj_stat'][step] = {
                        'mean': float((inj - donor_A2).abs().mean()),
                        'max': float((inj - donor_A2).abs().max())}
            if step % EVERY == 0 or step == 1:
                A['w2'][step] = (extras['attn'][:, :, 2, :]
                                 .mean(dim=0).cpu().numpy())
            LOSS(logits, y).backward()
        # ---- phase 2: step all arms (fp2<=20 convention) ----
        for name in ARMS:
            A = arms[name]
            if step <= 20:
                A['m'].pos.grad[2].zero_()
            A['opt'].step()
        # ---- eval (fixed bank + plain), post-update ----
        if step % EVAL_EVERY == 0 or step == CAP:
            with torch.no_grad():
                for name in ARMS:
                    A = arms[name]
                    A['m'].eval()
                    if name == 'natS':
                        lg, _ = routed_forward2(
                            A['m'], va[0], va[1], ROWS_OWN,
                            A['Wq0'], A['Wk0'])
                        vs = float((lg.argmax(-1) == va[2])
                                   .float().mean())
                    elif name == 'swapfix':
                        inj = donor_image_pi(mN, id_va, pi_fix_dev)
                        lg, _ = routed_forward2(
                            A['m'], va[0], va[1], ROWS_REPLAY,
                            A['Wq0'], A['Wk0'], donor_A2=inj)
                        vs = float((lg.argmax(-1) == va[2])
                                   .float().mean())
                    else:  # swapres: bank mean + per-draw + cur-pi
                        draws = []
                        for k in range(BANK_K):
                            inj = donor_image_pi(mN, id_va,
                                                 bank_dev[k])
                            lg, _ = routed_forward2(
                                A['m'], va[0], va[1], ROWS_REPLAY,
                                A['Wq0'], A['Wk0'], donor_A2=inj)
                            draws.append(float(
                                (lg.argmax(-1) == va[2])
                                .float().mean()))
                        inj = donor_image_pi(mN, id_va, pi_t_dev)
                        lg, _ = routed_forward2(
                            A['m'], va[0], va[1], ROWS_REPLAY,
                            A['Wq0'], A['Wk0'], donor_A2=inj)
                        res_rec['va_surg_curpi'].append(float(
                            (lg.argmax(-1) == va[2]).float().mean()))
                        vs = float(np.mean(draws))
                        res_rec['va_surg_bank_draws'].append(draws)
                        res_rec['bank_sd'].append(
                            float(np.std(draws)))
                    lp = float((A['m'](va[0], va[1]).argmax(-1)
                                == va[2]).float().mean())
                    A['t'].append(step)
                    A['va_surg'].append(vs)
                    A['va_plain'].append(lp)
                    if vs > 0.95 and A['cross95_surg'] is None:
                        A['cross95_surg'] = step
                    if lp > 0.9 and A['cross90_plain'] is None:
                        A['cross90_plain'] = step
        # ---- stratified probe every 500 (+ steps 50/200) ----
        if step % 500 == 0 or step in (50, 200):
            with torch.no_grad():
                rec = {}
                for name in ARMS:
                    A = arms[name]
                    A['m'].eval()
                    if name == 'natS':
                        lg_g, _ = routed_forward2(
                            A['m'], ga.long().to(DEV),
                            gb.long().to(DEV), ROWS_OWN,
                            A['Wq0'], A['Wk0'])
                        rec[name] = _strat_stats(
                            lg_g.argmax(-1).cpu() == gy,
                            self_in_grid, swap_in_grid, diag_grid)
                    elif name == 'swapfix':
                        inj_g = donor_image_pi(mN, torch.from_numpy(
                            table_ids_grid), pi_fix_dev)
                        rec[name] = strat_probe(
                            A['m'], A['Wq0'], A['Wk0'], ROWS_REPLAY,
                            inj_g, self_in_grid, swap_in_grid,
                            diag_grid, ga, gb)
                    else:  # swapres: bank-averaged surgery-active
                        keys = ('train_acc', 'va_grid', 'm1', 'm2',
                                'm3')
                        acc = {k: 0.0 for k in keys}
                        for k in range(BANK_K):
                            inj_g = donor_image_pi(
                                mN, torch.from_numpy(table_ids_grid),
                                bank_dev[k])
                            st = strat_probe(
                                A['m'], A['Wq0'], A['Wk0'],
                                ROWS_REPLAY, inj_g, self_in_grid,
                                swap_in_grid, diag_grid, ga, gb)
                            for kk in keys:
                                acc[kk] += st[kk] / BANK_K
                        rec[name] = acc
                strat[step] = rec
        if step % 1000 == 0:
            live = {n: (arms[n]['cross95_surg'],
                        arms[n]['va_surg'][-1]) for n in ARMS}
            print(f"  [seed{seed} t={step}] " +
                  " ".join(f"{n}:{(c if c else '-')}/{v:.2f}"
                           for n, (c, v) in live.items()), flush=True)

    # G4 end-of-run swapfix pi constancy check
    pi2, ok_fp2, _, _, _, _ = build_swap_pi(seed)
    g3['constant'] = bool((pi2 == pi_fix).all())

    # overlap statistics (empirical vs per-stratum theory, NOT gated)
    n_steps = max(CAP - 1, 1)
    theo = {'diag': 1.0 / (P - 1),
            'off': 1.0 / (2 * (REP_IDX.size - 1)),
            'all': (DIAG_IDX.size / (P - 1) +
                    OFF_IDX.size / (2 * (REP_IDX.size - 1))) / N_PAIRS}
    ov = {'all_emp': ov_all / (n_steps * N_PAIRS),
          'diag_emp': ov_diag / (n_steps * DIAG_IDX.size),
          'off_emp': ov_off / (n_steps * OFF_IDX.size),
          'theo': theo}

    res = {'gate0': gate0, 'g3': g3, 'g3_steps_fail': g3_steps_fail,
           'bank_checks': bank_checks, 'bank_hashes': bank_hashes,
           'bank_y_same': bank_y_same, 'res_hashes': res_hashes,
           'res_y_same': res_y_same, 'overlap': ov,
           'arms': {}, 'strat': strat}
    for name in ARMS:
        A = arms[name]
        res['arms'][name] = {
            't': A['t'], 'va_surg': A['va_surg'],
            'va_plain': A['va_plain'],
            'cross95_surg': A['cross95_surg'],
            'cross90_plain': A['cross90_plain'],
            'w2': {k: (v if isinstance(v, np.ndarray) else
                       v.cpu().numpy())
                   for k, v in A['w2'].items()},
            'inj_stat': A['inj_stat']}
    res['arms']['swapfix']['pi'] = pi_fix
    res['arms']['swapres'].update(res_rec)
    res['pi_fix'] = pi_fix
    res['bank'] = bank
    print(f"[seed{seed} done in {time.time() - t0:.0f}s] "
          f"g3_steps_fail={g3_steps_fail} "
          f"overlap all={ov['all_emp']:.2e} "
          f"(theo {theo['all']:.2e}) "
          f"diag={ov['diag_emp']:.2e} (theo {theo['diag']:.2e}) "
          f"off={ov['off_emp']:.2e} (theo {theo['off']:.2e})",
          flush=True)
    return res


def continuity_gate(out):
    """G2': natS + swapfix bitwise vs ARCHIVED R80 pkl (full-run)."""
    with open(os.path.join(REPRO_DIR,
                           'r80_swap_fixedpi_results.pkl'),
              'rb') as f:
        r80 = pickle.load(f)
    ok_all = True
    for seed in SEEDS:
        for arm in ('natS', 'swapfix'):
            a_new = out[seed]['arms'][arm]
            a_old = r80[seed]['arms'][arm]
            same_t = a_new['t'] == a_old['t']
            d_vs = max((abs(x - y) for x, y in
                        zip(a_new['va_surg'], a_old['va_surg'])),
                       default=0.0)
            d_vp = max((abs(x - y) for x, y in
                        zip(a_new['va_plain'], a_old['va_plain'])),
                       default=0.0)
            same_c95 = a_new['cross95_surg'] == a_old['cross95_surg']
            same_c90 = a_new['cross90_plain'] == a_old['cross90_plain']
            ok = same_t and d_vs == 0.0 and d_vp == 0.0 and \
                same_c95 and same_c90
            print(f"G2' seed{seed} {arm}: bitwise={ok} "
                  f"(t={same_t} maxdv_surg={d_vs:.2e} "
                  f"maxdv_plain={d_vp:.2e} c95={same_c95} "
                  f"c90={same_c90})", flush=True)
            ok_all &= ok
    return ok_all


def main():
    out = {}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_r84.pkl')
    for seed in SEEDS:
        print(f'=== seed{seed} ===', flush=True)
        out[seed] = run_seed(seed)
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)

    print('\n=== summary ===', flush=True)
    all_pass = True
    for seed in SEEDS:
        r = out[seed]
        c_n = r['arms']['natS']['cross95_surg']
        c_f = r['arms']['swapfix']['cross95_surg']
        c_r = r['arms']['swapres']['cross95_surg']
        g1 = (c_n == GATE_NATS[seed])
        g4ok = all([r['g3']['bijective'], r['g3']['equivariant'],
                    r['g3']['no_swap_self'], r['g3']['constant'],
                    r['g3']['fixed_points'] == 0])
        g3ok = r['g3_steps_fail'] == 0
        if SMOKE:
            g1 = 'skipped(smoke)'
        all_pass &= (g1 is True or g1 == 'skipped(smoke)') and \
            g4ok and g3ok and \
            (r['gate0'] is not None and r['gate0'] <= 1e-5)
        sr = r['arms']['swapres']
        strat_final = r['strat'].get(max(r['strat'].keys())) \
            if r['strat'] else {}
        sr_m2 = strat_final.get('swapres', {}).get('m2')
        print(f"seed{seed}: natS={c_n} swapfix={c_f} swapres={c_r} | "
              f"G0={r['gate0']:.2e} G1={g1} G3'={g3ok} G4={g4ok} | "
              f"swapres final strat m2={sr_m2} "
              f"va_plain cross90={sr['cross90_plain']}", flush=True)
        print(f"  swapres inj@1={sr['inj_stat'].get(1)}", flush=True)
    if not SMOKE:
        g2 = continuity_gate(out)
        all_pass &= g2
    else:
        eq = sampler_equivalence_gate()
        all_pass &= eq
    print('ALL GATES PASS' if all_pass else 'GATE FAILURE', flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results_r84.pkl', flush=True)


if __name__ == '__main__':
    main()
