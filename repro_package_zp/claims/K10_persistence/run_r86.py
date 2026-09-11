"""K10 run_r86.py — lifetime dose: does routing-address lifetime L
causally control delay length inside the equivariant family?

Package port of repro/r86_lifetime_dose.py (M2.5, 2026-09-02).
Verbatim except: common/ imports, repro paths via REPRO_DIR (G-C
continuity reads the ARCHIVED r84 pkl), claim-dir output, SMOKE env
R86_SMOKE -> K10R86_SMOKE.

Arms (SEEDS=(0,1), CAP=14000, r84 harness family): natS anchor /
swapfix (L=inf) / swapL1/10/50/200 (block-aligned redraw at
t=1, L+1, 2L+1, ...; dedicated stream 4000+seed PER ARM -- swapL1's
draw sequence is bitwise-identical to r84 swapres).  Fixed eval bank
K=8 (stream 5000+seed) identical to r84.  Gates: G0, G1 (13825/9300),
G-C continuity (natS/swapfix/swapL1 vs archived r84 pkl), per-redraw
V1-V4 + sha256, swapfix constancy.  Readout (locked §6cc):
MONOTONE / THRESHOLD-ABOVE-200 / NON-MONOTONE / CENSORED per seed.

Claim (ledger K10, part 2): lifetime dose response registered --
threshold-like启动 + ρ-不敏感逃逸时钟 (archive: NON-MONOTONE narrow
band 18025-20375 for the R83 extension family; r86 CAP=14000 window).
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
SMOKE = bool(os.environ.get('K10R86_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 120 if SMOKE else 14000
EVAL_EVERY = 25
EVERY = 5
BATCH = 512
SEEDS = (0,) if SMOKE else (0, 1)
GATE_NATS = {0: 13825, 1: 9300}
BANK_K = 8

P = task.GROUP['p']
N_PAIRS = task.N_PAIRS
N_TRAIN = int(round(N_PAIRS * 0.30))
IDX = np.arange(N_PAIRS, dtype=np.int64)
S_ARR = (IDX % P) * P + IDX // P
DIAG_IDX = IDX[S_ARR == IDX]
OFF_IDX = IDX[S_ARR != IDX]
REP_IDX = np.unique(np.minimum(OFF_IDX, S_ARR[OFF_IDX]))
ARMS = ('natS', 'swapfix', 'swapL1', 'swapL10', 'swapL50', 'swapL200')
LMAP = {'swapfix': None, 'swapL1': 1, 'swapL10': 10,
        'swapL50': 50, 'swapL200': 200}
ROWS_OWN = {0: 'own', 1: 'own', 2: 'own'}
ROWS_REPLAY = {0: 'own', 1: 'own', 2: 'replay'}

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K10-r86] SMOKE=1 but CUDA devices visible -- '
                     'aborting before any GPU touch')
print(f'[K10-r86] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} CAP={CAP} '
      f'BATCH={BATCH}', flush=True)


def fresh_model(seed):
    torch.manual_seed(seed)
    return D57Model(pos_mode='zeros', arch='abeq', ln_eps=1e-5,
                    eq_alpha=0.0).to(DEV)


def val_probe(seed):
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
    """arch='abeq' forward with per-row attention surgery (r84
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
    """R80 constructor, VERBATIM (swapfix arm; dedicated RNG 3000+seed,
    drawn once)."""
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
    ok_fp = int((pi == IDX).sum())
    ok_bij = int(np.unique(pi).size) == N_PAIRS
    ok_equiv = bool((pi[S_ARR] == S_ARR[pi]).all())
    ok_no_swap_self = bool(np.all(pi[OFF_IDX] != S_ARR[OFF_IDX]))
    all_a, all_b, all_y = task.build_tables()
    y_same_frac = float((np.asarray(all_y)[pi] == np.asarray(all_y))
                        .mean())
    return pi, ok_fp, ok_bij, ok_equiv, ok_no_swap_self, y_same_frac


def build_swap_pi_vec(rng):
    """Vectorized form; identical RNG consumption order (r84-verified
    bitwise-equal to the loop constructor)."""
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
    fp = int((pi == IDX).sum())
    bij = bool(np.bincount(pi, minlength=N_PAIRS).max() == 1)
    eqv = bool((pi[S_ARR] == S_ARR[pi]).all())
    nos = bool(np.all(pi[OFF_IDX] != S_ARR[OFF_IDX]))
    return fp, bij, eqv, nos


def pi_hash(pi):
    return hashlib.sha256(pi.tobytes()).hexdigest()


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
    gidx = torch.arange(P * P)
    table_ids_grid = ((gidx % P) * P + (gidx // P)).numpy()

    # ---- fixed equivariant pi (swapfix, L=infinity) ----
    pi_fix, ok_fp, ok_bij, ok_equiv, ok_noswap, y_same = \
        build_swap_pi(seed)
    pi_fix_dev = torch.from_numpy(pi_fix).long().to(DEV)
    g3 = {'fixed_points': ok_fp, 'bijective': ok_bij,
          'equivariant': ok_equiv, 'no_swap_self': ok_noswap,
          'y_same_frac': y_same}
    print(f"G3 swapfix pi checks: fixed_points={ok_fp} "
          f"bijective={ok_bij} equivariant={ok_equiv} "
          f"no_swap_self={ok_noswap} y_same_frac={y_same:.4f}",
          flush=True)

    # ---- fixed eval bank (identical construction to r84) ----
    bank_rng = np.random.default_rng(5000 + seed)
    bank = [build_swap_pi_vec(bank_rng) for _ in range(BANK_K)]
    bank_dev = [torch.from_numpy(b).long().to(DEV) for b in bank]
    bank_checks = [pi_checks(b) for b in bank]
    bank_ok = all(fp == 0 and bij and eqv and nos
                  for fp, bij, eqv, nos in bank_checks)
    print(f"eval bank K={BANK_K}: all_checks_pass={bank_ok}",
          flush=True)
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
                      'L': LMAP.get(name), 't': [], 'va_surg': [],
                      'va_plain': [], 'cross95_surg': None,
                      'cross90_plain': None, 'w2': {}, 'inj_stat': {}}
        if name in LMAP and LMAP[name] is not None:
            arms[name]['rng'] = np.random.default_rng(4000 + seed)
            arms[name]['pi'] = None
            arms[name]['hashes'] = []
            arms[name]['y_same'] = []
            arms[name]['ov_all'] = 0
            arms[name]['ov_diag'] = 0
            arms[name]['ov_off'] = 0
            arms[name]['n_draws'] = 0
            arms[name]['g3_fail'] = 0

    def donor_image_pi(mN, ids, pi_arr):
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
        # ---- per-arm pi redraw (block-aligned) ----
        for name in ARMS:
            A = arms[name]
            if A['L'] is None:
                continue
            if (step - 1) % A['L'] == 0:
                pi_new = build_swap_pi_vec(A['rng'])
                fp, bij, eqv, nos = pi_checks(pi_new)
                if not (fp == 0 and bij and eqv and nos):
                    A['g3_fail'] += 1
                if A['pi'] is not None:
                    same = (pi_new == A['pi'])
                    A['ov_all'] += int(same.sum())
                    A['ov_diag'] += int(same[DIAG_IDX].sum())
                    A['ov_off'] += int(same[OFF_IDX].sum())
                A['pi'] = pi_new
                A['pi_dev'] = torch.from_numpy(pi_new).long().to(DEV)
                A['hashes'].append((step, pi_hash(pi_new)))
                A['y_same'].append(float((y_np[pi_new] == y_np)
                                         .mean()))
                A['n_draws'] += 1

        idx = torch.randint(tr_a.shape[0], (BATCH,))
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        ids = id_train[idx]
        mN = arms['natS']['m']
        with torch.no_grad():
            inj = {name: (donor_image_pi(mN, ids, pi_fix_dev)
                          if name == 'swapfix' else
                          (donor_image_pi(mN, ids,
                                          arms[name]['pi_dev'])
                           if arms[name]['L'] is not None else None))
                   for name in ARMS}
        donor_A2 = None
        # ---- phase 1: forward + backward for all arms ----
        for name in ARMS:
            A = arms[name]
            m, opt = A['m'], A['opt']
            m.train()
            opt.zero_grad()
            if name == 'natS':
                logits, extras = routed_forward2(
                    m, a, b, ROWS_OWN, A['Wq0'], A['Wk0'])
                if step == 1:
                    gate0 = float((logits - m(a, b)).abs().max()
                                  .detach())
                donor_A2 = extras['attn'][:, :, 2, :].detach().clone()
            else:
                logits, extras = routed_forward2(
                    m, a, b, ROWS_REPLAY, A['Wq0'], A['Wk0'],
                    donor_A2=inj[name])
            if step == 1 or step % 1000 == 0:
                if inj[name] is not None:
                    A['inj_stat'][step] = {
                        'mean': float((inj[name] - donor_A2).abs()
                                      .mean()),
                        'max': float((inj[name] - donor_A2).abs()
                                     .max())}
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
        # ---- eval (bank + plain), post-update ----
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
                        inj_va = donor_image_pi(mN, id_va,
                                                pi_fix_dev)
                        lg, _ = routed_forward2(
                            A['m'], va[0], va[1], ROWS_REPLAY,
                            A['Wq0'], A['Wk0'], donor_A2=inj_va)
                        vs = float((lg.argmax(-1) == va[2])
                                   .float().mean())
                    else:  # swapL*: bank mean + cur-block-pi aux
                        draws = []
                        for k in range(BANK_K):
                            inj_va = donor_image_pi(mN, id_va,
                                                    bank_dev[k])
                            lg, _ = routed_forward2(
                                A['m'], va[0], va[1], ROWS_REPLAY,
                                A['Wq0'], A['Wk0'],
                                donor_A2=inj_va)
                            draws.append(float(
                                (lg.argmax(-1) == va[2])
                                .float().mean()))
                        inj_va = donor_image_pi(mN, id_va,
                                                A['pi_dev'])
                        lg, _ = routed_forward2(
                            A['m'], va[0], va[1], ROWS_REPLAY,
                            A['Wq0'], A['Wk0'], donor_A2=inj_va)
                        vs = float(np.mean(draws))
                        A.setdefault('va_curpi', []).append(float(
                            (lg.argmax(-1) == va[2]).float().mean()))
                        A.setdefault('bank_draws', []).append(draws)
                        A.setdefault('bank_sd', []).append(
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
                        inj_g = donor_image_pi(
                            mN, torch.from_numpy(table_ids_grid),
                            pi_fix_dev)
                        rec[name] = strat_probe(
                            A['m'], A['Wq0'], A['Wk0'], ROWS_REPLAY,
                            inj_g, self_in_grid, swap_in_grid,
                            diag_grid, ga, gb)
                    else:  # swapL*: bank-averaged surgery-active
                        keys = ('train_acc', 'va_grid', 'm1', 'm2',
                                'm3')
                        acc = {k: 0.0 for k in keys}
                        for k in range(BANK_K):
                            inj_g = donor_image_pi(
                                mN,
                                torch.from_numpy(table_ids_grid),
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
                           for n, (c, v) in live.items()),
                  flush=True)

    # swapfix pi constancy check
    pi2, _, _, _, _, _ = build_swap_pi(seed)
    g3['constant'] = bool((pi2 == pi_fix).all())

    res = {'gate0': gate0, 'g3': g3, 'arms': {}, 'strat': strat}
    for name in ARMS:
        A = arms[name]
        d = {'t': A['t'], 'va_surg': A['va_surg'],
             'va_plain': A['va_plain'],
             'cross95_surg': A['cross95_surg'],
             'cross90_plain': A['cross90_plain'],
             'w2': {k: (v if isinstance(v, np.ndarray)
                        else v.cpu().numpy())
                    for k, v in A['w2'].items()},
             'inj_stat': A['inj_stat']}
        if name in LMAP and LMAP[name] is not None:
            ncmp = max(A['n_draws'] - 1, 1)
            d['pi_audit'] = {
                'L': A['L'], 'n_draws': A['n_draws'],
                'g3_fail': A['g3_fail'],
                'hash_head': A['hashes'][:5],
                'overlap': {
                    'all': A['ov_all'] / (ncmp * N_PAIRS),
                    'diag': A['ov_diag'] / (ncmp * DIAG_IDX.size),
                    'off': A['ov_off'] / (ncmp * OFF_IDX.size)},
                'y_same_mean': float(np.mean(A['y_same']))}
            d['va_curpi'] = A.get('va_curpi', [])
            d['bank_sd'] = A.get('bank_sd', [])
        if name == 'swapfix':
            d['pi'] = pi_fix
        res['arms'][name] = d
    print(f"[seed{seed} done in {time.time() - t0:.0f}s]", flush=True)
    return res


def classify(seed, out):
    """Preregistered §6cc readout on T_cross^bank."""
    def T(name):
        return out[seed]['arms'][name]['cross95_surg']
    t1, t10, t50, t200 = T('swapL1'), T('swapL10'), T('swapL50'), \
        T('swapL200')
    tfx, tns = T('swapfix'), T('natS')
    print(f"  seed{seed} dose: natS={tns} L1={t1} L10={t10} "
          f"L50={t50} L200={t200} Linf={tfx}", flush=True)
    if None in (t1, t10, t50, t200):
        return 'CENSORED'
    tol = EVAL_EVERY
    if (t200 > t50 + tol and t50 > t10 + tol
            and t10 >= t1 - tol):
        return 'MONOTONE'
    if max(t10, t50, t200) <= 1.5 * t1:
        return 'THRESHOLD-ABOVE-200'
    return 'NON-MONOTONE'


def continuity_gate(out):
    """G-C: natS/swapfix/swapL1 bitwise vs ARCHIVED r84 pkl."""
    with open(os.path.join(REPRO_DIR,
                           'r84_persistence_missing_cell_results.pkl'),
              'rb') as f:
        r84 = pickle.load(f)
    ok_all = True
    for seed in SEEDS:
        for name, ref in (('natS', 'natS'), ('swapfix', 'swapfix'),
                          ('swapL1', 'swapres')):
            a_new = out[seed]['arms'][name]
            a_old = r84[seed]['arms'][ref]
            same_t = a_new['t'] == a_old['t']
            d_vs = max((abs(x - y) for x, y in
                        zip(a_new['va_surg'], a_old['va_surg'])),
                       default=0.0)
            d_vp = max((abs(x - y) for x, y in
                        zip(a_new['va_plain'], a_old['va_plain'])),
                       default=0.0)
            ok = (same_t and d_vs == 0.0 and d_vp == 0.0
                  and a_new['cross95_surg'] == a_old['cross95_surg']
                  and a_new['cross90_plain']
                  == a_old['cross90_plain'])
            print(f"G-C seed{seed} {name}(~{ref}): bitwise={ok} "
                  f"(t={same_t} maxdv_surg={d_vs:.2e} "
                  f"maxdv_plain={d_vp:.2e})", flush=True)
            ok_all &= ok
    return ok_all


def main():
    out = {}
    seeds_run = list(SEEDS)
    classes = {}
    i = 0
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results_r86.pkl')
    while i < len(seeds_run):
        seed = seeds_run[i]
        print(f'=== seed{seed} ===', flush=True)
        out[seed] = run_seed(seed)
        classes[seed] = classify(seed, out)
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)
        i += 1
        if (i == len(SEEDS) and len(seeds_run) == len(SEEDS)
                and not SMOKE
                and classes[seeds_run[0]] != classes[seeds_run[1]]):
            print('[K10-r86] classification conflict seed0/1 -> '
                  'prereg-authorized seed2', flush=True)
            seeds_run.append(2)

    print('\n=== summary ===', flush=True)
    all_pass = True
    for seed in seeds_run:
        r = out[seed]
        g0ok = r['gate0'] is not None and r['gate0'] <= 1e-5
        g3ok = all([r['g3']['bijective'], r['g3']['equivariant'],
                    r['g3']['no_swap_self'], r['g3']['constant'],
                    r['g3']['fixed_points'] == 0])
        aud = {n: r['arms'][n].get('pi_audit', {}).get('g3_fail')
               for n in r['arms'] if 'pi_audit' in r['arms'][n]}
        g3c = all(v == 0 for v in aud.values())
        all_pass &= g0ok and g3ok and g3c
        ns = r['arms']['natS']
        print(f"seed{seed}: natS={ns['cross95_surg']} "
              f"(c90={ns['cross90_plain']}) | G0={r['gate0']:.2e} "
              f"G3={g3ok} redraw_fails={aud} | "
              f"class={classes[seed]}", flush=True)
        for name in ARMS[1:]:
            pa = r['arms'][name].get('pi_audit')
            if pa:
                print(f"  {name}: draws={pa['n_draws']} "
                      f"ov_all={pa['overlap']['all']:.2e} "
                      f"y_same={pa['y_same_mean']:.4f}", flush=True)
        strat_final = r['strat'].get(max(r['strat'].keys()))
        if strat_final:
            print("  final strat m2: " +
                  " ".join(f"{n}={v['m2']:.3f}"
                           for n, v in strat_final.items()),
                  flush=True)
    if not SMOKE:
        g = continuity_gate(out)
        all_pass &= g
        same = len(set(classes.values())) == 1
        print(f"classes: {classes} same={same}", flush=True)
        all_pass &= same
    print('ALL GATES PASS' if all_pass else 'GATE FAILURE',
          flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results_r86.pkl', flush=True)


if __name__ == '__main__':
    main()
