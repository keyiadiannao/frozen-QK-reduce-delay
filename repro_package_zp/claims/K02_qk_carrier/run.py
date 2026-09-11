"""K02 run.py — state-factorial transplant at t=200 (QK value /
non-QK state / co-adaptation decomposition).

Package port of repro/r97_state_factorial.py (M2.5, 2026-09-02).
Verbatim except: bridge paths via common.states, val_probe from
common.readouts, claim-dir output, SMOKE env R97_SMOKE -> K02_SMOKE.

BATCH CHAIN (load-bearing, verbatim R51 mechanism): global torch RNG —
manual_seed(seed) -> build model(s) -> load ckpt -> DISCARD 200 draws ->
continue with torch.randint each step.  QK FROZEN in all arms (grad
=None each step, AdamW skips entirely).  Q0 semantics: the fresh QK
model is the SECOND D57Model construction after manual_seed(seed)
(call-order-dependent; do not "fix").

Claim (ledger K2): QK-CARRIER — slow fate is carried by the trained QK
values themselves (fate follows QK, 3/3 seeds); W_Q and W_K need joint
training (R99 JOINT is the reciprocal follow-up).  Preregistered §6dj;
verdict §6dk; wording §6dn.
"""
import os
import sys
import pickle
import copy

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
SMOKE = bool(os.environ.get('K02_SMOKE'))
if SMOKE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEV = 'cpu' if SMOKE else ('cuda' if torch.cuda.device_count() > 0
                           else 'cpu')
LOSS = nn.CrossEntropyLoss()
CAP = 60 if SMOKE else 4000
SNAP_EVERY = 25
BATCH = 128 if SMOKE else 512
SEEDS = (0,) if SMOKE else tuple(
    int(s) for s in os.environ.get('CLAIM_SEEDS', '0,1,2').split(','))
P = task.GROUP['p']
QK = ('Wq.weight', 'Wk.weight')
ARMS = [('S', 'Qs'), ('F', 'Q0'), ('S', 'Q0'), ('F', 'Qs')]
ARM_NAME = {('S', 'Qs'): 'S-Qs', ('F', 'Q0'): 'F-Q0',
            ('S', 'Q0'): 'S-Q0reinit', ('F', 'Qs'): 'F-QsTransplant'}

torch.set_num_threads(4)
if SMOKE and torch.cuda.device_count() > 0:
    raise SystemExit('[K02] K02_SMOKE=1 but CUDA visible -- abort')
print(f'[K02] DEV={DEV} SMOKE={SMOKE} seeds={SEEDS} CAP={CAP} '
      f'BATCH={BATCH}', flush=True)

R = ZPReadouts(DEV)
FAMILY = {'S': 'A', 'F': 'C'}   # r97 state codes -> bridge family letters


def ckpt(state, seed):
    return states.bridge_ckpt_path(REPRO_DIR, FAMILY[state], seed, None,
                                   'full0000200')


def build_arm(state, qsrc, seed):
    """Body = state ckpt (full non-QK state incl. opt); QK = qsrc."""
    torch.manual_seed(seed)
    m = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd = torch.load(ckpt(state, seed),
                    map_location='cpu', weights_only=False)
    body = sd['model']
    if qsrc == 'Q0':
        q0 = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        qk = {'Wq.weight': q0.Wq.weight.detach().cpu(),
              'Wk.weight': q0.Wk.weight.detach().cpu()}
    else:  # 'Qs' = S ckpt's trained Wq/Wk
        s_sd = torch.load(ckpt('S', seed),
                          map_location='cpu', weights_only=False)
        qk = {'Wq.weight': s_sd['model']['Wq.weight'],
              'Wk.weight': s_sd['model']['Wk.weight']}
    body = dict(body)
    body.update(qk)
    m.load_state_dict(body)
    opt, _, _ = make_optimizer(m, 'wd_0011')
    opt.load_state_dict(sd['opt'])
    return m, opt


def run_arm(state, qsrc, seed, probe_batch, ref_codes):
    va = R.val_probe(seed)
    m, opt = build_arm(state, qsrc, seed)
    all_a, all_b, all_y = task.build_tables()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perm = rng.permutation(task.N_PAIRS)
    tr_a = torch.from_numpy(all_a[perm[:R.N_TRAIN]]).long()
    tr_b = torch.from_numpy(all_b[perm[:R.N_TRAIN]]).long()
    tr_y = torch.from_numpy(all_y[perm[:R.N_TRAIN]]).long()
    for _ in range(200):     # discard frozen-phase draws (R51)
        torch.randint(tr_a.shape[0], (BATCH,))

    rec = {'t': [], 'val_acc': [], 'loss': [], 'crossed': None}
    for t in range(1, CAP + 1):
        idx = torch.randint(tr_a.shape[0], (BATCH,))
        m.train()
        opt.zero_grad()
        a = tr_a[idx].to(DEV)
        b = tr_b[idx].to(DEV)
        y = tr_y[idx].to(DEV)
        LOSS(m(a, b), y).backward()
        m.Wq.weight.grad = None
        m.Wk.weight.grad = None
        opt.step()
        if t % SNAP_EVERY == 0 or t == CAP:
            m.eval()
            with torch.no_grad():
                logits = m(va[0], va[1])
                vacc = float((logits.argmax(-1) == va[2])
                             .float().mean())
                vloss = float(LOSS(logits, va[2]).item())
            rec['t'].append(t)
            rec['val_acc'].append(vacc)
            rec['loss'].append(vloss)
            if vacc > 0.9:
                rec['crossed'] = t
                break
    # acute readout (computed post-hoc from the trained arm's copy --
    # verbatim: r97 computes it from the arm AFTER training)
    with torch.no_grad():
        m.eval()
        pa, pb, py = probe_batch
        lg = m(pa, pb)
        rec['acute'] = {
            'loss': float(LOSS(lg, py).item()),
            'va': float((lg.argmax(-1) == py).float().mean()),
            'logit_d_vs_SQs': float(
                (lg - ref_codes['logits']).abs().max()),
            'A2_cos_vs_SQs': float(torch.nn.functional.cosine_similarity(
                lg, ref_codes['logits'], dim=-1).mean())}
    print(f"  [{ARM_NAME[(state, qsrc)]} s{seed}] "
          f"crossed={rec['crossed']} last_va="
          f"{rec['val_acc'][-1]:.3f} acute_loss="
          f"{rec['acute']['loss']:.3f} acute_va="
          f"{rec['acute']['va']:.3f} "
          f"A2cos={rec['acute']['A2_cos_vs_SQs']:.3f}", flush=True)
    return rec


def classify(rec):
    if rec['crossed'] is not None and rec['crossed'] <= 3000:
        return 'fast'
    if rec['crossed'] is None and rec['val_acc'][-1] <= 0.5:
        return 'slow'
    return 'intermediate/other'


def main():
    out = {}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.pkl')
    for seed in SEEDS:
        print(f'=== seed{seed} ===', flush=True)
        va = R.val_probe(seed)
        ggen = torch.Generator()
        ggen.manual_seed(888)
        pa = torch.randint(0, P, (BATCH,), generator=ggen).to(DEV)
        pb = torch.randint(0, P, (BATCH,), generator=ggen).to(DEV)
        py = (pa + pb) % P
        probe_batch = (pa, pb, py)
        # reference arm (S,Qs) first: acute logits as reference
        m_ref, _ = build_arm('S', 'Qs', seed)
        m_ref.eval()
        with torch.no_grad():
            ref_logits = m_ref(pa, pb)
        for state, qsrc in ARMS:
            rec = run_arm(state, qsrc, seed, probe_batch,
                          {'logits': ref_logits})
            out[f'{ARM_NAME[(state, qsrc)]}-{seed}'] = rec
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)

    print('\n=== classification & verdict (locked §6dj) ===', flush=True)
    cls = {}
    for seed in SEEDS:
        row = {}
        for state, qsrc in ARMS:
            row[ARM_NAME[(state, qsrc)]] = classify(
                out[f'{ARM_NAME[(state, qsrc)]}-{seed}'])
        cls[seed] = row
        print(f"  seed{seed}: {row}", flush=True)
    if not SMOKE:
        n_sq0_fast = sum(cls[s]['S-Q0reinit'] == 'fast' for s in SEEDS)
        n_fqs_slow = sum(cls[s]['F-QsTransplant'] == 'slow'
                         for s in SEEDS)
        n_sq0_slow = sum(cls[s]['S-Q0reinit'] == 'slow' for s in SEEDS)
        n_fqs_fast = sum(cls[s]['F-QsTransplant'] == 'fast'
                         for s in SEEDS)
        mism = ['S-Q0reinit', 'F-QsTransplant']
        n_mismatch_abnormal = sum(
            1 for s in SEEDS
            if any(cls[s][a] not in ('fast', 'slow') for a in mism))
        if n_sq0_fast >= 2 and n_fqs_slow >= 2:
            print("VERDICT: QK-CARRIER -- trained QK values themselves "
                  "carry the slow fate", flush=True)
        elif n_sq0_slow >= 2 and n_fqs_fast >= 2:
            print("VERDICT: N-CARRIER -- QK plasticity shaped the "
                  "non-QK state; current QK values are not the fate "
                  "carrier (delta1 tightened)", flush=True)
        elif n_mismatch_abnormal >= 2:
            print("VERDICT: CO-ADAPTATION -- mismatch arms are neither "
                  "native-slow nor clean-fast; the matching relation "
                  "dominates (gamma_NQ)", flush=True)
        else:
            print("VERDICT: MIXED -- register per cell", flush=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print('saved results.pkl', flush=True)


if __name__ == '__main__':
    main()
