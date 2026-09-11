"""scripts/gen_qk_lineage.py — Stage 2 of the 10-seed server plan.

Generates the r107-lineage QK transplant weights per seed ONCE, as
checkpoint files, so that the replay (Stage 3) and S1/S2 (Stage 6)
load files instead of re-running the r107 construction inside the
package runtime.  This removes the r107 order-locked construction
sequence (torch.manual_seed interleavings) from every downstream
environment dependency: the weights are materialized here and hashed.

Verbatim from repro/r113_body_rnd_factorial.py build_qk() lines
130-207, with two deviations (both registered):
  1. results are SAVED instead of returned in-process;
  2. the install gates (analytic v-target checks) are kept and must
     PASS before the file is written -- same gate, same tolerance.

Only the arm the replay actually consumes is needed:  F_QS = (WqS, WkS)
(the S-native QK on the F body -- the r131c replay's Wq/Wk source).
We also emit F_Q0 for completeness of the lineage record.

Output: <out>/qk_lineage/seed{N}_F_QS.pt / seed{N}_F_Q0.pt
        each {'Wq': tensor, 'Wk': tensor, 'seed': N, 'source': 'r107'}
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from common import task                                    # noqa: E402
from common.model import D57Model                          # noqa: E402
from common.states import bridge_ckpt_path                 # noqa: E402

DEV = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'


def ckpt(repro_dir, family, seed):
    return torch.load(bridge_ckpt_path(repro_dir, family, seed, None,
                                       'full0000200'),
                      map_location='cpu', weights_only=False)


def build_qk(repro_dir, seed):
    """Verbatim r107 lineage (r113 lines 130-207).

    Order-locked: torch.manual_seed(seed) -> mS (1st construction) ->
    m0 (2nd construction); rng = default_rng(1234+seed) created once,
    consumed per-head inside the loop in r107's order.
    """
    torch.manual_seed(seed)
    mS = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    sd_S = ckpt(repro_dir, 'A', seed)
    mS.load_state_dict(dict(sd_S['model']))
    m0 = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    mF = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
    mF.load_state_dict(dict(ckpt(repro_dir, 'C', seed)['model']))

    with torch.no_grad():
        tok = torch.tensor([[0, 0, task.N_EQ]], device=DEV)
        h2_S = mS.ln1(mS.emb(tok) + mS.pos[None, :, :])[0, 2, :].detach()
        h2_F = mF.ln1(mF.emb(tok) + mF.pos[None, :, :])[0, 2, :].detach()

    Wq0, Wk0 = m0.Wq.weight.detach(), m0.Wk.weight.detach()
    WqS, WkS = mS.Wq.weight.detach(), mS.Wk.weight.detach()
    Hh, dh = mS.n_heads, mS.d_head
    d_model = Wk0.shape[1]

    qhat0_S = Wq0 @ h2_S
    qhat0_F = Wq0 @ h2_F
    q2S = WqS @ h2_S
    h2n_S = h2_S / float(h2_S @ h2_S) ** 0.5
    h2n_F = h2_F / float(h2_F @ h2_F) ** 0.5

    # S-side random directions (verbatim r107 order) + registered norms
    v_rand_S, dv_norms, v0_S = [], [], []
    rng = np.random.default_rng(1234 + seed)
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        v0_h = Wk0[sl, :].T @ qhat0_S[sl]
        vS_h = WkS[sl, :].T @ q2S[sl]
        dv_h = vS_h - v0_h
        nrm = float(dv_h @ dv_h) ** 0.5
        g = torch.from_numpy(rng.standard_normal(dv_h.shape)).to(
            device=DEV, dtype=dv_h.dtype)
        g = g - float(g @ h2n_S) * h2n_S
        g = g / (float(g @ g) ** 0.5 + 1e-30) * nrm
        v0_S.append(v0_h)
        v_rand_S.append(g)
        dv_norms.append(nrm)

    # F-side rematch directions: same seed, orthogonal to h2_F, scaled
    # to the REGISTERED S-side per-head delta norms.
    v_rand_F, v0_F = [], []
    rngF = np.random.default_rng(1234 + seed)
    for h in range(Hh):
        sl = slice(h * dh, (h + 1) * dh)
        v0_F.append(Wk0[sl, :].T @ qhat0_F[sl])
        g = torch.from_numpy(rngF.standard_normal((d_model,))).to(
            device=DEV, dtype=Wk0.dtype)
        g = g - float(g @ h2n_F) * h2n_F
        g = g / (float(g @ g) ** 0.5 + 1e-30) * dv_norms[h]
        v_rand_F.append(g)

    def install(base_wk, qhat, delta_list):
        Wk_e = base_wk.clone()
        for h in range(Hh):
            sl = slice(h * dh, (h + 1) * dh)
            q_h = qhat[sl]
            Wk_e[sl, :] += q_h.outer(delta_list[h]) / float(q_h @ q_h)
        return Wk_e

    Wk_rnd_S = install(Wk0, qhat0_S, v_rand_S)
    Wk_rnd_F = install(Wk0, qhat0_F, v_rand_F)

    # install gates (r113 verbatim): S_QRND_S / F_QRND_F have analytic
    # targets; they must hit before we bless the files.
    gates = {}
    for name, wk_e, qhat, targets in (
            ('S_QRND_S', Wk_rnd_S, qhat0_S, v0_S + v_rand_S),
            ('F_QRND_F', Wk_rnd_F, qhat0_F, v0_F + v_rand_F)):
        torch.manual_seed(seed)
        mtmp = D57Model(pos_mode='zeros', arch='abeq').to(DEV)
        body_family = 'A' if name == 'S_QRND_S' else 'C'
        body = dict(ckpt(repro_dir, body_family, seed)['model'])
        body['Wk.weight'] = wk_e.clone()
        mtmp.load_state_dict(body)
        h2 = h2_S if body_family == 'A' else h2_F
        q2 = mtmp.Wq.weight.detach() @ h2
        worst = 0.0
        with torch.no_grad():
            for h in range(Hh):
                sl = slice(h * dh, (h + 1) * dh)
                v_act = (mtmp.Wk.weight.detach()[sl, :].T @ q2[sl])
                worst = max(worst, float(
                    (v_act - targets[h]).abs().max()))
        gates[name] = worst
    ok = all(w < 1e-4 for w in gates.values())
    return {'S_L000': (Wq0, Wk0), 'F_Q0': (Wq0, Wk0),
            'F_QS': (WqS, WkS)}, gates, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro-dir', required=True,
                    help='dir containing bridge_zp_{A,C,...}/')
    ap.add_argument('--out', required=True,
                    help='output dir for qk_lineage/')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'qk_lineage'), exist_ok=True)
    summary = {}
    all_ok = True
    for seed in [int(s) for s in args.seeds.split(',')]:
        arms, gates, ok = build_qk(args.repro_dir, seed)
        all_ok &= ok
        rec = {'seed': seed, 'gates': gates, 'gates_ok': ok,
               'files': {}}
        for name in ('F_QS', 'F_Q0'):
            wq, wk = arms[name]
            fp = os.path.join(args.out, 'qk_lineage',
                              f'seed{seed}_{name}.pt')
            torch.save({'Wq': wq.cpu(), 'Wk': wk.cpu(), 'seed': seed,
                        'arm': name, 'source': 'r107'}, fp)
            h = hashlib.sha256()
            with open(fp, 'rb') as f:
                h.update(f.read())
            rec['files'][name] = {'path': fp, 'sha256': h.hexdigest()}
        summary[seed] = rec
        print(f'seed{seed}: gates={gates} ok={ok}', flush=True)
    with open(os.path.join(args.out, 'qk_lineage', 'summary.json'),
              'w') as f:
        json.dump(summary, f, indent=2)
    print('ALL GATES PASS' if all_ok else 'GATE FAILURE', flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
