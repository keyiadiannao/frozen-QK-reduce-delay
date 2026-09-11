# -*- coding: utf-8 -*-
"""phenotype_table.py — seed-heterogeneity table behind the S1 post hoc
note (paper §mech-leverage, "A post hoc reading ties the failures to a
structural seed variable").

Question (2026-09-06): do the seeds that fail individual gates share a
latent phenotype?  Answer: yes, two orthogonal axes.

  Structure axis  -- diag_share_L (select_k_modes.py stage-4 output):
    the fraction of the pre-crossing interaction write's 2D spectrum
    on same-|k| frequency pairs.  Bimodal across the 9 analysable
    seeds ({0.07..0.25} vs {0.56..0.61}); every gate anomaly
    (S1 below-gate 0/4/6, K03 LATE 3/6, K03 natS-censored 0/8,
    M11 minimum 3) falls in the low group; the K set captures
    15-20% of the write's spectral energy in the below-gate seeds
    (4-5% in the two remaining low seeds) against 52-60% in the four
    clean seeds.  Post hoc, n=9, indicative.
  Slowness axis   -- seed9: native crossing censored everywhere, yet
    the cleanest structure of all (diag 0.66 at its latest
    checkpoint); K03 natS censoring for 0/8/9 is pure horizon
    (native cross > CAP 14000: 15075/18750/>20000).

Inputs (all produced by the standard pipeline, no new training):
  --tenseed-out  tenseed_out/ runtime dir (frozen_k/, replay_summary.json)
  --claims-dir   claims/ (M10/M11/K03 result pkls)
  --out          output JSON path (default: alongside this script)
  --bridge-a     optional bridge_zp_A dir for base-summary features
  --r131c-ckpt   optional r131c_ckpt dir for the seed9 spectral check

Run:  python claims/M10_leverage_trace/phenotype_table.py
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenseed-out", required=True,
                    help="tenseed_out runtime dir (frozen_k/, "
                         "replay_summary.json)")
    ap.add_argument("--claims-dir", required=True,
                    help="package claims/ dir (M10/M11/K03 pkls)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: seed_phenotype_analysis.json "
                         "next to this script)")
    ap.add_argument("--bridge-a", default=None,
                    help="optional bridge_zp_A dir (base summaries)")
    ap.add_argument("--r131c-ckpt", default=None,
                    help="optional r131c_ckpt dir (seed9 spectral check)")
    args = ap.parse_args()

    seeds = list(range(9))
    frozen = {s: json.load(open(os.path.join(
        args.tenseed_out, "frozen_k", f"seed{s}.json"))) for s in seeds}
    diag = {s: frozen[s]["diag_share_L"] for s in seeds}
    # K5 capture: what fraction of the write's total spectral energy the
    # frozen top-5 diagonal set accounts for
    k5 = {s: diag[s] * sum(v for _, v in frozen[s]["top8"][:5])
          for s in seeds}
    rep = json.load(open(os.path.join(args.tenseed_out,
                                      "replay_summary.json")))
    cross = {s: rep[str(s)]["cross"] for s in range(10)}

    m10 = pickle.load(open(os.path.join(
        args.claims_dir, "M10_leverage_trace",
        "results_9seed.pkl"), "rb"))["seeds"]
    m11 = pickle.load(open(os.path.join(
        args.claims_dir, "M11_wr_factorial",
        "results_9seed.pkl"), "rb"))["seeds"]
    k03 = pickle.load(open(os.path.join(
        args.claims_dir, "K03_identity_geometry",
        "results_10seed.pkl"), "rb"))
    k03_seed = {s: k03[s] if s in k03 else k03[str(s)] for s in range(10)}

    rows = {}
    for s in seeds:
        rows[s] = {
            "diag_share_L": round(diag[s], 3),
            "K5_capture_of_write_energy": round(k5[s], 3),
            "replay_cross": cross[s],
            "J_P": round(m10[s]["J_P_window"], 2),
            "J_L": round(m10[s]["J_L_window"], 2),
            "S1_gate": bool(m10[s]["gate"]),
            "I_match": round(m11[s]["I_match"], 2),
            "K03_branch": k03_seed[s].get("branch"),
            "K03_t_div": k03_seed[s].get("t_div"),
            "K03_natS_cross": k03_seed[s]["arms"]["natS"]["cross90"],
            "K03_natF_cross": k03_seed[s]["arms"]["natF"]["cross90"],
        }
    if args.bridge_a:
        for s in range(10):
            fs = glob.glob(os.path.join(
                args.bridge_a, f"seed{s}_*_summary.json"))
            if not fs:
                continue
            j = json.load(open(fs[0]))
            rows.setdefault(s, {})["base"] = {
                "first_cross": j["first_cross_step"],
                "best_va": round(j["best_val_acc"], 3),
                "final_va": round(j["final_val_acc"], 3),
                "collapsed": bool(j["best_val_acc"] > 0.95
                                  and j["final_val_acc"] < 0.6)}

    low = [s for s in seeds if diag[s] < 0.3]
    hi = [s for s in seeds if diag[s] >= 0.3]
    s1_fail = [s for s in seeds if not m10[s]["gate"]]
    late = [s for s in seeds
            if k03_seed[s].get("branch") == "LATE"]
    gap = {s: m10[s]["J_L_window"] - m10[s]["J_P_window"] for s in seeds}

    corr = {
        "diag_vs_J_P": round(spearman([diag[s] for s in seeds],
                                      [m10[s]["J_P_window"] for s in seeds]), 3),
        "diag_vs_I_match": round(spearman([diag[s] for s in seeds],
                                          [m11[s]["I_match"] for s in seeds]), 3),
        "diag_vs_crossing": round(spearman([diag[s] for s in seeds],
                                           [cross[s] for s in seeds]), 3),
        "diag_vs_JL_minus_JP": round(spearman([diag[s] for s in seeds],
                                              [gap[s] for s in seeds]), 3),
    }

    s9_spectral = None
    if args.r131c_ckpt:
        # optional: diag share at seed9's latest checkpoint (needs
        # common/ + torch; CPU is fine, ~1 min)
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        import torch
        from common.model import D57Model
        from common import task
        task.set_group("zp", 113)
        P = 113
        fs = sorted(glob.glob(os.path.join(args.r131c_ckpt, "s9_t*.pt")),
                    key=lambda f: int(f.split("_t")[1].replace(".pt", "")))
        ck = torch.load(fs[-1], map_location="cpu", weights_only=False)
        m = D57Model(pos_mode="zeros", arch="abeq")
        m.load_state_dict(ck["model"])
        m.eval()
        aa = np.arange(P)[None, :].repeat(P, 0).ravel()
        bb = np.arange(P)[:, None].repeat(P, 1).ravel()
        m_all = []
        for i in range(0, P * P, 4096):
            a = torch.from_numpy(aa[i:i + 4096]).long()
            b = torch.from_numpy(bb[i:i + 4096]).long()
            with torch.no_grad():
                eq = torch.full_like(a, P)
                tok = torch.stack([a, b, eq], dim=1)
                x = m.emb(tok) + m.pos[None, :, :]
                h = m.ln1(x)
                H, dh = m.n_heads, m.d_head
                q = m.Wq(h).view(-1, 3, H, dh).transpose(1, 2)
                k = m.Wk(h).view(-1, 3, H, dh).transpose(1, 2)
                v = m.Wv(h).view(-1, 3, H, dh).transpose(1, 2)
                att = ((q @ k.transpose(-1, -2)) / (dh ** 0.5))\
                    .softmax(dim=-1)
                o = (att @ v).transpose(1, 2).contiguous()\
                    .view(-1, 3, H * dh)
                x = x + m.Wo(o)
                m_all.append(m.mlp2(torch.nn.functional.gelu(
                    m.mlp1(m.ln2(x)[:, 2, :]))).numpy())
        mw = np.concatenate(m_all).astype(np.float64)
        fa = mw.mean(axis=1, keepdims=True)
        fb = mw.mean(axis=0, keepdims=True)
        fg = mw.mean(axis=(0, 1), keepdims=True)
        F = np.fft.fft2((mw - fa - fb + fg).reshape(P, P, -1), axes=(0, 1))
        E = (np.abs(F) ** 2).sum(axis=2)
        tot = E[1:, 1:].sum()
        dg = sum(E[k, l] for k in range(1, P) for l in range(1, P)
                 if k == l or k == (P - l) % P)
        s9_spectral = {"t": int(fs[-1].split("_t")[1].replace(".pt", "")),
                       "diag_share": round(float(dg / tot), 3)}

    out = {
        "question": "do gate-failing seeds share a latent phenotype?",
        "answer": "two orthogonal axes: a structure axis (diag_share_L "
                  "bimodal; all gate anomalies in the low group) and an "
                  "independent slowness axis (seed9; K03 natS censoring "
                  "= pure horizon)",
        "table": {str(s): rows[s] for s in sorted(rows)},
        "groups": {"structure_low": low, "structure_high": hi,
                   "S1_fail": s1_fail, "K03_LATE": late},
        "K5_capture_ranges": {
            "below_gate_seeds": [0.15, 0.20],
            "other_low_seeds": [0.04, 0.05],
            "clean_seeds": [0.52, 0.60]},
        "gap_JL_minus_JP": {
            "clean_group_range": [round(min(gap[s] for s in hi), 2),
                                  round(max(gap[s] for s in hi), 2)],
            "low_group_range": [round(min(gap[s] for s in low), 2),
                                round(max(gap[s] for s in low), 2)]},
        "correlations_spearman": corr,
        "K03_natS_censored_is_horizon": {
            "CAP": 14000,
            "native_cross_of_censored": {str(s): cross[s]
                                         for s in (0, 8, 9)}},
        "seed9_spectral_check": s9_spectral,
        "caveat": "post hoc, n=9; groups found while investigating gate "
                  "failures, hypergeometric p~1/126 indicative only",
    }
    fp = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "seed_phenotype_analysis.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=1)
    print(f"written {fp}")
    for s in seeds:
        r = rows[s]
        print(f"  s{s}: diag={r['diag_share_L']:.2f} "
              f"K5cap={r['K5_capture_of_write_energy']:.2f} "
              f"J_P={r['J_P']:+.2f} J_L={r['J_L']:+.2f} "
              f"gate={r['S1_gate']} branch={r['K03_branch']}")


if __name__ == "__main__":
    main()
