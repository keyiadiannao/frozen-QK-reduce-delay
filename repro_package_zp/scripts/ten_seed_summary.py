"""Summarize the 10-seed (and 9-seed) confirmatory artifacts into copy-paste tables.

Usage:
    python ten_seed_summary.py --dir <folder containing *_10seed.pkl / *_9seed.pkl>

Reads the archived result pickles only; runs nothing.
"""
import argparse
import glob
import os
import pickle
import statistics as st


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def q1q3(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None, None
    n = len(xs)
    return xs[n // 4], xs[(3 * n) // 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    files = {}
    for p in glob.glob(os.path.join(a.dir, "*.pkl")):
        files[os.path.basename(p)] = p

    def get(prefix):
        for k, v in files.items():
            if k.startswith(prefix):
                return v
        return None

    # ---- K01 fate lock ------------------------------------------------
    p = get("K01")
    if p:
        d = load(p)
        print("### K01 fate lock (10 seeds) -- crossing step, cap 4000, vacc>0.9")
        arms = ["F-train", "F-frozen", "S-train", "S-frozen"]
        for arm in arms:
            cs = []
            for s in range(10):
                r = d[f"{arm}-{s}"]
                c = r.get("crossed")
                if c is None:
                    # fall back: first t with val_acc > 0.9
                    c = next((t for t, v in zip(r["t"], r["val_acc"]) if v > 0.9), None)
                cs.append(c)
            ncross = sum(c is not None for c in cs)
            vals = [c for c in cs if c is not None]
            print(f"  {arm:9s} cross {ncross}/10  per-seed={cs}")
            if vals:
                print(f"            median={med(cs)}  IQR={q1q3(cs)}  range={min(vals)}-{max(vals)}")
        print()

    # ---- K02 QK carrier ----------------------------------------------
    p = get("K02")
    if p:
        d = load(p)
        print("### K02 QK carrier (10 seeds) -- fate follows the QK values")
        arms = ["S-Qs", "F-Q0", "S-Q0reinit", "F-QsTransplant"]
        for arm in arms:
            cs = [d[f"{arm}-{s}"].get("crossed") for s in range(10)]
            ncross = sum(c is not None for c in cs)
            vals = [c for c in cs if c is not None]
            rng = f"{min(vals)}-{max(vals)}" if vals else "-"
            print(f"  {arm:16s} cross {ncross}/10  per-seed={cs}  range={rng}")
        print()

    # ---- K03 identity geometry ---------------------------------------
    p = get("K03")
    if p:
        d = load(p)
        print("### K03 identity geometry (10 seeds) -- fixed-lag main criterion")
        print("  (see k03_offline_verdict.py for the delta-P computation)")
        for s in range(10):
            r = d[s]
            arms = r.get("arms", {})
            row = {k: (v.get("cross90"), v.get("cross95"), v.get("branch"))
                   for k, v in arms.items() if isinstance(v, dict)}
            print(f"  seed{s}: t_div={r.get('t_div')} branch={r.get('branch')} {row}")
        print()

    # ---- K07 maintenance ---------------------------------------------
    p = get("K07")
    if p:
        d = load(p)
        print("### K07 maintenance (10 seeds)")
        print("  top keys:", list(d.keys()))
        arms = d.get("arms", {})
        for s, rec in arms.items():
            line = []
            for arm, v in rec.items():
                if isinstance(v, dict) and "I" in v:
                    line.append(f"{arm}: I_end={v['I'][-1]:.3f} rho_end={v['rho_m'][-1]:.3f}")
            print(f"  seed{s}: " + " | ".join(line))
        print()

    # ---- K14 dose response -------------------------------------------
    p = get("K14")
    if p:
        d = load(p)
        print("### K14 dose response (10 seeds) -- crossing step vs lambda, cap 12000")
        order = ["L000", "L025", "L050", "L075", "L100", "SELF", "OPER", "RND", "SS"]
        arms = d["arms"]
        for nm in order:
            cs = [arms[s][nm]["cross"] if nm in arms[s] else None for s in range(10)]
            vals = [c for c in cs if c is not None]
            ncens = sum(c is None for c in cs)
            q1, q3 = q1q3(cs)
            print(f"  {nm:5s} cens {ncens}/10  per-seed={cs}")
            if vals:
                print(f"         median={med(cs)} IQR=[{q1},{q3}] range={min(vals)}-{max(vals)}")
        # monotone check per seed over L000..L100
        mono = 0
        for s in range(10):
            seq = [arms[s][nm]["cross"] for nm in ["L000", "L025", "L050", "L075", "L100"]]
            ok = True
            for x, y in zip(seq, seq[1:]):
                if x is None and y is not None:
                    ok = False
                if x is not None and y is not None and y < x:
                    ok = False
            mono += ok
        print(f"  monotone non-decreasing in lambda: {mono}/10 seeds")
        print()

    # ---- M10 / M11 ----------------------------------------------------
    for pref, label in [("M10", "M10 leverage trace (S1)"), ("M11", "M11 wr factorial (S2)")]:
        p = get(pref)
        if not p:
            continue
        d = load(p)
        print(f"### {label}")
        print("  verdict:", d.get("verdict"))
        print()


if __name__ == "__main__":
    main()
