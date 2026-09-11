# Reproduction package — Early Query–Key Learning Determines the Grokking Delay in Modular Addition

Code and archived measurements for a study of the **grokking delay** in a
one-layer Transformer trained on modular addition (`c = a + b (mod 113)`).

The claim under test is that a brief early window of query–key learning is
causally responsible for much of the delay:

- **Fate is fixed early.** Freezing `W_Q W_K` at initialization yields early
  crossing (validation accuracy > 0.9 at steps 1375–1925, 10/10 seeds), while
  freezing it only *after* the first 200 training steps leaves the network on a
  long structured-lookup plateau (censored 10/10 seeds). Later query–key
  plasticity is neither necessary for the fast branch nor sufficient to release
  the early slow state.
- **The carrier is the query–key values, not the body.** In a 2×2
  state-factorial transplant at `t = 200`, the two mismatched arms invert with
  the query–key block in *both* directions and in every seed.
- **The effective content is an operand-scoring code.** Its seed-specific
  assignment is causally dispensable; its amplitude grades the delay length;
  the plateau is maintained jointly by the query–key values and the rest of the
  network.

This repository is the evidence base for that preprint. It is released so the
numbers can be recomputed rather than taken on faith.

- Preprint (full manuscript and four-page brief): DOI
  [`10.5281/zenodo.22708281`](https://doi.org/10.5281/zenodo.22708281)
- **Before citing or reusing anything here, read [`ERRATA.md`](ERRATA.md).** It
  lists one known error in a sealed summary file, one omitted data file, and
  the retained invalid runs.

---

## Repository layout

```
.
├── README.md                  this file
├── ERRATA.md                  known errors, omissions, caveats  ← read this
├── LICENSE                    Apache 2.0 (the full licence text)
├── NOTICE                     licence scope: code Apache 2.0, data CC BY 4.0
├── CITATION.cff               citation metadata
└── repro_package_zp/          the sealed reproduction package
    ├── README.md              package-internal guide (mixed Chinese/English)
    ├── env/                   requirements.txt + the two-tier contract
    ├── common/                shared library: task, model, states, gates, readouts
    ├── base/                  base-training entry point + snapshot spec
    ├── claims/                one directory per claim (K01–K16, M10, M11)
    ├── scripts/               drivers: 10-seed pipeline, WD1111 battery, seal
    ├── manifests/             MANIFEST.json (sha256 of every file + anchor)
    └── tenseed_out/           recorded 10-seed verdicts
```

Each `claims/<ID>/` directory holds:

| file | role |
|---|---|
| `README.md` | the claim statement, its source script, and its status |
| `run.py` | re-runs the experiment and applies the claim's own gate |
| `verify.py` | checks a rerun against the archived anchor |
| `results*.pkl` | the archived per-seed measurement (the anchor) |

---

## Two verification tiers

The package draws a distinction that matters if you intend to re-run anything.
It is defined in `repro_package_zp/env/ENVIRONMENT.md` and restated here.

| tier | what it checks | where it is meaningful |
|---|---|---|
| **Tier A** — bit-exact anchor | every field of a rerun matches the archived `results*.pkl` bitwise (`max|Δ| = 0`) | **only on the seal machine**, with the same torch build and GPU model |
| **Tier B** — verdict / statistical | the *behavioural* verdict reproduces: crossing falls in the recorded band, the fate classification agrees, the dose structure holds, preregistered gates pass, means ± SD agree over seeds | **any environment**, including CPU fallback and rented GPU servers |

Tier A is an accounting tool — it proves the archived numbers were extracted
without drift. It is *not* a requirement for using the package.

**Cross-hardware bit flips are expected and do not constitute a failure.** The
10-seed rerun and any external reproduction are Tier B. A rerun that reproduces
the verdicts while disagreeing in the last bits has succeeded.

The seal machine was: numpy 2.4.3 / torch 2.11.0+cu128, RTX 5060 8 GB,
`torch.set_num_threads(4)`.

---

## Quick start

```bash
git clone <this repository>
cd repro_package_zp
pip install -r env/requirements.txt     # numpy>=2.1, torch>=2.4; CPU fallback works
```

Re-run a single claim and apply its gate:

```bash
cd claims/K01_fate_lock
python run.py
```

Verify an archived anchor directly, without retraining — this is the fastest
way to check a number quoted in the paper:

```bash
python verify.py            # compares against results*.pkl in this directory
```

Two independent checks are available.

**1. Extraction reconciliation** — confirms the shipped `common/` library
reproduces the original research code (task tables, split construction, model
initialisation, optimizer grouping, sampler gates, C1–C5):

```bash
cd repro_package_zp
python verify_extract.py
# -> EXTRACT RECONCILIATION: ALL PASS
```

The `repro/ not found` line and the `skip ... ckpt missing` lines are expected:
both refer to the author's raw training tree, which is not part of this
distribution.

**2. Seal integrity** — confirms you are holding the sealed bytes. The expected
result is **122 files verified, 0 hash mismatches**:

```bash
cd repro_package_zp
grep -v '^repro/' manifests/CHECKSUMS.txt > /tmp/pkg.sums
sha256sum -c /tmp/pkg.sums --ignore-missing
```

> ⚠️ **Do not run `sha256sum -c manifests/CHECKSUMS.txt` unfiltered.**
> `CHECKSUMS.txt` holds 222 lines: 123 package files plus **99 external
> anchors** — raw training checkpoints and result files that live in the
> author's `repro/` tree and are deliberately *not* shipped here. Unfiltered,
> the command reports ~100 failures, of which all but one are the absent
> external anchors. The single genuine omission is
> `claims/K13_closure_boundary/results.pkl`. See
> [`ERRATA.md`](ERRATA.md) §2.

`manifests/SEAL.txt` records the seal, and the sha256 of `MANIFEST.json`
itself, so the manifest can be authenticated without trusting the file list:

```
seal: v1.16-tierA-final
sealed_at_utc: 2026-09-07T16:33:44Z
n_files: 123
n_external_anchors: 99
MANIFEST.json sha256: c90212decd5dc97fa6817506dfa40d4e74303c5b4cec1f22bea680b9acf5752c
```

### The full 10-seed pipeline

```bash
python scripts/run_ten_seed.py --repro-dir /path/bridges --out /path/tenseed_out
```

Every stage writes a completion marker, so re-running the same command resumes
rather than restarts. The stages can also be run separately
(`--stage 1` … `--stage 7`), which is recommended for monitoring. Indicative
costs on the seal machine: stage 1 ≈ 50 min, stage 2 ≈ 2 min, stage 3 ≈ 40 min,
stage 4 ≈ 1 min, stage 5 ≈ 3.5 h, stage 6 ≈ 3–4 h; about 3 GB of disk.

---

## Claim inventory

The four-page brief's tables are produced by **K01, K02, K03, K07 and K14**.
The remaining claims support the full manuscript. "Gate" is the preregistered
verdict recorded in `tenseed_out/ten_seed_verdict.json`.

| ID | claim | run | gate |
|---|---|---|---|
| K01 | fate lock at `t = 200` | yes | PASS (10 seeds) |
| K02 | query–key carrier transplant (+ joint-training requirement) | yes | PASS (10 seeds) |
| K03 | identity geometry: maintained under S, self-destructs under F | yes | PASS 10/10 (preregistered fixed-lag criterion) |
| K04 | operand-level scalar scoring functional | yes | discovery set (3 seeds) |
| K05 | formation = early direction selection + amplification | yes | discovery set (3 seeds) |
| K06 | Add arm sufficiency / remove arm confounded | yes | discovery set (3 seeds) |
| K07 | freezing the trained carrier gives identity-maintaining dynamics | yes | PASS (frozen gates) |
| K08 | restorative stability + basin boundary | yes | discovery set |
| K09 | endogeneity: crutch lock-in and delay reset | verify only | research |
| K10 | persistence mediation (equivariant family) | verify only | research |
| K11 | Z/F convergence in the absence of S-type maintenance | yes | research |
| K12 | surrogate mismatch collapses learning (**negative result**) | — | negative, retained |
| K13 | closure + restorativity assay boundary | yes | ledger row; **data not in this repo** (see ERRATA §2) |
| K14 | the write sets the delay length (dose–response) | yes | PASS (10 seeds) |
| K15 | plastic release: does installed amplitude survive? | yes | research |
| K16 | released-family split: the body controls re-acquisition | yes | research |
| M10 | S1 leverage trace (secondary replication port, 9 seeds) | yes | BELOW GATE 6/9, recorded |
| M11 | S2 write–reader factorial (secondary replication port, 9 seeds) | yes | PASS 9/9 |

Claims marked *discovery set* come from three seeds and are reported as
supporting characterisation, not as replicated findings — the brief and the
manuscript both label them this way.

**On K03's gate.** An earlier run recorded a `GATE FAILURE` for K03 that came
from an ad-hoc tier-B substitute criterion rather than the preregistered one.
Under the preregistered fixed-lag criterion — re-evaluated offline from the
saved anchor — K03 passes 10/10 (`ΔP@100 = +0.215 ± 0.062`,
`ΔP@500 = +0.536 ± 0.109`, both positive in 10/10 seeds; frozen-window
`snap=50` reversal 10/10; natF post-crossing collapse 8/8). `run.py` was
corrected and the correction is documented. Separately, a *branch-homogeneity*
observation (8 SHORT-RANGE-ONLY + 2 LATE) is recorded as a characterisation of
the 10-seed corpus; it is not a failure of the preregistered criterion.

### Additional experiments in `scripts/`

`run_wd1111_battery.py` (+ `train_wd1111_bridge.py`, `cb2_transplant.py`,
`cb34_readouts.py`) runs a four-part battery, CB1–CB4, checking that the
primary claims survive under attention-inclusive weight decay (`wd_1111`)
rather than the primary `wd_0011` regime. It is exploratory and non-gate: three
seeds, run to completion. CB2's early invalid runs are retained and labelled —
see [`ERRATA.md`](ERRATA.md) §3.

---

## What is *not* claimed

- No explanation of why optimization selects this organization.
- No complete mechanistic account of grokking, and no claim that the
  interpretation here is final.
- No closed bidirectional causal account against prior work that *removes*
  adaptive routing. The two intervention families act at different levels
  (wholesale replacement of the attention mechanism versus a parameter-level
  transplant inside one training regime).
- One task family (`c = a + b (mod 113)`, `a b =` format), one architecture
  (one-layer pre-norm Transformer, `d_model = 128`, 4 heads), and a primary
  regime that exempts the query–key block from weight decay.

---

## AI use disclosure

AI tools were used extensively during code development, experimental iteration,
analysis assistance and manuscript preparation. The author designed the
studies, verified all reported results, and is responsible for them.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the preprint for the scientific
claims and this repository for the code and data.

## License

Apache License 2.0 for the program source; the archived `*.pkl` measurement
data is CC BY 4.0. The split is stated in [`NOTICE`](NOTICE) — Apache 2.0 is a
software licence and does not sensibly cover measurement data, so the two are
licensed separately. The full Apache text is in [`LICENSE`](LICENSE).
