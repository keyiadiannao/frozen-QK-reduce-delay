# Reproduction package — Early Query–Key Learning Determines the Grokking Delay in Modular Addition

Code and archived per-seed measurements for a study of the **grokking delay** in
a one-layer Transformer trained on modular addition (`c = a + b (mod 113)`).

The claim under test: a brief early window of query–key learning is causally
responsible for much of the delay. Freezing `W_Q W_K` at initialization gives
early crossing, while freezing it only *after* the first 200 steps leaves the
network on a long structured-lookup plateau; a 2×2 state-factorial transplant at
`t = 200` inverts with the query–key block in both directions and in every seed.
This repository is the evidence base for the published documents, released so
the numbers can be recomputed rather than taken on faith.

- **Preprint** (four-page brief + full manuscript):
  [`10.5281/zenodo.22708281`](https://doi.org/10.5281/zenodo.22708281)
- **Read [`ERRATA.md`](ERRATA.md) before citing or reusing anything here.** It
  records one known error in a sealed summary file, one omitted data file, the
  retained invalid runs, and the findings of a pre-release audit.

---

## Layout

```
├── ERRATA.md                  known errors, omissions, caveats  ← read this
├── NOTICE                     licence scope: code Apache 2.0, data CC BY 4.0
├── CITATION.cff               citation metadata
└── repro_package_zp/          the sealed reproduction package
    ├── env/                   requirements.txt + the two-tier contract
    ├── common/                shared library: task, model, states, gates, readouts
    ├── base/                  base-training entry point + snapshot spec
    ├── claims/                one directory per claim (K01–K16, M10, M11)
    ├── scripts/               drivers: 10-seed pipeline, WD1111 battery, seal
    ├── manifests/             MANIFEST.json — sha256 of every file and anchor
    └── tenseed_out/           recorded 10-seed verdicts
```

Each `claims/<ID>/` directory holds `README.md` (claim statement and status),
`run.py` (re-runs the experiment and applies its gate), `verify.py` (checks a
rerun against the archived anchor), and `results*.pkl` (the archived per-seed
measurement — the anchor).

The model class is named `D57Model`. The name is legacy: the architecture was
transcribed verbatim from the authors' earlier dihedral-training script. This
package is entirely about `c = a + b (mod 113)`; the dihedral cross-family
replication is separate prior work and is not imported here.

---

## First: verify what you downloaded

```bash
git clone https://github.com/keyiadiannao/frozen-QK-reduce-delay
cd frozen-QK-reduce-delay/repro_package_zp
pip install -r env/requirements.txt     # numpy>=2.1, torch>=2.4; CPU fallback works
```

**Seal integrity** — the expected result is **122 files verified, 0 mismatches**:

```bash
grep -v '^repro/' manifests/CHECKSUMS.txt > /tmp/pkg.sums
sha256sum -c /tmp/pkg.sums --ignore-missing
```

> ⚠️ Do **not** run `sha256sum -c manifests/CHECKSUMS.txt` unfiltered.
> `CHECKSUMS.txt` has 222 lines: 123 package files plus **99 external anchors** —
> raw training checkpoints from the author's `repro/` tree, deliberately not
> shipped here. Unfiltered it reports ~100 failures, all but one of which are
> those absent anchors. The one genuine omission is
> `claims/K13_closure_boundary/results.pkl`. See [`ERRATA.md`](ERRATA.md) §2.

**Extraction reconciliation** — confirms the shipped `common/` library
reproduces the original research code (C1–C5):

```bash
python verify_extract.py     # -> EXTRACT RECONCILIATION: ALL PASS
```

The `repro/ not found` and `skip ... ckpt missing` lines are expected; both refer
to the author's raw training tree.

`manifests/SEAL.txt` pins the seal and the sha256 of `MANIFEST.json` itself, so
the manifest can be authenticated without trusting the file list:

```
seal: v1.16-tierA-final        n_files: 123
sealed_at_utc: 2026-09-07      n_external_anchors: 99
MANIFEST.json sha256: c90212decd5dc97fa6817506dfa40d4e74303c5b4cec1f22bea680b9acf5752c
```

---

## Checking a number without retraining

This is the fastest way to check a figure quoted in the paper. It compares the
archived anchor against itself and reports per-field agreement:

```bash
cd claims/K01_fate_lock
python verify.py
```

To re-run an experiment from scratch and apply its gate:

```bash
cd claims/K01_fate_lock
python run.py
```

### The full 10-seed pipeline

```bash
python scripts/run_ten_seed.py --repro-dir /path/bridges --out /path/tenseed_out
```

Every stage writes a completion marker, so re-running resumes rather than
restarts. Stages can be run separately (`--stage 1` … `--stage 7`), which is
recommended for monitoring. Indicative cost on the seal machine: stage 1 ≈ 50 min,
2 ≈ 2 min, 3 ≈ 40 min, 4 ≈ 1 min, 5 ≈ 3.5 h, 6 ≈ 3–4 h; about 3 GB of disk.

---

## Two verification tiers

Defined in `repro_package_zp/env/ENVIRONMENT.md`.

| tier | what it checks | where it is meaningful |
|---|---|---|
| **Tier A** — bit-exact anchor | every field of a rerun matches the archived `results*.pkl` bitwise (`max\|Δ\| = 0`) | **only on the seal machine**, with the same torch build and GPU model |
| **Tier B** — verdict / statistical | the *behavioural* verdict reproduces: crossing falls in the recorded band, the fate classification agrees, the dose structure holds, preregistered gates pass, means ± SD agree over seeds | **any environment**, including CPU fallback and rented GPU servers |

Tier A proves the archived numbers were extracted without drift; it is not a
requirement for using the package. Cross-hardware bit flips are expected and are
not a failure — a rerun that reproduces the verdicts while disagreeing in the
last bits has succeeded. Seal machine: numpy 2.4.3 / torch 2.11.0+cu128,
RTX 5060 8 GB, `torch.set_num_threads(4)`.

---

## Claim inventory

The brief's tables come from **K01, K02, K03, K07, K14**; the rest support the
full manuscript. Verdicts are as recorded in `tenseed_out/ten_seed_verdict.json`.

| ID | claim | evidence |
|---|---|---|
| K01 | fate lock at `t = 200` | PASS 10 seeds |
| K02 | query–key carrier transplant (+ joint-training requirement) | PASS 10 seeds |
| K03 | identity geometry: maintained under S, self-destructs under F | PASS 10/10 |
| K04 | operand-level scalar scoring functional | discovery set, 3 seeds |
| K05 | formation = early direction selection + amplification | discovery set, 3 seeds |
| K06 | Add arm sufficiency / remove arm confounded | discovery set, 3 seeds |
| K07 | freezing the trained carrier gives identity-maintaining dynamics | PASS (frozen gates) |
| K08 | restorative stability + basin boundary | discovery set |
| K09 | endogeneity: crutch lock-in and delay reset | research |
| K10 | persistence mediation (equivariant family) | research |
| K11 | Z/F convergence in the absence of S-type maintenance | research |
| K12 | surrogate mismatch collapses learning | **negative result**, retained |
| K13 | closure + restorativity assay boundary | data omitted — see ERRATA §2 |
| K14 | the write sets the delay length (dose–response) | PASS 10 seeds |
| K15 | plastic release: does installed amplitude survive? | research |
| K16 | released-family split: the body controls re-acquisition | research |
| M10 | S1 leverage trace (9 seeds) | BELOW GATE 6/9, recorded |
| M11 | S2 write–reader factorial (9 seeds) | PASS 9/9 |

Claims marked *discovery set* come from three seeds and are reported as
supporting characterisation, not replicated findings — the brief and the
manuscript label them the same way. `scripts/run_wd1111_battery.py` additionally
runs an exploratory CB1–CB4 battery under attention-inclusive weight decay
(three seeds, non-gate); its early invalid CB2 runs are retained and labelled, as
is an earlier withdrawn `GATE FAILURE` string for K03 — see
[`ERRATA.md`](ERRATA.md) §3.

---

## Scope

One task family (`c = a + b (mod 113)`, `a b =` format), one architecture
(one-layer pre-norm Transformer, `d_model = 128`, 4 heads), and a primary regime
that exempts the query–key block from weight decay. The paper states what is and
is not claimed; this repository makes no claim beyond it.

AI tools were used extensively during code development, experimental iteration,
analysis assistance and manuscript preparation. The author designed the studies,
verified all reported results, and is responsible for them.

## Citation and licence

Cite the preprint for the scientific claims and this repository for the code and
data — see [`CITATION.cff`](CITATION.cff). Program source is Apache 2.0; the
archived `*.pkl` measurement data is CC BY 4.0. The split is stated in
[`NOTICE`](NOTICE).
