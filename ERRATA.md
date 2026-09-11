# Errata and release notes

This repository distributes the reproduction package **as sealed**. Files under
`repro_package_zp/` are byte-identical to the sealed archive whose checksums
are recorded in `repro_package_zp/manifests/MANIFEST.json`, so that
`repro_package_zp/verify_extract.py` verifies end to end.

That constraint has one consequence worth stating plainly: where a sealed file
is wrong, this release **records the error rather than silently rewriting the
evidence**. The items below are the complete list of known deviations, caveats
and omissions.

---

## 1. K14 verdict string disagreed with its own anchor — **corrected in seal v1.17**

**Where.** `repro_package_zp/tenseed_out/ten_seed_verdict.json`, key
`gate_verdicts.K14_dose_response`.

**What it says.**

```
censoring fraction 0/2/7/8/9 out of 10 across lambda=0/.25/.5/.75/1.0
```

**What the anchor says.** Recomputing from the authoritative per-seed anchor
`repro_package_zp/claims/K14_dose_response/results_10seed.pkl` (where a run is
censored iff its `cross` field is `None`, cap = 12000 steps) gives:

| arm | censored | observed crossing steps | median |
|---|---|---|---|
| L000 | **0/10** | 1700, 1800, 1875, 1875, 1925, 1975, 2000, 2050, 2050, 2300 | 1950 |
| L025 | **2/10** | 7025, 7575, 8525, 8625, 10600, 11775, 11900, 11975 | 9612 |
| L050 | **6/10** | 10025, 10625, 10975, 11525 | 10800 |
| L075 | **8/10** | 10850, 11925 | 11387 |
| L100 | **9/10** | 12000 | 12000 |
| SELF | **0/10** | 1975, 2325, 2375, 2375, 2425, 2450, 2500, 2575, 2925, 3025 | 2438 |
| RND  | **0/10** | 5675, 6375, 6675, 7225, 7500, 7725, 7800, 9175, 10350, 11200 | 7612 |
| OPER | **5/10** | 9300, 10375, 11000, 11700, 11850 | 11000 |

The dose series is therefore **0 / 2 / 6 / 8 / 9** out of 10 (monotone
non-decreasing, 10/10 seeds), **not** 0 / 2 / 7 / 8 / 9.

**Which number is right.** The `results_10seed.pkl` value. It is the anchor:
`claims/K14_dose_response/verify.py` checks against it, and the manuscript and
the four-page brief both report `6/10` for L050 with "4 observed" crossings,
which is internally consistent with the table above. The `.../7/...` in the
verdict JSON is an error in a prose convenience summary, not in the measurement.

**Status: corrected in seal v1.17-reconciliation (2026-09-11).** The string now
reads `0/2/6/8/9`, re-derived from the anchor by script rather than retyped. It
was *not* edited silently into seal v1.16: that would have changed the file's
sha256 and broken the hash every other file is verified against. Instead the
package was re-sealed, and the new `MANIFEST.json` `change` field records exactly
what was reconciled — see §1c for the full accounting of the two-file delta.

**To check it yourself.**

```python
import pickle
d = pickle.load(open('repro_package_zp/claims/K14_dose_response/results_10seed.pkl', 'rb'))
for cond in ['L000', 'L025', 'L050', 'L075', 'L100']:
    # a run is censored iff it never crossed within the window
    n = sum(1 for s in d['arms'] if d['arms'][s][cond]['cross'] is None)
    print(cond, f'{n}/10 censored')
# L000 0/10   L025 2/10   L050 6/10   L075 8/10   L100 9/10
```

---

## 1b. The WD1111 battery summary carries the **invalid** CB2 run

**Where.** `repro_package_zp/wd1111_battery/wd1111_battery_summary.json`, key
`cb2_transplant`.

**What it says** — against what the corrected per-claim file says:

| | summary JSON | `cb2_transplant/cb2_results.json` (corrected) |
|---|---|---|
| seed 0 | `SQS=None`, verdict `NOT-CARRIER` | `SQS=10875`, verdict `CARRIER-HOLDS` |
| seed 1 | `SQS=None`, `FQS=9025`, verdict `MIXED` | `SQS=8400`, `FQS=13275`, verdict `CARRIER-HOLDS` |
| seed 2 | `SQS=None`, verdict `NOT-CARRIER` | `SQS=15925`, verdict `CARRIER-HOLDS` |

The summary's `cb2_transplant` section is **identical to the content of
`cb2_results_INVALID_zerograd_bug.json`** — verified by direct comparison
(`summary == invalid` → true; `summary == corrected` → false). Its verdict is
therefore the *opposite* of the measurement.

**Which is right.** The corrected per-claim file. The invalid run is the one
described in `repro_package_zp/README.md`: two early `cb2_transplant.py` runs
omitted `opt.zero_grad()`, so gradients accumulated across steps and training
stalled at chance. The corrected rerun scores `CARRIER-HOLDS` in 3/3 seeds, which
is what `repro_package_zp/README.md` states.

**What the manuscript says.** It quotes the **corrected** numbers — "the
transplant factorial scores `Carrier-Holds` in 3/3 ... on the slow body (crossing
8400–15925) and on the fast body alike (11050–13275) ... while the initial
construction imposes the fast fate on either body (1150–1950)" — and all three
ranges match `cb2_results.json` exactly. Paper, anchor and package README agree;
only the aggregate disagrees.

**Status: corrected in seal v1.17-reconciliation (2026-09-11).** The section now
carries the corrected values, `CARRIER-HOLDS` in 3/3 seeds. Its three sibling
sections (`cb1_fate`, `cb3_lookup`, `cb4_fourier`) were checked and **did** match
their per-claim files, so the defect was isolated to CB2 — the one CB section
corrected after the summary was written.

---

## 1c. The pattern behind §1 and §1b — read the anchors, not the aggregates

Both defects have the same root cause: **an aggregate file was written before a
correction and never rebuilt.** The K14 verdict string predates a recount; the
CB2 section predates the `opt.zero_grad()` fix. Both are now corrected.

| file | status |
|---|---|
| `claims/*/results*.pkl` | **authoritative** — the per-claim anchors |
| `claims/*/` per-experiment JSON (e.g. `cb2_results.json`) | **authoritative** |
| `repro_package_zp/README.md` | **authoritative** — states the corrected conclusions |
| the manuscript and the brief | **authoritative** — quote the corrected numbers |
| `tenseed_out/ten_seed_verdict.json` | corrected in v1.17 (§1) |
| `wd1111_battery/wd1111_battery_summary.json` | corrected in v1.17 (§1b) |

**Rule that still applies: treat the per-claim anchors and the package README as
the record; treat the aggregate summaries as convenience digests.** If an
aggregate and an anchor ever disagree, the anchor wins — the v1.17 corrections
were themselves made by re-deriving from the anchors, not by retyping.

### The v1.17 delta, stated exactly

`seal v1.17-reconciliation` differs from `seal v1.16-tierA-final` in **exactly
two files**:

```
tenseed_out/ten_seed_verdict.json            (K14 verdict string)
wd1111_battery/wd1111_battery_summary.json   (cb2_transplant section)
```

The other **121 file hashes are unchanged**, the 123-entry key set is identical,
the 99 external anchors are identical, and `n_files` is unchanged at 123. The
new `MANIFEST.json` sha256 is
`54819e8233d84ea8ff0495850406489984d9edaa10868558931889b8c0bb90ea`.

Because the v1.16 file set and the v1.17 file set are the same 123 paths, a
verifier holding either manifest can diff them and confirm the delta is exactly
those two entries and nothing else.

This is exactly the class of defect a self-verifying package cannot catch:
`verify.py` re-runs the same code and compares against the anchor, so it never
reads the aggregates at all.

---

## 2. What `manifests/CHECKSUMS.txt` covers, and why most of it is absent here

`manifests/CHECKSUMS.txt` has **222 lines**, and they are not all package files:

| group | count | present in this repository |
|---|---|---|
| package files | 123 | 122 present, 1 omitted (see below) |
| external anchors | 99 | none — by design |

The 99 **external anchors** are raw training artefacts from the author's
`repro/` tree: bridge checkpoints (`bridge_zp_A`…`bridge_zp_D`, 4 arms × 6 seeds
× 4–5 files) and raw result pickles (`r80`–`r109`). They are the inputs from
which the package's archived `results*.pkl` files were extracted, and
`manifests/MANIFEST.json` records `external_anchors_missing: 0`, meaning all 99
were present on the seal machine when the seal was taken. They are several GB of
checkpoints and are not redistributed here.

**Consequence.** `sha256sum -c manifests/CHECKSUMS.txt` run unfiltered reports
~100 failures, all but one of which are these absent external anchors. Filter
them out first:

```bash
grep -v '^repro/' manifests/CHECKSUMS.txt > /tmp/pkg.sums
sha256sum -c /tmp/pkg.sums --ignore-missing
# expected: 122 OK, 0 FAILED
```

Verified against seal v1.17-reconciliation: **122 files hash-matched, 0
mismatches, 1 expected absence** (K13, below), and the sha256 of `MANIFEST.json`
matches the value recorded in `manifests/SEAL.txt`
(`54819e8233d84ea8ff0495850406489984d9edaa10868558931889b8c0bb90ea`).

**The one genuine omission.** `claims/K13_closure_boundary/results.pkl` is a
package file, not an external anchor, and it is missing. It is 144 MB and
exceeds GitHub's hard limit of 100 MB per file. Its sha256 is recorded in
`MANIFEST.json`, so its absence is detectable rather than silent.

K13 is an appendix ledger row in the full manuscript ("frozen-carrier closure").
It is **not** part of the evidence base of the four-page brief, whose headline
tables are produced by K01, K02, K03, K07 and K14 — all present and complete
here. The complete package, including this file, is deposited with the preprint
on Zenodo.

`repro_package_zp/verify_extract.py` is a different check: it is an *extraction
reconciliation* (C1–C5: task tables, split construction, model initialisation,
optimizer grouping, sampler gates) and it passes. It does not check file hashes.

---

## 3. Known-corrected runs retained in the archive

These are deliberate, not defects in the release.

- **CB2, `opt.zero_grad()` bug.** Two early `cb2_transplant.py` runs were
  invalid: the script omitted `opt.zero_grad()`, so gradients accumulated
  across steps and training stalled at chance. Their output is retained as
  `wd1111_battery/cb2_transplant/cb2_results_INVALID_zerograd_bug.json`; the
  corrected rerun is authoritative. See `repro_package_zp/README.md`.
- **K03: a `GATE FAILURE` string that was later withdrawn.** The package's
  Chinese-language `README.md` still records a failed *branch-homogeneity* gate
  (8 SHORT-RANGE-ONLY + 2 LATE). That failure string came from an ad-hoc tier-B
  substitute criterion, **not** from K03's preregistered criterion, and
  `tenseed_out/ten_seed_verdict.json` records K03 as
  **PASS (10/10)** under the preregistered fixed-lag criterion, re-evaluated
  offline from the saved anchor:
  `ΔP@100 = +0.215 ± 0.062` and `ΔP@500 = +0.536 ± 0.109`, both positive in
  10/10 seeds; frozen-window (`snap=50`) reversal 10/10; natF post-crossing
  collapse 8/8. `run.py` was corrected accordingly. The branch-homogeneity
  statistic remains a legitimate descriptive observation about the 10-seed
  corpus; it is simply not a gate, and the package README's wording predates
  the correction. **Take the verdict JSON, not the package README, as
  authoritative on K03.**
- **S1 (`J_K` trajectory): BELOW GATE, 6/9.** Downgraded as recorded; the
  failing seeds belong to a low-structure phenotype documented in
  `claims/M10_leverage_trace/`. M10 and M11 are 9-seed (`results_9seed.pkl`),
  not 10-seed.
- **K12 is a negative result** (forward/backward surrogate mismatch collapses
  learning) and is kept in the package as such.

---

## 3b. Findings of the pre-release audit (2026-09-11)

An independent audit of the load-bearing paths behind the four-page brief
(K01, K02, K03, K07, K14) was run before release. It recomputed every stated
number from the archived anchors using fresh code written from the brief's own
definitions, and separately tested whether the claim-specific `run.py` files
implement the operations the brief describes.

**Outcome: the headline result reproduces.** 22 of 23 checked cells match
exactly — including K03's two statistics to the printed digits, K07's two
`I(500)` ranges, K01's four cells plus the 19/19 durability count, and all
eight K14 cells together with their interquartile ranges. The code's crossing
rule was independently confirmed against the recorded `crossed` field in
**80/80** runs. The defects found are in the *prose describing* the operations,
not in the operations themselves — with one exception, which is flagged below
as blocking.

**These five defects were invisible to this package's own verification.** They
are divergences between the code and the prose describing it, or between an
archived quantity and the prose citing it. `verify.py` re-runs the same code,
and none of them is a numerical self-consistency failure.

**Resolution.** All five were corrected in the preprint documents
(`note.tex` / `main.tex`, both rebuilt) before release. **None required a code
change** — the `run.py` files were found to implement the operations correctly;
it was the write-up that misdescribed them. This package is therefore unchanged
and its seal still verifies.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | `K02` Table 2, `N_F + Q_0` lower bound is 1425; the anchor says **1400** | low (typo) | corrected to 1400 |
| 2 | "the initialization ($Q_0$)" is a *second* random init, not the donor's own | low (wording) | reworded to "a fresh initialization" |
| 3 | "Every transplant replays the donor batch chain exactly" is false for the two `Q_0` arms | medium | restated to what is true |
| 4 | "Acute readouts taken at transplantation (zero training) show the mismatched arms are locally functional at $t{=}200$" — the cited quantity is post-training, and measured at transplantation the mismatched arms are near chance | **high (blocking)** | sentence replaced with the at-transplantation measurement; conclusion retained on a stronger footing |
| 5 | The brief cites "the $1/113$ chance level"; the code records a macro chance of ≈0.030 | low | baseline corrected to the measured chance |

**1.** Arm `F-Q0` crossings over ten seeds are
`[1400, 1425, 1425, 1475, 1500, 1500, 1575, 1600, 1625, 1950]`. Seed 5 crosses
at 1400 and the anchor's own `crossed` field records 1400, so the table's
`1425–1950` is the range over nine seeds while the same row reports `10/10`.

**2.** `build_arm()` constructs a *second* `D57Model` to source the `Q_0` block.
Measured: `|1st construction − F ckpt Wq| = 0.000000e+00` but
`|2nd construction − F ckpt Wq| = 1.752440e-01`. The docstring flags the
call-order dependence. Both faithful arms of the 2×2 are unaffected; only the
matched control `(F, Q_0)` is sourced differently.

**3.** Because a `Q_0` arm constructs one extra model, it consumes extra global
torch RNG before the 200-discard loop. Seed 0 first training batch indices:
`Qs` arms `[825, 242, 3497, …]`; `Q_0` arms `[3097, 3392, 871, …]`. Within each
pair the chain is matched; across the pairs it is not, and the native donor used
a one-construction chain. Batch ordering is not a plausible cause of a
categorical 10/10 flip, but the claim as written is inaccurate.

**4 (blocking).** `run.py` computes `rec['acute']` *after* the training loop
(its own comment says so); `A2_cos_vs_SQs` is 0.350 for the `(S, Q_S)` arm
compared against its own reference, where it would be 1.0 at $t{=}200$. Measured
at transplantation with zero training steps (fixed probe batch, chance
$1/113 = 0.0088$), over the six seeds whose $t{=}200$ checkpoints were retained:
matched `S-Q_S` 0.152, matched `F-Q_0` 0.038, and the two **mismatched** arms —
which the brief had called "locally functional" — 0.013 and 0.028, i.e.
**1.4× and 3.1× chance**, the lowest of the four. The brief's claim that
transplanting carries no immediate behavioural effect was withdrawn. The
intended conclusion survives by a better route, and is what the documents now
state: both mismatched arms are *specifically* degraded at transplantation yet
diverge in fate, so the shared immediate effect cannot be what separates them.

**5.** `K03/run.py`'s `retrieval()` returns a macro-averaged chance over the
actual masked same-sum pools; the anchor records ≈0.030, implying an effective
pool near 33 rather than 113. `P8 ≈ 0.95` clears both baselines, so no claim is
at risk, but 1/113 is the more flattering of the two.

Full report, recomputation scripts and raw output: the author's audit working
directory (`audit_20260911/`), not part of this repository.

---

## 4. Language of the per-claim notes

`repro_package_zp/claims/*/README.md` and the `env/`, `base/` and
`common/` notes are the author's internal working documents. They are written
in mixed Chinese and English: the claim statements, status lines and audit
notes are in Chinese, while identifiers, file names, commands and verdict
strings are in English. This top-level `README.md` and this file are in
English. The archive is distributed as-is rather than translated, for the same
byte-identity reason as item 1.

---

## 5. Seal-machine details recorded verbatim

`repro_package_zp/env/ENVIRONMENT.md` records the measured environment of the
machine on which the seal was produced, including that interpreter's absolute
local path and the GPU model. This is deliberate: the bit-exact (Tier A)
verification tier is only meaningful on that machine, so the environment is
part of the record. It is left unmodified to preserve byte-identity. It
contains no credentials.

---

## 6. References to documents not included here

`repro_package_zp/README.md` cites the author's internal planning documents
(`REPRO_PLAN.md`, `CLAIM_LEDGER.md`, `TEN_SEED_RUN_PLAN.md`,
`EXPERIMENT_CENSUS.md`) by relative path. Those documents govern *which* claims
were admitted to the package; they are not needed to run or verify it, and they
are not part of this release. Nothing in the package's execution or
verification depends on them, with one exception:
`repro_package_zp/scripts/seal.py` reaches outside the package to re-hash raw
training outputs, so it can only re-seal on the original machine. Re-running a
claim does not use `seal.py`.

The reproducibility contract those documents define is restated in the
top-level `README.md` under "Two verification tiers".
