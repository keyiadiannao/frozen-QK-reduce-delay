# K14 — the write sets the delay length (dose–response)

- **Claim**: the installed row-2 operand code *causally lengthens* the
  delay, with a graded (monotone) response to the installed amplitude —
  a knob, not a switch. The delay is carried by the operand-addressing
  code, **not** by the self-position suppression.
- **Source**: `repro/r107_code_amplitude.py` (archived run, §6eh);
  package port `run.py` is a verbatim copy with only the import shim,
  `REPRO_DIR`, `K14_SMOKE` and the output path changed.
- **Preregistration**: EXTENSION_SUMMARY §6eh (design + decision rule
  locked before the run).
- **Archived output**: `repro/r107_code_amplitude_results.pkl`
  (3 seeds, cap 12000, QK frozen, install gate ≤ 1.9e-06, HARNESS PASS).
- **Status**: DONE (2026-09-03) — ported, smoke-tested, full run and
  `verify.py` bit-exact reconciliation.

## Design

| Arm | Content installed | Purpose |
|-----|-------------------|---------|
| L000/L025/L050/L075/L100 | λ·(operand code + self term) | dose–response ladder |
| SS | natural $S$ state | positive control |
| SELF | $h_2$-aligned part (full self-term change) | isolates self-suppression |
| OPER | $h_2$-orthogonal part (zero self-term change) | isolates the operand code |
| RND | random injective code, matched norm, $\perp h_2$ | form vs particular code |

Query–key block frozen throughout; body from the $S_{200}$ checkpoint.
Primary readout: first step with plain validation accuracy $> 0.9$.

## Result (3 seeds, cap 12000)

| seed | L000 | L025 | L050 | L075 | L100 | SS | SELF | OPER | RND |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1875 | 10950 | 12200 | cens | cens | cens | 2325 | cens | 10325 |
| 1 | 2050 | 9050 | 10500 | cens | cens | cens | 2375 | 10025 | 7675 |
| 2 | 1925 | 11050 | cens | cens | cens | cens | 2575 | 11900 | 7950 |

Final $m_2$ (accuracy on validation pairs whose swapped twin was held
out): L000 0.99–1.00; **SELF 0.99–1.00** (fully generalized, like the
baseline); L100 0.05–0.85; SS 0.04–0.65.

**Verdict: DOSE-RESPONSE** (3/3 seeds monotone). Self-suppression
alone leaves generalization intact; the operand code alone, or an
arbitrary injective code, reproduces most of the delay.

## Boundary (state with the claim)

The applicability condition is the **frozen query–key block**: with the
block frozen the model cannot revise the installed routing, so what is
measured is the delay *imposed* by that routing. Amplitude scaling after
release is **not** tested.
