# texture_provenance Sidecar Parameter Ledger

The point of this sidecar: classify every CKM/PMNS texture factor as **derived**
(machine-checked source) or **free input**, then count free inputs vs observables.

## Inherited (read-only)

From `boundary_response/`:
1. Quark Clebsch/coin data: `C_F = 4/3`, BCC `sqrt(2)` and `1/sqrt(2)`, coin sum
   `Gamma_q^2 = 5I`, phase `atan(sqrt(5))` (`quark_clebsch_factors.py`,
   `quark_boundary_shell.py`).
2. Quark depth embedding `{1:0, 2:2, 3:6}` and transition depths `{2,4,6}`
   (`quark_transfer_hierarchy.py`).
3. Charged-lepton residual port `e1 = sqrt(2/3) a + 1/sqrt(3) u`, rotation sine
   `sin(theta_e) = sqrt(3/2) eps^2` (`charged_lepton_leakage.py`).
4. V10 leptonic phase word, verdict `LEPTONIC_PHASE_WORD_DERIVED_PASS`, phase
   `5*pi/12` (`leptonic_boundary_holonomy.py`).
5. Ergodicity-prior remaining input `physical_vacuum_order_parameter_exists`
   (`local_boundary_fiber.py`).

## B1 — derived factors (7, each with a machine-checked source)

| factor | value | source check |
|---|---|---|
| color Casimir `C_F` | `4/3` | `sum_A T^A T^A == (4/3) I_3` |
| coin base | `sqrt(5)` | `Gamma_q^2 == 5 I` and `5 == 2_BCC + 3_color` |
| quark CP phase | `atan(sqrt(5))` | from coin base, given `r=1` |
| BCC symmetric | `sqrt(2)` | two-path Clebsch |
| BCC antisymmetric | `1/sqrt(2)` | two-path Clebsch |
| charged-lepton mixing | `sqrt(3/2)` | `1/<a|e1> = sqrt(3/2)` from residual port |
| leptonic phase word | `-5*pi/12` | holonomy verdict `LEPTONIC_PHASE_WORD_DERIVED_PASS` |

## B2 — free inputs (4)

1. **quark depth embedding** `{0,2,6}` — fit to the CKM hierarchy. Even+additive
   structure is *checked* (so given that constraint, the integers are the knob).
2. **charged-lepton two-step depth** (= 2).
3. **ergodicity prior `r=1`** — the single branching input (Jaynes max-entropy /
   equal-degeneracy / regular fiber -> `physical_vacuum_order_parameter_exists`).
4. **CP-phase branch** — a discrete sign.

## B3 — count

- `N_free = 4`
- `N_observables = 8` (CKM 3 angles + 1 phase; PMNS 3 angles + 1 phase)
- `surplus = 4 > 0` -> `TEXTURE_PREDICTIVE` (not numerology).
- Genuine derived predictions: the two CP phases (`atan(sqrt(5))`, `5*pi/12`),
  the PMNS angle structure, and the CKM additivity relation
  `depth_13 = depth_12 + depth_23`.

## Continuous parameters

None. The audit is exact-symbolic; no fitting parameters. The "free inputs" are
discrete structural choices (integer depths, a binary branch, a max-entropy prior),
not tuned continuous knobs.

## Remaining declared input (the deferral)

- `generation_depth_embedding_derived` — derive the depth embedding `{0,2,6}` from
  a generation mechanism. Open and tied to `N=3` being empirical per the closed
  `triality/broken_triality/exceptional/topology` kill-sidecars. **Not attempted**
  here: A3b decides derive-vs-count for the texture factors; deriving the
  generation hierarchy is a separate (open) problem.
