# texture_provenance — Status

**Status**: A3b texture "derive or count" audited.
Verdict: **TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT**.

Roadmap gate A3b. A3a (`unified_boundary/`) unified the transfer boundary `H_Q`
and deferred deriving the quark depths `{0,2,6}` and the Clebsch/coin factors.
A3b is the honest derive-or-count ledger for those texture factors — where theory
vs numerology is decided.

**Result (the ledger is mixed):** the texture's group-theoretic / geometric
factors are *derived*; the generation *hierarchy* is free input.

- **Derived (7 factors, B1):** `C_F = 4/3` (SU(3) Casimir), coin base `sqrt(5)`
  from `Gamma_q^2 = 5I = (2_BCC + 3_color)`, BCC `sqrt(2)`/`1/sqrt(2)`,
  charged-lepton `sqrt(3/2)` from the residual port, the V10 leptonic phase word,
  and the quark CP phase `atan(sqrt(5))`. These give the texture STRUCTURE — the
  CP phases `atan(sqrt(5))` and `5 pi/12` and the PMNS angle structure.
- **Free input (4, B2):** the quark depth embedding `{0,2,6}` (fit to the CKM
  hierarchy; even+additive checked), the charged-lepton two-step depth, the
  flat-coin `r=1` ergodicity prior (-> `physical_vacuum_order_parameter_exists`),
  and the CP-phase branch. These set the HIERARCHY.
- **Count (B3):** `N_free = 4 < N_observables = 8` (surplus 4) -> predictive, not
  numerology.

**Honest scope:** the framework predicts the texture *structure*, conditional on
the free depth embedding. It does **not** derive the hierarchy: deriving `{0,2,6}`
is a generation mechanism (`N=3` is empirical per the closed
`triality/broken_triality/exceptional/topology` kills), recorded as the remaining
input `generation_depth_embedding_derived` and **not attempted** here.

## Gate status

| Gate | Status | Verdict |
|---|---|---|
| B1 — derived-factor ledger | done | **DERIVED_FACTORS_CATALOGUED** — 7 factors, each checked against its source |
| B2 — free-input enumeration | done | **FREE_INPUTS_ENUMERATED** — 4 inputs with sources |
| B3 — parameter count | done | **TEXTURE_PREDICTIVE** — `N_free=4 < N_obs=8`, surplus 4 |
| B4 — combined verdict | done | **TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT** |

## What this does and does not establish

- **Does:** the CKM/PMNS texture's group-theory/geometry factors and the CP phases
  are genuine derived predictions, with fewer free inputs than observables — so the
  textures are predictive for *structure*, not pure numerology.
- **Does not:** derive the magnitude hierarchy. The quark depth embedding is fit to
  the CKM hierarchy; deriving it requires a generation mechanism, which the
  algebraic/topological kills show does not exist (`N=3` empirical). So the CKM/PMNS
  *magnitudes* remain conditional on the depth embedding.

## Cross-module dependency

Reuses `boundary_response` texture-factor functions (quark Clebsch/coin/depth,
charged-lepton leakage, V10 leptonic holonomy) and the `local_boundary_fiber`
ergodicity-prior remaining-input marker. All imports via `reuse.py`.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/texture_provenance/tests -q
```

Expected: 16 passing. A decisive negative control
(`test_verdict_kills_when_inputs_match_or_exceed_observables`) confirms the count
gate can fail: `parameter_count_verdict(8, 8) == TEXTURE_NUMEROLOGY_KILL`.
