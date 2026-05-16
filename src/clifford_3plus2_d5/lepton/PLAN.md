# Leptonic Bridge Laboratory — Brainstorm v9

Status: 9th iteration. Incorporates v8 reviewer fixes. Combinatorial intertwiner search before symbolic; BLOCK_SUPPORT mask corrected; char-poly spectrum matching; explicit gauge pairing; side-restriction J compatibility; Lab A primary verdict renamed.

## Context

Eight prior iterations refined the plan structurally. v9 closes seven implementation-level issues from v8 and reorders the intertwiner search to mitigate the "symbolic solver stall" risk.

### v9 fixes

1. **`BLOCK_SUPPORT` was no-op**. `product(c.blocks, c.blocks)` with `blocks = ((0,2), (2,6))` permits all entries in `[0,6) × [0,6)` — no constraint. **Fix**: use an explicit allowed-pair mask, e.g. `allowed_block_pairs: tuple[tuple[int, int], ...] = ((0,0), (1,1), (0,1))` for "diagonal blocks plus singlet→doublet off-diagonal."
2. **Continuous intertwiner families are common, not rare**. Degenerate spectra (two doublet pairs at same angle in R^6) generically give parametric orthogonal solutions. Returning `not_solved` stalls scans. **Fix**: combinatorial pre-search first — signed permutations, block permutations, split-reassignment matrices — before invoking the generic quadratic solver. The first two cover most physical wall transitions.
3. **`sorted_eigenvalues` fragile on equivalent exact expressions**. SymPy may distinguish `1/sqrt(3)` from `sqrt(3)/3`. **Fix**: use characteristic polynomial comparison: `U_left.charpoly() == U_right.charpoly()` (or `.minpoly()` if multiplicity is irrelevant).
4. **Gauge generator ordering assumption**. `zip(left_gauge_gens, right_gauge_gens)` assumes matched indexing and basis. **Fix**: store `gauge_pairs: tuple[tuple[sp.Matrix, sp.Matrix], ...]` explicitly. Or compute `right_g = T · left_g · T^T` from the wall and compare spans, not list order.
5. **J compatibility check too strict in domain wall**. Checking `J == ±J_left` as full matrices assumes the same global frame. **Fix**: side-restriction test — `P_left · J · P_left == ±J_left`, where `P_left` is the projector onto the left side's support. Document the frame convention explicitly.
6. **`load_bearing_qca_bridge` semantics**. `domain_wall_candidate` is a different bridge standard. **Fix**: keep `load_bearing_qca_bridge = False` for domain_wall_candidate regardless. The field's invariant is "satisfies the original carrier-first global (2,4) bridge contract." Domain-wall results report `load_bearing_domain_wall_candidate` as a separate field instead.
7. **Lab A primary verdict naming**. v8 renamed Lab A honestly. v9 enforces it at the verdict-label level: Lab A's positive outcome is `CLOCK_PLANE_CLOSURE_CANDIDATE` unless `max_primitive_class == ARBITRARY_REAL_ORTHOGONAL`, in which case the verdict promotes to `BRIDGE_CANDIDATE_J_BLIND`.

### Implementation-order change (per v8 reviewer advice)

Domain-wall intertwiner enumeration runs in three tiers, cheapest first:

- **Tier 1**: explicit split-reassignment matrices. Catalogued forms: permutation matrices that swap singlet/doublet block assignments. Finite enumeration (≤ 24 for R^6 with 3 mode pairs).
- **Tier 2**: signed permutations and block permutations. Finite, larger but still tractable (≤ 2^6 × 6! = 46080; usually much smaller after locality and intertwining constraints).
- **Tier 3**: generic quadratic intertwiner solver (the symbolic risk path). Used only after Tier 1+2 are exhausted.

If Tier 1 produces a `DOMAIN_WALL_CANDIDATE`, Tier 3 is unnecessary for the main research question.

## Goal (unchanged from v8)

Three labs on the additive v2 path:

- Lab A (R^4): clock-plane primitive closure to `M_2(ℂ)`. Primary verdict: `CLOCK_PLANE_CLOSURE_CANDIDATE`.
- Lab B strict (R^6) + structural wall (R^6): regression verification of Route 1 at R^6.
- Lab B domain wall (R^6): the new mechanism. Verdict: `DOMAIN_WALL_CANDIDATE` (distinct from `BRIDGE_CANDIDATE`).

## Pre-implementation: additive v2 path

All v8 sections V1–V11 carry forward unchanged unless noted.

### V8.fix1 (revised) — `LocalityModel.BLOCK_SUPPORT` with explicit pair mask

```python
@dataclass(frozen=True)
class LocalityConstraint:
    model: LocalityModel
    sites: tuple[int, ...] = ()
    blocks: tuple[tuple[int, int], ...] = ()             # (start, end) per block
    allowed_block_pairs: tuple[tuple[int, int], ...] = ()  # (block_idx_row, block_idx_col)
    radius: int = 0

def locality_constraints(T: sp.Matrix, model: LocalityModel, c: LocalityConstraint) -> list:
    n = T.shape[0]
    eqs = []
    if model == LocalityModel.BLOCK_SUPPORT:
        # Build site → block-index map
        block_idx_of_site = [-1] * n
        for idx, (s, e) in enumerate(c.blocks):
            for site in range(s, e):
                block_idx_of_site[site] = idx

        for i in range(n):
            for j in range(n):
                row_block = block_idx_of_site[i]
                col_block = block_idx_of_site[j]
                if row_block < 0 or col_block < 0:
                    eqs.append(T[i, j])   # outside any block — must vanish
                    continue
                if (row_block, col_block) not in c.allowed_block_pairs and \
                   (col_block, row_block) not in c.allowed_block_pairs:
                    eqs.append(T[i, j])
    # ... other models as v8
    return eqs
```

For Lab B domain wall at R^6 with singlet block `[0,2)` and doublet block `[2,6)`:
- `blocks = ((0, 2), (2, 6))`.
- For a transition that swaps singlet ↔ doublet assignment: `allowed_block_pairs = ((0, 1), (1, 0))` (off-diagonal only).
- For a transition that reorients within blocks: `allowed_block_pairs = ((0, 0), (1, 1))` (block-diagonal).
- For a fully general intertwiner mixing both: `allowed_block_pairs = ((0,0), (1,1), (0,1), (1,0))`.

Tier 1's split-reassignment matrices use the off-diagonal mask.

### V12 (new) — Char-poly spectrum match

```python
def spectrum_matches(A: sp.Matrix, B: sp.Matrix) -> bool:
    """Compare characteristic polynomials, robust to algebraic-equivalent expressions."""
    return sp.simplify(A.charpoly().as_expr() - B.charpoly().as_expr()) == 0
```

Replaces `sorted_eigenvalues(A) == sorted_eigenvalues(B)` everywhere in the intertwiner solver.

### V13 (new) — Tier 1: explicit split-reassignment enumeration

```python
def enumerate_split_reassignment_intertwiners_r6(
    U_left: sp.Matrix,
    U_right: sp.Matrix,
    *,
    left_blocks: tuple[tuple[int, int], ...],
    right_blocks: tuple[tuple[int, int], ...],
) -> tuple[sp.Matrix, ...]:
    """Tier 1: explicit permutation matrices that reassign mode-pairs between
    singlet and doublet blocks, then verify intertwining property.
    """
    n = U_left.shape[0]
    candidates = []
    # Enumerate permutations of mode-pair indices that swap the singlet pair
    # for each doublet pair, with sign options.
    mode_count = n // 2

    for perm in mode_pair_permutations(mode_count):
        for sign_pattern in sign_patterns(mode_count):
            T = build_mode_pair_permutation_matrix(perm, sign_pattern, n)
            if not is_real_orthogonal(T):
                continue
            if (T * U_left - U_right * T).applyfunc(sp.simplify) == sp.zeros(n):
                candidates.append(T)
    return tuple(candidates)
```

Cheap, finite. Most physical wall transitions are in this class.

### V14 (new) — Tier 2: signed/block permutation enumeration

Wider catalogue: signed permutations of basis vectors, block permutations of full blocks, etc. Still finite (bounded by `2^n × n!`); for n=6 that's ~46k, but locality + intertwining cut it dramatically.

### V15 (revised) — Tier 3: generic quadratic intertwiner solver (v8's V7)

Used only if Tier 1 and Tier 2 produce zero candidates.

### V16 (new) — Tiered `solve_T_intertwiner_orthogonal`

```python
def solve_T_intertwiner_orthogonal(
    U_left, U_right, *,
    locality_constraint, dimension=None, max_intertwiner_basis=16,
) -> tuple[tuple[sp.Matrix, ...], str]:
    """Returns (intertwiners, source_tier).

    source_tier: "split_reassignment" | "signed_permutation" | "block_permutation" | "generic_quadratic" | "underdetermined" | "no_solutions"
    """
    if not spectrum_matches(U_left, U_right):
        return (), "no_solutions"

    # Tier 1
    tier1 = enumerate_split_reassignment_intertwiners_r6(U_left, U_right, ...)
    if tier1:
        return tier1, "split_reassignment"

    # Tier 2
    tier2 = enumerate_signed_block_permutation_intertwiners(U_left, U_right, ...)
    if tier2:
        return tier2, "signed_permutation"   # or "block_permutation"

    # Tier 3 (symbolic)
    tier3, status = solve_quadratic_intertwiner_generic(U_left, U_right, ...)
    if status == "parametric_family":
        return (), "underdetermined"
    return tier3, "generic_quadratic" if tier3 else "no_solutions"
```

The `source_tier` is recorded on every wall candidate result, so reports show which tier supplied the bridge candidate.

### V17 (revised) — Explicit gauge pairing in `WallContext`

```python
@dataclass(frozen=True)
class WallContext:
    gauge_pairs: tuple[tuple[sp.Matrix, sp.Matrix], ...]  # (g_left, g_right) explicit
    left_complex_structure: sp.Matrix
    right_complex_structure: sp.Matrix
    transition: sp.Matrix
    locality_constraint: LocalityConstraint

    # Side projectors (for V18 J compatibility)
    left_side_projector: sp.Matrix
    right_side_projector: sp.Matrix

    def consistency_certified(self) -> bool:
        n = self.transition.shape[0]
        T = self.transition

        # T is orthogonal
        if (T.T * T - identity(n)).applyfunc(sp.simplify) != sp.zeros(n):
            return False

        # Intertwining for each gauge pair
        for g_left, g_right in self.gauge_pairs:
            if (T * g_left - g_right * T).applyfunc(sp.simplify) != sp.zeros(n):
                return False

        # Intertwining for J
        if (T * self.left_complex_structure - self.right_complex_structure * T).applyfunc(sp.simplify) != sp.zeros(n):
            return False

        # Side projectors are complementary and J-commuting
        if (self.left_side_projector + self.right_side_projector - identity(n)).applyfunc(sp.simplify) != sp.zeros(n):
            return False

        return True
```

### V18 (revised) — Side-restriction J compatibility

```python
def side_local_gauge_with_wall_transition_v9(
    profile, algebra, center, target_projectors,
    j_candidates, wall_context,
):
    """J-candidate passes iff its side-restrictions match ±J_left and ±J_right
    (with consistent global sign), and J commutes with each gauge_pair's left side.
    """
    if not wall_context.consistency_certified():
        return CommutantPolicyResult.GAUGE_MISALIGNMENT

    P_L = wall_context.left_side_projector
    P_R = wall_context.right_side_projector
    J_L = wall_context.left_complex_structure
    J_R = wall_context.right_complex_structure

    passing = []
    for j_cand in j_candidates:
        J = j_cand.matrix

        # Side restriction tests
        left_restriction = (P_L * J * P_L).applyfunc(sp.simplify)
        right_restriction = (P_R * J * P_R).applyfunc(sp.simplify)

        sign_left = +1 if left_restriction == (P_L * J_L * P_L).applyfunc(sp.simplify) else \
                    -1 if left_restriction == (-P_L * J_L * P_L).applyfunc(sp.simplify) else 0
        sign_right = +1 if right_restriction == (P_R * J_R * P_R).applyfunc(sp.simplify) else \
                     -1 if right_restriction == (-P_R * J_R * P_R).applyfunc(sp.simplify) else 0

        if sign_left == 0 or sign_right == 0:
            continue   # J restrictions don't match the wall's declared J's
        if sign_left != sign_right:
            continue   # mismatched global sign on the two sides

        # Gauge compatibility on left side only (right follows by consistency)
        ok_gauge = all(
            commutator(J, g_left).applyfunc(sp.simplify) == sp.zeros(*J.shape)
            for g_left, _ in wall_context.gauge_pairs
        )

        if ok_gauge:
            passing.append(j_cand)

    if len(passing) == 2:
        return CommutantPolicyResult.PASSED_UNIQUE_PM
    elif len(passing) > 2:
        return CommutantPolicyResult.PASSED_MULTIPLE_ALIGNED
    else:
        return CommutantPolicyResult.GAUGE_MISALIGNMENT
```

### V19 (revised) — Verdict labels with strict separation

```python
class V2Verdict(StrEnum):
    BRIDGE_CANDIDATE = "bridge_candidate"
        # global (2,4) split + ±J + SM commutant alignment
    BRIDGE_CANDIDATE_J_BLIND = "bridge_candidate_j_blind"
        # bridge_candidate AND Lab A's max_primitive_class == ARBITRARY_REAL_ORTHOGONAL
    CLOCK_PLANE_CLOSURE_CANDIDATE = "clock_plane_closure_candidate"
        # Lab A positive with max_primitive_class != ARBITRARY_REAL_ORTHOGONAL
    DOMAIN_WALL_CANDIDATE = "domain_wall_candidate"
        # central ℂ + side-local gauge alignment + ±J intertwining; NOT the same as bridge
    CANDIDATE_ONLY_J_NOT_FORCED = "candidate_only_j_not_forced"
    STRUCTURAL_CANDIDATE_ORBIT = "structural_candidate_orbit"
    NOT_SOLVED = "not_solved"
    FALSIFIED = "falsified"

# load_bearing_qca_bridge invariant: True only for BRIDGE_CANDIDATE or BRIDGE_CANDIDATE_J_BLIND.
# DOMAIN_WALL_CANDIDATE sets a separate field load_bearing_domain_wall_candidate.
```

The result dataclass now carries both fields:

```python
@dataclass(frozen=True)
class RuleToVerdictV2Result:
    # ... (all v8 fields) ...
    load_bearing_qca_bridge: bool = False
    load_bearing_domain_wall_candidate: bool = False
```

This distinction is mechanical, not philosophical: an external reader scanning `load_bearing_qca_bridge` for "the project's original goal" will not be misled by a domain-wall positive.

## Lab A — clock-plane primitive closure (verdict label refined)

Lab A's positive verdict is `CLOCK_PLANE_CLOSURE_CANDIDATE` unless the rule's `max_primitive_class == ARBITRARY_REAL_ORTHOGONAL`, in which case it is `BRIDGE_CANDIDATE_J_BLIND`. The latter is empty by v9 design (Lab A's primitive family doesn't include arbitrary real-orthogonal generators; that's deferred to a future iteration).

So Lab A v1 will produce only `CLOCK_PLANE_CLOSURE_CANDIDATE` positives at most. The honest reading: "clock-plane primitives close to `M_2(ℂ)`," not "J-blind discovery."

## Lab B — three profiles, refined verdict labels

- B.1 strict: positive is `BRIDGE_CANDIDATE`. Predicted: `CANDIDATE_ONLY_J_NOT_FORCED` (Route 1's 4-orbit).
- B.2 structural wall: positive is `BRIDGE_CANDIDATE`. Predicted: same as B.1.
- B.3 domain wall: positive is `DOMAIN_WALL_CANDIDATE`. *Distinct* from BRIDGE_CANDIDATE.

`load_bearing_qca_bridge = True` is set only if B.1 or B.2 unexpectedly passes (extremely unlikely). `load_bearing_domain_wall_candidate = True` is set if B.3 passes.

## Cross-lab synthesis (v9 final)

| Lab A | Lab B strict | Lab B str wall | Lab B domain wall | Reading |
|---|---|---|---|---|
| CLOCK_PLANE_CLOSURE | CANDIDATE_ONLY (predicted) | CANDIDATE_ONLY (predicted) | **DOMAIN_WALL** | Best honest case: clock-plane primitives close at R^4 AND a side-local domain wall mechanism at R^6 gives unique ±J. Neither is the original carrier-first bridge, but both are honest mechanisms. Next research target: investigate the domain-wall mechanism in detail and whether it extends to R^10 with side-local SM structure. |
| CLOCK_PLANE_CLOSURE | CANDIDATE_ONLY | CANDIDATE_ONLY | CANDIDATE_ONLY / FALSIFIED | Clock-plane closure works; even the side-local wall can't supply a unique ±J at R^6. Domain-wall mechanism is empty. |
| FALSIFIED | CANDIDATE_ONLY (predicted) | CANDIDATE_ONLY (predicted) | (any) | Clock-plane primitive closure to `M_2(ℂ)` empty in `F_A`. Theorem target inside the primitive class. |
| any | NOT_PREDICTED | NOT_PREDICTED | (any) | Bug — debug. |
| any | any | any | **BRIDGE_CANDIDATE** | Impossible by definition of `DOMAIN_WALL_CANDIDATE` ≠ `BRIDGE_CANDIDATE`. Not in this table. |

## Effort estimate

| Phase | Lines | Days |
|---|---|---|
| Additive v2 path (v1-v18 with v9 corrections) | ~1800 | 5.5 |
| Combinatorial intertwiner enumeration (Tier 1, 2) | ~400 | 1 |
| Lab carrier + gauge + profile glue | ~450 | 1.5 |
| Lab A scans | ~700 | 2.5 |
| Lab B strict + structural wall regression | ~400 | 1.5 |
| Lab B domain wall (main; Tier 1 first, generic only if needed) | ~700 | 3 |
| Tests (incl. tiered intertwiner tests, WallContext consistency tests) | ~800 | 1.5 |
| Reports | ~400 | 1 |
| **Total** | **~5650** | **~17.5** |

About 3.5 weeks. The combinatorial pre-search adds 1 day but avoids the symbolic-solver stall.

## Session outline

1. **Session 1** — Additive v2 path scaffolding + verdict labels with strict separation (v19).
2. **Session 2** — Intertwiner solver: spectrum-by-char-poly, BLOCK_SUPPORT with allowed_pair_mask, Tier 1+2+3 tiered enumeration, WallContext consistency precondition. **Gate**: `r10_chiral_16_profile()` regression matches legacy R^10 on selected reference rules.
3. **Session 3** — Lab A: carrier, clock-plane primitive family, on-site scan with primitive-class tagging.
4. **Session 4** — Lab A: Bloch + wall (same-spectrum).
5. **Session 5** — Lab B strict + structural wall small regression (~30 candidates).
6. **Session 6** — Lab B domain wall: Tier 1 (split-reassignment) enumeration first, then Tier 2 (signed permutations) if needed, then Tier 3 only as final fallback. WallContext consistency tests, side-local gauge data, side-projector setup.
7. **Session 7** — Consolidate.

## Critical files (additive)

Same as v8:
- `src/clifford_3plus2_d5/qca/rule_verdict.py` (additions)
- `src/clifford_3plus2_d5/qca/profiles.py`, `predicates.py`, `wall.py` (new)
- `src/clifford_3plus2_d5/algebra/real_carrier.py`, `commutants.py` (additions)
- `src/clifford_3plus2_d5/lepton/`, `scripts/lepton_*.py`, `tests/test_*.py`, `docs/literature/lepton_*.md`

## Open decisions (final — minor implementation notes)

1. **Tier 1 split-reassignment catalogue size**: for R^6 with 3 mode pairs, ~24 permutations × 8 sign patterns = ~192 candidates. Run Tier 1 for *every* `(U_left, U_right)` pair in the scan, then aggregate.
   - Recommendation: enumerate all 192 per `(U_left, U_right)` pair, filter by intertwining + locality, expect ~1-5 to pass.

2. **Tier 3 underdetermined family handling**: report as `NOT_SOLVED` with metadata noting the parametric dimension; do not attempt to sample.
   - Recommendation: confirmed.

3. **WallContext gauge_pairs validation order**: in `consistency_certified`, check orthogonality first, then intertwining for J, then intertwining for each gauge_pair, then complementarity of side projectors. Bail on first failure.
   - Recommendation: confirmed.

4. **Side projectors for R^6**: `P_left = diag(1,1,1,1,1,1,0,0,0,0,0,0)`? No — for a 1D-cyclic 2-site wall, the side projectors live in the *lattice* support, not the internal carrier. For Lab B domain wall on a single carrier R^6 with internal block structure, the "sides" must be defined more carefully.
   - **Open** — needs design choice. Recommendation: **treat the wall as an internal-block transition at one lattice site**. P_left and P_right are then the side-local *internal block* projectors that the WallContext declares (e.g., `P_left = singlet_proj_left + doublet_proj_left = I_6`, since both blocks live in the same R^6 site). This makes the "side restriction" tests degenerate to identity — and the bridge mechanism is then about *internal* block-reassignment, not spatial-side switching.
   - **Alternative**: model the wall as a 2-site cyclic lattice (R^6 ⊗ R^2_site), with side projectors as `P_left = I_6 ⊗ |0⟩⟨0|`, `P_right = I_6 ⊗ |1⟩⟨1|`. The carrier is then R^12, and the wall transition is a 12×12 matrix.
   - Recommendation: **decide before Session 6**. The 2-site cyclic model is more physical but doubles the dimension.
