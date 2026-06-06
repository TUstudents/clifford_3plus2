# universal_bath Plan

## Purpose

Turn the bath idea into a computable sidecar:

```text
sector response = finite Lanczos head + universal silver tail
```

The sidecar tests whether flavor sectors differ only in the first few floors of
their boundary spectral measure while sharing one retarded BCC/BB tail.

## Session 01 - Bath spine

Pass only if:

- the period-one tail satisfies `t = 1 / (z - t)`;
- `t(2 sqrt(2)) = sqrt(2) - 1`;
- finite scalar moments round-trip through a Jacobi head;
- finite-head Schur response equals the continued fraction;
- replacing the silver tail changes the response;
- the reduction taxonomy separates positive Jacobi, indefinite look-ahead
  Jacobi, and CMV/OPUC sectors;
- the CMV/OPUC free-tail criterion is `alpha_n = 0` after the finite head.

Verdict target:

```text
UNIVERSAL_BATH_SPINE_PASS
```

Session 01 proves the common algebraic spine. It does not derive any sector
masses.

## Session 02 - Write-once source dictionary

Freeze sector source anchors before computing any masses:

```text
V_f = {charge slot, residual port geometry, normal-depth placement}
```

Pass only if:

- every frozen source is fixed without using flavor data;
- every frozen source has a declared charge anchor;
- every frozen source has a declared residual BCC boundary port vector;
- every frozen source has a declared normal-depth placement;
- the radial first-hop survival weight is universal, inherited from the BB
  same-normal identity.

Session 02 freezes only anchors that are supported by existing sidecars:

- `neutrino_collective_u`: $u$, depth 1;
- `neutrino_edge_b`: $b$, depth 0;
- `charged_lepton_active_e1`: $e_1$, depth 2.

It records up/down quark charge anchors but leaves their BCC source vectors
unfrozen.  That is the intended anti-circularity behavior: do not invent a quark
source to make the universal-bath program look complete.

Verdict target:

```text
SOURCE_DICTIONARY_CORE_PASS
```

Abort only if a frozen source fails normalization, uses flavor data, or does not
share the BB survival branch.

## Session 03 - Neutrino product bath

Compute cross-return moments in the framed sterile source model:

```text
<u|H_Q^k|b>, k = 1,2,3,4
```

The product bath becomes a theorem only if the cross returns vanish from the
graph/source structure rather than from an inserted tensor ansatz.

Session 03 uses the frozen Session 02 neutrino anchors and the product bath

```text
H_Q = H_chain tensor I_family
```

to verify:

- diagonal returns for `u` and `b` are equal;
- cross-return moments vanish for `k = 0..4`;
- the normalized Weyl-tail response is `t(z)^2 P_u + P_b`;
- at `z = 2 sqrt(2)`, the response is `epsilon^2 P_u + P_b`;
- rank-one, wrong-source, and alternate-tail controls fail.

Verdict target:

```text
NEUTRINO_PRODUCT_BATH_INTERNAL_PASS
```

This is `C:9` inside the product half-line bath. Session 09 blocks the physical
upgrade until a microscopic BCC family-port graph defines the `u/b` cross
moments without inserting `I_family`.

## Session 04 - Charged-lepton CMV/OPUC head

Use CMV/OPUC seeded by the charged-lepton source. The charged-lepton holonomy
should appear as a finite-head Schur/Verblunsky coefficient, not as an inserted
rotation.

Session 04 uses:

- frozen source `charged_lepton_active_e1`;
- decomposition `e1 = sqrt(2/3) a + 1/sqrt(3) u`;
- source depth 2 and the universal tail value `epsilon`;
- primitive holonomy word `SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2`.

It verifies:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
alpha_e = sin(theta_e) exp(-5 pi i / 12)
|alpha_e|^2 < 1
```

and builds an exact two-state CMV/Givens head followed by the free OPUC tail
`alpha_n = 0`.

Verdict target:

```text
CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS
```

This session does not derive charged-lepton masses and does not assemble PMNS.

## Session 05 - Charged-lepton 2/9 torsion gate

Decide whether the charged-lepton torsion value is a real source moment or a
post-hoc arithmetic identity.

Pass only if the frozen source

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

has exact occupations

```text
p_a = 2/3, p_u = 1/3, p_b = 0
```

so the incoherent transition weight is

```text
p_a p_u = 2/9.
```

The coherent amplitude control `sqrt(2)/3`, equal-weight control `1/4`, and
one-port controls must be rejected. This session does not rederive the CMV
phase or charged-lepton masses.

Verdict target:

```text
CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS
```

## Session 06 - Up-quark nilpotent CMV head

Model the finite nilpotent head and test whether silver-tail injection forces
the coherent entry value:

```text
x = 1 / sqrt(2)
```

The pass verdict is conditional because the up-quark BCC source vector remains
unfrozen in the Session 02 dictionary.

Verdict target:

```text
UP_NILPOTENT_HEAD_CONDITIONAL_PASS
```

## Session 07 - Down-quark indefinite symmetric head

Compare the 3-port permutation graph with the 6-element `S_3` Cayley graph.
The physical down bath is whichever graph produces the correct first weights
without inserting the answer. Use look-ahead indefinite Lanczos if a real
signature breakdown occurs; do not clamp negative $b_n^2$.

The implemented count-level gate separates the clean `(6,2,4)` baseline from
the available-but-unforced `(6,2,5)` candidate. It does not select the bottom
rank-5 line.

Verdict target:

```text
DOWN_HEAD_FORK_LOCALIZED_PASS
```

## Session 08A - Quark height-door audit

Before freezing quark sources, separate the electroweak charge theorem from the
height-dynamics premise.

Pass only if:

- hypercharge forces `H_tilde` for up and `H` for down;
- both neutral Higgs components have `Q_em = Y + T3 = 0`;
- the declared up repair is the oriented length-3 nilpotent;
- the declared down repair is the Hermitian path closure with spectrum
  `{0,1,3}`;
- swapping repair modes is still hypercharge-allowed, proving the repair split
  does not come from electroweak charges alone.

Verdict target:

```text
QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT
```

This session does not freeze `V_u` or `V_d`.

## Session 08B - Quark color-lift audit

Audit color as visible gauge covariance versus hidden return multiplicity.

Pass only if:

- a fixed rank-one visible color source is rejected by `SU(3)_c` covariance;
- a color-scalar spectator embedding preserves visible color but stays on the
  three-port shell;
- an active hidden color-return embedding also preserves visible color while
  reaching the six-channel shell `1_direct + 2_BCC + 3_color`;
- the active lift makes `(6,2,5)` available but not forced;
- gauge covariance alone does not select active over spectator.

Verdict target:

```text
QUARK_COLOR_LIFT_NO_SELECTION_AUDIT
```

## Session 09 - Neutrino microscopic BCC moment audit

Attack the strongest product-bath premise before assembling more sectors.
Session 03 vanishing cross moments are internal to

```text
H_Q = H_chain tensor I_family
```

and are automatic from the inserted family identity.  The microscopic gate is:

```text
<u|H_BCC^k|b>, k = 0,1,2,3,4
```

computed from a BCC boundary graph carrying the `u` and `b` family ports.

The implemented Session 09 audit checks the currently available microscopic BB
edge update:

- exact q=0 same-normal norm `I/2`;
- exact q=+-2 mixed-normal leakage norm `I/2`;
- total edge norm `I`;
- Session 03 product-bath cross moments still vanish internally;
- the rank-one no-family control still has cross return;
- the current microscopic BB edge graph has spinor/q-depth labels but no `u,b`
  family-port labels.

Verdict target:

```text
NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT
```

This is not a pass of the neutrino crown. It records that the exact BB edge
update exists, but the BCC family-port graph needed to define
`<u|H_BCC^k|b>` has not been built.

## Session 10 - Selected neutrino family-port graph

Build the missing family-port graph without using flavor data:

```text
H_fam = H_chain tensor (P_u + P_b) + I tensor Lambda P_a
```

The selected radial mode `a` is separated. The active neutrino plane `span(u,b)`
gets two isomorphic radial scar fibers.

Pass only if:

- `P_u + P_b + P_a = I`;
- the graph differs from `H_chain tensor I_family`;
- direct graph moments give `<u|H_fam^k|b> = 0` for checked finite powers;
- direct graph moments give equal `u` and `b` diagonal returns;
- radial-active moments such as `<a|H_fam^k|u>` vanish;
- the full residual `K3` graph control fails equal `u/b` returns;
- the full product identity control propagates the radial `a` mode and is
  therefore not the selected graph;
- closing the active plane with the universal retarded tail gives
  `epsilon^2 P_u + P_b`.

Verdict target:

```text
NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS
```

This completes the internal family-port graph needed by Session 09. It does not
yet prove that the selected active-plane condition is forced by the microscopic
BB edge update.

## Session 11 - Derive selected active-plane boundary condition

Try to derive

```text
P_act = P_u + P_b, P_rad = P_a
```

from the microscopic BB edge update, selected boundary normal, and residual
port incidence. This is the physical gate behind Session 10.

## Session 12 - Mixing from Krylov overlaps

Treat CKM and PMNS as overlaps of sector left-Krylov bases. The powers are the
first target; prefactors remain conditional until they are head coefficients.
