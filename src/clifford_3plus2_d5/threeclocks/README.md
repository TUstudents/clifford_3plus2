# threeclocks

Sidecar on hold for a simpler quark model built from three finite clocks.

The sidecar currently contains infrastructure and one D3 source certificate:

```text
three independent cyclic clocks
exact shift/phase matrices
exact clock words and closure orders
```

It does **not** claim a quark mass theorem yet.  The purpose is to freeze a
small algebraic substrate before trying to attach up/down quark sectors.

## Motivation

The parked `universal_bath` sidecar reduced many quark structures, but it also
ran into a real dynamical obstacle: Higgs-door orientation did not follow from
the available symmetry/current/flag data.  `threeclocks` starts over with a
smaller ansatz and fewer moving parts.

The working idea is:

```text
quark structure = simple clock closure data first,
                  sector interpretation second.
```

## Session 01 - Clock Spine

The first certificate builds the finite-clock spine:

```text
default prototype: Z3 x Z3 x Z3
dimension: 27
closure order: 3
```

For each local clock, exact shift and phase matrices satisfy:

```text
Z X = omega X Z.
```

Clock words compose exactly, have inverses, and report closure order.  The
infrastructure also supports non-uniform clocks such as:

```text
Z2 x Z3 x Z5
```

so the default `Z3^3` prototype is not hard-wired as a theorem.

Verdict:

```text
THREECLOCKS_INFRASTRUCTURE_PASS
```

## Session 02 - D3 Clock Source

The first physics certificate attaches a single chiral `D3 ~= S3` clock to the
selected three-port boundary tooth.  With

```text
u = (1,1,1)/sqrt(3)
a = (2,-1,-1)/sqrt(6)
b = (0,1,-1)/sqrt(2)
```

and `C e1=e2, C e2=e3, C e3=e1`, the exact identities are:

```text
(C e1 - C^-1 e1)/sqrt(2) = b
(2 e1 - C e1 - C^-1 e1)/sqrt(6) = a
```

So `b` is the oriented tangent current of the selected clock tooth, while `a`
is the radial second difference.  This gives a non-mass-data reason to try the
quark source `b`.

The same session keeps two controls explicit:

```text
spec(2I - C - C^-1) = {0,3,3}
```

so the three-port Laplacian is not the down shell; and a literal port-basis cut
`e1->e2->e3->0` does not equal the desired repair flag
`N=|u><a|+|a><b|` in the `(u,a,b)` frame.

Verdict:

```text
D3_CLOCK_SOURCE_IDENTITY_PASS
```

## Current Boundary

What is implemented:

- exact `ClockSpec`;
- exact `ThreeClockSystem`;
- exact `ClockWord`;
- local Weyl shift/phase matrices;
- exact `D3` clock-source identities selecting `b`;
- controls separating the source identity from the up repair flag and down shell;
- Session 01/02 scripts and tests.

What is not implemented:

- derivation of the representation-basis repair flag `N`;
- full unitary dilation of the effective same-normal clock branch;
- active quark shell selection over the spectator embedding;
- retarded bottom contact veto;
- quark mass exponents from clock closure;
- CKM/PMNS mixing;
- Higgs-door assignments;
- color or hypercharge coupling.

Those should be added only after the clock model says something simple and
falsifiable.

## Run

```bash
uv run python -m clifford_3plus2_d5.threeclocks.scripts.session_01_clock_spine
uv run python -m clifford_3plus2_d5.threeclocks.scripts.session_02_d3_clock_source
uv run pytest src/clifford_3plus2_d5/threeclocks/tests -q
```
