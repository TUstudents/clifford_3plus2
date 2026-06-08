# threeclocks Plan

## Purpose

Build a much simpler quark model from three finite clocks.

The sidecar should stay small.  The first rule is:

```text
clock algebra before flavor interpretation
```

## Session 01 - Clock spine

Pass only if:

- exactly three finite clocks are represented;
- the default prototype is `Z3 x Z3 x Z3`;
- each local clock has exact shift/phase matrices;
- each local clock satisfies `Z X = omega X Z`;
- words compose, invert, and close exactly;
- non-uniform clock orders are supported as controls;
- no quark masses or mixings are claimed.

Verdict:

```text
THREECLOCKS_INFRASTRUCTURE_PASS
```

## Completed Gates

### Session 02 - D3 clock source identity

Attach one selected `D3 ~= S3` clock to the residual three-port boundary.

Pass only if:

- the exact D3 relations hold;
- the selected tooth decomposes as `e1 = u/sqrt(3) + sqrt(2/3) a`;
- the oriented first difference gives `b`;
- the radial second difference gives `a`;
- the three-port Laplacian control has spectrum `{0,3,3}`;
- a literal port-basis cut does not equal the target repair flag `N`;
- no quark mass theorem is claimed.

Verdict:

```text
D3_CLOCK_SOURCE_IDENTITY_PASS
```

## Next Gates

### Session 03 - D3 repair-flag bridge

Ask whether the clock defect produces the representation-basis flag
`N=|u><a|+|a><b|` rather than only a port-basis cut.

Pass only if:

- the finite Schur reduction starts from the effective clock branch, not from
  the desired `N`;
- the output is compared against the target flag in the `(u,a,b)` frame;
- the literal cut and three-port Laplacian controls remain rejected;
- if `N` is not produced, the up-clock ansatz is marked conditional rather than
  repaired by hand.

### Session 04 - Same-normal clock branch dilation

Audit the effective branch:

```text
B_+ tensor C + B_- tensor C^-1
```

as a subunitary same-normal scar branch inside a full unitary dilation with
outgoing mixed-normal channels.

Pass only if:

- the same-normal norm is exactly the BB survival half;
- leakage channels complete the norm without recurrent contamination;
- the clock acts on an independent family/clock factor;
- the result does not rely on quark mass data.

### Session 05 - Retarded down shell contact veto

Build the active shell:

```text
1_direct + 2_BCC + 3_color
```

and audit whether the bottom current vetoes the direct/contact return.

Pass only if:

- active-over-spectator selection is stated as a theorem target or derived by a
  finite Gauss/current argument;
- the contact-allowed control gives `sqrt(2/3)`;
- the contact-veto branch gives `sqrt(5/6)`;
- the microscopic rule, not data preference, selects one branch.

### Session 06 - Closure-order mass skeleton

Ask whether a hierarchy can come from clock closure order alone.

Pass only if:

- exponents are derived from word order, stabilizer order, or orbit length;
- the model either produces a nontrivial up/down hierarchy or fails cleanly;
- no Clebsch prefactors are introduced beyond identities already certified by
  previous sessions.

### Session 07 - CKM from clock-kernel left frames

Build the two-sided clock kernels and compute the relative left-frame
holonomy:

```text
V_CKM = U_u,L^dagger U_d,L.
```

Pass only if:

- the left frames come from actual finite heads;
- no CKM angle or phase is inserted by hand;
- failures are labeled as missing dynamics rather than absorbed into
  prefactors.

### Session 08 - Clock parity / odd-shell replacement

If a down-sector odd shell is needed, express it as a clock parity or
non-identity clock sector rather than importing the previous primitive shell.

Pass only if:

- the odd/even split is intrinsic to the three-clock system;
- the count is determined before comparing to quark masses;
- the old `(6,2,5)` logic is used only as a control, not as an input.

## Reuse Boundary

Allowed imports:

- exact algebra utilities from package root if needed;
- `sympy`.

Avoid imports from `universal_bath`, `scalar_clebsch`, or `radial_response`
during the first physics passes.  This sidecar should test whether a simpler
clock ansatz has independent content.
