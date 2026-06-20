# QCA_SMv0 Carrier and Fibre

QCA_SMv0 is the constructive simulator layer.  Its carrier is the local
Standard-Model charge register used by the field update, not the full theory
claim about the microscopic Clifford substrate.

## Physical Fibre Versus Simulator Register

The compact theory-side one-generation matter carrier is the Spin(10) chiral
spinor

```text
complex 16 = Q_L(3*2) + u^c(3) + d^c(3) + L_L(2) + e^c(1) + nu^c(1).
```

In the real Clifford/Pati-Salam sidecars this same object is represented as a
real chiral-16 block of dimension `32`, together with a compatible complex
structure `J`:

```text
R^32 with J  ==  C^16.
```

That is the compact internal matter fibre.  The QCA_SMv0 arrays are a
simulation register built around that carrier, with extra axes for spacetime
Dirac streaming and family.  These axes should not be counted as new internal
Spin(10) degrees of freedom.

## Actual Simulator Layout

The current production-facing field states use these layouts:

```text
free Dirac field:
  psi[x,y,z, dirac]
  shape = (nx, ny, nz, 4)
  local fibre dimension = 4

SM internal field:
  psi[x,y,z, dirac, internal]
  shape = (nx, ny, nz, 4, 32)
  local fibre dimension = 4 * 32 = 128

three-family SM field:
  psi[x,y,z, dirac, internal, family]
  shape = (nx, ny, nz, 4, 32, 3)
  local fibre dimension = 4 * 32 * 3 = 384
```

Here `SM_INTERNAL_DIM = 32` is the simulator's complex internal register.  It is
not the minimal irreducible Spin(10) matter fibre.  It is a pragmatic doubled
presentation of the one-generation chiral-16 label set used by the current
local gauge/Yukawa conventions.  With family included, the non-Dirac simulator
register is `32 * 3 = 96`.

The axes have separate jobs:

```text
spatial axes:
  BCC lattice sites

dirac axis:
  spacetime Weyl/Dirac streaming and chirality-flipping mass collision

internal axis:
  one-generation SM/Spin(10)-derived charge register in the simulator basis

family axis:
  Froggatt-Nielsen / recirculation matrices
```

For physically labeled initial conditions, use
`sm_qca_family_carrier_basis_state(...)`.  It writes one amplitude into the
visible field using the sector labels

```text
Q, u_c, d_c, L, e_c, nu_c
```

together with the Dirac component, family index, internal-copy index, and
color/weak labels where those labels apply.  This is the preferred entry point
for sector-specific QCA experiments; `deterministic_qca_family_state(...)` is
only a mixed audit/smoke-test pattern.

For a calibrated quark/FN field-response experiment, use
`sm_run_jitted_qca_calibrated_carrier_basis_probe(...)`.  It creates such a
labeled basis state, calibrates the compressed production update from
masses/CKM, runs the rollout, and returns initial/final carrier populations.
For a batched quark-family response matrix, use
`sm_qca_prepared_quark_family_response(...)` on an already prepared calibrated
production setup; it evolves pure weak-up and weak-down `Q` family basis states
and returns complex target amplitudes plus target-sector populations into
`u_c` and `d_c`.  For one-tick single-site probes it also reports the
structured-cache one-tick amplitudes and residuals against the measured field
response.  `sm_qca_calibrated_quark_family_response(...)` is the one-call
wrapper that prepares the setup first.

## Lepton Carrier Map

The lepton sector uses the same production field fibre as the quark sector:

```text
psi[x,y,z, dirac, internal, family]
shape = (nx, ny, nz, 4, 32, 3)
```

The lepton carrier labels are:

```text
L    : weak-doublet lepton carrier, weak index 0 or 1
e_c  : charged-lepton singlet carrier
nu_c : right-handed / sterile-neutrino singlet carrier
```

The family axis is the same three-family register used by the quark FN
collision.  The lepton inputs in QCA_SMv0 are explicit simulator Yukawa
matrices, currently supplied as diagonal triples through the phenomenology CLI
or as `FamilyLeptonYukawas` in Python.

With the current unitary-gauge Higgs convention, the two clean lepton carrier
probes are:

```text
L(weak=0) -> nu_c   via the neutrino / H_tilde door
L(weak=1) -> e_c    via the charged-lepton / H door
```

These probes are production-field tests.  They check that the existing
`Dirac 4 * internal 32 * family 3` simulator fibre routes lepton Yukawa inputs
to the correct carrier sectors while preserving the lean production contract.
They do not yet derive charged-lepton masses, neutrino masses, or PMNS mixing.

Gauge links act on the 32-dimensional internal simulator register.  The BCC
streaming acts on the Dirac axis and spatial sites.  The local Higgs/FN
collision acts site-locally on the Dirac, internal, and family axes.

The `4` in the field layout is ordinary spacetime Dirac spin:

```text
Dirac 4 = left Weyl 2 + right Weyl 2.
```

It is not the Spin(10) internal spinor.  The internal Spin(10)-side matter
content is the chiral `16`; the spacetime Dirac axis is present because the
simulator implements BCC transport and local Higgs/Yukawa chirality flips.

## Not Complex 10

This carrier is not `complex 10`.

The Spin(10) vector `10` is not a Standard-Model generation.  It is useful for
vector data, Clifford generators, and sometimes Higgs representations, but the
matter generation is the Spin(10) chiral spinor `16`.

The theory-side Clifford/Pati-Salam carrier used in the lepton work is the
chiral Clifford block: real dimension 32, or complex dimension 16 after
choosing a compatible complex structure.  That object is the cleaner candidate
for a minimal one-generation matter carrier.

QCA_SMv0 currently uses an expanded simulator register with 32 complex
components for the internal gauge-charge axis.  This is a pragmatic field
layout for exact local update rules, not a claim that the microscopic minimal
carrier is a 32-complex-dimensional irreducible representation.

## FN Recirculation Is Not The Physical Fibre

The explicit `fn_dilation` path networks are reference dilations.  They prove
that a Froggatt-Nielsen factor can be realized by a finite local unitary path:

```text
path length n  ->  endpoint amplitude lambda^n.
```

For the default quark charges, the exact pair-path reference network has hidden
dimensions

```text
up hidden paths:   sum_ij (Q_i + U_j + 1) = 45
down hidden paths: sum_ij (Q_i + D_j + 1) = 27
```

and those slots are replicated over the visible spin/color/weak doors in the
exact validation path.  That large auxiliary space is not the local
Standard-Model fibre.  It is a verifier for the recirculation interpretation.

The production path is the compressed effective Yukawa collision:

```text
Y_ij = c_ij lambda^(Q_i + R_j),
U_Y = exp(-i step_size beta Y(H)).
```

A future dynamic recirculation backend should use shared charge ladders or
clock registers, not one independent hidden path for every matrix entry copied
into every spacetime cell.

## Relation To Sidecars

`lepton`
: Works closest to the minimal Clifford/Pati-Salam carrier.  Its natural
  matter carrier is the chiral Clifford block, real 32 = complex 16 after a
  complex structure is chosen.

`sim`
: Provides shared simulator infrastructure.  It should stay carrier-agnostic.

`spacetime_qca`
: Contains broader BCC/QCA prototypes and production experiments.  QCA_SMv0
  copies patterns when useful, but keeps its own code because the SM/FN layout
  will evolve differently.

`universal_bath`, `threeclocks`, `cusp`
: Work with small flavor/recirculation spaces.  They are theory sidecars, not
  full Standard-Model field fibres.

`qca_smv0`
: The constructive simulator sidecar.  It uses an explicit BCC field state,
  gauge links, Higgs collision, FN recirculation matrices, and center-holonomy
  CP coefficients.

## Current Honest Status

QCA_SMv0 has a working expanded carrier and a compact rollout runner.  The
current production-facing fibre is compact at the field level:

```text
Dirac 4 * simulator-internal 32 * family 3 = 384 complex amplitudes per site.
```

The exact hidden FN dilation is intentionally outside that fibre and should be
used as a reference/validation mode.  The remaining bridge is architectural:
either compress the simulator internal register back to the minimal Clifford
`C^16` carrier, or keep the practical `32`-complex basis with a documented
adapter to the Clifford carrier.
