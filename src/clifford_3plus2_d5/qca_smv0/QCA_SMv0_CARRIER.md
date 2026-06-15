# QCA_SMv0 Carrier and Fibre

QCA_SMv0 is the constructive simulator layer.  Its carrier is the local
Standard-Model charge register used by the field update, not the full theory
claim about the microscopic Clifford substrate.

## Actual Simulator Fibre

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

So the simulator's internal charge fibre is `SM_INTERNAL_DIM = 32`.
With family included, the non-Dirac internal fibre is `32 * 3 = 96`.

The axes have separate jobs:

```text
spatial axes:
  BCC lattice sites

dirac axis:
  split-step BCC/Weyl/Dirac streaming

internal axis:
  Standard-Model-like gauge representation register

family axis:
  Froggatt-Nielsen / recirculation matrices
```

Gauge links act on the 32-dimensional internal register.  The BCC streaming
acts on the Dirac axis and spatial sites.  The local Higgs/FN collision acts
site-locally on the Dirac, internal, and family axes.

## Not Complex 10

This carrier is not `complex 10`.

The theory-side Clifford/Pati-Salam carrier used in the lepton work is better
thought of as a chiral Clifford block: real dimension 32, or complex dimension
16 after choosing a compatible complex structure.  That object is the cleaner
candidate for a minimal one-generation matter carrier.

QCA_SMv0 currently uses an expanded simulator register with 32 complex
components for the internal gauge-charge axis.  This is a pragmatic field
layout for exact local update rules, not a claim that the microscopic minimal
carrier is a 32-complex-dimensional irreducible representation.

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
remaining bridge is conceptual and architectural: decide whether the expanded
32-complex internal register should be compressed back into the minimal
Clifford chiral-16 carrier, or kept as the practical simulation basis with a
documented adapter to the Clifford carrier.

