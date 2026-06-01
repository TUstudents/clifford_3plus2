"""Boundary repair-scar sidecar for the quark depth spectrum.

This sidecar tests whether the declared quark family depths ``{0, 2, 6}``
can be represented by a graph-native boundary repair defect:

    D_scar = 2 Delta(P3).

It does not derive the microscopic origin of the scar.  A pass means the depth
spectrum has an exact path-defect Laplacian operator origin, a nilpotent repair
flag realization, and a shortest-repair variational characterization once the
edge-count cost principle is supplied.  V9 refines that cost principle into a
conditional microscopic locality theorem: one-tick residual geometry forbids the
shortcut support, while equal weights still require repair-isometry saturation.
V10 proves that repair-isometry saturation is exactly the no-leakage condition
for the active repair block.
V11 proves that a unique allowed microscopic successor condition is sufficient
for no leakage, leaving explicit successor enumeration as the next gate.
V12 implements that finite successor certificate for the modeled local boundary
basis, leaving basis completeness as the remaining microscopic gap.
"""
