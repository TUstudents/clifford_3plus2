# Strong CP

The `strongcp` sidecar asks whether the BCC/QCA construction generates a QCD
vacuum angle. This is not the CKM CP question. It belongs to the color
topological sector:

$$
SU(3)_c\subset SU(4)_{\rm PS},\qquad
Q={1\over 32\pi^2}\epsilon^{\mu\nu\rho\sigma}
\operatorname{tr}(F_{\mu\nu}F_{\rho\sigma}).
$$

The synthesis verdict is narrower than the strongest wording in the sidecar:
the BCC construction is strong-CP trivial at $O(\epsilon)$ and $O(\epsilon^2)$
in the implemented carrier and spatial gauge sector, and higher-order effects
are safe at the SME-scale value of $\epsilon$. An all-order zero theorem for a
full $3+1$D gauge extension is not yet proved.

## The theta irrep

On the cubic lattice, the pseudoscalar momentum shape is

$$
k_xk_yk_z\in A_{2u}.
$$

The degree-three polynomial sector used by the sidecar decomposes as

$$
\mathcal P_3=A_{2u}\oplus T_{2u}\oplus 2T_{1u},
$$

where the implemented $T_{1u}$ projector is the six-dimensional isotypic
block. The theta channel is the one-dimensional $A_{2u}$ line. C:8.

The carrier is BCC-centrosymmetric:

$$
r\mapsto -r,\qquad
\gamma^0D(k)(\gamma^0)^{-1}=D(-k).
$$

This pins the parity of the effective-Hamiltonian corrections. The sidecar
checks

$$
H^{(1)}\in T_{2g},
\qquad
H^{(2)}\in T_{1u},
\qquad
\Pi_{A_{2u}}H^{(2)}=0.
$$

Thus the direct $A_{2u}$ theta source is absent at $O(\epsilon)$ and
$O(\epsilon^2)$. C:8.

## Gauge-sector computation

The implemented `spacetime_qca` gauge sector is spatial. Its six canonical BCC
plaquettes have no temporal Wilson leg. Therefore the literal four-dimensional
topological density is dimensionally zero in the implemented spatial-only
sector:

$$
\epsilon^{ijkl}F_{ij}F_{kl}=0,\qquad i,j,k,l\in\{1,2,3\}.
$$

This is a theorem about the present code's gauge carrier. C:9.

The sidecar also verifies that the six-plaquette representation is parity-even
under spatial inversion. Hence the tested tensors
$
T_{ab}=\operatorname{tr}(F_aF_b)
$
live in even cubic parity and have zero $A_{2u}$ projection for
$SU(2)_L$, $SU(2)_R$, and $SU(4)_{\rm PS}$. Since $SU(3)_c$ is inside
$SU(4)_{\rm PS}$, this is the relevant color-sector check available in the
current carrier. C:8.

The limitation is equally important: a true $3+1$D topological-charge audit
needs a temporal gauge-link convention. The present result does not yet compute
a full lattice $F\tilde F$ for dynamical QCD. C:6 for extension to full QCD
without extra assumptions.

## Chiral-anomaly route

A chiral rotation
$
\psi\mapsto e^{i\alpha\gamma^5}\psi
$
shifts
$
\bar\theta=\theta+\arg\det M_q
$
through the Fujikawa Jacobian. The relevant diagnostic is the scalar
$\gamma^5$ trace of the Hamiltonian corrections.

The sidecar finds

$$
\operatorname{tr}(\gamma^5H^{(1)})=0,\qquad
\operatorname{tr}(\gamma^5H^{(2)})=0,\qquad
\operatorname{tr}(\gamma^5(H^{(1)})^2)=0.
$$

Therefore neither $H^{(1)}$ nor $H^{(2)}$ induces a direct
$\bar\theta$ shift. C:8.

It also records a nonzero cross-trace:

$$
\operatorname{tr}(\gamma^5H^{(1)}H^{(2)})
=-{8\over 3}k_xk_y^3k_z.
$$

This is not a direct theta shift, but it is a warning against overstating the
selection rule. Higher-order axial structures exist. At the representative
scale
$
\epsilon\lesssim 2\times 10^{-33}\ {\rm m},
$
an $O(\epsilon^3)$ effect is far below the neutron-EDM bound, but it is not an
all-order zero certificate. C:8 for the cross-trace; C:6 for safety by scale.

## Correction to the naive selection rule

The parity rule
$
g\times g=g,\quad g\times u=u,\quad u\times u=g
$
is necessary but not sufficient to exclude $A_{2u}$. It only tells us whether a
product is even or odd. The theta irrep $A_{2u}$ is itself odd.

In ordinary cubic representation algebra,
$
T_{2g}\otimes T_{1u}
$
has an $A_{2u}$ channel. Therefore the statement "parity alone forbids
$A_{2u}$ at all orders" is not a theorem. What is proved is the direct absence
of the theta channel in the audited leading structures. C:9 for the correction;
C:3 for any future all-order vanishing claim until the full representation
product and temporal gauge extension are controlled.

## Synthesis role

The strong-CP sidecar gives a useful separation:

$$
\text{weak/CKM CP phase}\neq \text{QCD vacuum angle}.
$$

The BCC lattice can have a CP-odd $T_{2g}$ fermion correction while still
avoiding a QCD $\theta$ term at the leading audited orders. This is the right
physics: CKM-type CP violation and strong CP are different sectors.

Paper-level statement:

> In the implemented BCC carrier, the leading CP-odd lattice correction does
> not induce a QCD vacuum angle at $O(\epsilon)$ or $O(\epsilon^2)$: the
> $A_{2u}$ theta channel is absent, the spatial plaquette topological density
> is dimensionally trivial, and the chiral-anomaly trace vanishes through the
> audited orders. Higher-order effects are scale-suppressed rather than proven
> identically zero.

Certainty: C:8 for the leading-order triviality; C:6 for higher-order safety;
C:3 for a complete all-order strong-CP solution in a full $3+1$D gauge theory.

