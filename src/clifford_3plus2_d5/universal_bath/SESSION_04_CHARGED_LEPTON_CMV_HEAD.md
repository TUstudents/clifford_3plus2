# Session 04 - Charged-Lepton CMV/OPUC Head

Session 04 builds the finite charged-lepton head in the unitary
CMV/OPUC language.  The frozen Session 02 charged-lepton source is

$$
e_1=(1,0,0)
=
\sqrt{\frac23}\,a+\frac1{\sqrt3}\,u+0\,b.
$$

The source is a chiral/unitary sector, so its finite data belong in a
Verblunsky coefficient on the disk, not in a real-line Jacobi residue.

## Leakage Magnitude

The frozen source depth is two.  With the universal tail

$$
t(2\sqrt2)=\epsilon=\sqrt2-1,
$$

the two-step leakage amplitude is

$$
\epsilon^2.
$$

The radial component of the charged-lepton source is $\sqrt{2/3}$, so

$$
\sqrt{\frac23}\sin\theta_e=\epsilon^2,
\qquad
\sin\theta_e=\sqrt{\frac32}\,\epsilon^2,
\qquad
\sin^2\theta_e=\frac32\,\epsilon^4.
$$

Depth-one and depth-three controls are rejected.

## Phase Head

The primitive charged-lepton boundary word is imported as an upstream
holonomy prerequisite:

$$
\text{SCHUR\_RETURN}\to\text{PARENT\_A3}\to\text{RESIDUAL\_A2}.
$$

Its phase is

$$
\exp\!\left(-\frac{5\pi i}{12}\right).
$$

The session does not claim this word from nothing.  It uses the already-gated
boundary-loop model and records that microscopic loop dynamics remains the
physical premise.

## Finite CMV Head

The finite charged-lepton Verblunsky coefficient is

$$
\alpha_e
=
\sin\theta_e\,\exp\!\left(-\frac{5\pi i}{12}\right).
$$

It satisfies

$$
|\alpha_e|^2=\sin^2\theta_e=\frac32\,\epsilon^4<1.
$$

The associated two-state CMV/Givens head is

$$
G_e=
\begin{pmatrix}
\rho_e&\alpha_e\\
-\overline{\alpha_e}&\rho_e
\end{pmatrix},
\qquad
\rho_e=\sqrt{1-|\alpha_e|^2},
$$

and the certificate checks

$$
G_e^\dagger G_e=I.
$$

After this finite head, the universal CMV tail is free:

$$
\alpha_n=0.
$$

## Verdict

The certificate verdict is:

```text
CHARGED_LEPTON_CMV_HEAD_PASS
```

Meaning:

- `C:9` exact algebra for the finite CMV head given the frozen source and
  primitive holonomy word;
- `C:7` as physical boundary theory until the charged-lepton loop-selection
  dynamics is derived microscopically;
- charged-lepton masses, PMNS assembly, quarks, and CKM remain parked.
