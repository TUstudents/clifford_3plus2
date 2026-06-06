# Session 01 - Bath Spine

The first session does not derive a mass spectrum. It builds the common
spectral language in which every later sector should be tested.

The proposed structure is

$$
\Sigma_f(z)=
\cfrac{1}{z-a_0^{(f)}
-\cfrac{(b_1^{(f)})^2}{z-a_1^{(f)}
-\cfrac{(b_2^{(f)})^2}{z-a_2^{(f)}-\cdots}}}.
$$

The coefficients are the Lanczos coefficients of the unresolved bath seen from
the sector source:

$$
\mu_k^{(f)}
=
\langle V_f|H_Q^k|V_f\rangle.
$$

The finite head is sector-specific. The tail is universal:

$$
t(z)=\frac{z-\sqrt{z^2-4}}2,
\qquad
t=\frac1{z-t}.
$$

At the BB marginal point from the band-edge sidecar,

$$
z_*=2\sqrt2,
$$

this gives

$$
t(z_*)=\sqrt2-1.
$$

So the tail is not a fitted closure. It is the retarded period-one Weyl tail
selected by the BCC/BB marginal-stability theorem.

## Certificate

The Python certificate checks:

1. $t=1/(z-t)$;
2. $t(2\sqrt2)=\sqrt2-1$;
3. a toy positive measure round-trips through a finite Jacobi head;
4. finite-head Schur response equals the continued fraction;
5. replacing the terminator changes the response.

The fifth item is important. It proves that "silver tail" is a physical closure
principle, not a tautology of continued fractions.

The same certificate fixes the reduction taxonomy:

$$
\text{positive}\to\text{scalar Jacobi},
\qquad
\text{real indefinite}\to\text{look-ahead Jacobi},
\qquad
\text{chiral unitary}\to\text{CMV/OPUC}.
$$

For CMV/OPUC sectors, the universal tail is expressed as

$$
\alpha_n=0
$$

after the finite head. The phase-bearing quantities belong in the finite
Verblunsky coefficients, not in a hand-inserted real-line rotation.

## Verdict

Session 01 verdict:

```text
UNIVERSAL_BATH_SPINE_PASS
```

Certainty: `C:9` for the algebra above. `C:6` for applying the universal tail
to physical sectors before their source moments are computed.

The next session must freeze the source dictionary:

$$
V_f=\{\text{charge slot},\text{ residual port geometry},\text{ normal-depth
placement}\}.
$$

Changing a source after looking at masses is a falsification, not a tuning
move.
