# Session 03 - Neutrino Product Bath

Session 03 applies the universal bath spine to the frozen neutrino sources from
Session 02:

$$
u=\frac{(1,1,1)}{\sqrt3},
\qquad
b=\frac{(0,1,-1)}{\sqrt2}.
$$

The product-bath ansatz is

$$
H_Q=H_{\rm chain}\otimes I_{\rm fam}.
$$

The chain supplies the radial retarded tail.  The residual family label lives
inside the unresolved sector.

## Return Moments

For finite truncations, the hidden product states are

$$
|U\rangle=|0\rangle\otimes u,
\qquad
|B\rangle=|0\rangle\otimes b.
$$

Because the hidden Hamiltonian is a tensor product with $I_{\rm fam}$,

$$
\langle U|H_Q^k|B\rangle
=
\langle 0|H_{\rm chain}^k|0\rangle\,\langle u|b\rangle
=0.
$$

The diagonal returns are equal:

$$
\langle U|H_Q^k|U\rangle
=
\langle B|H_Q^k|B\rangle
=
\langle 0|H_{\rm chain}^k|0\rangle.
$$

The certificate checks these exact identities for $k=0,1,2,3,4$.

Certainty inside the product bath: `C:9`.

## Weyl Tail Readout

The semi-infinite chain Weyl function is the universal tail

$$
m(z)=t(z)=\frac{z-\sqrt{z^2-4}}2,
\qquad
m=\frac1{z-m}.
$$

The normalized neutrino response is therefore

$$
\widehat\Sigma_\nu(z)=m(z)^2P_u+P_b.
$$

In the residual basis $(a,u,b)$ this is

$$
\widehat\Sigma_\nu(z)
\sim
\operatorname{diag}(0,m(z)^2,1).
$$

At the BB marginal probe

$$
z_*=2\sqrt2,
\qquad
m(z_*)=\epsilon=\sqrt2-1,
$$

so

$$
\widehat\Sigma_\nu(z_*)=\epsilon^2P_u+P_b.
$$

Thus

$$
\frac{m_2}{m_3}=\epsilon^2,
\qquad
\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=\epsilon^4.
$$

## Controls

The certificate rejects three shortcuts:

1. Removing the family factor gives a rank-one response with $u$-$b$ cross
   terms.
2. Replacing $b$ by the radial $a$ source changes the target response.
3. Replacing the silver tail by a constant terminator changes the target
   response.

These controls are the point of the session.  The neutrino prediction survives
because the source factorization, source geometry, and universal tail all act
together.

## Verdict

The certificate verdict is:

```text
NEUTRINO_PRODUCT_BATH_INTERNAL_PASS
```

Meaning:

- `C:9` inside the stated product half-line bath;
- physical upgrade blocked by Session 09 until a microscopic BCC family-port
  graph defines $\langle u|H_{\rm BCC}^k|b\rangle$ without inserting
  $I_{\rm fam}$;
- PMNS, charged leptons, quarks, and CKM remain parked.
