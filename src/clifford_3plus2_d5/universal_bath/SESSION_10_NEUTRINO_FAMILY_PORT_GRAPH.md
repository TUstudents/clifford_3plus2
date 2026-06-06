# Session 10 - Neutrino Family-Port Graph

Session 09 showed the missing object:

$$
\langle u|H_{\rm BCC}^k|b\rangle
$$

was not defined because the microscopic BB edge update had q-depth and spinor
labels, but no explicit family-port graph.  Session 10 supplies the minimal
selected-port family graph for the neutrino gate.

## Construction

Use the residual family basis

$$
u=\frac{(1,1,1)}{\sqrt3},\qquad
a=\frac{(2,-1,-1)}{\sqrt6},\qquad
b=\frac{(0,1,-1)}{\sqrt2}.
$$

The selected BCC boundary separates the radial mode $a$ from the active
neutrino plane:

$$
P_{\rm act}=P_u+P_b,\qquad P_{\rm rad}=P_a,
$$

with

$$
P_{\rm act}+P_{\rm rad}=I,\qquad
\operatorname{rank}P_{\rm act}=2,\qquad
\operatorname{rank}P_{\rm rad}=1.
$$

The finite family-port graph is

$$
H_{\rm fam}^{(N)}
=
H_{\rm chain}^{(N)}\otimes P_{\rm act}
+ I_N\otimes \Lambda P_{\rm rad}.
$$

Thus the active $u,b$ plane carries two isomorphic radial scar fibers, while
the radial $a$ mode is separated by the selected boundary penalty.

This is not the product bath

$$
H_{\rm chain}\otimes I_{\rm family},
$$

because the $a$ mode is not propagated as a third active copy.

## Moment Gate

The certificate computes the moments directly from the finite graph:

$$
\langle u|H_{\rm fam}^k|b\rangle=0,
\qquad k=0,\ldots,4,
$$

and

$$
\langle u|H_{\rm fam}^k|u\rangle
=
\langle b|H_{\rm fam}^k|b\rangle,
\qquad k=0,\ldots,4.
$$

It also checks the radial-active separation:

$$
\langle a|H_{\rm fam}^k|u\rangle=0,
\qquad k=0,\ldots,4.
$$

Closing the active plane with the universal retarded tail then gives

$$
\widehat\Sigma_\nu
=
\epsilon^2 P_u+P_b.
$$

## Controls

The full residual $K_3$ graph is rejected: it does not keep the $u$ and $b$
diagonal returns equal, because $u$ and $b$ sit in different $K_3$ eigenvalue
sectors.

The full product identity graph is also distinguished: it propagates the
radial $a$ mode as a third active copy, which the selected family-port graph
does not.

The old rank-one no-family control still has a nonzero $u/b$ cross return, and
the alternate-tail control still fails the target response.

## Verdict

```text
NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS
```

Meaning:

```text
The explicit selected family-port graph has u and b family nodes, direct graph
moments give zero u/b cross returns and equal u/b diagonal returns, and the
universal tail readout gives epsilon^2 P_u + P_b.
```

## Honest Boundary

This completes the **internal** neutrino family-port graph that Session 09
identified as missing.  It does not yet prove that the selected active-plane
condition

$$
P_{\rm act}=P_u+P_b,\qquad P_{\rm rad}=P_a
$$

is forced by the microscopic BB edge update.  That is now the next physical
gate.

## Certainty Ledger

| Statement | Status |
|---|---|
| $P_u+P_b+P_a=I$ and ranks are $2+1$ | `C:9` |
| finite selected graph has zero $u/b$ cross moments | `C:9` |
| finite selected graph has equal $u/b$ diagonal moments | `C:9` |
| selected graph differs from $H_{\rm chain}\otimes I_{\rm family}$ | `C:9` |
| $K_3$ control fails equal $u/b$ returns | `C:9` |
| physical BB update forces $P_{\rm act}=P_u+P_b$ | `C:3` |
