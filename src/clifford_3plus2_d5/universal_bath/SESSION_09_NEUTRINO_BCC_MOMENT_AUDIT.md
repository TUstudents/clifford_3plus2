# Session 09 - Neutrino BCC Moment Audit

Session 09 tests the critique of Session 03.

Session 03 used

$$
H_Q = H_{\rm chain}\otimes I_{\rm family}.
$$

Then

$$
\langle u|H_Q^k|b\rangle=0
$$

is automatic, because the family factor is an identity and
$\langle u|b\rangle=0$.  That is an internal product-bath result, not a
microscopic BCC derivation.  The real gate is:

$$
\langle u|H_{\rm BCC}^k|b\rangle
$$

computed from the actual BCC boundary graph, before inserting the product
family factor.

## Exact BB Edge Data

The currently modeled microscopic edge update contains the pinned BB Weyl
same-normal and mixed-normal blocks.

The same-normal $q=0$ blocks obey

$$
B_+^\dagger B_+ + B_-^\dagger B_- = \frac12 I.
$$

The mixed-normal leakage blocks obey

$$
M_{+2}^\dagger M_{+2} + M_{-2}^\dagger M_{-2} = \frac12 I.
$$

Therefore the local edge update has the expected exact split

$$
\frac12 I + \frac12 I = I.
$$

This is a real microscopic BB result: `C:9`.

## What Is Missing

The edge graph currently has labels

```text
spinor
q0_same_normal
q_plus2_leakage
q_minus2_leakage
```

It does not have the family-port labels

```text
family_port_u
family_port_b
```

Therefore the true microscopic moments

$$
\langle u|H_{\rm BCC}^k|b\rangle
$$

are not defined in the currently modeled graph.  If we add an identity family
factor, we are back to Session 03's product ansatz, where the vanishing is
automatic.

## Negative Control

The rank-one no-family control still has a nonzero $u/b$ cross return.  Thus
the product result is not a tautology of the residual basis.  It is sensitive
to the family-factor assumption.

## Verdict

```text
NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT
```

Meaning:

```text
The exact BB edge q=0/leakage update exists, but it is not yet a microscopic
family-port graph.  The neutrino epsilon^4 result remains product-ansatz
protected until a BCC boundary graph with u and b family ports is built and its
cross moments vanish without inserting I_family.
```

## Certainty Ledger

| Statement | Status |
|---|---|
| BB edge same-normal norm is $I/2$ | `C:9` |
| BB edge mixed-normal norm is $I/2$ | `C:9` |
| Session 03 product-bath cross moments vanish internally | `C:9` |
| Product-bath vanishing is a microscopic BCC family-port theorem | `C:2` |
| Current BCC edge graph contains no $u/b$ family-port nodes | `C:9` |
| Neutrino $\epsilon^4$ is upgraded beyond the product ansatz | `C:1` |

## Next Object

The next required object is a microscopic BCC family-port boundary graph:

$$
H_{\rm BCC,family}
$$

with explicit $u$ and $b$ source nodes or source vectors.  Only then can the
sidecar compute

$$
\langle u|H_{\rm BCC,family}^k|b\rangle
$$

as a walk-counting observable.
