# Session 02 - D3 Clock Source

**Verdict:** `D3_CLOCK_SOURCE_IDENTITY_PASS`

This session records the first physics move of the `threeclocks` sidecar:
replace the imported quark source guess by a local clock identity.

The claim proven here is narrow:

```text
the selected D3 clock tooth has a radial second-difference line a
and an oriented tangent-current line b.
```

The quark-source proposal is therefore:

```text
quark source = tangent current b,
not the selected scalar port e1.
```

No quark mass theorem is claimed in this session.

## Clock Setup

Use the three-port basis:

```text
e1=(1,0,0), e2=(0,1,0), e3=(0,0,1).
```

The residual frame is:

```text
u = (1,1,1)/sqrt(3)
a = (2,-1,-1)/sqrt(6)
b = (0,1,-1)/sqrt(2)
```

Let `C` be the clock step:

```text
C e1 = e2, C e2 = e3, C e3 = e1.
```

Let `rho` fix `e1` and swap `e2,e3`.  The exact D3 relations pass:

```text
C^3 = I
rho^2 = I
rho C rho = C^-1
```

## Exact Source Identities

The selected tooth decomposes as:

```text
e1 = u/sqrt(3) + sqrt(2/3) a
```

with no `b` component.

The clock first and second differences are exact:

```text
(C e1 - C^-1 e1)/sqrt(2) = b
(2 e1 - C e1 - C^-1 e1)/sqrt(6) = a
```

Interpretation:

```text
u = trace / density
a = radial curvature of the selected tooth
b = oriented tangent current of the clock
```

This is the strongest result of the session.  It distinguishes `a` and `b`
without using quark masses.

## Control: Three-Port Laplacian Is Not the Down Shell

Let:

```text
J = C - C^-1
J^dagger J = 2I - C - C^-1.
```

On the three-port space:

```text
spec(J^dagger J) = {0,3,3}.
```

This is the unbroken D3 doublet degeneracy.  It is not a three-generation down
hierarchy and must not be conflated with the six-label active quark shell.

## Control: A Literal Clock Cut Is Not the Repair Flag

The desired up-sector finite head uses the target flag:

```text
N = |u><a| + |a><b|
```

so:

```text
b -> a -> u -> 0.
```

Session 02 explicitly checks that a literal port-basis cut:

```text
e1 -> e2 -> e3 -> 0
```

does **not** become this `N` after transforming into the `(u,a,b)` frame.

Therefore:

```text
the clock derives the source b;
the b -> a -> u repair flag remains a theorem target.
```

## Conditional Up Head

If the representation-basis repair flag `N` is supplied, then:

```text
N^3 = 0
n(u,a,b) = (2,1,0)
```

and, with the BCC same-normal survival amplitude:

```text
x = 1/sqrt(2)
```

the finite head gives:

```text
exp(xN)|b> = |b> + x|a> + x^2|u>/2
           = |b> + |a>/sqrt(2) + |u>/4.
```

The conditional up profile is:

```text
C_u = (1/4, 1/sqrt(2), 1)
```

in light-to-heavy order `(u,a,b)`.

## Conditional Down Shell

The proposed down readout is not `J^dagger J` on three ports.  It is a
retarded Hermitian return shell on the active quark shell:

```text
S_q = 1_direct + 2_BCC + 3_color.
```

If the bottom current vetoes the direct/contact return, the ranks are:

```text
d: 6
s: 2
b: 5
```

and the conditional profile is:

```text
C_d = (1, 1/sqrt(3), sqrt(5/6)).
```

The contact-allowed control remains:

```text
C_d = (1, 1/sqrt(3), sqrt(2/3)).
```

The finite audit still owed is:

```text
does the retarded boundary rule actually veto the identity/contact return?
```

## Open Theorem Targets

The exact source identity does not close the quark theory.  The next gates are:

- derive the representation-basis repair flag `N` from the D3 clock defect;
- embed `B_+ tensor C + B_- tensor C^-1` into a full unitary dilation;
- select the active quark shell over the spectator embedding;
- derive the bottom contact veto from the retarded boundary current;
- derive closure exponents `k_u=3n` and `k_d=2(n+1)`;
- compute CKM from the two-sided clock kernels.

## Meaning

This session gives the proposal its first non-arbitrary hook:

```text
b is the clock tangent current.
```

It does not yet prove:

```text
up = coherent nilpotent head,
down = retarded six-label shell,
CKM = left-frame mismatch.
```

That boundary is the point.  The simpler clock model is now falsifiable at the
next finite gates instead of being another imported quark ansatz.
