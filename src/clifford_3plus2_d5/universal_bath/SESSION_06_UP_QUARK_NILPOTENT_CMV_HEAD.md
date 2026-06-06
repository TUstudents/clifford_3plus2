# Session 06 - Up-Quark Nilpotent CMV Head

Session 06 tests the finite up-sector head, not the full up-quark source
problem.  The Session 02 source dictionary still records
`up_quark_boundary_source` as unresolved.  This is intentional:
the BCC source vector and normal depth must be derived without using mass data.

## Survival Injection

The BB same-normal edge identity gives

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12 I.
$$

For a normalized $q=0$ source, the first-hop survival weight is therefore

$$
\frac12,
$$

so the finite-head injection amplitude is

$$
x=\sqrt{\frac12}=\frac1{\sqrt2}.
$$

## Nilpotent Head

The up-sector repair flag is a length-three nilpotent:

$$
N^3=0,\qquad N^2\ne 0.
$$

The coherent finite head is the truncated Taylor kernel

$$
\exp(xN)=I+xN+\frac{x^2}{2}N^2.
$$

Its grade readout is

$$
\left(\frac{x^2}{2},x,1\right)
=
\left(\frac14,\frac1{\sqrt2},1\right).
$$

The geometric control at the same injection amplitude would give

$$
\left(x^2,x,1\right)
=
\left(\frac12,\frac1{\sqrt2},1\right),
$$

so the factorial Taylor structure is a real discriminator.

## CMV Tail

The finite CMV head is recorded as

$$
\left(\frac1{\sqrt2},\frac14\right),
$$

with both coefficients inside the unit disk.  After this finite head, the
universal CMV tail is free:

$$
\alpha_n=0.
$$

## Verdict

The certificate verdict is:

```text
UP_NILPOTENT_HEAD_CONDITIONAL_PASS
```

Meaning:

- `C:9` exact finite-head algebra once the up sector is assigned the nilpotent
  CMV reduction;
- `C:7` that $x=1/\sqrt2$ is inherited from the BB survival branch;
- `C:4` as a complete quark theory because the microscopic up-quark source
  vector is not yet frozen.

The result closes the up finite-head algebra, but not the up source theorem.
