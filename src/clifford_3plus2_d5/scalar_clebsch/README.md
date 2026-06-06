# scalar_clebsch

Study sidecar for scalar quark mass Clebsch coefficients.

This sidecar asks a narrow question:

```text
Can the quark mass Clebsches be derived from scalar boundary-response rules
rather than borrowed from CKM current amplitudes?
```

## Result

```text
SCALAR_CLEBSCH_THEORY_CONDITIONAL_PASS
```

The sidecar separates three scalar readouts:

```text
up sector:
  nilpotent Taylor response exp(xN), x = 1/sqrt(2)
  -> (1/4, 1/sqrt(2), 1)

down sector, natural baseline:
  S3/projector count structure
  -> (1, 1/sqrt(3), sqrt(2/3))

down sector, data-improved candidate:
  odd-shell bottom repair
  -> (1, 1/sqrt(3), sqrt(5/6))
```

The up-sector story is the cleanest part. The down-sector story is deliberately
split: `(6,2,4)` is the natural S3/projector baseline, while `(6,2,5)` is the
candidate that matches the common-scale mass study better but still needs a
theorem for the bottom `+1`.

## Up Sector

Let `N` be the length-3 nilpotent repair flag:

```text
N^3 = 0,  N^2 != 0.
```

The scalar Taylor response is:

```text
exp(xN) = I + xN + (x^2/2)N^2.
```

The grade readout is:

```text
(x^2/2, x, 1).
```

With the normalized scalar repair amplitude:

```text
x = 1/sqrt(2),
```

this gives:

```text
C_u = (1/4, 1/sqrt(2), 1).
```

The nearby empirical rational vector:

```text
(1/4, 3/4, 1)
```

is retained as a data-oriented control, not the preferred theorem. The older
vector:

```text
(1/2, sqrt(2), 1)
```

is rejected as a positive scalar grade profile because `sqrt(2) > 1`; it belongs
to coherent current-amplitude logic, not scalar mass-response logic.

## Down Sector

The primitive quark shell from `boundary_response` has:

```text
1_direct + 2_BCC + 3_color = 6 primitive labels.
```

The natural S3/projector baseline gives counts:

```text
d : full shell       -> n = 6
s : BCC odd doublet -> n = 2
b : S3 projector    -> n = 4
```

With the overlap rule:

```text
C_d,i = sqrt(n_i / 6),
```

the baseline is:

```text
C_d,baseline = (1, 1/sqrt(3), sqrt(2/3)).
```

The data-improved candidate uses the full odd shell for the bottom coefficient:

```text
d : full shell       -> n = 6
s : BCC odd doublet -> n = 2
b : odd shell        -> n = 5
```

giving:

```text
C_d,candidate = (1, 1/sqrt(3), sqrt(5/6)).
```

The candidate ratio formulas are:

```text
m_d/m_s = sqrt(3) eta^2
m_s/m_b = sqrt(2/5) eta^2.
```

## S3 Projector Audit

The finite regular-representation audit gives a sharper verdict on `(6,2,5)`:

```text
S3_PROJECTOR_COUNT_AVAILABILITY_PASS
```

In the regular S3 shell:

```text
Reg(S3) = 1_triv + 1_sign + 2_std + 2_std.
```

The central isotypic ranks are:

```text
trivial = 1
sign = 1
standard_isotypic = 4
regular = 6
```

So rank `5` is available as either:

```text
regular - trivial
regular - sign
```

but S3 alone does not choose which one-dimensional line is excluded. The middle
rank `2` is also available as a defect-polarized standard copy, but it is not a
central S3 projector; the central rank-2 object is `trivial + sign`, not the
BCC standard doublet.

Therefore `(6,2,5)` is available in the regular S3 algebra, but not forced by
S3 alone. The missing theorem is the defect-selection rule that chooses:

```text
s : a specific standard copy
b : a specific excluded one-dimensional line
```

## Full Theory Ledger

The proposed scalar coefficient architecture is:

```text
depth_scar:
  integer powers / hierarchy exponents

scalar_clebsch up:
  Taylor nilpotent grade response, x = 1/sqrt(2)

scalar_clebsch down:
  S3 baseline plus an open odd-shell bottom repair candidate

boundary_response CKM:
  coherent current amplitudes and CP holonomy
```

This keeps scalar masses separate from CKM current amplitudes. The same `sqrt(2)`
can appear in CKM as a coherent two-path enhancement while `1/sqrt(2)` appears
in scalar masses as a normalized one-step Taylor amplitude.

## What Remains Open

1. Derive `x = 1/sqrt(2)` from the scalar/Higgs repair channel rather than
   using it as a normalized amplitude axiom.
2. Derive or kill the defect-selection rule that turns S3 availability into
   the specific down counts `(6,2,5)`.
3. Integrate this sidecar with the common-scale quark mass audit.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/scalar_clebsch/tests -q
```
