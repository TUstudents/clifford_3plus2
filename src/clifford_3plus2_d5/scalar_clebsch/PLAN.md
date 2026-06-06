# scalar_clebsch Plan

## Purpose

Test whether quark mass Clebsches can be given a scalar boundary-response
origin:

```text
C_u = (1/4, 1/sqrt(2), 1)
C_d,baseline = (1, 1/sqrt(3), sqrt(2/3))
C_d,candidate = (1, 1/sqrt(3), sqrt(5/6)).
```

This sidecar is intentionally separate from `boundary_response`'s CKM current
Clebsches. It treats scalar mass coefficients as positive response weights, not
coherent path amplitudes.

## Implemented Gates

### V2a - Up-sector nilpotent Taylor response

Pass only if:

- the length-3 nilpotent flag has `N^3=0`, `N^2 != 0`;
- the scalar Taylor kernel is `exp(xN)=I+xN+x^2 N^2/2`;
- the normalized repair amplitude is `x=1/sqrt(2)`;
- the grade profile `(x^2/2,x,1)` gives `(1/4,1/sqrt(2),1)`;
- the nearby rational `(1/4,3/4,1)` is kept as a control, not the theorem;
- `(1/2,sqrt(2),1)` is rejected as a scalar positive grade profile.

### V2b - Down-sector baseline and odd-shell candidate

Pass only if:

- the primitive quark shell prerequisite passes in `boundary_response`;
- the S3/projector baseline has counts `(6,2,4)`;
- `sqrt(n/6)` gives `(1,1/sqrt(3),sqrt(2/3))` for the baseline;
- the data-improved odd-shell candidate has counts `(6,2,5)`;
- `sqrt(n/6)` gives `(1,1/sqrt(3),sqrt(5/6))` for the candidate;
- the candidate ratio formulas are `sqrt(3) eta^2` and `sqrt(2/5) eta^2`;
- the bottom `4 -> 5` shift remains explicitly open.

### V3 - S3 projector availability audit

Pass only if:

- the regular S3 shell has central ranks `1`, `1`, `4`, and `6`;
- rank `5` is available as `regular - trivial` and `regular - sign`;
- rank `2` is available as a defect-selected standard copy;
- rank `2` standard is not supplied by central S3 alone;
- rank `5` is not unique without choosing which one-dimensional line is
  excluded.

## Next Gates

- V4: derive `x=1/sqrt(2)` from scalar/Higgs boundary repair normalization.
- V5: derive or kill the defect-selection rule that chooses the standard copy
  and the excluded one-dimensional line for the down candidate `(6,2,5)`.
- V6: connect to a common-scale quark mass audit.
- V7: decide whether the down candidate or baseline is the correct scalar mass
  coefficient once V5 is settled.
