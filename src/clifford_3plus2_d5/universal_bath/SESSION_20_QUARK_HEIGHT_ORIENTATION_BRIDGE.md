# Session 20 - Quark Height-Orientation Bridge Audit

Session 08A separated two facts:

```text
hypercharge forces H_tilde for up and H for down
```

but

```text
hypercharge does not force up -> nilpotent, down -> Hermitian.
```

Session 20 imports the stronger depth-scar successor certificate and asks
whether the height premise can be reduced.

## Certified Repair Flag

The depth-scar certificate gives the oriented active successors

```text
a -> u
b -> a
```

so the repair flag is

$$
N = |u\rangle\langle a| + |a\rangle\langle b|.
$$

This is exactly the up-sector oriented nilpotent:

$$
N^3=0,\qquad N^2\ne0.
$$

The down-sector Hermitian closure is not an independent object.  It is the
flag Laplacian

$$
\Delta_N = N N^T + N^T N - (N+N^T).
$$

Thus:

```text
up repair   = oriented nilpotent flag N
down repair = Hermitian closure Delta_N
```

The two quark repair objects are two readouts of one certified successor flag.

## What Still Does Not Follow

The SM charge doors remain:

```text
up   -> H_tilde
down -> H
```

and both neutral Higgs components have $Q_{\rm em}=Y+T_3=0$.  But Session 08A's
negative control still stands: the swapped repair assignment is hypercharge
allowed.  The successor certificate derives the oriented flag; it does not
derive which Higgs door couples to the retarded flag and which couples to the
Hermitian closure.

The remaining premise is sharpened to:

```text
higgs_door_orientation_couples_H_tilde_to_retarded_flag_and_H_to_flag_closure
```

## Verdict

```text
QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT
```

What is now exact:

- the depth-scar successor certificate supplies $a\to u$, $b\to a$;
- the declared up repair is exactly this nilpotent flag;
- the declared down repair is exactly the Hermitian closure of the same flag;
- the up/down repair objects are not independent assumptions.

What remains open:

- the microscopic Higgs-door orientation coupling;
- the actual quark source vectors and normal-depth placements.

## Certainty

- `C:9` for the algebraic identity between the successor flag, up nilpotent,
  and down Hermitian closure.
- `C:9` for the hypercharge-door audit and swapped-assignment control.
- `C:3` for the remaining Higgs-door orientation-coupling premise.
