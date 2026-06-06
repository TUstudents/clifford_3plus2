# Session 24 - Quark Higgs-Door Orientation Coupling

This session attacks the remaining quark keystone:

```text
H_tilde -> retarded/oriented flag N
H       -> Hermitian closure Delta_N
```

The available inputs are strong:

- SM hypercharge forces the door labels $H_\sim$ for up and $H$ for down;
- the depth-scar successor certificate supplies one oriented flag;
- Session 22 selects the quark current source as the selected-$S_2$ odd line
  $b$;
- Session 23 selects the rank-five down bottom profile inside the retarded
  current model.

The question is whether these facts force the door-to-readout assignment.

## Reflection Control

Use residual order $(u,a,b)$ and the endpoint reflection

$$
R=
\begin{pmatrix}
0&0&1\\
0&1&0\\
1&0&0
\end{pmatrix}.
$$

The certified flag is

$$
N=|u\rangle\langle a|+|a\rangle\langle b|.
$$

Endpoint reflection gives

$$
RNR=N^T.
$$

So the available conjugation geometry supplies **orientation reversal**.

The down operator is not $N^T$.  It is the paired Hermitian closure:

$$
\Delta_N = NN^T+N^TN-(N+N^T).
$$

It is reflection invariant:

$$
R\Delta_N R=\Delta_N.
$$

But reflection alone does not produce it:

$$
RNR\neq\Delta_N.
$$

Therefore Higgs conjugation/reversal supplies the reverse flag, not the extra
pairing operation that makes the Hermitian down readout.

## Assignment Control

The desired assignment is:

```text
H_tilde -> N
H       -> Delta_N
```

The swapped control is:

```text
H_tilde -> Delta_N
H       -> N
```

Under the currently available constraints, both are constructible:

- both use the SM-forced $H_\sim/H$ door labels;
- both preserve neutral Higgs components;
- both use the same selected-$S_2$ odd current source;
- both use readout objects already supplied by the depth-scar flag.

Thus the available constraints do **not** select a unique assignment.

## Verdict

```text
QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT
```

What is gained:

- the obstruction is now exact: conjugation gives $N^T$, not $\Delta_N$;
- the needed extra operation is named as **pairing/Hermitian closure**, not
  merely "down is different";
- the swapped assignment remains the live negative control.

What remains open:

```text
higgs_door_orientation_couples_H_tilde_to_retarded_flag_and_H_to_flag_closure
```

This is a real dynamical boundary premise.  A future proof must derive why the
direct Higgs door reads the paired current covariance while the conjugate door
reads the one-way retarded flag.  Symmetry and conjugation alone do not do it.

## Certainty

- `C:9` for $RNR=N^T$.
- `C:9` for $R\Delta_N R=\Delta_N$.
- `C:9` that $RNR\neq\Delta_N$.
- `C:9` that the swapped assignment survives the currently encoded gauge,
  current, and flag constraints.
- `C:3` for the door-to-readout assignment as a physical derivation until a
  boundary dynamics supplies the missing pairing rule.
