# Session 17 - Quark Active Color-Return Microcanonical Audit

Session 08B showed that visible color covariance rejects a fixed color vector
but does not choose between:

```text
color-scalar spectator return  -> residual 3-port shell
active hidden color return     -> six primitive quark labels
```

Session 17 asks whether the upstream primitive-shell thermodynamics selects the
active branch.

## Primitive Shell

The primitive quark shell is

```text
1_even + 5_odd = 1_direct + 2_BCC + 3_color
```

with labels

```text
direct_even_return
bcc_odd_quadrature_1
bcc_odd_quadrature_2
color_odd_red
color_odd_green
color_odd_blue
```

The active hidden color-return lift reaches exactly this shell:

```text
even_direct = 1
bcc_odd     = 2
color_odd   = 3
odd_total   = 5
total       = 6
```

The spectator shell has only three residual ports, so it is a compressed
control from the primitive-shell point of view.

## Microcanonical Reduction

Boundary-response V23 proves:

```text
equal primitive-label degeneracy
  -> rho_label = I_6 / 6
```

Thus every primitive label has weight

$$
\left(\frac16,\frac16,\frac16,\frac16,\frac16,\frac16\right).
$$

Compressed macrochannel counting gives the wrong branch:

$$
r=\frac1{\sqrt5},
\qquad
\phi=\frac\pi4,
$$

and is rejected by the upstream control.

## Verdict

```text
QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS
```

Meaning:

```text
Inside the six-label primitive-shell model, equal-degeneracy microcanonical
reduction selects active hidden color return over the compressed spectator
shell.  This reduces the active-color blocker to the physical
equal-degeneracy / max-entropy prior.
```

## Honest Boundary

This is not a gauge-covariance theorem and it does not by itself freeze
`V_u,V_d`.  The remaining physical input is:

```text
equal_boundary_degeneracy_or_max_entropy_prior
```

along with the upstream `vacuum_framing` and `transfer_probe` inputs of the
primitive-shell theorem.

Certainty: `C:9` for the exact microcanonical reduction and compressed-control
rejection; `C:5` for using it as the active hidden color-return selection,
because the equal-degeneracy / max-entropy prior remains physical input.
