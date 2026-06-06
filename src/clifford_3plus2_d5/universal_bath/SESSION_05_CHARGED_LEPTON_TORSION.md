# Session 05 - Charged-Lepton $2/9$ Torsion Gate

Session 05 answers a narrow question left open by the charged-lepton CMV head:
does the number $2/9$ appear as a boundary object, or was it only arithmetic?

The frozen Session 02 source is still

$$
e_1=\sqrt{\frac23}\,a+\frac1{\sqrt3}\,u+0\,b.
$$

Therefore the port occupations are

$$
p_a=\frac23,\qquad p_u=\frac13,\qquad p_b=0.
$$

The incoherent two-channel transition weight is

$$
p_a p_u=\frac23\frac13=\frac29.
$$

So $2/9$ does jump out as a source-geometry torsion moment.

## Negative Controls

The coherent amplitude is not $2/9$:

$$
\left(\sqrt{\frac23}\right)\left(\frac1{\sqrt3}\right)
=\frac{\sqrt2}{3}.
$$

An equal two-port source would give

$$
\frac12\frac12=\frac14,
$$

and one-port controls give zero transition weight.  These controls reject the
interpretation that any two-channel source automatically gives $2/9$.

## Verdict

The certificate verdict is:

```text
CHARGED_LEPTON_TORSION_2_OVER_9_PASS
```

Meaning:

- `C:9` exact algebra that the frozen $e_1$ source gives $p_a p_u=2/9$;
- `C:7` as boundary theory because the source itself is upstream-gated;
- the CMV phase and the full charged-lepton mass angle are not rederived by
  this session.

The honest statement is: $2/9$ is a real occupation-transition invariant of the
frozen colorless active source, but the dynamical conversion from that moment
into the observed charged-lepton holonomy remains a separate gate.
