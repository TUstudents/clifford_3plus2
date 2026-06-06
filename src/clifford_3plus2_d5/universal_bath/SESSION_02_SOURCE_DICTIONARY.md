# Session 02 - Write-Once Source Dictionary

Session 02 freezes the sources that can be fixed before any sector mass
moments are computed.  The rule is strict:

$$
V_f=\{\text{charge anchor},\text{ residual boundary port},
\text{ normal-depth placement}\}.
$$

If any part of $V_f$ is chosen after looking at a mass ratio, the universal bath
program has failed for that sector.

## Universal First Hop

The BB same-normal boundary blocks are

$$
B_+=
\begin{pmatrix}
1/2&i/2\\
0&0
\end{pmatrix},
\qquad
B_-=
\begin{pmatrix}
0&0\\
i/2&1/2
\end{pmatrix}.
$$

They obey the exact identity

$$
B_+^\dagger B_+ + B_-^\dagger B_- = \frac12 I.
$$

Therefore every normalized $q=0$ boundary source has the same radial
first-hop survival weight $1/2$.  This is a source-independent BB identity,
not a flavor fit.

Certainty: `C:9`.

## Frozen Anchors

The frozen source anchors are:

| Label | Sector | Source | Depth | Reduction | Certainty |
|---|---:|---|---:|---|---:|
| `neutrino_collective_u` | neutrino | $u=(1,1,1)/\sqrt3$ | 1 | scalar Jacobi | `C:7` |
| `neutrino_edge_b` | neutrino | $b=(0,1,-1)/\sqrt2$ | 0 | scalar Jacobi | `C:7` |
| `charged_lepton_active_e1` | charged lepton | $e_1=(1,0,0)$ | 2 | CMV/OPUC | `C:6` |

The charged-lepton source is recorded in the residual basis as

$$
e_1=\sqrt{\frac23}\,a+\frac1{\sqrt3}\,u+0\,b.
$$

The two-step depth is the existing charged-lepton leakage gate.  It is not a
charged-lepton mass fit and it does not derive the charged-lepton phase.

## Unresolved Anchors

The quark source records are present but not frozen:

| Label | Sector | Known anchor | Missing anchor | Certainty |
|---|---:|---|---|---:|
| `up_quark_boundary_source` | up quark | $Q_L\to u_R$ via $\tilde H$ | BCC source vector and depth | `C:3` |
| `down_quark_boundary_source` | down quark | $Q_L\to d_R$ via $H$ | BCC source vector and depth | `C:3` |

The existing quark sidecar supplies conditional shell, transfer-depth, color,
and BCC Clebsch gates.  It does not yet give the universal-bath source vector
$V_f$ as a charge-plus-boundary-geometry object.  Session 02 therefore refuses
to freeze quark sources.

This is not a failure of the source dictionary.  It is the anti-circularity
gate doing its job.

## Verdict

The certificate verdict is:

```text
SOURCE_DICTIONARY_CORE_PASS
```

Meaning:

- the lepton-side anchors currently supported by the repo are frozen;
- all frozen normalized sources share the exact BB survival weight $1/2$;
- no frozen or unresolved source record uses observed flavor data;
- quark sources remain explicitly unresolved until a microscopic BCC source
  vector is derived.

Session 03 may now attack neutrino cross-return moments.  It must not import a
quark source from this dictionary, because none is frozen yet.
