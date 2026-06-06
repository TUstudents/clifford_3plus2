# Session 23 - Down Identity-Return Veto

Session 18 localized the down bottom fork:

```text
(6,2,4)/6 -> (1,1/sqrt(3),sqrt(2/3))
(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))
```

Session 21 reframed the down readout as a Hermitian current covariance, not a
scalar vector.  Session 22 showed that the quark source line is the selected
$S_2$ odd current line $b$.  The remaining question is whether the down current
may use the identity/direct contact return.

## Retarded Current Criterion

The finite criterion tested here is:

```text
a down mass event must leave the visible sheet before returning.
```

Inside the active primitive quark shell:

```text
1_even + 5_odd = 1_direct + 2_BCC + 3_color
```

the only zero-excursion return is:

```text
direct_even_return
```

It has return order $0$ and is rejected by the retarded-current criterion.

The allowed retarded returns are exactly:

```text
2_BCC + 3_color = 5 odd channels.
```

Therefore the bottom readout becomes the full primitive odd shell:

$$
C_b=\sqrt{\frac56}.
$$

The strange readout remains the BCC odd doublet:

$$
C_s=\sqrt{\frac26}=\frac1{\sqrt3}.
$$

Thus the retarded down profile is:

$$
C_d=
\left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

## Control

The contact/baseline profile is:

$$
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right).
$$

It has bottom count $4$, not the five non-identity odd current returns, so it
fails the retarded predicate.  This does not make the baseline algebraically
wrong; it means the baseline is the contact-allowed/S3-projector readout, while
the retarded-current readout selects the odd shell.

## Verdict

```text
DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS
```

What is gained:

- the down $4\to5$ fork is decided inside the retarded-current model;
- the identity/direct line is the unique rejected zero-excursion channel;
- the rank-five bottom coefficient is no longer an arbitrary S3 complement but
  the full non-identity odd current shell;
- the strange coefficient remains the BCC odd doublet.

What remains open:

```text
down_mass_event_requires_nonidentity_hidden_excursion_before_return
colored_quark_mass_source_is_selected_S2_odd_boundary_current
```

The session does not derive the retarded-current criterion from bare BB block
algebra.  It proves that **if** the down Hermitian mass readout is a non-contact
retarded current covariance, the bottom coefficient is $\sqrt{5/6}$.

## Certainty

- `C:9` for the primitive shell decision once the retarded criterion is stated.
- `C:9` for rejecting the direct identity line under that criterion.
- `C:9` for the resulting count vector $(6,2,5)$.
- `C:6` for the physical retarded-current interpretation.
- `C:4` as a from-bare-BB theorem, because the material/dynamical origin of the
  non-contact down current criterion is still open.
