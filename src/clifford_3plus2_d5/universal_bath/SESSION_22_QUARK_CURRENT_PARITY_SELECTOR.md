# Session 22 - Quark Current-Parity Selector

Session 21 used the source hypothesis

$$
\text{quark source}=\text{colored active current line }b.
$$

This session asks whether the line $b$ is merely a convenient active direction
or whether it is selected by the residual boundary symmetry.

## Selected $S_2$

After selecting the first family port $e_1$, the residual symmetry preserving
that selection is the swap of the two unselected ports:

$$
e_2 \leftrightarrow e_3.
$$

In the residual basis

$$
u=\frac{(1,1,1)}{\sqrt3},\qquad
a=\frac{(2,-1,-1)}{\sqrt6},\qquad
b=\frac{(0,1,-1)}{\sqrt2},
$$

the selected $S_2$ action is exactly

$$
S_2|_{(u,a,b)}=
\operatorname{diag}(+1,+1,-1).
$$

Therefore

$$
P_{\rm even}=P_u+P_a,\qquad P_{\rm odd}=P_b.
$$

The odd line is the oriented current across the two unselected ports:

$$
J_{23}=\frac{e_2-e_3}{\sqrt2}=b.
$$

So $b$ is not just a non-scalar active line.  It is the unique selected-$S_2$
odd current line.

## Relation To Active Plane

Session 11 derived the selected active incidence plane:

$$
P_{\rm act}=P_u+P_b,\qquad P_{\rm rad}=P_a.
$$

The active plane alone is not sufficient, because it still contains the even
collective scalar line $u$.  The selected-odd current condition removes that
ambiguity:

$$
P_{\rm act}P_{\rm odd}=P_b.
$$

Controls:

- the selected scalar port $e_1$ has no $b$ component;
- $u$ is active but even/scalar, so it is not a current;
- $a$ is even and radial/gapped, so it is not a current;
- active incidence alone selects a plane, not a line.

## Verdict

```text
QUARK_CURRENT_PARITY_SELECTOR_PASS
```

What is gained:

- the Session 21 current line $b$ is now an exact selected-$S_2$ odd-current
  representation;
- the source-selector statement is reduced from "choose the active non-scalar
  line" to "colored quark mass events are selected-$S_2$ odd boundary
  currents";
- the up first-passage source used in Session 21 is no longer a raw geometric
  choice once the odd-current premise is admitted.

What remains open:

```text
colored_quark_mass_source_is_selected_S2_odd_boundary_current
down_reads_hermitian_current_covariance_not_scalar_b_vector
one_tick_retarded_down_return_vetoes_identity_word
```

The first line is smaller and sharper than the old current-source premise, but
it is still a physical statement about colored boundary current dynamics.  The
session does not derive the Higgs-door orientation coupling, the down
Hermitian covariance readout, or the identity-return veto.

## Certainty

- `C:9` for the selected-$S_2$ parity decomposition.
- `C:9` for $J_{23}=b$.
- `C:9` for $P_{\rm act}P_{\rm odd}=P_b$.
- `C:6` for the physical quark-current interpretation.
- `C:4` for using this as a complete quark source freeze, because the down
  readout and Higgs-door orientation gates remain open.
