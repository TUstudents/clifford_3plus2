# Simulator infrastructure

The `sim` package is shared infrastructure. It is not a physics sidecar.

It provides generic JAX helpers for:

- dtype/backend handling;
- periodic pull-roll lattice utilities;
- generic state/link containers;
- finite-value diagnostics;
- observable selection and stacking;
- Python-loop and `jax.lax.scan` recorded runners;
- `.npz` plus JSON sidecar persistence;
- benchmark and warm profiling payloads.

The key boundary is:

$$
\texttt{sim}=\text{generic runner infrastructure},
\qquad
\texttt{spacetime\_qca}=\text{BCC/SM physics policy}.
$$

Thus `sim` contains no BCC Weyl walk, no Pati-Salam data, no Wilson-force
normalization, no Higgs/Yukawa rule, and no Gauss-law physics. C:9.

The simulator split matters for synthesis only because it prevents simulator
bookkeeping from being mistaken for theory. The scan-backed runner can support
future numerical campaigns, but it does not add a theorem about flavor or
mass.

Paper-level statement:

> The simulator stack is an implementation substrate for running and profiling
> QCA field prototypes. It carries no independent flavor, CP, or mass
> mechanism.

Certainty: C:9.

