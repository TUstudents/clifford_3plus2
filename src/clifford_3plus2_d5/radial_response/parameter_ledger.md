# radial_response Parameter Ledger

| Item | Value | Status |
|---|---:|---|
| Mass mechanism | `Sigma(z)=V^T(z-H_Q)^-1V` | R1 exact Schur/Feshbach identity |
| Recirculation path | `P -> Q -> ... -> Q -> P` | R1 exact series expansion |
| Silver transfer root | `epsilon = sqrt(2) - 1` | R11 inherited from `boundary_response.transfer` / V26 |
| Radial amplitude factor | `eta = epsilon^2 = 3 - 2 sqrt(2)` | R11 inherited exact power |
| Intensity/mixing factor | `r = epsilon^4 = 17 - 12 sqrt(2)` | R11 inherited exact power |
| Shared transfer source | residual K3 graph + sterile-chain Weyl function | R11 inherited from `boundary_response` and `flavor_a_track` |
| Independent fitted eta | `0.172089` | R11 rejected as source of truth |
| Alternate transfer roots | K2/K4 roots | R11 rejected |
| Exponential up invariant | `1/2` | R2 exact stacking result |
| Geometric up invariant | `1` | R2 exact control |
| Up repair amplitude | `x = 1/sqrt(2)` | R5 conditional on two equal no-leakage scalar repair successors |
| Two-channel repair norm | `2 x^2 = 1` | R5 exact isometry result |
| Scalar repair successors | `triality_plus`, `triality_minus` | R7 finite modeled certificate |
| Scalar successor controls | same-state, wrong-height, two-tick, leakage, asymmetric, third successor | R7 vetoed |
| Scalar successor basis completeness inside S3 | `A3 \ {e}` | R8 exact S3 census |
| BCC/vacuum-framed reduction to S3 | selected-exit stabilizer induces residual S3 | R9 conditional on automorphism premise |
| Full QCA scalar boundary completeness | not derived | R9 remaining burden |
| Literal `exp(N)` singular values | `(2,1,1/2)` at `x=1` | R3 kill control |
| Down rank-5 source | regular S3 minus one 1D line | R4 available |
| Down rank-5 selection | trivial vs sign exclusion | open |
| Down rank-2 source | defect-polarized standard copy | R4 available |
| Down rank-2 selection | choice of standard copy | open |
| Minimal Floquet defect form | `U = S C` | R6 exact unitary toy |
| Unitary self-energy | `U_PQ (z I - U_QQ)^-1 U_QP` | R6 exact Schur form |
| Phase/radial values forced by `U=S C` | false | R6 and R11 controls keep them open |
| One-level triality-head bath | `Sigma_1(z) = 1/z` | R12 admissible spectral measure |
| Two-level symmetric tail bath | `Sigma_2(z) = (z - 1)/(z^2 - z - 1)` | R12 admissible spectral measure |
| Radial pole/residue rigidity from finite data | false | R12 no-go |
| Up target spectral measure | poles `{eta^6/4, eta^3/sqrt(2), 1}`, weights `{1/25, 8/25, 16/25}` | R13 positive finite measure |
| Down S3-baseline target measure | poles `{sqrt(3/2) eta^4, eta^2/sqrt(2), 1}`, weights `{1/2, 1/6, 1/3}` | R13 positive finite measure |
| Down odd-shell target measure | poles `{sqrt(6/5) eta^4, sqrt(2/5) eta^2, 1}`, weights `{6/13, 2/13, 5/13}` | R13 positive finite measure |
| Target-measure Jacobi bath | unique inverse finite Jacobi reconstruction | R13 exact reconstruction |
| Existing simple forward bath selects target measures | false | R13 controls reject R12 baths, P3 Jacobi, silver-tail Jacobi, minimal unitary toy |
| Additional spectral-density principle | required | open after R13 |

## Interpretation

Finite S3/BCC data controls channels, projectors, and transfer-depth grammar.
The silver transfer root is no longer a radial-side input: it is inherited from
the already-derived boundary transfer stack. R12 shows that the current finite
S3/silver-transfer data do not force radial mass poles or residues. R13 shows
that the desired mass textures can be encoded as positive finite spectral
measures and inverse Jacobi baths, but that this encoding is reconstruction
only: the existing simple forward baths do not select the target measures.
Radial masses therefore require a forward spectral-density principle. This
sidecar keeps that unresolved input named instead of duplicating transfer
calculations.
