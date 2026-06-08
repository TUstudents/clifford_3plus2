"""Session 01 certificate for the Stage 1 free BCC Weyl/Dirac walk."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import (
    bare_bcc_walk_diagnostics,
    bcc_dirac_step,
    bcc_weyl_step,
)


def main() -> None:
    key = jax.random.PRNGKey(11)
    shape = (4, 3, 2, 2)
    real = jax.random.normal(key, shape, dtype=jnp.float32)
    imag = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float32)
    state = (real + 1j * imag).astype(jnp.complex64)
    dirac_state = jnp.concatenate((state, state), axis=-1)

    diagnostics = bare_bcc_walk_diagnostics(state)
    jitted_weyl = jax.jit(bcc_weyl_step)
    jitted_dirac = jax.jit(bcc_dirac_step)
    weyl_jit_delta = jnp.max(jnp.abs(jitted_weyl(state) - bcc_weyl_step(state)))
    dirac_jit_delta = jnp.max(jnp.abs(jitted_dirac(dirac_state) - bcc_dirac_step(dirac_state)))

    print("QCA_SMv0 Session 01 - Stage 1 free BCC Weyl/Dirac walk")
    print(f"hop_count: {diagnostics.hop_count}")
    print(f"weyl_hop_completeness_residual: {float(diagnostics.hop_completeness_residual):.3e}")
    print(f"weyl_symbol_unitarity_residual: {float(diagnostics.symbol_unitarity_residual):.3e}")
    print(f"weyl_norm_drift: {float(diagnostics.norm_drift):.3e}")
    print(f"weyl_small_k_speed_error: {float(diagnostics.small_k_speed_error):.3e}")
    print("anisotropy_spreads:", [float(value) for value in diagnostics.anisotropy_spreads])
    print("anisotropy_halving_ratios:", [float(value) for value in diagnostics.anisotropy_halving_ratios])
    print(f"dirac_hop_completeness_residual: {float(diagnostics.dirac_hop_completeness_residual):.3e}")
    print(f"dirac_symbol_unitarity_residual: {float(diagnostics.dirac_symbol_unitarity_residual):.3e}")
    print(f"dirac_norm_drift: {float(diagnostics.dirac_norm_drift):.3e}")
    print(f"dirac_small_k_speed_error: {float(diagnostics.dirac_small_k_speed_error):.3e}")
    print(f"weyl_jit_delta: {float(weyl_jit_delta):.3e}")
    print(f"dirac_jit_delta: {float(dirac_jit_delta):.3e}")
    print("verdict: QCA_SMV0_STAGE1_FREE_BCC_PASS")


if __name__ == "__main__":
    main()
