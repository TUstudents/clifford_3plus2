# spacetime_qca.lab

Prototype runners and tiny deterministic controls live here.

Use this package for:

- debug-friendly Python-stepped runs;
- deterministic tiny-lattice probes;
- experimental runner changes before they are stable enough for the main
  simulator API.

The current lab entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.lab.scripts.run_tiny_sim
```

The lab runner is intentionally not the production path.  Stable simulation
campaigns should use `clifford_3plus2_d5.spacetime_qca.simulator`.
