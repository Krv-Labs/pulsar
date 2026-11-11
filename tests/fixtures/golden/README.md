# Golden fixtures

These files are the frozen reference the Rust port was proven against. They
were written by the Python implementation **thema-pulsar 0.5.0** at commit
`afe7941` (recorded under `reference` in `manifest.json`), through
`scripts/golden/generate.py`. That generator is still on the `v0.6.0` tag,
the last release that contains Python. Behaviour added in 0.6.0 after
`afe7941` (preprocessing plan, dossier payload budgets, multi-imputation) has
no fixture here; it is covered by Rust known-answer tests next to the code.
`manifest.json` declares each artifact's tolerance; `crates/pulsar-parity/tests/`
holds the parity tests that read them.

Rules:

- **Never regenerate or edit a fixture to make a test pass.** A parity failure
  is a bug in the port (or a divergence to document in the module header).
- The Python reference is gone, so there are no new fixtures. New behaviour is
  covered by Rust known-answer and edge-case tests next to the code.
- To inspect how a fixture was produced, read the generator
  (`git show v0.6.0:scripts/golden/generate.py`) or check out `afe7941`.
