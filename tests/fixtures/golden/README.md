# Golden fixtures

These files are the frozen reference the Rust port was proven against. They
were written by the Python implementation (Pulsar v0.5/v0.6, git tag
`python-reference`) through generator scripts that lived under
`legacy/python/scripts/golden/` until Python was removed from the repository
(the last commit that contains them is `508035a`). `manifest.json` declares
each artifact's tolerance; `crates/pulsar-parity/tests/` holds the parity
tests that read them.

Rules:

- **Never regenerate or edit a fixture to make a test pass.** A parity failure
  is a bug in the port (or a divergence to document in the module header).
- The Python reference is gone, so there are no new fixtures. New behaviour is
  covered by Rust known-answer and edge-case tests next to the code.
- To inspect how a fixture was produced, read its generator in git history
  (`git show 508035a:legacy/python/scripts/golden/<name>.py`) or check out the
  `python-reference` tag.
