## What changed

<!-- One paragraph: the problem and the approach. Link an issue if there is one. -->

## Checklist

- [ ] `./idol-test --cli-only` passes (this is what CI runs)
- [ ] `./idol-test` passes (real weights — needed for anything touching generation or the viewer)
- [ ] `ruff check .` is clean
- [ ] New flag or behaviour has a check in `idol_tools/smoke_test.py`
- [ ] `README.md` (and `--help`) document the change
- [ ] No edits inside `idol-sdk/`
- [ ] No gated weights, tokens, or machine-specific paths committed

## What you looked at

<!-- e.g. "generated 3 objects at --preset draft and checked _input.png", or "n/a, CLI-only change" -->
