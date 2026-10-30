# Contributing

Thanks for taking an interest. Small, focused PRs are easiest to take.

## Before you open a PR

```bash
scripts/setup.sh          # environment
./idol-test --cli-only    # fast: no GPU, no weights — this is what CI runs
./idol-test               # full: real weights, GPU required
ruff check .              # style
```

Both test modes must pass. If you add a flag or change behaviour, add a check for it in
`idol_tools/smoke_test.py` — the tests are the contract, and the CLI section runs anywhere.

## Layout

```
idol_tools/generate.py    the idol-gen CLI: presets, validation, batch, --serve
idol_tools/viewer.py      the idol-view browser viewer (viser) and the .ply -> arrays conversion
idol_tools/smoke_test.py  the checks
scripts/setup.sh          environment setup and diagnostics
idol-sdk/                 upstream clone — treat as read-only
docs/                     archived model card, notes
```

## House rules

- **Don't patch `idol-sdk/`.** It is a clone of upstream and gets overwritten. If something there is wrong,
  work around it in `idol_tools/`, pin a commit with `SDK_TAG`, and file the issue upstream.
- **Keep the gated encoders gated.** `idol_tools/smoke_test.py` stubs DINOv3 and the FLUX.2 VAE so tests run
  without Hugging Face access. Never vendor, mirror, or commit those weights, and never route around
  `require_encoder_access` — that check is the licence boundary.
- **Keep `.venv` isolated.** No `--system-site-packages`; the launchers keep `PYTHONNOUSERSITE=1`. A stale
  user-site `peft` shadowing the venv breaks `diffusers` and is miserable to debug.
- **New flag → new check, and document it** in `README.md` (the levers table) and in `--help`.
- **Numbers need sources.** Timings and VRAM claims belong in `docs/` with the GPU and settings that produced
  them, not in the README as vibes.

## Ideas that would be welcome

- A `--compare` mode that loads two seeds side by side in the viewer with a sync-able camera.
- Turningtable renders (orbit a camera, export frames) so people without a splat viewer can still see results.
- Batch presets: a TOML file of named settings per input type (product shot, figurine, logo).
- Anything that makes the first-run experience clearer — the gated-encoder step is where people get stuck.

Please don't add: model fine-tuning, training code, or downloads of anything from the gated repos.
