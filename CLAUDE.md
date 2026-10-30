# idol — project notes

Local wrapper around the upstream `speridlabs/idol` image-to-3D SDK. Read `README.md` first.

## What lives where

- `idol-sdk/` — **upstream clone, treat as read-only.** Don't patch it to get something working; change
  `scripts/generate.py` or pin a commit via `SDK_TAG` in `scripts/setup.sh` and file the issue upstream.
- `scripts/generate.py` — our CLI. All new flags go here, not into the SDK.
- `scripts/smoke_test.py` — the test. `./idol-test` (add `--eager` for the non-compiled path).
- `idol-gen`, `idol-test` — thin launchers into `.venv`; they also re-exec from the venv if you run the
  scripts directly with another python.
- `docs/model-card.md` — archived model card (benchmarks, per-GPU timings, official flag table).

## Non-negotiables for this repo

- **Run the test before calling anything done.** `./idol-test` must pass; it covers the real weights, both
  sampling paths, PLY validity, determinism, and the CLI's preset/validation/batch logic. If you add a flag,
  add a check for it.
- **Keep the gated encoders gated.** The stub in `smoke_test.py` exists so tests run without HF access. Never
  vendor, commit, or mirror DINOv3 / FLUX.2 VAE weights, and never route around `require_encoder_access` —
  that check is the license boundary, not an obstacle.
- **Keep `.venv` isolated.** No `--system-site-packages`, and the launchers keep `PYTHONNOUSERSITE=1`: this
  machine's `~/.local` has an old `peft` that breaks `diffusers` when it leaks in.
- **Don't trust search results about this model.** It is days old; the model card and the SDK source are the
  only reliable references.

## Levers you are usually tuning

`--num_gaussians` (density) › `--steps` (detail, linear cost) › `--cfg_strength` (fidelity to the input) ›
`--rescale_t` (shape vs detail in the schedule) › `--mode`/`--crop_margin` (framing — often the real cause of
a bad result). Presets bundle the first four; explicit flags override presets.

## Gotchas

- First generation in a process pays `torch.compile`; don't measure speed or "hangs" on image #1.
- `--num_gaussians` must be a multiple of 32 and ≥ 32,768 — the SDK would otherwise silently round down, so
  the CLI validates and suggests the nearest valid value.
- A bad result is usually framing, not sampling: look at `<name>_input.png` before touching `--steps`.
- Empty alpha from BiRefNet → `no object found in the image`; the image likely has no single dominant object.
- OOM is caught and reported with the specific levers to pull rather than a stack trace.
