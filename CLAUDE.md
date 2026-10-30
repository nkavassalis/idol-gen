# idol — project notes

Public CLI + web app around the upstream `speridlabs/idol` image-to-3D SDK. Read `README.md` first.

## What lives where

- `idol_tools/generate.py` — the `idol-gen` CLI: presets, `--density`, validation, batch, `--serve`.
- `idol_tools/viewer.py` — the `idol-view` browser viewer (viser) and the `.ply` → arrays conversion
  (quaternion → covariance, SH-DC → RGB). `build_scene()` is split from `serve()` so it is testable.
- `idol_tools/app.py` — the `idol-app` web UI: upload, quality levers, generate, view, download, all in one
  viser page. It reuses `generate.PRESETS`/`DENSITY` (so CLI and app cannot drift) and displays results by
  re-reading the `.ply` it just wrote. Generations are serialised on `AppState.lock`; errors go to the page.
- `idol_tools/smoke_test.py` — the checks. `./idol-test`, `--eager`, or `--cli-only` (no GPU/weights; CI uses it).
  `FakeViser` drives the viewer and the app without a browser; `FakePipeline` drives the app without weights.
- `idol-gen`, `idol-view`, `idol-app`, `idol-test` — bash launchers into `.venv` (prefer the installed console
  scripts, fall back to `python -m idol_tools.<mod>`). Windows users take the console scripts / `-m` route.
- `scripts/setup.sh` — env setup + diagnostics: `--check`, `--rebuild`, `--no-weights`, `--cpu`, `SDK_TAG=`.
- `idol-sdk/` — **upstream clone, read-only.** Never patch it to get something working; change `idol_tools/`,
  or pin with `SDK_TAG` and file the issue upstream.
- `pyproject.toml` — our package (`idol-gen` console scripts, `[view]`/`[dev]` extras); depends on the SDK by
  git URL, and setup.sh then installs the local clone editable so it wins.
- `docs/model-card.md` — archived model card (benchmarks, per-GPU timings, official flag table).
- `.github/workflows/ci.yml` — ubuntu + windows, CPU torch, `ruff check .`, `smoke_test --cli-only`.

## Non-negotiables for this repo

- **Run the test before calling anything done.** `./idol-test` must pass (and `--cli-only` for CI). It covers
  the real weights, both sampling paths, PLY validity, determinism, the quaternion/covariance maths, the
  `.ply` round trip incl. the z-up→y-up convention, and the CLI's preset/validation/batch logic. New flag →
  new check.
- **Keep the gated encoders gated.** The stub in `smoke_test.py` exists so tests run without HF access. Never
  vendor, commit, or mirror DINOv3 / FLUX.2 VAE weights, and never route around `require_encoder_access` —
  that check is the licence boundary, not an obstacle.
- **Keep `.venv` isolated.** No `--system-site-packages`, and the launchers keep `PYTHONNOUSERSITE=1`: this
  machine's `~/.local` has an old `peft` that breaks `diffusers` when it leaks in.
- **This is a public repo.** No machine-specific paths, hostnames, or "works on my GPU" claims in the README;
  timings belong in `docs/` with the hardware and settings that produced them.
- **Don't trust search results about this model.** It is days old; the model card and the SDK source are the
  only reliable references.

## Levers you are usually tuning

`--num_gaussians` (density) › `--steps` (detail, linear cost) › `--cfg_strength` (fidelity to the input) ›
`--rescale_t` (shape vs detail in the schedule) › `--mode`/`--crop_margin` (framing — often the real cause of
a bad result). Presets bundle the first four; explicit flags override presets. To change density alone use
`--density {32k,64k,128k,256k,512k,1m}` (fixed rungs, beats the preset); `--num_gaussians` is the exact
knob and also accepts shorthand like `256k`/`0.5m`. The two are mutually exclusive.

## Gotchas

- First generation in a process pays `torch.compile`; don't measure speed or "hangs" on image #1.
- `--num_gaussians` must be a multiple of 32 and ≥ 32,768 — the SDK would otherwise silently round down, so
  the CLI validates and suggests the nearest valid value. `--density` avoids the question entirely.
- `k`/`m` in density values are 1024/1024² (`256k` = 262,144), matching the model's power-of-two rungs.
- **Up axis.** The SDK's `Gaussian` is z-up (verified: rendering from the object's own predicted camera with
  world up +z reproduces the input image). Upstream's `save_ply` rotates +90° about x, which sends up to **-y**
  — upside down for every y-up viewer. We write the file ourselves (`idol_tools/ply.py`) with -90° about x so
  up is **+y**; `--ply_convention upstream` reproduces upstream's bytes exactly, and `UP_AXIS` is the single
  source of truth the viewer and app read. `test_ply_convention` pins all of it — do not "simplify" the
  rotation back without checking that test and the README section.
- A bad result is usually framing, not sampling: look at `<name>_input.png` before touching `--steps`.
- Empty alpha from BiRefNet → `no object found in the image`; the image likely has no single dominant object.
- OOM is caught and reported with the specific levers to pull rather than a stack trace.
- `serve()` blocks forever by design; tests must use `build_scene()` with the fake viser, never `serve()`.
- The ground is a **floor**: its plane must be perpendicular to the up axis (`xz` for `+y`, `xy` for `+z` —
  asking for `xy` with a y-up scene gives a vertical wall through the object) and it sits at the lowest point
  of the visible objects, not at the origin, because Idol centres objects on the origin. `viewer.UP_INDEX` /
  `GROUND_PLANE` / `floor_height` / `add_ground` are the only places that decide this; the app reuses them.
- The app shows results by parsing the `.ply` back off disk, not from the in-memory `Gaussian` — deliberate, so
  the viewport provably matches the file. The in-memory object is z-up and the `.ply` is y-up; don't shortcut.
- Real viser handles are needed to catch API drift (`add_upload_button`, `send_file_download`, `atomic`,
  `sleep_forever`). `--no-model` boots the page without weights for exactly that kind of check.
