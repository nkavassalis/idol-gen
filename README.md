# idol-gen — one image → 3D Gaussian splat

Turn a single picture of an object into a 3D Gaussian splat `.ply`, from the command line, on your own GPU —
then look at it in your browser.

This is a friendly CLI and viewer around **[Idol](https://huggingface.co/speridlabs/idol)** by
[Sperl Labs](https://speridlabs.com/research/idol), an open-weights image-to-3D model that removes the
background, frames the object, and predicts up to 1,048,576 Gaussians including the sides the photo never
showed. ~7 GB VRAM, seconds per object on a consumer card.

```bash
./idol-gen photo.jpg --mode real --preset high      # -> outputs/photo.ply
./idol-view outputs/                                # orbit it in the browser
./idol-app                                          # or skip the terminal: upload, generate, view in one page
```

**What this repo adds** on top of the (excellent) upstream SDK: a **browser app** that does the whole thing
without a command line, presets and a readable density ladder, argument validation that fails before you burn
a minute of GPU time, batch folders with resume, seed variants, a local viewer, a setup script that tells you
exactly what is missing, and a test suite that runs without Hugging Face access.

**What it is not:** the model, the training code, or an affiliation with Sperl Labs. The model is theirs —
please cite their [project page](https://speridlabs.com/research/idol).

## Contents

- [Requirements](#requirements) · [Install](#install) · [Quickstart](#quickstart)
- [Web app](#web-app) · [Quality levers](#quality-levers) · [Viewing](#viewing) · [Batch](#batch)
- [How it works](#how-it-works) · [Tests](#tests) · [Troubleshooting](#troubleshooting) · [License](#license)

## Requirements

- NVIDIA GPU with CUDA and ~7 GB free VRAM (works on 16 GB consumer cards; CPU inference exists but is
  impractical).
- Python 3.10+.
- A Hugging Face account with the two **gated** image encoders accepted — the install step covers this.

## Install

```bash
git clone https://github.com/YOUR_USER/idol-gen.git && cd idol-gen
scripts/setup.sh
```

The script clones the upstream SDK, builds an isolated `.venv` (uv if you have it, pip otherwise), installs
dependencies, prefetches the ~3.4 GB of model weights, and checks whether you can reach the gated encoders.
It is idempotent; `--check` re-runs the diagnostics, `--rebuild` starts from scratch.

**The gated encoders.** Idol reads its input through two models that require you to accept their terms:

- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (only its VAE is used)

Accept both, then `hf auth login` (or `export HF_TOKEN=hf_...`). Verify with:

```bash
./idol-gen some.png --dry-run     # prints resolved settings + encoder access, generates nothing
```

<details>
<summary>Windows, or installing without the script</summary>

```bash
python -m venv .venv && source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
git clone --depth 1 https://github.com/speridlabs/idol-sdk.git
pip install -e ./idol-sdk -e ".[view]"
python -m idol_tools.smoke_test --cli-only              # sanity check
```

The `./idol-gen` wrappers are bash; on Windows call `python -m idol_tools.generate` / `idol_tools.viewer`,
or the installed `idol-gen` / `idol-view` console scripts.
</details>

## Quickstart

```bash
./idol-gen photo.jpg --mode real                 # photograph (looser crop)
./idol-gen render.png                            # render or AI image (tight crop, the default)
./idol-gen render.png --preset draft             # 20 steps, 65k — cheap, iterate here
./idol-gen render.png --preset high              # 75 steps, 524k
./idol-gen render.png --density 32k              # density only
./idol-gen render.png --serve --open             # generate, then view in the browser
```

Every image writes:

| File | What it is |
|---|---|
| `<name>.ply` | the object, standard 3DGS — opens in [gsplat.tech](https://gsplat.tech), SuperSplat, Blender add-ons |
| `<name>_input.png` | the 1024² image the model actually saw, after background removal and cropping |
| `<name>_camera.json` | predicted yaw/pitch/radius/fov of the input view |

`<name>_input.png` is the fastest way to diagnose a bad result — if the object is badly framed or bitten into
there, no sampling setting will save it.

## Web app

If you would rather not touch a terminal at all:

```bash
./idol-app                      # http://localhost:7860
./idol-app --image photo.jpg --open
./idol-app --port 8080 --out shots/ --host 0.0.0.0     # e.g. on a headless box you reach over the network
```

One page: drop in a picture, pick a preset or pull the same levers the CLI exposes (density, steps, CFG,
time shift, mode, crop margin, seed), press **Generate**, and the splat appears in the 3D viewport beside the
exact 1024² image the model saw. Results are written to `outputs/web/` as `.ply` + `_input.png` +
`_camera.json`, and a **Download** button hands you the `.ply` in the browser.

Still fully local — the browser does the rendering and the only network traffic is downloading weights.
`--host 0.0.0.0` lets others on your network generate on your GPU, which the app will remind you of. The CLI
remains the way to batch a folder or script things; the app and the CLI share the same preset, density and
validation code, so they cannot drift apart.

<details>
<summary>Upstream also ships a Gradio demo</summary>

`idol-sdk/demo/app.py` is Sperl Labs' own Gradio page (with an example gallery). It works, but it is a second
UI with its own pinned Gradio version and its own copy of the settings, so this repo's `idol-app` reuses our
presets, validation and viewer instead. Use whichever you prefer.
</details>

## Quality levers

Ordered roughly by impact. `--dry-run` prints what would run without spending GPU time.

| Flag | Default | What it does |
|---|---|---|
| `--preset` | `standard` | Bundle of the four levers below: `draft` 20 steps/65k, `standard` 50/262k, `high` 75/524k, `max` 100/1,048,576. Explicit flags override it. |
| `--density` | from preset | The easy density switch: `32k` `64k` `128k` `256k` `512k` `1m`. Independent of the preset, so `--preset draft --density 1m` means fast sampling, full density. |
| `--num_gaussians` | from preset | Exact count, 32,768–1,048,576, multiple of 32. Takes shorthand too (`256k`, `0.5m`, `262,144`). Mutually exclusive with `--density`. |
| `--steps` | from preset | Euler steps on the flow; cost is linear, detail smears below ~25. |
| `--cfg_strength` | 3.0 | How strictly the object follows the input. Higher hugs the image (and can artifact); `0` is unconditional. |
| `--rescale_t` | 3.0 | Noise-schedule shift: >1 spends more steps on overall shape, <1 on fine detail. |
| `--mode` | `synthetic` | Crop room around the object: `synthetic` 1.2 for renders, `real` 1.7 for photos. |
| `--crop_margin` | from mode | Override it directly: square side = object bbox × this. |
| `--seed` / `--variants N` | 42 / 1 | Same seed reproduces exactly. `--variants 3` writes three consecutive seeds so you can pick. |
| `--dinov3_dtype fp32` | `bf16` | Full-precision image encoder, ~1.7 GB more VRAM. |
| `--no_optimize` | optimized | Disable torch.compile + batched guidance + cuDNN attention + TF32 (~1.7× slower, same output). |

Also: `-o/--out`, `--device cuda:1`, `--weights <repo-or-folder>`, `--skip-existing`, `--no-preview`,
`--no-camera`, `--dry-run`, `-q`.

**A loop that works:** `--preset draft --variants 3` until you like the shape → check `_input.png` if you
don't → re-run the winning seed at `--preset high`. The first generation in a process pays `torch.compile`,
so don't judge speed on image #1.

### Input formats

Anything PIL opens: `.png .jpg .jpeg .webp .bmp .tif .tiff`. What matters is whether the file has real
alpha. A transparent PNG/WebP cutout is trusted as-is and **skips background removal entirely**; JPEGs and
opaque images get matted by BiRefNet. So a clean hand-made cutout is often the best input. With JPEGs, keep
the background distinct from the object and quality ≥ 90 — compression ringing on high-contrast edges is the
usual cause of nibbled silhouettes.

## Viewing

```bash
./idol-view outputs/                    # every .ply in the folder, checkbox per object
./idol-view outputs/donut.ply --port 9000 --open
./idol-view outputs/ --info             # just list counts and extents
./idol-gen render.png --serve --open    # generate, then serve, and block until Ctrl-C
```

Uses [viser](https://viser.studio): a local web server, rendering in your browser, **nothing uploaded**.
Orbit with the drag, zoom with the scroll, toggle objects and splat scale in the side panel. Bind beyond
localhost with `--host 0.0.0.0` if you are on a remote box.

No install at all: drag a `.ply` onto [gsplat.tech](https://gsplat.tech) or the
[SuperSplat editor](https://superspl.at/editor) — both parse the file client-side.

## Batch

```bash
./idol-gen renders/ --out out/ --skip-existing          # resumable
./idol-gen renders/ --preset draft --variants 3         # 3 seeds each
./idol-gen renders/ --mode real --crop_margin 1.4       # override framing for a whole set
```

## How it works

`idol_tools/generate.py` resolves preset + flags into a settings dict, loads
`IdolPipeline` from the upstream SDK once, and loops. Per image: BiRefNet removes the background, the image
is cropped to a square around the object and resized to 1024², DINOv3 tokens and FLUX.2 VAE tokens condition
a DiT that is sampled with flow matching (Euler, with classifier-free guidance and a shifted time grid),
then an octree decoder places `num_gaussians / 32` anchors and a Gaussian decoder turns each into 32
Gaussians, written out as a standard 3DGS `.ply` (rotated z-up → y-up for viewers).

`idol_tools/viewer.py` reads that `.ply` back into centres, covariances (`R·diag(s²)·Rᵀ`), colours
(`0.5 + SH_C0·f_dc`) and opacities, and hands them to viser. The conversions are unit-tested against the
SDK's own `Gaussian` object, so a silent axis or quaternion mistake would fail the build.

`idol_tools/app.py` is the browser front end: a viser server whose side panel holds the upload button and the
levers, and whose scene holds the results. It calls the same `IdolPipeline` as the CLI and displays results by
re-reading the `.ply` it just wrote — so what you see on screen is provably what landed on disk. Generations
are serialised behind a lock (one GPU, one job at a time) and failures are reported in the page rather than
the console.

## Tests

```bash
./idol-test --cli-only     # no GPU, no weights, no Hugging Face access: CLI, batch, viewer maths
./idol-test                # + real weights: BiRefNet, crop, sampler, decoders, PLY validity, determinism
./idol-test --eager        # the un-optimized sampling path
```

The full test loads the **real** Idol weights and only stubs the two gated encoders (random tokens of the
right shape), so it runs offline. It checks Gaussian counts, finiteness, standard 3DGS PLY attributes, seed
determinism vs. seed sensitivity, crop behaviour per mode, the quaternion/covariance maths, the `.ply`
round trip including the z-up → y-up convention, every CLI flag's precedence, and the web app's controls,
lever resolution, upload, generation, download and error handling against a fake viser server. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Idol needs two image encoders ... gated` | Accept the two model pages above and `hf auth login`. `./idol-gen x.png --dry-run` re-checks. |
| `no object found in the image` | BiRefNet's alpha came out empty: no single dominant object, or the background matches the object. Crop tighter, or supply a transparent PNG. |
| Object is cut off or tiny in the result | Framing, not sampling. Look at `<name>_input.png`; adjust `--mode` or `--crop_margin`. |
| Blobby / smeared detail | More `--steps`, more `--density`, or `--dinov3_dtype fp32`. |
| Over-sharpened, weird colours | Lower `--cfg_strength` (try 1.5–2). |
| `CUDA out of memory` | The CLI prints the levers: fewer `--num_gaussians`, `--dinov3_dtype bf16`, `--preset draft`. |
| First image is slow | `torch.compile`. Subsequent images in the process are the real speed. |
| `Failed to import diffusers... peft` | A user-site package is shadowing the venv. Keep `PYTHONNOUSERSITE=1` (the launchers set it) and re-run `scripts/setup.sh --rebuild`. |
| `idol-app` page opens but Generate does nothing | Look at the terminal running it — usually the gated encoders (`scripts/setup.sh --check`). The page itself reports model errors inline. |
| Result looks like it's lying on its back | Up-axis mismatch in *your* viewer, not the file: Idol writes y-up like other 3DGS tools. `./idol-view x.ply --up z` for SDK-native files. |

## Privacy and licensing

Images are processed locally; the only network traffic is downloading weights. The model and SDK are
Apache-2.0 by Sperl Labs; this wrapper is Apache-2.0 too (see [LICENSE](LICENSE) and
[NOTICE](NOTICE)). The DINOv3 and FLUX.2 weights are gated under their own terms, which you accept
yourself — this repository never contains or mirrors them. Back sides of objects are hallucinated: a
plausible guess, not a measurement.

## Cite

```bibtex
@misc{idol,
  title  = {Idol: a single image to 3D Gaussians},
  author = {Sperl Labs},
  year   = {2026},
  url    = {https://speridlabs.com/research/idol}
}
```
