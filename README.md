# idol — one image → 3D Gaussian splat

Local setup + a `idol-gen` CLI around [speridlabs/idol](https://huggingface.co/speridlabs/idol), an
image-to-3D model that turns a single picture of an object into a standard 3D Gaussian Splatting `.ply`
(background removed, object framed, hidden sides hallucinated). Up to 1,048,576 Gaussians, ~7.2 GB VRAM.

Upstream code lives in `idol-sdk/` (a clone of <https://github.com/speridlabs/idol-sdk>, Apache-2.0) and the
model card is archived at `docs/model-card.md`. Everything in `scripts/` and the two launchers here is ours.

## Requirements

- NVIDIA GPU with CUDA (verified on an RTX 5070 Ti 16 GB, torch 2.14.1+cu130, sm_120). ~4.5 GB of weights in
  VRAM, 7.2 GB peak at default settings.
- Python 3.10+.
- A Hugging Face account with the two **gated** image encoders accepted (see below).

## Setup

```bash
scripts/setup.sh          # clones idol-sdk, builds .venv, installs deps, prefetches 3.4 GB of weights
```

Then get access to the two encoders Idol reads its input through — accept the terms on each page with your
HF account, then log in once:

- <https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m> (DINOv3 ViT-H+/16)
- <https://huggingface.co/black-forest-labs/FLUX.2-dev> (only its VAE is downloaded)

```bash
hf auth login            # or: export HF_TOKEN=hf_...
./idol-gen photo.jpg --dry-run     # verifies settings + encoder access without generating
```

The Idol weights themselves are ungated and prefetched by `setup.sh`; the encoders download on first real run.
If access is missing, Idol stops **before** downloading anything and prints which pages to accept.

## Use

```bash
./idol-gen photo.jpg --mode real                      # photograph
./idol-gen render.png                                 # render / AI image (default --mode synthetic)
./idol-gen render.png --preset draft                  # 20 steps, 65k Gaussians — iterate on framing/seed
./idol-gen render.png --preset max                    # 100 steps, 1,048,576 Gaussians
./idol-gen render.png --steps 80 --gaussians 524288 --cfg 2.0
./idol-gen renders/ --out out/ --skip-existing        # whole folder, resumable
./idol-gen render.png --variants 3 --seed 100         # seeds 100,101,102 to pick the best
```

Each image writes `<name>.ply` (open in any splat viewer — [gsplat viewer](https://gsplat.tech),
SuperSplat), plus `<name>_input.png` (the 1024² image the model actually saw) and `<name>_camera.json`
(predicted yaw/pitch/radius/fov of the input).

## Quality levers

| Flag | Default | What it does |
|---|---|---|
| `--preset` | `standard` | Bundle of the four levers below: `draft` 20/65k, `standard` 50/262k, `high` 75/524k, `max` 100/1,048,576. Explicit flags override it. |
| `--num_gaussians` | preset | Splat density. 32,768 → 1,048,576, must be a multiple of 32 (rejected early with the nearest valid value). Biggest knob for detail-vs-size. |
| `--steps` | preset | Euler steps on the flow. Cost is linear; below ~25 detail starts to smear. |
| `--cfg_strength` | 3.0 | Classifier-free guidance. Higher hugs the input image harder (and can over-sharpen/artifact); `0` is unconditional. |
| `--rescale_t` | 3.0 | Time shift. >1 spends more steps on overall shape, <1 on fine detail. |
| `--mode` | `synthetic` | Crop room around the object: `synthetic` 1.2 (renders, one object filling the frame), `real` 1.7 (photos). |
| `--crop_margin` | from mode | Override the crop directly: square side = object bbox × this. |
| `--seed` / `--variants N` | 42 / 1 | Same seed = same object. `--variants` renders consecutive seeds into separate files. |
| `--dinov3_dtype` | `bf16` | `fp32` runs the DINOv3 encoder in full precision, ~1.7 GB more VRAM. |
| `--no_optimize` | optimized | Disable torch.compile + batched guidance + cuDNN attention + TF32 (~1.7× slower, same output). |

Other flags: `-o/--out`, `--device` (e.g. `cuda:1`), `--weights` (HF repo or local folder),
`--skip-existing`, `--no-preview`, `--no-camera`, `--dry-run`, `-q/--quiet`.

**Suggested loop:** `--preset draft` to lock in mode/crop/seed, then re-run the winner at `--preset high`
(or `max`). The first generation in a process pays the `torch.compile` cost.

## Tests

```bash
./idol-test            # optimized path (what idol-gen uses by default)
./idol-test --eager    # eager sampling path
```

The smoke test loads the **real** Idol weights and runs BiRefNet background removal, cropping, the DiT
sampler, both decoders and the PLY writer, checking Gaussian counts, finiteness, standard 3DGS PLY
attributes, seed determinism, crop behaviour, and the CLI's preset/validation logic. The two gated
encoders are replaced by a stub that returns random tokens of the right shape — so the test runs offline
and without HF access, and cannot tell you whether the *encoders'* output is good.

## Layout

```
idol-gen            launcher: .venv python + isolation, -> scripts/generate.py
idol-test           launcher -> scripts/smoke_test.py
scripts/generate.py the CLI (presets, validation, batch, previews, camera json)
scripts/setup.sh    reproducible environment
scripts/smoke_test.py
idol-sdk/           upstream SDK (clone, unmodified)
docs/model-card.md  archived HF model card (benchmarks, per-GPU timings)
```

## Notes and boundaries

- **Isolation.** `.venv` is created *without* system site-packages and the launchers set `PYTHONNOUSERSITE=1`.
  This matters: `~/.local/lib/python3.10/site-packages` on this machine has an old `peft` that shadows into
  venvs with system site-packages and breaks `diffusers`' `AutoencoderKLFlux2` import.
- **Nothing is uploaded.** Images are processed locally; the only network use is downloading weights.
- **Gated upstream weights stay gated.** This project does not mirror, cache, or redistribute DINOv3 or the
  FLUX.2 VAE; access is granted per-account by Meta and Black Forest Labs on their own terms.
- **Single object, object-shaped inputs.** Idol is not a scene or human reconstructor; a busy background or
  several touching objects confuse BiRefNet's crop and the model will invent geometry. If the alpha comes out
  empty you get `no object found in the image`.
- **Back sides are hallucinated.** Anything the picture never showed is a plausible guess, not a measurement.
- **VRAM.** 16 GB is comfortable at `max` in bf16; if you OOM the CLI prints the exact levers to pull.
- `torch.compile` (the default `optimize` path) is what makes the 5070 Ti fast enough to be pleasant; the
  first call in a process compiles, so don't judge speed on image #1.
