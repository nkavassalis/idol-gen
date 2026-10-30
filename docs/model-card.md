---
license: apache-2.0
library_name: idol
pipeline_tag: image-to-3d
tags:
- gaussian-splatting
- 3d
- image-to-3d
---

<h1 align="center">Idol</h1>

<p align="center"><b>A single image to a 3D Gaussian splat.</b><br>One picture of an object in, up to a million 3D Gaussians out, including the sides the picture never showed.</p>

<p align="center">
  <a href="https://speridlabs.com/research/idol"><img alt="Project Page" src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-1f6feb?style=for-the-badge"></a>
  <a href="https://github.com/speridlabs/idol-sdk"><img alt="Code" src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github"></a>
  <a href="https://huggingface.co/spaces/speridlabs/idol"><img alt="Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-ffcc4d?style=for-the-badge"></a>
</p>

Idol turns one picture of an object into a 3D Gaussian splat. Give it a render, a generated image or a photo: it removes the background, frames the object and generates up to 1,048,576
Gaussians, written as a standard 3DGS `.ply`. It runs on consumer GPUs, using 7.2 GB of VRAM and can run in under 2 seconds.

<div align="center">
  <video src="https://huggingface.co/speridlabs/idol/resolve/main/assets/reel.mp4" width="720" autoplay loop muted playsinline controls></video>
</div>

## Examples

Idol turns one picture of an object into a 3D Gaussian splat. Give it a render, a generated image or a photo: it removes the background, frames the object and generates up to 1,048,576 Gaussians, including the sides the picture never showed. It runs on consumer GPUs, using 7.2 GB of VRAM and can run in under 2 seconds.

<table>
<tr>
<td width="50%"><img src="assets/gallery/armchair.webp"></td>
<td width="50%"><img src="assets/gallery/avocado.webp"></td>
</tr>
<tr>
<td width="50%"><img src="assets/gallery/lamp.webp"></td>
<td width="50%"><img src="assets/gallery/axolotl.webp"></td>
</tr>
<tr>
<td width="50%"><img src="assets/gallery/grandma.webp"></td>
<td width="50%"><img src="assets/gallery/lollipop.webp"></td>
</tr>
<tr>
<td width="50%"><img src="assets/gallery/grapes.webp"></td>
<td width="50%"><img src="assets/gallery/castle.webp"></td>
</tr>
</table>

## Get started

**1. Get access to the image encoders.** Idol reads its input through two gated models on Hugging Face. Accept their
terms with your Hugging Face account, then log in once with `hf auth login` (or set `HF_TOKEN`):

- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (only its VAE is downloaded)

**2. Install the code**

```bash
git clone https://github.com/speridlabs/idol-sdk.git
cd idol-sdk
pip install -e .          # Python 3.10+, PyTorch 2.5+, an NVIDIA GPU with CUDA
```

There is no separate download step: on the first run, Idol's weights (about 3 GB, from this repository) and the two
image encoders download automatically into the Hugging Face cache. If the encoders are not accessible yet, Idol stops
before downloading anything and says what is missing.

**3. Generate a 3D object**

```bash
python -m idol.sample --image photo.jpg --mode real
python -m idol.sample --image renders/ --mode synthetic --out outputs
```

For each image this writes `<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the image the
model saw, after background removal and cropping) and `<name>_camera.json` (the predicted camera of the input).

Pick the mode for your input. It sets how much room the crop leaves around the object:

| `--mode` | for | crop |
|---|---|---|
| `synthetic` (default) | renders and generated images of a single object | object fills 1/1.2 of the frame |
| `real` | photographs | object fills 1/1.7 of the frame |

From Python:

```python
from idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
output = pipeline("photo.jpg", mode="real", seed=42, num_gaussians=1_048_576)
output.gaussian.save_ply("photo.ply")
```

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--seed` | 42 | Fix it to get the same object again. |
| `--steps` | 50 | Sampling steps. Fewer are faster but lose some detail. |
| `--cfg_strength` | 3.0 | Classifier-free guidance: how strictly the object follows the image. |
| `--rescale_t` | 3.0 | Time shift: higher spends more steps on the overall shape, lower on fine detail. |
| `--num_gaussians` | 262144 | 32,768 to 1,048,576, a multiple of 32. |
| `--dinov3_dtype` | bf16 | `fp32` runs the DINOv3 encoder in full precision (1.7 GB more memory). |
| `--weights` | `speridlabs/idol` | A Hugging Face repo or a local folder. |
| `--no_optimize` | optimized | Turn off the faster sampling. |

Faster sampling is on by default on GPU (~1.7x, same output quality): the denoiser is compiled with `torch.compile`,
the guided and unguided passes run as one batch, and attention uses cuDNN with TF32 matmuls while sampling. The first
image in a process is slower while it compiles.

## More details about the model

| Component | Parameters | Precision on GPU | Weights in VRAM |
|---|---|---|---|
| DiT (denoiser) | 370.6M | FP16 transformer blocks; FP32 embeddings and norms | ~0.8–1.0 GB |
| Octree decoder | 50.8M | FP16 transformer blocks; FP32 embeddings and norms | ~0.1–0.2 GB |
| Gaussian decoder | 237.3M | FP16 transformer blocks; FP32 embeddings and norms | ~0.5–0.9 GB |
| BiRefNet (background removal) | 220.7M | FP16 | ~0.44 GB |
| DINOv3 ViT-H+/16  | 840.6M | bf16 | ~1.68 GB |
| FLUX.2 VAE  | 84.1M  | bf16 (FP32 checkpoint, cast on load) | ~0.17 GB |
| **Full request** | **~1.80B** | mostly FP16 / bf16, some FP32 | **~3.7–4.4 GB of weights, 7.2 GB peak** |

## Benchmark

Idol, TripoSplat, TRELLIS and LGM, scored at the input view (PSNR, SSIM, LPIPS, IoU) and at seven new views
(DINOv2, CLIP, sharpness distance). Mean scores, 1 × H100 per model. Bold is the best in each column.

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | DINOv2 ↑ | CLIP ↑ | Sharpness distance ↓ | Request time ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Idol** | 50 steps, 262,144 Gaussians | 12.25 | **0.726** | **0.213** | **0.862** | **0.781** | **0.885** | **0.81** | 6.8 s |
| TripoSplat | 50 steps, 262,144 Gaussians | 11.76 | 0.712 | 0.225 | 0.860 | 0.743 | 0.868 | 1.03 | 9.9 s |
| TRELLIS | Paper setting, 50 + 50 steps | 11.89 | 0.716 | 0.235 | 0.845 | 0.754 | 0.868 | 0.97 | 7.1 s |
| LGM | Default, 65,536 Gaussians | **12.46**\* | 0.722 | 0.237 | 0.834 | 0.575 | 0.784 | 1.22 | **1.1 s** |

\* LGM has the highest PSNR despite visibly blurry results, because PSNR averages squared pixel error.

## Across GPUs

One image to 262,144 Gaussians in 3.4 s. Request time at 50 steps: median over 8 images, with the same code and
settings on every GPU, one GPU per machine.

| GPU | VRAM | Request time |
|---|---|---|
| B300 | 269 GB | 3.4 s |
| B200 | 179 GB | 3.6 s |
| H200 | 140 GB | 6.6 s |
| H100 | 80 GB | 6.7 s |
| RTX 5090 | 32 GB | 14.1 s |
| A100 | 80 GB | 16.3 s |
| L40S | 45 GB | 16.6 s |
| RTX 4090 | 24 GB | 19.1 s |
| RTX 3080 | 10 GB | 46.2 s |
| L4 | 22 GB | 54.6 s |

**Made with ❤️ by [speridlabs.com](https://speridlabs.com)**
