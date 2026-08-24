# DA3 (+Camera) on the NVIDIA PhysicalAI-AV rig — minimal demo

One self-contained notebook ([`demo.ipynb`](demo.ipynb)) showing, end to end, how to:

1. **Download frames, video, and LiDAR** from the gated
   [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
   dataset (7 synchronized cameras, 360°, metric LiDAR).
2. **Load the camera parameters properly** — the dataset's lenses are f-theta
   (no pinhole focal exists); DA3's conditioning port only accepts a pinhole FOV, so the
   K must be built with the lens's *true* FOV (the naive axis fit announces the 121.5°
   lenses as 92°), while all real geometry uses the exact f-theta rays.
3. Run **Depth Anything 3** conditioned on the calibrated rig — the camera port is what
   makes cross-camera assembly work at all.
4. Recover **metric scale classically**: SIFT + exact f-theta rays + midpoint
   triangulation over the rig's calibrated baselines (1.3–2.5 m). No LiDAR in the loop.
5. **Judge on LiDAR**: per-camera AbsRel / δ<1.25 / bias on valid projected points,
   plus a fused BEV against the LiDAR spin.

Write-up with the full 14-scene study, ablations, and honest limitations:
**[wbjang.github.io](https://wbjang.github.io/blog/posts/av-rig-recon/)**.

## Setup

```bash
git clone https://github.com/wbjang/av-rig-da3-demo.git && cd av-rig-da3-demo
python3 -m venv .venv && source .venv/bin/activate     # required on modern Debian/Ubuntu (PEP 668)
pip install -r requirements.txt
pip install torch torchvision                          # or the CUDA build for your platform, see pytorch.org
pip install --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
python -m ipykernel install --user --name av-rig-demo  # register the kernel for Jupyter
```

**Why `--no-deps` for DA3:** its declared dependency list pins `numpy<2` and pulls
`open3d` (no Linux-aarch64 wheels exist), `xformers`, and a raft of server/export
extras -- none of which inference needs. `requirements.txt` already contains DA3's
actual runtime deps, verified by running the notebook end to end. Afterwards, every
`pip install` in the venv will print resolver warnings like "depth-anything-3 requires
e3nn / fastapi / open3d / opencv-python / pillow-heif / pre-commit / pycolmap / typer /
uvicorn / xformers, which is not installed" and "requires numpy<2". **All benign**:
those are unused extras (and `opencv-python-headless` provides the same `cv2` module
pip claims is missing).

Alternatively, instead of pip-installing DA3, clone its repo and point the notebook at
it: `export DA3_SRC=/path/to/Depth-Anything-3/src`.

Two harmless things you may see: a one-time `gsplat` warning when DA3 is imported
(3DGS rendering support, unused here), and -- on aarch64 -- `pycolmap` cannot be
installed at all (no wheels exist); the notebook stubs it before importing DA3, since
it is only needed for COLMAP export. If you import `depth_anything_3.api` outside the
notebook, stub it the same way:
`python -c "import sys, types; sys.modules.setdefault('pycolmap', types.ModuleType('pycolmap')); import depth_anything_3.api"`.

## Data access

The dataset is **gated**: request access on the
[dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
then:

```bash
huggingface-cli login
```

That is all the notebook needs: it **streams** exactly what it uses (byte-range reads)
and caches locally, a few hundred MB for the demo clip. There is also a bulk
pre-download API (`PhysicalAIAVDatasetInterface().download_clip_features(clip_id,
[features...])`), but be aware it fetches entire multi-clip chunk files -- about
**38 GB** for the demo clip's chunks -- so prefer the streaming default unless you want
the data offline.

The DA3-Large weights (~1.4 GB) download automatically from HF on first
`from_pretrained`. A GPU with ~4 GB runs DA3-Large at the resolution used here.

## Expected results (clip `07721315`)

DA3's own camera-translation scale leaves depth 2–5× short; after the classical
per-camera anchoring the depth is essentially unbiased vs LiDAR (median ratio ≈ 0.9–1.0
per camera) with AbsRel ≈ 0.15–0.32 depending on camera — details and the honest read in
the write-up.

## License note

The notebook ships with outputs cleared: the dataset license does not permit
redistributing imagery, so run it yourself to see frames and results. Please respect the
same constraint with anything you generate from the data.
