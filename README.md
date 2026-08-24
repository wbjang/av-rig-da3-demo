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
python3 -m venv .venv && source .venv/bin/activate   # required on modern Debian/Ubuntu (PEP 668)
pip install -r requirements.txt
pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
# torch: install per https://pytorch.org for your platform
python -m ipykernel install --user --name av-rig-demo   # register the kernel for Jupyter
```

Alternatively, instead of pip-installing DA3, clone its repo and point the notebook at
it: `export DA3_SRC=/path/to/Depth-Anything-3/src`.

The dataset is **gated**: request access on the
[dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
then `huggingface-cli login`. First access streams from HF and caches locally
(a clip's worth of data is a few hundred MB). A GPU with ~4 GB runs DA3-Large at the
resolution used here.

## Expected results (clip `07721315`)

DA3's own camera-translation scale leaves depth 2–5× short; after the classical
per-camera anchoring the depth is essentially unbiased vs LiDAR (median ratio ≈ 0.9–1.0
per camera) with AbsRel ≈ 0.15–0.32 depending on camera — details and the honest read in
the write-up.

## License note

The notebook ships with outputs cleared: the dataset license does not permit
redistributing imagery, so run it yourself to see frames and results. Please respect the
same constraint with anything you generate from the data.
