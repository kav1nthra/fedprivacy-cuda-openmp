# Parallel acceleration of privacy-preserving federated learning (serial C vs CUDA)

Course project based on:

> G. A. Kaissis, M. R. Makowski, D. Rueckert, R. F. Braren. **Secure, privacy-preserving and federated machine
> learning in medical imaging.** *Nature Machine Intelligence* 2, 305-311 (2020).
> https://doi.org/10.1038/s42256-020-0186-1

**The idea:** hospitals keep their patient images and only share model updates. Those updates are protected with
differential privacy (DP) and secure aggregation. Both cost extra computation. This project asks: **how much faster
does a GPU (CUDA) make them compared with plain serial C?**

| Kernel | What it does (plain words) | Backends |
|---|---|---|
| `dp_grad` | clips every image's gradient and sums them (core of DP-SGD) | serial C, CUDA naive, CUDA optimised |
| `secagg`  | hides each hospital's update behind random masks that cancel in the sum | serial C, CUDA |

"CUDA with and without" is covered two ways: **CPU vs GPU**, and **GPU with vs without transfer cost** (kernel-only time
vs time including host<->device copies). The notebook also trains a small federated model end to end (5 hospitals, DP
noise, secure aggregation).

## Quick start

**Google Colab (recommended, gives you a GPU)**
1. Upload `FedPrivacy_CUDA.ipynb` and this zip to Colab (or just upload the zip when the notebook asks).
2. *Runtime > Change runtime type > T4 GPU*.
3. Run the first code cell (dependencies), fill in the team cell, then *Runtime > Run all*. About 2-5 minutes.
4. `report/report.pdf` downloads automatically.

**Local (Linux / WSL / macOS with Homebrew gcc)**
```bash
pip install -r requirements.txt
jupyter notebook FedPrivacy_CUDA.ipynb      # Run all
```
Requirements: `gcc` (needed). `nvcc` + an NVIDIA GPU (optional: without them the CUDA parts are skipped and the report says so).

## Layout
```
FedPrivacy_CUDA.ipynb          the only thing you need to run (Run all)
src/fl_cpu.c                   serial C kernels
src/fl_cuda.cu                 CUDA kernels (naive + optimised) and timing hooks
src/fedkernels.py              build + ctypes wrappers + numpy reference + DP epsilon (zCDP)
scripts/make_dataset.py        regenerates the synthetic dataset (deterministic, seed 42)
data/                          synthetic_chest_xray_32.npz, hospital_summary.csv, DATASET_CARD.md
results/                       CSVs written by the notebook (after running)
report/                        report.pdf and figs/ (after running)
Makefile                       optional: build the libraries without the notebook
```

## Filling in the team details
Edit the team cell of the notebook: project title, course, institution, and for each member name, ID, role,
contribution and reflection. Suggested roles are pre-filled (GPU lead, privacy and data lead, results lead); change or
delete entries to match your team. Anything left empty prints as `[to be filled]` in the PDF. `OBSERVATIONS` is your
own paragraph for the findings section.

## Dataset
Synthetic 32x32 chest-X-ray-like images (10,000) across 5 simulated hospitals with different size, pneumonia rate,
contrast and noise (non-IID). It is synthetic on purpose, so **accuracy numbers are not clinical results**. See
`data/DATASET_CARD.md`.

## What the benchmarks measure
* **Time** per call: median of several runs after a warm-up.
* **Speed-up** vs the serial C baseline (compiled `-O3 -march=native -ffast-math`, so the compiler vectorises it: a strong baseline).
* **Throughput**: GFLOP/s (about 4*N*D flops) and effective GB/s (2 passes over the N x D matrix).
* **CUDA with vs without transfer**: kernel-only time, per step including the small w/g copies, one-time data upload,
  and the number of calls after which the upload is paid back relative to serial C.
* **Secure aggregation** time vs model size (up to 10 clients x 10M parameters).
* **Correctness** of every backend against a float64 NumPy reference.

## Honest limitations
* Synthetic data; logistic regression, not a CNN.
* The mask generator in `secagg` is a fast hash for benchmarking and is **not cryptographically secure**; the protocol
  is simplified (no dropouts, no key agreement).
* The epsilon values are a conservative zCDP bound for full-batch DP-GD; use a vetted accountant for real claims.
* Numbers depend on the machine; always report the hardware table that the PDF prints.
* The serial C path was tested end to end. The CUDA code could not be executed in the environment used to prepare this
  version (no GPU), so the first run on your GPU is its real test; the notebook compares every backend with a reference
  and warns if anything deviates.

## Git
```bash
git init
git add .
git commit -m "Federated DP/secure-aggregation kernels: serial C vs CUDA"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
Commit `report/report.pdf` after running on the GPU machine. `src/build/` (compiled libraries) is git-ignored.
