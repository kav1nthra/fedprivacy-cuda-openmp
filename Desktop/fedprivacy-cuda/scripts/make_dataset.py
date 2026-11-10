"""Generate the synthetic multi-hospital chest-X-ray-like dataset.

Why synthetic?  Real chest X-ray data (e.g. the paediatric pneumonia sets used in
follow-up work to the Kaissis et al. paper) cannot be redistributed in a zip and
is large.  This generator makes small 32x32 grey-scale "radiographs" with
lung fields, spine/rib structure and optional pneumonia-like opacities, split
across 5 simulated hospitals with DIFFERENT class prevalence, brightness/contrast
and noise (non-IID, as in real federated medical imaging).

Run:  python scripts/make_dataset.py            (writes data/*)
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_image(rng, size, pneumonia, gain, offset, noise):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    img = np.full((size, size), 0.55)
    # lung fields (darker ellipses) with small random shifts / sizes
    for cx in (-0.40, 0.40):
        cx += rng.normal(0, 0.03)
        cy = rng.normal(0, 0.03)
        rx, ry = 0.27 + rng.normal(0, 0.02), 0.62 + rng.normal(0, 0.03)
        lung = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        img[lung] -= 0.25
    # spine / mediastinum (bright central band), ribs (horizontal ripples), abdomen
    img += 0.18 * np.exp(-(xx / 0.10) ** 2)
    img += 0.04 * np.sin(14 * yy + rng.uniform(0, 6.28)) * (np.abs(xx) > 0.15)
    img += 0.15 * (yy > 0.75)
    if pneumonia:  # 1-3 soft "consolidation" blobs inside the lung fields
        for _ in range(rng.integers(1, 4)):
            side = rng.choice([-1, 1])
            cx = side * 0.40 + rng.normal(0, 0.08)
            cy = rng.uniform(-0.45, 0.45)
            s = rng.uniform(0.09, 0.20)
            amp = rng.uniform(0.10, 0.28)
            img += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
    img = img * gain + offset + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    # hospital: (n_images, pneumonia prevalence, gain, offset, noise)  -> non-IID
    hospitals = {
        0: (2600, 0.50, 1.00, 0.00, 0.05),
        1: (2200, 0.35, 0.90, 0.05, 0.06),
        2: (1800, 0.65, 1.10, -0.04, 0.05),
        3: (2000, 0.45, 0.95, 0.02, 0.08),
        4: (1400, 0.55, 1.05, -0.02, 0.07),
    }
    X, y, h = [], [], []
    for hid, (n, prev, gain, off, noise) in hospitals.items():
        for _ in range(n):
            lab = int(rng.random() < prev)
            img = make_image(rng, a.size, lab, gain * rng.normal(1, 0.04), off, noise)
            X.append((img * 255).astype(np.uint8))
            y.append(lab)
            h.append(hid)
    X = np.stack(X)
    y = np.array(y, np.uint8)
    h = np.array(h, np.int8)
    split = (rng.random(len(y)) < 0.2).astype(np.uint8)  # 1 = global held-out test set
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f"synthetic_chest_xray_{a.size}.npz"),
                        X=X, y=y, hospital=h, split=split)
    df = pd.DataFrame({"hospital": h, "y": y, "split": split})
    summ = (df.groupby("hospital")
              .agg(images=("y", "size"), pneumonia_rate=("y", "mean"), test_images=("split", "sum"))
              .round(3).reset_index())
    summ.to_csv(os.path.join(a.out, "hospital_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print("X", X.shape, "saved to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
