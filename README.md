# The First Lunar Pathloss Radio Map Prediction Challenge

Baseline code and reference tooling for the **ICASSP 2027 Grand Challenge** on
lunar radio map prediction.

Given a 256×256 lunar heightmap and a one-hot transmitter position, predict the
256×256 pathloss map in dB, at **415 MHz** and **5.8 GHz**. Ground truth comes
from Sionna RT ray-tracing over synthetic lunar topography at 1 m/px.

- **Data and leaderboard:** [Kaggle competition](https://www.kaggle.com/competitions/the-first-lunar-pathloss-radio-map-prediction-challenge)
- **Challenge site:** https://lunarradiomapchallenge.github.io

---

## The metric, in one paragraph

**RMSE in dB over valid (ray-traced) pixels only**, computed separately per band
and weighted 50/50.

The ray tracer could not resolve every cell, and the unresolved cells were
gap-filled — at 5.8 GHz with a static placeholder of exactly 145.0 dB, covering
about 11.4% of pixels. Those values are placeholders, not physics, and they are
excluded from scoring. Every training and validation map ships with a **validity
mask**: 1 = ray-traced, 0 = filled.

Train with a masked loss. Training on every pixel measurably improves the
all-pixel diagnostic while making the ranked metric *worse* — the voids are
geometrically structured, so a model can score by learning where the tracer
failed instead of learning propagation.

You submit full 256×256 maps. The organizers apply the mask at scoring time, so
you never need the test masks.

---

## Quick start

**On Kaggle** — open `notebooks/lunar_starter.ipynb`, attach the competition
data, run it top to bottom. It loads the arrays, trains a small baseline, and
writes a correctly formatted `submission.csv`.

**Locally:**

```bash
python -m venv .venv && .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python train.py    --data-root <data> --band 415 --loss masked --phase firstU
python evaluate.py --ckpt runs/<run>_best.pt --data-root <data> --split val
```

`<data>` is wherever you unpacked the Kaggle download.

---

## What's here

| file | what it is |
|---|---|
| `lunar_dataset.py` | reference PyTorch dataloader — also ships with the data |
| `train.py` | trains the RadioWNet baseline, both stages |
| `evaluate.py` | scores a checkpoint on `val`, in dB, masked and unmasked |
| `score_submission.py` | validates a submission directory; `--validate-only` needs no ground truth |
| `metrics.py` | RMSE / NMSE / PSNR / SSIM, with masked variants |
| `radiounet.py` | seam onto the vendored upstream model |
| `RadioUNet/` | upstream RadioWNet model code, MIT (see `RadioUNet/README.md`) |
| `notebooks/lunar_starter.ipynb` | end-to-end starter, Kaggle-ready |

---

## The data

Download from Kaggle and unpack at one root. It is **consolidated arrays**, one
file per split per field — not one file per sample:

```
train_index.csv   train_hm.npy   train_pl415.npy   train_mask415.npy
val_index.csv     val_hm.npy     val_pl415.npy     val_mask415.npy
                                 ..._pl58.npy      ..._mask58.npy
metadata.json     README.md      lunar_dataset.py
```

**Row *i* of `train_pl415.npy` is row *i* of `train_index.csv`.** That ordering
is the contract. Three things are compressed out:

```python
# heightmaps are deduplicated -- 51 transmitters on a terrain share one
hm = np.load("train_hm.npy", mmap_mode="r")[int(row["hm_row"])]

# TX maps are not stored; a one-hot 256x256 is two integers
tx = np.zeros((256, 256), np.float32)
tx[int(row["tx_row"]), int(row["tx_col"])] = 1.0

# masks are bit-packed
mask = np.unpackbits(np.load("train_mask415.npy", mmap_mode="r")[i]).reshape(256, 256)
```

`lunar_dataset.py` does all of this for you. Always use `mmap_mode="r"` —
`train_pl415.npy` is 3.4 GB.

| split | terrains | samples/band | targets | masks |
|---|---|---|---|---|
| train | 6,120 | 25,720 | ✅ | ✅ |
| val | 680 | 2,330 | ✅ | ✅ |
| test | 1,200 | 4,950 | withheld | withheld |

Splitting is at **terrain** level: a terrain with 51 transmitters contributes 51
samples sharing one heightmap, so a per-sample split would leak.

---

## Baseline

RadioWNet (Levie et al.), trained with a masked loss. Reference numbers on the
withheld test set, over valid pixels:

| model | 415 MHz | 5.8 GHz | combined (50/50) |
|---|---|---|---|
| two band-specific models | 4.66 dB | 6.28 dB | **5.47 dB** |
| one conditional model | 4.76 dB | 6.42 dB | **5.59 dB** |

Two stages: `--phase firstU` trains the coarse U-Net, `--phase secondU` the
refinement on top of it (`--init-from auto` picks up the matching firstU run).

---

## Submitting

**Now — the Kaggle leaderboard.** One row per pixel:

```
ID,PL
pl415_00047_00_3,88.19
```

`ID` is `pl<band>_<sample_id>_<flat_pixel_index>`, row-major over 256×256
(`arr.reshape(-1)` order). Start from `sample_submission.csv` and overwrite `PL`.

> This leaderboard is a **format checker and plays no part in the final
> ranking.** It scores a subset of the *validation* split, whose ground truth
> ships with the data — anyone can submit the true values and score 0.00. Use it
> to prove your pipeline works.

**December — the real submission.** Not a CSV. One `.npy` per test sample per
band (`pl415_<sample_id>.npy`, `pl58_<sample_id>.npy`), each 256×256 in dB, plus
your trained model and inference code. Inference must stay under **500 ms** per
map. Check your directory before sending:

```bash
python score_submission.py --submission <dir> --data-root <data> --band both --validate-only
```

---

## Licences

- **This code:** MIT — see `LICENSE`.
- **`RadioUNet/`:** MIT, Copyright (c) 2019 Ron Levie. Unmodified upstream model
  code; see `RadioUNet/README.md` for the source and citation.
- **The dataset:** CC BY 4.0, distributed via Kaggle.

If you modify or redistribute that model code, keep its MIT notice intact.

## Citing

```
R. Levie, C. Yapar, G. Kutyniok, G. Caire, "RadioUNet: Fast Radio Map Estimation
with Convolutional Neural Networks", IEEE Transactions on Wireless
Communications, vol. 20, no. 6, pp. 4001-4015, 2021.
```
