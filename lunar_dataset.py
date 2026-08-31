"""
Reference dataloader for the First Lunar Pathloss Radio Map Prediction Challenge.

Deliberately minimal: heightmap + TX one-hot in, pathloss out.

    from lunar_dataset import LunarRadioMapDataset
    from torch.utils.data import DataLoader

    train = LunarRadioMapDataset("/kaggle/input/the-first-lunar-pathloss-radio-map-prediction-challenge",
                                 split="train", band="415", augment=True)
    val   = LunarRadioMapDataset(root, split="val", band="415")

    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
    x, y = next(iter(loader))     # x: (B, 2, 256, 256)   y: (B, 1, 256, 256)

``band="both"`` serves both frequencies from one dataset and appends a third
input channel, 0.0 for 415 MHz, 1.0 for 5.8 GHz, for a single conditional
model. ``return_mask=True`` adds the validity mask, which is what you want for a
masked loss. ``augment=True`` applies random flips and 90-degree rotations
consistently across every map in a sample; use it for training only.

Storage
-------
The released data is stored as **consolidated arrays**, one per split per field,
rather than one file per sample. Row *i* of ``pl415.npy`` is row *i* of
``index.csv``. Two consequences the loader hides from you:

* Heightmaps are deduplicated. A terrain carrying 51 transmitters has 51
  samples sharing one heightmap, so ``hm.npy`` has one row per *terrain* and
  ``index.csv`` carries an ``hm_row`` column pointing into it.
* **TX maps are not stored.** A one-hot 256x256 array is fully described by two
  integers, so ``index.csv`` carries ``tx_row``/``tx_col`` and the one-hot is
  rebuilt on load, this removes ~2.2 GB of pure redundancy.
* **Masks are bit-packed** with ``np.packbits`` - exact, 8x smaller.

Layouts
-------
Three on-disk layouts are recognised, so the same loader works against the
Kaggle bundle and against a local build:

    flat-arrays    root/train_index.csv, root/train_pl415.npy      (Kaggle)
    nested-arrays  root/train/index.csv, root/train/pl415.npy      (local repack)
    per-file       root/train/index.csv -> pl415/pl_00000_00.npy   (master tree)

Detection is automatic. ``dataset.layout`` reports which was found.

The test split ships inputs only; its targets and validity masks are withheld.
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

BANDS = ("415", "58")
SPLITS = ("train", "val", "test")
GRID = (256, 256)


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


class LunarRadioMapDataset(Dataset):
    """Lunar pathloss radio maps: (heightmap, TX one-hot) -> pathloss in dB.

    Parameters
    ----------
    root : str
        Directory holding the split's arrays and ``metadata.json``.
    split : {"train", "val", "test"}
    band : {"415", "58", "both"}
    normalize : bool
        Scale inputs and targets to [0, 1] using the ranges in ``metadata.json``,
        which are derived from the training split only. Use
        :meth:`denormalize_pathloss` to get dB back.
    augment : bool
        Random flips and 90-degree rotations, applied identically to every map in
        a sample. Training only, leaving it on for val moves the numbers.
    return_name : bool
        Append the sample name to each item.
    return_mask : bool
        Append the validity mask: 1 where the ray tracer produced a value, 0
        where it was gap-filled. Unavailable for the test split.

    Items are ``(input, target[, mask][, name])``, mask before name, and
    ``target`` is absent for the test split.
    """

    def __init__(self, root, split="train", band="415", normalize=True,
                 augment=False, return_name=False, return_mask=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if band not in BANDS + ("both",):
            raise ValueError(f"band must be '415', '58' or 'both', got {band!r}")

        self.root = root
        self.split = split
        self.band = band
        self.normalize = normalize
        self.augment = augment
        self.return_name = return_name
        self.return_mask = return_mask

        meta_path = _first_existing(os.path.join(root, "metadata.json"))
        if meta_path is None:
            raise FileNotFoundError(
                f"metadata.json not found in {root}. Point root at the directory "
                f"the archive unpacked into.")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.hm_min, self.hm_max = self.meta["heightmap_range_m"]
        self.pl_min, self.pl_max = self.meta["pathloss_range_db"]

        index_path, self.layout, self._dir = self._resolve(root, split)
        with open(index_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"{index_path} is empty")
        self.rows = rows

        bands = BANDS if band == "both" else (band,)
        if split == "test":
            self.has_targets = False
        elif self.layout == "per-file":
            self.has_targets = all(f"pl{b}" in rows[0] for b in bands)
        else:
            self.has_targets = all(self._target_path(b) is not None for b in bands)

        if self.return_mask and split == "test":
            raise ValueError(
                "test masks are withheld -- they mark the gap-filled regions, and "
                "the organizers apply them at scoring time. Use return_mask=False "
                "for the test split.")

        self.bands = bands
        self.samples = [(i, b) for b in bands for i in range(len(rows))]

        self._cache = {}

    # -- layout resolution -------------------------------------------------

    def _resolve(self, root, split):
        """Find the index and work out which of the three layouts this is."""
        flat = os.path.join(root, f"{split}_index.csv")
        if os.path.exists(flat):
            return flat, "flat-arrays", root

        nested = os.path.join(root, split, "index.csv")
        if os.path.exists(nested):
            d = os.path.join(root, split)
            layout = "nested-arrays" if os.path.exists(os.path.join(d, "hm.npy")) \
                else "per-file"
            return nested, layout, d

        raise FileNotFoundError(
            f"no index for split {split!r} under {root}. Looked for "
            f"{split}_index.csv and {split}/index.csv. Unpack the archive at "
            f"this root, or point root at the unpacked directory.")

    def _array_path(self, field):
        """Path to a consolidated array, or None if this layout has none."""
        if self.layout == "flat-arrays":
            return _first_existing(os.path.join(self.root, f"{self.split}_{field}.npy"))
        if self.layout == "nested-arrays":
            return _first_existing(os.path.join(self._dir, f"{field}.npy"))
        return None

    def _target_path(self, b):
        return self._array_path(f"pl{b}")

    def _arr(self, field):
        """Memory-map a consolidated array once and keep it.

        Memory-mapping matters: the training targets are 3.4 GB per band, and
        reading them into RAM would defeat the point of the repack.
        """
        if field not in self._cache:
            p = self._array_path(field)
            if p is None:
                raise FileNotFoundError(f"{field}.npy not available in this layout")
            self._cache[field] = np.load(p, mmap_mode="r")
        return self._cache[field]

    # -- field readers -----------------------------------------------------

    def _heightmap(self, i):
        row = self.rows[i]
        if self.layout == "per-file":
            return np.load(os.path.join(self._dir, row["heightmap"])).astype(np.float32)
        return np.asarray(self._arr("hm")[int(row["hm_row"])]).astype(np.float32)

    def _tx(self, i):
        row = self.rows[i]
        if self.layout == "per-file":
            return np.load(os.path.join(self._dir, row["tx"])).astype(np.float32)
        tx = np.zeros(GRID, np.float32)
        tx[int(row["tx_row"]), int(row["tx_col"])] = 1.0
        return tx

    def _target(self, i, b):
        if self.layout == "per-file":
            return np.load(os.path.join(self._dir, self.rows[i][f"pl{b}"])).astype(np.float32)
        return np.asarray(self._arr(f"pl{b}")[i]).astype(np.float32)

    def _mask(self, i, b):
        if self.layout == "per-file":
            mi = os.path.join(self._dir, "mask_index.csv")
            if "_masks" not in self._cache:
                with open(mi, newline="") as f:
                    self._cache["_masks"] = {r["sample_id"]: r for r in csv.DictReader(f)}
            rel = self._cache["_masks"][self.rows[i]["sample_id"]][f"mask{b}"]
            return (np.load(os.path.join(self._dir, rel)) > 0).astype(np.float32)
        packed = np.asarray(self._arr(f"mask{b}")[i])
        return np.unpackbits(packed).reshape(GRID).astype(np.float32)

    # -- normalization -----------------------------------------------------

    def denormalize_pathloss(self, y):
        """Map a normalized prediction back to dB."""
        return y * (self.pl_max - self.pl_min) + self.pl_min

    def _norm(self, a, lo, hi):
        return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    # -- item --------------------------------------------------------------

    def sample_name(self, idx):
        """Identifier for item ``idx``: ``<sample_id>_<band>``.

        Use this instead of indexing ``.samples`` -- that attribute is
        internal and its shape differs between layouts.
        """
        i, band = self.samples[idx]
        return f"{self.rows[i]['sample_id']}_{band}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, band = self.samples[idx]
        name = f"{self.rows[i]['sample_id']}_{band}"

        hm = self._heightmap(i)
        if self.normalize:
            hm = self._norm(hm, self.hm_min, self.hm_max)
        channels = [hm, self._tx(i)]
        if self.band == "both":
            channels.append(np.full(GRID, 0.0 if band == "415" else 1.0, np.float32))

        target = None
        if self.has_targets:
            pl = self._target(i, band)
            target = self._norm(pl, self.pl_min, self.pl_max) if self.normalize else pl

        mask = self._mask(i, band) if self.return_mask else None

        # augment everything together so TX, target and mask stay pixel-aligned
        extra = [m for m in (target, mask) if m is not None]
        if self.augment:
            channels, extra = _augment(channels, extra)
        if target is not None:
            target = extra[0]
        if mask is not None:
            mask = extra[-1]

        out = [torch.from_numpy(np.stack(channels, axis=0))]
        if target is not None:
            out.append(torch.from_numpy(target).unsqueeze(0))
        if mask is not None:
            out.append(torch.from_numpy(mask).unsqueeze(0))
        if self.return_name:
            out.append(name)
        return out[0] if len(out) == 1 else tuple(out)


def _augment(channels, extra):
    """Random element of the dihedral group, applied to every map alike."""
    n = len(channels)
    maps = list(channels) + list(extra)

    if random.random() < 0.5:
        maps = [np.fliplr(m) for m in maps]
    if random.random() < 0.5:
        maps = [np.flipud(m) for m in maps]
    k = random.randint(0, 3)
    if k:
        maps = [np.rot90(m, k) for m in maps]

    maps = [np.ascontiguousarray(m) for m in maps]
    return maps[:n], maps[n:]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smoke-test the dataloader.")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--band", default="415", choices=["415", "58", "both"])
    a = p.parse_args()

    ds = LunarRadioMapDataset(a.root, split=a.split, band=a.band,
                              return_mask=(a.split != "test"), return_name=True)
    print(f"layout   : {ds.layout}")
    print(f"items    : {len(ds):,}  (rows {len(ds.rows):,} x bands {len(ds.bands)})")
    print(f"targets  : {ds.has_targets}")
    item = ds[0]
    names = ["input", "target", "mask", "name"]
    for v, n in zip(item, names if ds.has_targets else ["input", "mask", "name"]):
        if torch.is_tensor(v):
            print(f"  {n:<8} {tuple(v.shape)}  {v.dtype}  "
                  f"[{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {n:<8} {v}")
    if ds.has_targets:
        y = item[1]
        print(f"target in dB: [{ds.denormalize_pathloss(y).min():.2f}, "
              f"{ds.denormalize_pathloss(y).max():.2f}]")
