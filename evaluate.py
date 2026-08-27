"""
Score a trained checkpoint on the validation split, or on the withheld test set.

Reports the four metrics the training loop tracks -- RMSE (dB), PSNR, SSIM and
NMSE -- each with a masked variant where masking is meaningful (SSIM is
deliberately not masked; see metrics.py). Results go to stdout and to a JSON
file carrying a summary, a per-band breakdown and a per-sample table.

    rmse_db_masked   THE RANKING METRIC -- ray-traced pixels only
    rmse_db          diagnostic: every pixel, including the 5.8 GHz static fill

Scoring is over valid pixels only. The 5.8 GHz targets carry a 145.0 dB
placeholder where the ray tracer could not resolve a cell; it is not physics, and
counting it rewards reproducing the simulator's failure modes. Predictions are
submitted as full 256x256 maps -- the mask is applied here, at scoring time, so
nothing about void geometry has to be handed to entrants.

Splits
------
    --split val    targets ship with the data; this is what entrants use
    --split test   not scoreable here -- the targets and the validity masks are
                   both withheld. The organizers score the test set with
                   score_submission.py.

Examples
--------
    python evaluate.py --ckpt runs/radiownet_58_masked_secondU_best.pt \
        --data-root <data-root> --split val --out results/58_masked_val.json

A band=both checkpoint is scored on both bands at once; the summary then carries
a per-band breakdown as well as the pooled numbers, since 415 MHz and 5.8 GHz
are very different problems and a pooled RMSE hides which one moved.

The full model is the secondU checkpoint -- it carries both halves, so
``--phase firstU`` scores its coarse output and ``--phase secondU`` the refined
one, which is how you measure what the refinement bought.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricAccumulator, save_example_panels, ssim  # noqa: E402
from radiounet import RadioWNet  # noqa: E402

BANDS = ("415", "58")


def load_dataset_module(data_root):
    for cand in (data_root, os.path.dirname(os.path.abspath(__file__))):
        if os.path.exists(os.path.join(cand, "lunar_dataset.py")):
            sys.path.insert(0, cand)
            import lunar_dataset
            return lunar_dataset
    raise FileNotFoundError("lunar_dataset.py not found in --data-root or beside evaluate.py")


class ScoredSet(Dataset):
    """Inputs joined to their targets, wherever the targets live.

    On ``val`` the reference loader already returns the target, so this is a
    thin pass-through. On ``test`` the public tree has no targets
    (``metadata.json``: test ``has_targets`` false), so the target is read here
    from the ground-truth tree. Doing that inside the Dataset rather than in the
    eval loop means the DataLoader workers parallelize it -- it is one more .npy
    read per sample, and I/O, not the GPU, is the bottleneck.

    Items are ``(x, y, mask, name, band)``; mask is an empty tensor when masks
    are off.
    """

    def __init__(self, base, lo, hi, need_mask):
        self.base = base
        self.lo = float(lo)
        self.scale = float(hi - lo)
        self.need_mask = need_mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x, name = item[0], item[-1]
        # the loader names items "<sample_id>_<band>", which is the only place
        # the per-item band survives when band="both"
        sample_id, band = name.rsplit("_", 1)

        # (x, y[, mask], name)
        y = item[1]
        m = item[2] if self.need_mask else torch.zeros(0)
        return x, y, m, name, band


def main(args):
    ld = load_dataset_module(args.data_root)
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=device)
    saved = ck.get("args", {})
    band = args.band or saved.get("band", "58")
    phase = args.phase or saved.get("phase", "firstU")
    # match the output that was actually trained for this phase; a secondU
    # checkpoint holds both halves, so --phase firstU scores its coarse output
    out_idx = 0 if phase == "firstU" else 1

    in_ch = 3 if band == "both" else 2
    model = RadioWNet(inputs=in_ch, phase=phase).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')}, band {band}, phase {phase}, "
          f"val best {ck.get('best', float('nan')):.4f} dB)")

    if args.split == "test":
        raise SystemExit(
            "--split test cannot be scored here: the targets and the validity "
            "masks are both withheld. Score --split val instead -- it is "
            "held-out data you do have. The organizers score the test set with "
            "score_submission.py after the deadline.")

    need_mask = not args.no_mask
    if not need_mask:
        print("warning: --no-mask disables rmse_db_masked, which is the metric "
              "this challenge ranks on. The numbers below are diagnostic only.",
              flush=True)
    base = ld.LunarRadioMapDataset(args.data_root, split=args.split, band=band,
                                   return_name=True, return_mask=need_mask)
    lo, hi = base.pl_min, base.pl_max
    scale = float(hi - lo)
    ds = ScoredSet(base, lo, hi, need_mask)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"{args.split} items {len(ds)} (band={band}), scale {scale:.1f} dB")

    accs = {"all": MetricAccumulator(scale)}
    if band == "both":
        accs.update({b: MetricAccumulator(scale) for b in BANDS})
    per_sample = []
    amp = args.amp and device.type == "cuda"

    with torch.no_grad():
        for bi, (x, y, m, names, bands) in enumerate(loader):
            if args.limit and bi * args.batch_size >= args.limit:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True) if need_mask else None

            with torch.autocast("cuda", enabled=amp):
                pred = model(x)[out_idx]
            pred = pred.float()

            accs["all"].update(pred, y, m)
            if band == "both":
                # a batch straddles the band boundary exactly once, so split it
                for b in set(bands):
                    sel = torch.tensor([j for j, bb in enumerate(bands) if bb == b],
                                       device=device)
                    accs[b].update(pred[sel], y[sel],
                                   m[sel] if m is not None else None)

            # per-sample rows: the aggregate hides whether a difference is
            # concentrated in the heavily-filled maps
            se_n = (pred - y) ** 2
            mse_n = se_n.mean(dim=(1, 2, 3))
            rmse_db = (mse_n.clamp_min(0) ** 0.5) * scale
            psnr = 10.0 * torch.log10(1.0 / mse_n.clamp_min(1e-12))
            ssim_v = ssim(pred.clamp(0, 1), y.clamp(0, 1))
            for j, name in enumerate(names):
                row = {"sample_id": name.rsplit("_", 1)[0], "band": bands[j],
                       "rmse_db": float(rmse_db[j]), "psnr_db": float(psnr[j]),
                       "ssim": float(ssim_v[j])}
                if m is not None:
                    mj = m[j] > 0
                    if mj.any():
                        row["rmse_db_masked"] = float((se_n[j][mj].mean() ** 0.5) * scale)
                        row["valid_frac"] = float(mj.float().mean())
                per_sample.append(row)

            if bi % args.log_every == 0:
                print(f"  {len(per_sample)}/{len(ds)}", flush=True)

    summary = {"checkpoint": args.ckpt, "split": args.split, "band": band,
               "phase": phase, "epoch": ck.get("epoch"), "samples": len(per_sample),
               "official_metric": "rmse_db_masked" if need_mask else None}
    summary.update(accs["all"].compute())
    # a band with no items (only reachable under --limit, which walks 415 first)
    # would otherwise print an empty block
    per_band = ({b: accs[b].compute() for b in BANDS if accs[b].n}
                if band == "both" else {})

    print(f"\n--- {args.split} results ---")
    for k, v in summary.items():
        star = "  <- ranking metric" if k == "rmse_db_masked" else ""
        print((f"  {k}: {v:.4f}{star}" if isinstance(v, float)
               else f"  {k}: {v}"))
    for b, mb in per_band.items():
        print(f"  [{b}] " + "  ".join(f"{k} {v:.4f}" for k, v in mb.items()))

    if args.panels:
        k = min(args.panels, len(ds))
        idxs = [int(round(i * (len(ds) - 1) / max(k - 1, 1))) for i in range(k)]
        xs, ys, ms, names = [], [], [], []
        for i in idxs:
            xi, yi, mi, name, _ = ds[i]
            xs.append(xi)
            ys.append(yi)
            if need_mask:
                ms.append(mi)
            names.append(name)
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
        mb_ = torch.stack(ms).to(device) if ms else None
        with torch.no_grad():
            out = model(xb)
        path = args.panels_out or (
            os.path.splitext(args.out)[0] + "_panels.png" if args.out
            else "panels.png")
        p = save_example_panels(path, xb, out[out_idx].float(), yb, mb_, lo, scale,
                                names, max_examples=len(idxs),
                                base=out[0].float() if out_idx == 1 else None,
                                err_lim=args.panel_err_lim)
        if p:
            print(f"wrote {p}")

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_band": per_band,
                       "per_sample": per_sample}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Score a RadioWNet checkpoint on the val or test split.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="LunarRM")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--band", default=None, choices=["415", "58", "both"],
                   help="defaults to the band the checkpoint was trained on")
    p.add_argument("--phase", default=None, choices=["firstU", "secondU"],
                   help="which output to score; defaults to the phase the "
                        "checkpoint was trained with. A secondU checkpoint can "
                        "be scored either way.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-mask", action="store_true",
                   help="skip the mask reads. This disables rmse_db_masked, the "
                        "ranking metric, leaving only diagnostic all-pixel "
                        "numbers -- for quick sanity runs, not for reporting.")
    p.add_argument("--limit", type=int, default=0, help="stop after ~N samples")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default=None, help="results JSON")
    p.add_argument("--panels", type=int, default=8,
                   help="example panels to render (0 disables)")
    p.add_argument("--panels-out", default=None)
    p.add_argument("--panel-err-lim", type=float, default=10.0)
    main(p.parse_args())
