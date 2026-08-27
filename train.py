"""
Baseline training for the First Lunar Pathloss Radio Map Prediction Challenge.

Trains RadioWNet to predict a 256x256 pathloss map in dB from a lunar heightmap
and a one-hot transmitter position.

Two-stage training
------------------
RadioWNet is two U-Nets in series and they are trained one at a time, because
the refinement has nothing to refine until the first U-Net has converged:

    --phase firstU     trains the first U-Net (10.92M of 13.27M parameters)
    --phase secondU    freezes it and trains the refinement (2.35M)

The ``secondU`` checkpoint contains both halves and is the full model. It must
be seeded from a trained ``firstU`` run; ``--init-from auto`` (the default for
``--phase secondU``) finds ``<out>/<band>_<loss>_firstU_best.pt``.

Valid pixels only
-----------------
Training and scoring both happen on ray-traced pixels only. The 5.8 GHz targets
carry a static 145.0 dB fill where the ray tracer could not resolve a cell --
~11% of pixels, and a majority of a few maps. That value is a placeholder, not
physics, and counting it rewards a model for reproducing the simulator's failure
modes rather than lunar propagation.

    --loss masked    (default) MSE over ray-traced pixels only
    --loss plain     MSE over every pixel, including the fill

``--loss plain`` is kept as an ablation. It is NOT the challenge objective, and
a run trained that way is still selected and reported on the masked metric, so
the two remain directly comparable.

Both metrics are always computed:

    rmse_db_masked   ray-traced pixels only -- THE RANKING METRIC
    rmse_db          every pixel, fill included -- diagnostic only

Masks are exact, ship with the train and val archives, and are complete for both
bands, so masking filters no samples -- only pixels.

Validation
----------
``val`` is a frozen split, disjoint from ``train`` at the terrain level and
shipped as its own directory. Every entrant validates on the same maps, so
numbers reported against it are comparable across entries. Do not re-split.

Examples
--------
    # stage 1
    python train.py --data-root LunarRM --band 58 --epochs 450
    # stage 2, seeded from stage 1 automatically
    python train.py --data-root LunarRM --band 58 --phase secondU --epochs 150

    # one conditional model over both bands
    python train.py --data-root LunarRM --band both --epochs 300

    # resume an interrupted run (same --out, same --run-name)
    python train.py ... --resume auto

    # seconds-long smoke test of the whole path, no GPU
    python train.py --band 58 --epochs 1 --batch-size 2 --limit-batches 3 \
        --num-workers 0 --device cpu --no-amp --out /tmp/smoke --run-name smoke
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricAccumulator, save_example_panels  # noqa: E402
from radiounet import RadioWNet, trains_in_phase  # noqa: E402


def load_dataset_module(data_root):
    """Prefer the loader shipped with the data, so code and data never skew."""
    for cand in (data_root, os.path.dirname(os.path.abspath(__file__))):
        if os.path.exists(os.path.join(cand, "lunar_dataset.py")):
            sys.path.insert(0, cand)
            import lunar_dataset
            return lunar_dataset
    raise FileNotFoundError("lunar_dataset.py not found in --data-root or beside train.py")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_init_from(args):
    """Checkpoint whose weights seed this run; 'auto' means the matching firstU run.

    Weights only -- the optimizer, the LR schedule and the epoch counter all
    start fresh, because this is a new stage rather than a continuation.
    """
    if args.init_from != "auto":
        return args.init_from
    stem = (args.run_name.replace("secondU", "firstU") if "secondU" in args.run_name
            else f"radiownet_{args.band}_{args.loss}_firstU")
    for suffix in ("_best.pt", "_last.pt"):
        cand = os.path.join(args.out, stem + suffix)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"--init-from auto found neither {stem}_best.pt nor {stem}_last.pt in "
        f"{args.out}. Train the firstU stage first, or pass --init-from <path> / "
        f"--init-from none.")


def make_loaders(ld, args):
    # Masks are always loaded: the ranking metric is computed over valid pixels
    # regardless of which loss trained the run, so both arms read the same data.
    # The consolidated mask arrays are row-aligned with the index and cover
    # every row, so there is no partial-mask case left to guard against.
    common = dict(root=args.data_root, band=args.band, return_mask=True)
    train_ds = ld.LunarRadioMapDataset(split="train", augment=not args.no_augment,
                                       **common)
    val_ds = ld.LunarRadioMapDataset(split="val", augment=False, **common)
    print(f"samples: {len(train_ds)} train / {len(val_ds)} val")

    dl = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, persistent_workers=args.num_workers > 0,
              drop_last=False)
    return (DataLoader(train_ds, shuffle=True, **dl),
            DataLoader(val_ds, shuffle=False, **dl), train_ds)


def unpack(batch, device):
    x, y, m = batch[0], batch[1], batch[2]
    return (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
            m.to(device, non_blocking=True))


def compute_loss(pred, y, m, kind):
    if kind == "masked":
        se = (pred - y) ** 2 * m
        denom = m.sum()
        return se.sum() / denom if denom > 0 else se.sum() * 0.0
    return nn.functional.mse_loss(pred, y)


@torch.no_grad()
def validate(model, loader, device, scale, amp, out_idx, limit=0):
    """RMSE (dB), NMSE, PSNR and SSIM, each with its valid-pixels-only variant."""
    model.eval()
    acc = MetricAccumulator(scale)
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        x, y, m = unpack(batch, device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model(x)[out_idx]
        acc.update(pred.float(), y, m)
    return acc.compute(prefix="val/")


@torch.no_grad()
def write_panels(model, dataset, indices, device, out_idx, lo, scale,
                 path, err_lim):
    """Render a FIXED set of validation samples, so the panels show the same maps
    improving rather than a different random draw each time."""
    model.eval()
    xs, ys, ms, names = [], [], [], []
    for i in indices:
        item = dataset[i]
        xs.append(item[0])
        ys.append(item[1])
        ms.append(item[2])
        # Built from .rows/.samples rather than dataset.sample_name(),
        # which older copies of the shipped loader do not have.
        i_row, band = dataset.samples[i]
        names.append(f"{dataset.rows[i_row]['sample_id']}_{band}")
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    m = torch.stack(ms).to(device) if ms else None
    out = model(x)
    # In secondU the trained output is a correction on top of the frozen first
    # U-Net; pass that baseline so the panels can show the correction itself.
    base = out[0].float() if out_idx == 1 else None
    return save_example_panels(path, x, out[out_idx].float(), y, m, lo, scale,
                               names, max_examples=len(indices), base=base,
                               err_lim=err_lim)


def main(args):
    ld = load_dataset_module(args.data_root)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, ds = make_loaders(ld, args)

    # dB per unit of normalized target -- so RMSE is reported in real units
    lo, hi = ds.pl_min, ds.pl_max
    scale = float(hi - lo)
    in_ch = 3 if args.band == "both" else 2

    # RadioWNet returns [out1, out2]. In "firstU" the refinement is detached; in
    # "secondU" the first U-Net is. Train and score whichever one this phase
    # actually optimizes.
    out_idx = 0 if args.phase == "firstU" else 1
    model = RadioWNet(inputs=in_ch, phase=args.phase).to(device)

    # The frozen half is detached inside forward(), so it gets no gradient either
    # way; setting requires_grad to match makes that explicit and gives an honest
    # parameter count. The optimizer still takes every parameter, which keeps its
    # state_dict layout identical across phases so --resume stays compatible --
    # Adam skips parameters whose grad is None.
    for pname, prm in model.named_parameters():
        prm.requires_grad_(trains_in_phase(pname, args.phase))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train:,} / {n_total:,} ({args.phase} half)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine by default: step decay only fits a run whose length matches the
    # schedule. At 0.1-every-20-epochs it reaches 1e-7 by epoch 60, so a
    # 150-epoch run spends its last 90 epochs not learning. Cosine spreads the
    # same range over whatever --epochs is set to.
    if args.lr_sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr_min)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step,
                                                gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    os.makedirs(args.out, exist_ok=True)
    ckpt_last = os.path.join(args.out, f"{args.run_name}_last.pt")
    ckpt_best = os.path.join(args.out, f"{args.run_name}_best.pt")
    hist_path = os.path.join(args.out, f"{args.run_name}_history.csv")

    start_epoch = 0
    best = float("inf")
    resumed = False
    if args.resume:
        path = ckpt_last if args.resume == "auto" else args.resume
        if os.path.exists(path):
            ck = torch.load(path, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            # A checkpoint written under a different scheduler carries fields
            # that do not describe this one; replaying the epoch count is the
            # only faithful way to place a new schedule on a resumed run.
            #
            # The same applies when only the LENGTH changed, and there it is not
            # cosmetic: state_dict carries T_max, so loading it into a longer run
            # silently restores the OLD horizon and then steps past it. Cosine's
            # recursive form does not clamp -- resuming a finished 80-epoch
            # cosine with --epochs 200 walks the LR up to ~2.6e-1, 2600x base,
            # which destroys the model in one epoch. Replay instead.
            prev_sched = ck.get("lr_sched", "step")
            prev_len = ck["sched"].get("T_max") if args.lr_sched == "cosine" else None
            len_changed = prev_len is not None and prev_len != args.epochs
            if prev_sched == args.lr_sched and not len_changed:
                sched.load_state_dict(ck["sched"])
            else:
                why = (f"checkpoint used --lr-sched {prev_sched}"
                       if prev_sched != args.lr_sched else
                       f"checkpoint cosine ran to {prev_len} epochs, this run to "
                       f"{args.epochs}")
                print(f"{why}; replaying the new schedule to that epoch")
                # Cosine's recursive update reads the CURRENT group lr, and
                # opt.load_state_dict() just restored the OLD schedule's final
                # one -- for a cosine that ran to completion, exactly eta_min.
                # Replaying from there leaves the LR pinned at the floor for the
                # whole resumed run, because every subsequent step scales a value
                # that is already eta_min. Reset to the base LR so the replay
                # walks the NEW schedule from its start.
                for group, base_lr in zip(opt.param_groups, sched.base_lrs):
                    group["lr"] = base_lr
                # stepping without an intervening opt.step() is exactly what the
                # replay needs, so silence the warning torch raises about it
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for _ in range(ck["epoch"] + 1):
                        sched.step()
            start_epoch = ck["epoch"] + 1
            best = ck.get("best", best)
            resumed = True
            print(f"resumed from {path} at epoch {start_epoch}")
        elif args.resume != "auto":
            raise FileNotFoundError(path)
        else:
            print("no checkpoint to resume; starting fresh")

    # Seeding a new stage. Weights only: a resumed run already carries its own
    # optimizer state, so --init-from applies exactly when --resume found nothing.
    if args.init_from and not resumed:
        path = resolve_init_from(args)
        ck = torch.load(path, map_location=device)
        prev = ck.get("args", {})
        if prev.get("band", args.band) != args.band:
            raise SystemExit(f"--init-from {path} was trained on band "
                             f"{prev.get('band')}, this run is band {args.band}")
        model.load_state_dict(ck["model"])
        print(f"initialized weights from {path} (phase {prev.get('phase', '?')}, "
              f"epoch {ck.get('epoch')}, best {ck.get('best', float('nan')):.4f} dB); "
              f"optimizer and LR schedule start fresh")
    elif args.phase == "secondU" and not resumed:
        print("warning: training secondU from random init -- the frozen first "
              "U-Net is untrained, so its output is noise. Pass --init-from "
              "<firstU checkpoint>.")

    # Fixed example set, spread across the validation split so the panels cover a
    # range of fill fractions rather than clustering on similar maps.
    n_val = len(val_loader.dataset)
    k = min(args.panels, n_val)
    panel_idx = ([int(round(i * (n_val - 1) / max(k - 1, 1))) for i in range(k)]
                 if k > 0 else [])

    print(f"training {args.run_name}: band={args.band} loss={args.loss} "
          f"phase={args.phase} in_ch={in_ch} scale={scale:.1f} dB")

    hist_cols = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            x, y, m = unpack(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(x)[out_idx]
                loss = compute_loss(pred.float(), y, m, args.loss)
            scaler.scale(loss).backward()
            if args.clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            running += loss.item() * x.size(0)
            seen += x.size(0)
            if args.log_every and i % args.log_every == 0:
                print(f"  ep{epoch} [{i}/{len(train_loader)}] loss {loss.item():.6f}",
                      flush=True)
            if args.limit_batches and i + 1 >= args.limit_batches:
                break

        # read before stepping: get_last_lr() after the step is next epoch's LR
        epoch_lr = sched.get_last_lr()[0]
        sched.step()

        metrics = validate(model, val_loader, device, scale, args.amp, out_idx,
                           limit=args.limit_batches)
        train_loss = running / max(seen, 1)
        dt = time.time() - t0
        # the ranking metric leads; all-pixel RMSE trails as a diagnostic
        msg = (f"epoch {epoch}: train_loss {train_loss:.6f}  "
               f"rmse {metrics['val/rmse_db_masked']:.3f} dB (valid px)  "
               f"psnr {metrics['val/psnr_db']:.2f}  "
               f"ssim {metrics['val/ssim']:.4f}  "
               f"[all-px {metrics['val/rmse_db']:.3f} dB]")
        print(msg + f"  ({dt:.0f}s)", flush=True)

        row = {"epoch": epoch, "train_loss": train_loss, "lr": epoch_lr,
               "epoch_seconds": round(dt, 1),
               **{k: v for k, v in metrics.items()}}
        if hist_cols is None:
            hist_cols = list(row)
            new_file = not os.path.exists(hist_path) or start_epoch == 0
            with open(hist_path, "w" if new_file else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=hist_cols)
                if new_file:
                    w.writeheader()
                w.writerow(row)
        else:
            with open(hist_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=hist_cols).writerow(row)

        if panel_idx and args.panels_every and epoch % args.panels_every == 0:
            write_panels(model, val_loader.dataset, panel_idx, device, out_idx, lo,
                         scale,
                         os.path.join(args.out, f"{args.run_name}_panels.png"),
                         args.panel_err_lim)

        # Always select on the ranking metric, even for a --loss plain ablation.
        # Selecting a plain run on all-pixel RMSE would pick the checkpoint that
        # best reproduces the 5.8 GHz fill, which is not what it is scored on.
        sel = metrics["val/rmse_db_masked"]
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "lr_sched": args.lr_sched,
                 "epoch": epoch, "best": best, "args": vars(args),
                 "metrics": metrics}
        torch.save(state, ckpt_last)
        if sel < best:
            best = sel
            state["best"] = best
            torch.save(state, ckpt_best)
            print(f"  new best {best:.4f} dB -> {ckpt_best}")

    with open(os.path.join(args.out, f"{args.run_name}_summary.json"), "w") as f:
        json.dump({"run": args.run_name, "band": args.band, "loss": args.loss,
                   "phase": args.phase, "init_from": args.init_from,
                   "selection_metric": "val/rmse_db_masked",
                   "best": best, "epochs": args.epochs}, f, indent=2)
    print(f"done. best {best:.4f} dB (valid pixels only)")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train the RadioWNet baseline on the lunar pathloss dataset.")
    p.add_argument("--data-root", default="LunarRM")
    p.add_argument("--band", default="58", choices=["415", "58", "both"],
                   help="'both' trains one conditional model over both "
                        "frequencies and adds a band-flag input channel")
    p.add_argument("--loss", default="masked", choices=["masked", "plain"],
                   help="masked (default) trains on ray-traced pixels only, which "
                        "is the challenge objective. plain counts the 5.8 GHz "
                        "fill too and is kept only as an ablation; either way the "
                        "run is selected and ranked on valid pixels.")
    p.add_argument("--phase", default="firstU", choices=["firstU", "secondU"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-sched", default="cosine", choices=["cosine", "step"])
    p.add_argument("--lr-min", type=float, default=1e-6, help="cosine floor")
    p.add_argument("--lr-step", type=int, default=20, help="--lr-sched step only")
    p.add_argument("--lr-gamma", type=float, default=0.1, help="--lr-sched step only")
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8,
                   help="the arrays are memory-mapped, so decoding and "
                        "augmentation dominate loading -- raise this if "
                        "the GPU is starved")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    p.add_argument("--init-from", default=None,
                   help="seed model weights from a checkpoint -- weights only, no "
                        "optimizer/schedule/epoch. 'auto' (the default for "
                        "--phase secondU) picks the matching firstU run in --out; "
                        "'none' forces random init. Ignored when --resume finds a "
                        "checkpoint of this run.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--panels", type=int, default=6,
                   help="validation examples rendered to <run>_panels.png (0 off)")
    p.add_argument("--panels-every", type=int, default=5)
    p.add_argument("--panel-err-lim", type=float, default=10.0,
                   help="fixed +/- dB colour scale for the panel error map, so "
                        "panels stay comparable across epochs")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="stop each epoch after N batches; for smoke tests")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.run_name is None:
        a.run_name = f"radiownet_{a.band}_{a.loss}_{a.phase}"
    # The second stage is useless on top of a random first U-Net, so it seeds
    # itself from the firstU run by default; "none" is the explicit opt-out.
    if a.init_from is None and a.phase == "secondU":
        a.init_from = "auto"
    if a.init_from in ("none", "scratch", ""):
        a.init_from = None
    main(a)
