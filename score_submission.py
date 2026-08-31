"""
Score a participant submission against the withheld test ground truth.

This is the organizer-side counterpart to ``evaluate.py``. ``evaluate.py`` scores
a *checkpoint* by running the model itself; this scores a *directory of predicted
arrays*, which is what a submission actually is.

Submission format
-----------------
One ``.npy`` per test sample per band, flat in one directory:

    pl415_00000_00.npy      float32 or float16, shape (256, 256), pathloss in dB
    pl58_00000_00.npy
    ...

The stem after the band prefix is the ``sample_id`` from ``test/index.csv``.
Predictions are **full maps in dB**, the validity mask is applied here, at
scoring time, so entrants never receive the test masks. Values are not clipped
to the physical range: a prediction outside it is scored as the error it is.

Optionally the submission may carry a ``runtime.json`` recording inference time,
which is copied into the report:

    {"seconds_per_map": 0.184, "device": "NVIDIA L4", "notes": "..."}

The metric
----------
    rmse_db_masked   THE RANKING METRIC -- ray-traced pixels only, per band
    rmse_db          diagnostic: every pixel, including the 5.8 GHz static fill

``combined_rmse_db_masked`` is the plain 50/50 mean of the two per-band masked
RMSEs. It is NOT the final rank: the published ranking normalizes each band's
RMSE across all participating teams and sums, which cannot be computed from one
submission in isolation. This script produces the per-band numbers that feed the ranking.

Usage
-----
    # organizers, with the withheld ground truth
    python score_submission.py --submission subs/team_x \
        --gt-root LunarRM_test_groundtruth --data-root LunarRM \
        --out results/team_x.json

    # participants, to check their output is well-formed before submitting
    python score_submission.py --submission /kaggle/working --data-root LunarRM \
        --validate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch and metrics are imported lazily inside score(): --validate-only is the
# path participants run to check their own output, and it needs neither. Making
# them module-level would mean a format check fails on a machine without torch.

BANDS = ("415", "58")
GRID = (256, 256)


def read_index(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def expected_names(sample_ids, bands):
    """The exact set of files a complete submission contains."""
    return {f"pl{b}_{sid}.npy": (sid, b) for b in bands for sid in sample_ids}


def load_map(path):
    """Load one predicted map, rejecting anything that is not a scorable array.

    Every rejection here is a submission the organizers would otherwise have to
    chase by email, so the messages name the file and the actual value seen.
    """
    try:
        a = np.load(path, allow_pickle=False)
    except Exception as e: 
        raise ValueError(f"unreadable ({type(e).__name__}: {e})") from None
    if a.shape != GRID:
        raise ValueError(f"shape {a.shape}, expected {GRID}")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"dtype {a.dtype} is not numeric")
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        plural = "s" if n != 1 else ""
        raise ValueError(f"{n} non-finite value{plural}")
    return a


def validate(sub_dir, expect):
    """Check the submission is complete and well-formed before scoring any of it.

    Returns (problems, extras). A missing or malformed file is fatal; an extra
    file is only reported, since a stray notebook artifact is not cheating.
    """
    if not os.path.isdir(sub_dir):
        return [f"{sub_dir} is not a directory"], []

    present = {n for n in os.listdir(sub_dir) if n.endswith(".npy")}
    missing = sorted(set(expect) - present)
    extras = sorted(present - set(expect))

    problems = []
    if missing:
        problems.append(f"{len(missing)} missing file(s), e.g. {missing[:5]}")

    bad = []
    for name in sorted(present & set(expect)):
        try:
            load_map(os.path.join(sub_dir, name))
        except ValueError as e:
            bad.append(f"{name}: {e}")
            if len(bad) >= 10:
                bad.append("... (stopping after 10)")
                break
    if bad:
        problems.append("malformed file(s):\n    " + "\n    ".join(bad))
    return problems, extras


def score(args):
    import torch

    from metrics import MetricAccumulator

    gt_test = os.path.join(args.gt_root, "test")
    gt_rows = read_index(os.path.join(gt_test, "index.csv"))
    gt_by_id = {r["sample_id"]: r for r in gt_rows}

    mask_path = os.path.join(gt_test, "mask_index.csv")
    if not os.path.exists(mask_path):
        raise SystemExit(
            f"{mask_path} not found. The masked RMSE is the ranking metric, so "
            f"scoring without the test masks is refused rather than silently "
            f"reported as an all-pixel number.")
    masks_by_id = {r["sample_id"]: r for r in read_index(mask_path)}

    with open(args.meta) as f:
        meta = json.load(f)
    lo, hi = meta["pathloss_range_db"]
    scale = float(hi - lo)

    bands = BANDS if args.band == "both" else (args.band,)
    sample_ids = [r["sample_id"] for r in gt_rows]
    expect = expected_names(sample_ids, bands)

    problems, extras = validate(args.submission, expect)
    if extras:
        print(f"note: {len(extras)} unexpected file(s) ignored, "
              f"e.g. {extras[:3]}", file=sys.stderr)
    if problems:
        print("SUBMISSION REJECTED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return None, 1
    print(f"submission complete: {len(expect)} files, {len(sample_ids)} samples, "
          f"band(s) {'+'.join(bands)}")

    accs = {"all": MetricAccumulator(scale)}
    accs.update({b: MetricAccumulator(scale) for b in bands})
    per_sample = []

    for band in bands:
        buf_p, buf_t, buf_m, buf_id = [], [], [], []

        def flush():
            if not buf_id:
                return
            p = torch.from_numpy(np.stack(buf_p)).unsqueeze(1)
            t = torch.from_numpy(np.stack(buf_t)).unsqueeze(1)
            m = torch.from_numpy(np.stack(buf_m)).unsqueeze(1)
            # Normalization is affine and unclipped, so the dB error the
            # accumulator reports is exact; clipping here would quietly forgive
            # predictions that left the physical range.
            pn, tn = (p - lo) / scale, (t - lo) / scale
            accs["all"].update(pn, tn, m)
            accs[band].update(pn, tn, m)
            se = ((pn - tn) ** 2) * (scale ** 2)
            for j, sid in enumerate(buf_id):
                mj = m[j, 0] > 0
                row = {"sample_id": sid, "band": band,
                       "rmse_db": float(se[j].mean() ** 0.5),
                       "valid_frac": float(mj.float().mean())}
                if mj.any():
                    row["rmse_db_masked"] = float(se[j, 0][mj].mean() ** 0.5)
                per_sample.append(row)
            buf_p.clear()
            buf_t.clear()
            buf_m.clear()
            buf_id.clear()

        for sid in sample_ids:
            buf_p.append(load_map(os.path.join(args.submission, f"pl{band}_{sid}.npy")))
            buf_t.append(np.load(os.path.join(gt_test, gt_by_id[sid][f"pl{band}"]))
                         .astype(np.float32))
            buf_m.append(np.load(os.path.join(gt_test, masks_by_id[sid][f"mask{band}"]))
                         .astype(np.float32))
            buf_id.append(sid)
            if len(buf_id) >= args.batch_size:
                flush()
        flush()
        print(f"  scored band {band}: {len(sample_ids)} maps")

    report = {"summary": accs["all"].compute(),
              "per_band": {b: accs[b].compute() for b in bands},
              "per_sample": per_sample}

    masked = [report["per_band"][b].get("rmse_db_masked") for b in bands]
    if all(v is not None for v in masked):
        report["summary"]["combined_rmse_db_masked"] = float(np.mean(masked))
    report["summary"]["ranking_note"] = (
        "combined_rmse_db_masked is the 50/50 mean of the per-band masked RMSE. "
        "The published rank normalizes each band across all teams and sums, so "
        "it is computed over the full field, not from this file alone.")

    rt = os.path.join(args.submission, "runtime.json")
    if os.path.exists(rt):
        with open(rt) as f:
            report["runtime"] = json.load(f)
    else:
        print("note: no runtime.json in the submission; per-map runtime "
              "unverified", file=sys.stderr)
    return report, 0


def main():
    p = argparse.ArgumentParser(
        description="Score a participant submission against withheld ground truth.")
    p.add_argument("--submission", required=True,
                   help="directory of predicted pl<band>_<sample_id>.npy maps")
    p.add_argument("--gt-root", default="LunarRM_test_groundtruth",
                   help="withheld ground-truth tree (organizers only)")
    p.add_argument("--data-root", default="LunarRM",
                   help="public dataset root, for test/index.csv and metadata.json")
    p.add_argument("--meta", default=None,
                   help="metadata.json; defaults to <data-root>/metadata.json")
    p.add_argument("--band", default="both", choices=["415", "58", "both"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--validate-only", action="store_true",
                   help="check the submission is complete and well-formed, then "
                        "exit. Needs no ground truth, so participants can run it.")
    args = p.parse_args()

    if args.meta is None:
        args.meta = os.path.join(args.data_root, "metadata.json")

    if args.validate_only:
        idx = os.path.join(args.data_root, "test", "index.csv")
        ids = [r["sample_id"] for r in read_index(idx)]
        bands = BANDS if args.band == "both" else (args.band,)
        expect = expected_names(ids, bands)
        problems, extras = validate(args.submission, expect)
        if extras:
            print(f"note: {len(extras)} unexpected file(s) would be ignored, "
                  f"e.g. {extras[:3]}")
        if problems:
            print("NOT READY TO SUBMIT:", file=sys.stderr)
            for pr in problems:
                print(f"  {pr}", file=sys.stderr)
            sys.exit(1)
        print(f"submission looks good: {len(expect)} files, all "
              f"{GRID[0]}x{GRID[1]} and finite")
        sys.exit(0)

    report, rc = score(args)
    if rc:
        sys.exit(rc)

    s = report["summary"]
    nan = float("nan")
    print("\n" + "-" * 52)
    for b in report["per_band"]:
        pb = report["per_band"][b]
        print(f"  band {b:>3}  rmse_db_masked {pb.get('rmse_db_masked', nan):7.3f}"
              f"   (all-pixel {pb.get('rmse_db', nan):7.3f})")
    if "combined_rmse_db_masked" in s:
        print(f"  combined  {s['combined_rmse_db_masked']:7.3f} dB  (50/50)")
    print("-" * 52)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
