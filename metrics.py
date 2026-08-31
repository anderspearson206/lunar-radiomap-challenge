"""
Metrics for lunar pathloss radio maps.

Two domains are in play and mixing them silently would make numbers
incomparable, so each metric states which one it uses:

  dB domain         the physical target, pathloss in dB (~20-228)
  normalized [0,1]  the network's actual output, (pathloss - lo) / (hi - lo)

  rmse_db     dB domain          all pixels; diagnostic
  nmse        normalized         sum((p-t)^2) / sum(t^2), and its dB form
  psnr_db     normalized         10*log10(1 / mse), data_range = 1
  ssim        normalized         gaussian-window SSIM, 11x11 sigma 1.5

SSIM and PSNR are conventionally image-domain metrics, so they are computed on
the normalized maps with data_range=1. Reporting PSNR on raw dB values would
make it depend on the arbitrary pathloss range.

Masked variants (ray-traced pixels only) are provided for RMSE, NMSE and PSNR.
SSIM is deliberately NOT masked: it is a windowed statistic, so zeroing pixels
inside the window changes local means and variances and produces a number that
is not comparable to the unmasked one.

Plain torch, so nothing here needs torchmetrics or skimage.
"""

import torch
import torch.nn.functional as F

__all__ = ["ssim", "MetricAccumulator", "save_example_panels"]


def _gaussian_window(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(1, 1, size, size).contiguous()


def ssim(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    """Per-sample SSIM for (B,1,H,W) tensors. Returns (B,)."""
    pred = pred.float()
    target = target.float()
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, w, padding=pad)
    mu_y = F.conv2d(target, w, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, w, padding=pad) - mu_x2
    sigma_y = F.conv2d(target * target, w, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, w, padding=pad) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).mean(dim=(1, 2, 3))


class MetricAccumulator:
    """Streams over batches so nothing large is held in memory.

    Pixel-weighted for RMSE/NMSE (so big and small maps contribute per pixel),
    sample-averaged for PSNR/SSIM (the convention for image metrics).

    Parameters
    ----------
    scale_db : float
        dB per unit of normalized target, i.e. ``pl_max - pl_min``. The
        accumulator is fed normalized tensors and reports RMSE in dB.
    """

    def __init__(self, scale_db):
        self.scale = float(scale_db)
        self.se_db = 0.0        # squared error, dB domain
        self.n = 0
        self.se_db_m = 0.0      # masked
        self.n_m = 0
        self.se_n = 0.0         # squared error, normalized
        self.sq_t = 0.0         # target energy, normalized
        self.se_n_m = 0.0
        self.sq_t_m = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.samples = 0

    @torch.no_grad()
    def update(self, pred, target, mask=None):
        pred = pred.float()
        target = target.float()
        b = pred.shape[0]

        diff_n = pred - target
        se_n = diff_n ** 2
        se_db = se_n * (self.scale ** 2)

        self.se_db += se_db.sum().item()
        self.n += se_db.numel()
        self.se_n += se_n.sum().item()
        self.sq_t += (target ** 2).sum().item()

        if mask is not None:
            m = mask.float()
            self.se_db_m += (se_db * m).sum().item()
            self.n_m += m.sum().item()
            self.se_n_m += (se_n * m).sum().item()
            self.sq_t_m += ((target ** 2) * m).sum().item()

        # per-sample image metrics on the normalized maps
        mse_per = se_n.mean(dim=(1, 2, 3)).clamp_min(1e-12)
        self.psnr_sum += (10.0 * torch.log10(1.0 / mse_per)).sum().item()
        self.ssim_sum += ssim(pred.clamp(0, 1), target.clamp(0, 1)).sum().item()
        self.samples += b

    def compute(self, prefix=""):
        out = {}
        if self.n:
            out["rmse_db"] = (self.se_db / self.n) ** 0.5
        if self.sq_t > 0:
            nmse = self.se_n / self.sq_t
            out["nmse"] = nmse
            out["nmse_db"] = 10.0 * torch.log10(torch.tensor(max(nmse, 1e-20))).item()
        if self.samples:
            out["psnr_db"] = self.psnr_sum / self.samples
            out["ssim"] = self.ssim_sum / self.samples
        if self.n_m:
            out["rmse_db_masked"] = (self.se_db_m / self.n_m) ** 0.5
            mse_m = self.se_n_m / self.n_m
            out["psnr_db_masked"] = 10.0 * torch.log10(
                torch.tensor(max(1.0 / max(mse_m, 1e-20), 1e-20))).item()
            if self.sq_t_m > 0:
                nmse_m = self.se_n_m / self.sq_t_m
                out["nmse_masked"] = nmse_m
                out["nmse_db_masked"] = 10.0 * torch.log10(
                    torch.tensor(max(nmse_m, 1e-20))).item()
        return {f"{prefix}{k}": v for k, v in out.items()}


def save_example_panels(path, x, pred, target, mask, lo, scale, names,
                        max_examples=6, base=None, err_lim=10.0):
    """Write a PNG comparing inputs, prediction, ground truth and error.

    Returns the path written, or None if matplotlib is unavailable -- a missing
    plotting stack must never kill a training run.

    ``base`` is the first U-Net's output when scoring the secondU phase. The
    refinement is a small correction on top of it, so a panel of the composite
    prediction looks the same every epoch; passing ``base`` adds a panel of
    ``pred - base``, which is the part that actually trains.

    ``err_lim`` fixes the error color scale in dB. Auto-scaling it per epoch
    makes a shrinking error render identically every time, only the colorbar
    numbers move -- which is exactly the "nothing is changing" illusion.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (skipping example panels: {e})")
        return None

    import os

    n = min(max_examples, x.shape[0])
    rows = []
    for i in range(n):
        hm = x[i, 0].detach().cpu().numpy()
        tx = x[i, 1].detach().cpu().numpy()
        p_db = pred[i, 0].detach().cpu().float().numpy() * scale + lo
        t_db = target[i, 0].detach().cpu().float().numpy() * scale + lo
        m = mask[i, 0].detach().cpu().numpy() if mask is not None else None
        b_db = (base[i, 0].detach().cpu().float().numpy() * scale + lo
                if base is not None else None)
        vmin = min(p_db.min(), t_db.min())
        vmax = max(p_db.max(), t_db.max())

        panels = [
            ("heightmap", hm, "terrain", None, None),
            ("TX", tx, "hot", 0, 1),
            ("ground truth (dB)", t_db, "viridis", vmin, vmax),
            ("prediction (dB)", p_db, "viridis", vmin, vmax),
            (f"error (dB, +/-{err_lim:g})", p_db - t_db, "coolwarm", -err_lim, err_lim),
        ]
        if b_db is not None:
            ref = p_db - b_db
            rlim = max(abs(ref).max(), 1e-6)
            panels.append((f"refinement (dB, +/-{rlim:.2f})", ref, "coolwarm", -rlim, rlim))
        if m is not None:
            panels.append(("valid mask", m, "gray", 0, 1))

        rmse = float(((p_db - t_db) ** 2).mean() ** 0.5)
        label = f"{names[i]}   RMSE {rmse:.2f} dB"
        if b_db is not None:
            base_rmse = float(((b_db - t_db) ** 2).mean() ** 0.5)
            label += f"  (first U {base_rmse:.2f} dB, delta {rmse - base_rmse:+.2f})"
        rows.append((panels, label))

    ncol = max(len(p) for p, _ in rows)
    # Height per row is tuned to the square maps plus their colourbars giving
    # rows more than that leaves a band of white between every example.
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.1 * ncol, 2.95 * len(rows)),
                             squeeze=False)
    for r, (panels, label) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(panels):
                continue
            title, data, cmap, v0, v1 = panels[c]
            im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
            # column headings once, on the top row only
            if r == 0:
                ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(h_pad=1.6)

    # Row labels go on AFTER the layout pass, in figure coordinates. Placing them
    # as axes-relative text beforehand makes tight_layout try to fit an artist
    # whose offset is a fraction of the very axes height it is shrinking, and the
    # panels collapse to a fraction of their intended size.
    for r, (_, label) in enumerate(rows):
        pos = axes[r][0].get_position()
        # on the top row the label has to clear the column headings
        fig.text(pos.x0, pos.y1 + (0.030 if r == 0 else 0.008), label, fontsize=9,
                 fontweight="bold", ha="left", va="bottom")
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
