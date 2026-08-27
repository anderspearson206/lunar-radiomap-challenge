"""
Access to the upstream RadioUNet model code.

The RadioWNet architecture is not reimplemented here. A copy of the authors'
``lib/modules.py`` is vendored under ``third_party/RadioUNet/`` as unmodified
upstream bytes, and this module is the seam between that file and the rest of
the baseline: it locates it, checks it is the pinned version, and re-exports the
model.

    from radiounet import RadioWNet
    model = RadioWNet(inputs=2, phase="firstU")

Loading is lazy, so ``--help`` and argument parsing still work in an environment
without torch; the import only happens when a model is actually built.

``RadioWNet(inputs, phase)`` returns ``[out1, out2]`` from its forward pass.
``phase="firstU"`` puts gradients on the first U-Net and detaches the
refinement; anything else detaches the first and trains the refinement.

Provenance and licence (MIT, Copyright (c) 2019 Ron Levie) are recorded in
``third_party/RadioUNet/PROVENANCE.txt``. Run ``python radiounet.py --verify``
to check the vendored bytes against the pinned checksums.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "third_party", "RadioUNet")
MODULE_PATH = os.path.join(VENDOR, "modules.py")

PINNED_COMMIT = "36ab70663443af615c37051ce97da14d04925c7a"
# sha256 of the upstream blobs, LF line endings
CHECKSUMS = {
    "modules.py": "60ee896f812710f6710cc0f3fc2f573844c0ebaabb4c331b452ed259bfd24235",
    "LICENSE": "32b73603c81d657744fc0ce578d0779b68c53a34cda0796bd579c94b300b8f83",
}

_MISSING = """RadioUNet model code not found at
  {path}

It ships with this repository, so a missing file usually means an incomplete
clone or a stripped archive. Re-clone, or restore the file from
https://github.com/RonLevie/RadioUNet at commit {commit}."""

_MANGLED = """{name} does not match the pinned upstream checksum.

  expected  {want}
  found     {got}

The usual cause on Windows is git rewriting LF to CRLF on checkout. This
repository ships a .gitattributes marking these files -text to prevent that; if
it was bypassed, restore them with:

  git checkout -- third_party/RadioUNet/"""

_cached = None


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify(quiet=False):
    """Check the vendored files against the pinned checksums.

    A copy that does not match is not the code this baseline was validated
    against, so it is refused rather than silently used.
    """
    ok = True
    for name, want in CHECKSUMS.items():
        path = os.path.join(VENDOR, name)
        if not os.path.exists(path):
            raise SystemExit(_MISSING.format(path=path, commit=PINNED_COMMIT))
        got = sha256(path)
        if got != want:
            raise SystemExit(_MANGLED.format(name=name, want=want, got=got))
        if not quiet:
            print(f"  {name:<12} {got}  OK")
    return ok


def load_module(check=True):
    """Import the vendored upstream ``modules.py`` and return it."""
    global _cached
    if _cached is not None:
        return _cached
    if not os.path.exists(MODULE_PATH):
        raise SystemExit(_MISSING.format(path=MODULE_PATH, commit=PINNED_COMMIT))
    if check:
        verify(quiet=True)

    spec = importlib.util.spec_from_file_location("radiounet_modules", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cached = module
    return module


def RadioWNet(*args, **kwargs):  # noqa: N802 -- mirrors the upstream class name
    """Build an upstream ``RadioWNet``. Signature: ``(inputs=2, phase="firstU")``."""
    return load_module().RadioWNet(*args, **kwargs)


def trains_in_phase(param_name, phase):
    """Whether a parameter belongs to the half that ``phase`` trains.

    The upstream forward pass detaches the other half, so those parameters get no
    gradient regardless; setting ``requires_grad`` to match just makes it
    explicit and gives an honest trainable-parameter count. The optimizer is
    still handed every parameter, which keeps its ``state_dict`` layout identical
    across the two stages so a checkpoint stays resumable either way.

    Upstream names every layer of the refinement half with a leading ``W``.
    """
    return param_name.startswith("W") if phase == "secondU" else not param_name.startswith("W")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Verify the vendored RadioUNet source.")
    p.add_argument("--verify", action="store_true",
                   help="check vendored files against the pinned checksums")
    a = p.parse_args()

    print(f"vendored from https://github.com/RonLevie/RadioUNet")
    print(f"commit {PINNED_COMMIT}")
    verify()
    print("\nAll vendored files match the pinned upstream bytes.")
