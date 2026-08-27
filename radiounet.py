"""
Access to the RadioUNet model code.

The RadioWNet architecture is not reimplemented here. The authors' model code is
included unmodified under ``RadioUNet/modules.py`` (MIT, Copyright (c) 2019 Ron
Levie -- see ``RadioUNet/README.md``), and this module is the seam between that
file and the rest of the baseline.

    from radiounet import RadioWNet
    model = RadioWNet(inputs=2, phase="firstU")

Loading is lazy, so ``--help`` and argument parsing still work in an environment
without torch; the import only happens when a model is actually built.

``RadioWNet(inputs, phase)`` returns ``[out1, out2]`` from its forward pass.
``phase="firstU"`` puts gradients on the first U-Net and detaches the
refinement; anything else detaches the first and trains the refinement.
"""

from __future__ import annotations

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "RadioUNet", "modules.py")

_MISSING = """RadioUNet model code not found at
  {path}

It ships with this repository, so a missing file usually means an incomplete
clone or a stripped archive. Re-clone, or restore it from
https://github.com/RonLevie/RadioUNet"""

_cached = None


def load_module():
    """Import ``RadioUNet/modules.py`` and return it."""
    global _cached
    if _cached is not None:
        return _cached
    if not os.path.exists(MODULE_PATH):
        raise SystemExit(_MISSING.format(path=MODULE_PATH))

    spec = importlib.util.spec_from_file_location("radiounet_modules", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cached = module
    return module


def RadioWNet(*args, **kwargs):  # noqa: N802 -- mirrors the upstream class name
    """Build a ``RadioWNet``. Signature: ``(inputs=2, phase="firstU")``."""
    return load_module().RadioWNet(*args, **kwargs)


def trains_in_phase(param_name, phase):
    """Whether a parameter belongs to the half that ``phase`` trains.

    The forward pass detaches the other half, so those parameters get no
    gradient regardless; setting ``requires_grad`` to match just makes it
    explicit and gives an honest trainable-parameter count. The optimizer is
    still handed every parameter, which keeps its ``state_dict`` layout identical
    across the two stages so a checkpoint stays resumable either way.

    Every layer of the refinement half is named with a leading ``W``.
    """
    return param_name.startswith("W") if phase == "secondU" else not param_name.startswith("W")
