"""rewind/state.py

TrainerState is one exact snapshot of a training process: model and
optimizer state on CPU, plus the torch / python / numpy / CUDA RNG states,
tagged with the step and branch it was taken on.

Snapshots that leave the process (the tracker's `type="torch"` artifacts)
are written through `to_dict()` and read back through `from_dict()`. The
dict holds nothing but tensors, plain containers and Python scalars, so it
loads under `torch.load`'s default `weights_only=True` (PyTorch >= 2.6),
which refuses to unpickle arbitrary classes such as this dataclass or the
tuple-with-ndarray that `numpy.random.get_state()` returns.
"""

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch

STATE_FORMAT = 1


def _optim_state_to_cpu(optim_state):
    cpu_state = {"state": {}, "param_groups": copy.deepcopy(optim_state["param_groups"])}
    for pid, state in optim_state["state"].items():
        cpu_state["state"][pid] = {
            k: (v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in state.items()
        }
    return cpu_state


def _rng_to_plain(rng: dict) -> dict:
    """Encode the RNG states with tensors and lists only (see module doc)."""
    out = {"torch": rng.get("torch"), "cuda": rng.get("cuda")}
    py = rng.get("python")
    if py is not None:
        version, internal, gauss_next = py
        out["python"] = [int(version), [int(i) for i in internal], gauss_next]
    np_state = rng.get("numpy")
    if np_state is not None:
        if not isinstance(np_state, tuple):
            raise TypeError("expected numpy's legacy tuple RNG state; got "
                            f"{type(np_state).__name__}")
        name, keys, pos, has_gauss, cached = np_state
        out["numpy"] = [str(name), torch.from_numpy(np.asarray(keys).astype(np.int64)),
                        int(pos), int(has_gauss), float(cached)]
    return out


def _rng_from_plain(plain: dict) -> dict:
    rng = {"torch": plain.get("torch"), "cuda": plain.get("cuda"), "python": None, "numpy": None}
    py = plain.get("python")
    if py is not None:
        version, internal, gauss_next = py
        rng["python"] = (int(version), tuple(int(i) for i in internal), gauss_next)
    np_state = plain.get("numpy")
    if np_state is not None:
        name, keys, pos, has_gauss, cached = np_state
        keys = keys.numpy() if isinstance(keys, torch.Tensor) else np.asarray(keys)
        rng["numpy"] = (str(name), keys.astype(np.uint32), int(pos), int(has_gauss), float(cached))
    return rng


@dataclass
class TrainerState:
    step: int
    model_state: dict
    optim_state: dict
    lr: float
    rng_state: dict = field(default_factory=dict)
    branch_id: str = "root"

    @staticmethod
    def capture(step, model, optimizer, lr, branch_id="root"):
        return TrainerState(
            step=step,
            model_state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            optim_state=_optim_state_to_cpu(optimizer.state_dict()),
            lr=lr,
            rng_state={
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            branch_id=branch_id,
        )

    def restore(self, model, optimizer):
        model.load_state_dict(self.model_state)
        optimizer.load_state_dict(self.optim_state)
        torch.set_rng_state(self.rng_state["torch"])
        if self.rng_state.get("python") is not None:
            random.setstate(self.rng_state["python"])
        if self.rng_state.get("numpy") is not None:
            np.random.set_state(self.rng_state["numpy"])
        if torch.cuda.is_available() and self.rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(self.rng_state["cuda"])

    # ---------------- serialization ----------------

    def to_dict(self) -> dict:
        """A `torch.save`-able payload made of tensors, containers and
        scalars only, so `torch.load` accepts it with `weights_only=True`."""
        return {
            "format": STATE_FORMAT,
            "step": int(self.step),
            "lr": self.lr,
            "branch_id": self.branch_id,
            "model_state": dict(self.model_state),
            "optim_state": self.optim_state,
            "rng_state": _rng_to_plain(self.rng_state),
        }

    @classmethod
    def from_dict(cls, payload) -> "TrainerState":
        """Inverse of `to_dict`. A TrainerState passes through unchanged, so
        callers can hand over whatever the tracker returned."""
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise TypeError(f"not a TrainerState payload: {type(payload).__name__}")
        return cls(
            step=int(payload["step"]),
            model_state=payload["model_state"],
            optim_state=payload["optim_state"],
            lr=payload.get("lr"),
            rng_state=_rng_from_plain(payload.get("rng_state") or {}),
            branch_id=payload.get("branch_id", "root"),
        )
