import copy
import random
from dataclasses import dataclass, field
 
import numpy as np
import torch


def _optim_state_to_cpu(optim_state):
    cpu_state = {"state": {}, "param_groups": copy.deepcopy(optim_state["param_groups"])}
    for pid, state in optim_state["state"].items():
        cpu_state["state"][pid] = {
            k: (v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in state.items()
        }
    return cpu_state

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
        random.setstate(self.rng_state["python"])
        np.random.set_state(self.rng_state["numpy"])
        if torch.cuda.is_available() and self.rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all(self.rng_state["cuda"])