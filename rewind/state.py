import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch


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
            model_state=copy.deepcopy(model.state_dict()),
            optim_state=copy.deepcopy(optimizer.state_dict()),
            lr=lr,
            rng_state={
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            branch_id=branch_id,
        )

    def restore(self, model, optimizer):
        model.load_state_dict(self.model_state)
        optimizer.load_state_dict(self.optim_state)
        torch.set_rng_state(self.rng_state["torch"])
        random.setstate(self.rng_state["python"])
        np.random.set_state(self.rng_state["numpy"])