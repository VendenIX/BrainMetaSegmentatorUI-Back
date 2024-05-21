# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains some Learning Rate schedulers
to reduce the value of the learning rate depending on
the progress of the training process.
"""

import math
from typing import List
import warnings

from pytorch_lightning.utilities.cli import LR_SCHEDULER_REGISTRY
from torch import nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

__all__ = ["LinearLR", "ExponentialLR", "WarmupCosineSchedule", "LinearWarmupCosineAnnealingLR"]


class _LRSchedulerMONAI(_LRScheduler):
    """Base class for increasing the learning rate between two boundaries over a number
    of iterations.
    
    Attributes:
        end_lr: The final learning rate.
        num_iter: Number of iterations over which the test occurs
    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1) -> None:
        """
        Arguments:
            optimizer: Wrapped optimizer.
            end_lr: The final learning rate.
            num_iter: The number of iterations over which the test occurs.
            last_epoch: The index of last epoch.
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LRSchedulerMONAI, self).__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self) -> List[float]:
        """Computes learning rate using chainable form of the scheduler.
        
        Returns:
            lrs: Learning rate of each parameters.
        """
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self) -> List[float]:
        """Computes learning rate using chainable form of the scheduler.
        
        Returns:
            lrs: Learning rate of each parameters.
        """
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, t_total: int, cycles: float = 0.5, last_epoch: int = -1
    ) -> None:
        """
        Arguments:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of warmup iterations.
            t_total: Total number of training iterations.
            cycles: Cosine cycles parameter.
            last_epoch: The index of last epoch.
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        """Returns a learning rate multiplicative factor value at a considered step.
        
        Arguments:
            step: Considered step to make the computation.
        
        Returns:
            lr_lambda: A multiplicative factor given an integer parameter epoch.
        """
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


@LR_SCHEDULER_REGISTRY
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule 
    between `warmup_start_lr` and `base_lr` followed by a cosine annealing schedule 
    between `base_lr` and `eta_min`.

    Attributes:
        warmup_epochs: Maximum number of iterations for linear warmup.
        max_epochs: Maximum number of iterations.
        warmup_start_lr: Learning rate to start the linear warmup.
        eta_min: Minimum learning rate.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Arguments:
            optimizer: Wrapped optimizer.
            warmup_epochs: Maximum number of iterations for linear warmup.
            max_epochs: Maximum number of iterations.
            warmup_start_lr: Learning rate to start the linear warmup.
            eta_min: Minimum learning rate.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        
        Returns:
            lrs: Learning rate of each parameters.
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        
        Returns:
            lrs: Learning rate of each parameters.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
