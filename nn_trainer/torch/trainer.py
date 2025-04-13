import copy
from dataclasses import dataclass
from typing import Callable, Generic, Self, TypeVar

import numpy as np
import torch
from mlflow import ActiveRun
from torch import nn, optim
from tqdm import tqdm

from nn_trainer.core.exceptions import NotFittedError, NotSetEvaluatorError
from nn_trainer.core.logger import Logger
from nn_trainer.torch.evaluator.base_evaluator import BaseEvaluator
from nn_trainer.torch.loader import BaseLoader

Model = TypeVar("Model", bound=nn.Module)


@dataclass
class Trainer(Generic[Model]):
    """Trainer for torch model

    Attributes:
        model (Model): Torch model.
        optimizer (optim.Optimizer): Optimizer.
        train_loader (BaseLoader): Data loader used in training
        loss_fn (Callable[[Model, tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]): Loss function evaluated in training
        valid_evaluator (BaseEvaluator): (Optional) Evaluator for validation. Defaults to None.
        early_stopping_patience (int): (Optional) Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        epoch_num (int): (Optional) Number of training iterations. Defaults to 128.
        active_run (ActiveRun): (Optional) MLFlow's run state.
    """

    model: Model
    optimizer: optim.Optimizer
    train_loader: BaseLoader
    loss_fn: Callable[[Model, tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]
    valid_evaluator: BaseEvaluator | None = None
    early_stopping_patience: int = 0
    epoch_num: int = 128
    active_run: ActiveRun | None = None

    def __step_batch(
        self,
        model: Model,
        Xs: tuple[torch.Tensor, ...],
        y: torch.Tensor,
    ) -> float:
        model.train()
        self.optimizer.zero_grad()

        batch_loss = self.loss_fn(model, Xs, y)
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item()

    def __step_epoch(self, epoch_i: int) -> float:
        pbar = tqdm(self.train_loader.setup_epoch())
        pbar.set_description(f"[TRAIN {str(epoch_i).zfill(3)}]")

        batch_loss_buff: list[float] = []
        for Xs, y in pbar:
            batch_loss = self.__step_batch(self.model, Xs, y)
            batch_loss_buff.append(batch_loss)
            pbar.set_postfix({"batch_loss": batch_loss})

        epoch_loss = np.mean(batch_loss_buff).item()
        return epoch_loss

    def __valid_and_check_early_stopping(self, epoch_i: int) -> bool:
        if self.valid_evaluator is None:
            return False

        # Calculate and log valid loss/scores
        loss, metrics = self.valid_evaluator.evaluate(self.model)
        print(f"[VALID {str(epoch_i).zfill(3)}]:", f"{loss=}, {metrics=}")
        self.logger.log_valid_loss(loss, epoch_i)
        self.logger.log_valid_metrics(metrics, epoch_i)

        # Update best state dict if valid loss is best
        if epoch_i == self.logger.best_epoch_i:
            self.__best_state_dict = copy.deepcopy(self.model.state_dict())

        # Check early stopping
        early_stopping_flag = (self.early_stopping_patience > 0) and (
            (epoch_i - self.logger.best_epoch_i) >= self.early_stopping_patience
        )

        return early_stopping_flag

    def fit(self) -> Self:
        # Initialize logger
        self.logger = Logger(active_run=self.active_run)

        # Initialize best state dict
        self.__best_state_dict: dict | None = None

        # Validation
        self.__valid_and_check_early_stopping(epoch_i=0)

        # Iterate training {epoch_num} times
        for epoch_i in range(1, self.epoch_num + 1):
            # Train
            epoch_loss = self.__step_epoch(epoch_i)
            self.logger.log_train_loss(epoch_loss, epoch_i)

            # Validation and Check early stopping
            if self.__valid_and_check_early_stopping(epoch_i):
                break

        return self

    @property
    def best_model(self) -> Model:
        model = copy.deepcopy(self.model)
        model.load_state_dict(self.best_state_dict)
        return model

    @property
    def best_state_dict(self) -> dict:
        if self.valid_evaluator is None:
            raise NotSetEvaluatorError()

        best_state = getattr(self, "_Trainer__best_state_dict", None)
        if best_state is None:
            raise NotFittedError()

        return best_state
