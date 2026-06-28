import torch
import os
import time
import math
import warnings
import torch.nn as nn
from typing import Optional
from tqdm import trange


class Trainer:
    r"""
    Trainer class for training differentiable system with multiple loss functions.
    It handles the training step, validation steps, results logging, and the early stopping criterion.
    By default, it uses :meth:`torch.optim.Adam` as the optimizer, and :meth:`torch.optim.lr_scheduler.StepLR` as the learning rate scheduler.
    Each loss (criterion) can be registered using the :meth:`register_criterion` method.
    The training process can be started using the :meth:`train` method with the training and validation datasets.
    To each loss it is possible to assign a weight :math:`\alpha` and a flag indicating whether the loss function
    requires the model as an input, which might be needed when the loss depends on the model's parameters.

        **Arguments / Attributes**:
            - **net** (nn.Module): The differentiable system to be trained.
            - **max_epochs** (int): Maximum number of training epochs. Default: 10.
            - **lr** (float): Learning rate for the optimizer. Default: 1e-3.
            - **patience** (int): Number of epochs to wait for improvement in validation loss before early stopping. Default: 5.
            - **patience_delta** (float): Minimum improvement in validation loss to be considered as an improvement. Default: 0.01.
            - **step_size** (int): Period of learning rate decay. Default: 50.
            - **step_factor** (float): Multiplicative factor of learning rate decay. Default: 0.1.
            - **train_dir** (str): The directory for saving training outputs. Default: None.
            - **device** (str): Device to use for training. Default: 'cpu'.

        **Attributes**:
            - **min_val_loss** (float): Minimum validation loss to be updated by the early stopper.
            - **optimizer** (torch.optim.Optimizer): The optimizer.
            - **criterion** (list): List of loss functions.
            - **alpha** (list): List of weights for the loss functions.
            - **requires_model** (list): List of flags indicating whether the loss functions require the model as an input.
            - **scheduler** (torch.optim.lr_scheduler.StepLR): The learning rate scheduler.

        Examples::

            >>> trainer = Trainer(net)  # initialize the trainer with a trainable nn.Module net
            >>> alpha_1, alpha_2 = 1, 0.1
            >>> loss_1, loss_2 = torch.nn.MSELoss(), torch.nn.L1Loss()
            >>> trainer.register_criterion(loss_1, alpha_1)  # register the first loss function with weight 1
            >>> trainer.register_criterion(loss_2, alpha_2)  # register the second loss function with weight 0.1
            >>> trainer.train(train_dataset, valid_dataset)
    """

    def __init__(
        self,
        net: nn.Module,
        max_epochs: int = 10,
        lr: float = 1e-3,
        patience: int = 5,
        patience_delta: float = 0.01,
        step_size: int = 50,
        step_factor: float = 0.1,
        log: bool = True,
        train_dir: str = None,
        device: str = "cpu",
    ):
        self.device = device
        self.log = log
        self.net = net.to(device)
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.patience_delta = patience_delta
        self.min_val_loss = float("inf")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.n_loss = 0
        if self.log:
            assert os.path.isdir(train_dir), (
                "The directory specified in train_dir does not exist."
            )
        self.train_dir = train_dir

        self.criterion, self.alpha, self.requires_model = (
            [],
            [],
            [],
        )  # list of loss functions, weights, and parameter flags
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=step_factor
        )  # learning rate scheduler

    def register_criterion(
        self, criterion: nn.Module, alpha: int = 1, requires_model: bool = False
    ):
        r"""
        Register in the class a loss function (criterion) and its weight.

            **Arguments**:
                - **criterion** (nn.Module): The loss function.
                - **alpha** (float): The weight of the loss function. Default: 1.
                - **requires_model** (bool): Whether the loss function requires the model as an input. Default: False.
        """
        self.criterion.append(criterion.to(self.device))
        self.alpha.append(alpha)
        self.requires_model.append(requires_model)
        self.n_loss += 1

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
    ):
        r"""
        Train the neural network model.

            **Arguments**:
                - **train_dataset** (torch.utils.data.Dataset): The training dataset.
                - **valid_dataset** (torch.utils.data.Dataset): The validation dataset.
        """

        self.train_loss, self.valid_loss = [], []
        self.train_loss_log, self.valid_loss_log = {}, {}
        # initialize self.train_loss_log and self.valid_loss_log as dictionary with the name of each loss as key
        for i in range(self.n_loss):
            loss_name = self.criterion[i].__class__.__name__
            self.train_loss_log[loss_name] = []
            self.valid_loss_log[loss_name] = []

        st = time.time()  # start time
        pbar = trange(self.max_epochs, desc="Training")
        for epoch in pbar:
            st_epoch = time.time()

            # training
            epoch_loss = 0
            for data in train_dataset:
                epoch_loss += self.train_step(data)
            self.scheduler.step()
            self.train_loss.append(epoch_loss / len(train_dataset))
            # validation
            epoch_loss = 0
            for data in valid_dataset:
                epoch_loss += self.valid_step(data)
            self.valid_loss.append(epoch_loss / len(valid_dataset))
            et_epoch = time.time()

            # update progress bar in place (no newlines)
            pbar.set_postfix_str(
                get_str_results(
                    epoch=epoch,
                    train_loss=self.train_loss,
                    valid_loss=self.valid_loss,
                    time=et_epoch - st_epoch,
                )
            )

            # save checkpoints
            if self.log:
                self.save_model(epoch)
            if self.early_stop():
                break

        et = time.time()  # end time
        print(f"Early stopping at epoch: {epoch}")
        print(f"Training time: {et - st:.3f}s")

    def move_to_device(self, data: list | torch.Tensor):
        if isinstance(data, list):
            data = [x.to(self.device) for x in data]
        else:
            data = data.to(self.device)
        return data

    def train_step(self, data: tuple):
        r"""
        Perform a single training step.

            **Arguments**:
                - **data** (tuple): A tuple containing the input data and the target data :code:`(inputs, targets)`.

            **Returns**:
                - float: The loss value of the training step.
        """
        inputs, targets = data
        inputs = self.move_to_device(inputs)
        targets = self.move_to_device(targets)
        # batch processing
        self.optimizer.zero_grad()
        estimations = self.net(inputs)
        loss = 0
        for alpha, criterion, requires_model in zip(
            self.alpha, self.criterion, self.requires_model
        ):
            if requires_model:
                temp = criterion(estimations, targets, self.net)
                self.train_loss_log[criterion.__class__.__name__].append(temp.item())
                loss += alpha * temp
            else:
                temp = criterion(estimations, targets)
                self.train_loss_log[criterion.__class__.__name__].append(temp.item())
                loss += alpha * temp
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def valid_step(self, data: tuple):
        r"""
        Perform a single validation step.

            **Arguments**:
                - **data** (tuple): A tuple containing the input data and the target data :code:`(inputs, targets)`.

            **Returns**:
                - float: The loss value for the validation step.
        """
        # batch processing
        inputs, targets = data
        inputs = self.move_to_device(inputs)
        targets = self.move_to_device(targets)

        self.optimizer.zero_grad()
        estimations = self.net(inputs)
        loss = 0
        for alpha, criterion, requires_model in zip(
            self.alpha, self.criterion, self.requires_model
        ):
            if requires_model:
                temp = criterion(estimations, targets, self.net)
                self.valid_loss_log[criterion.__class__.__name__].append(temp.item())
                loss += alpha * temp
            else:
                temp = criterion(estimations, targets)
                self.valid_loss_log[criterion.__class__.__name__].append(temp.item())
                loss += alpha * temp
        return loss.item()

    def print_results(self, e: int, e_time: float):
        r"""Print a string with the training results for an epoch."""
        print(
            get_str_results(
                epoch=e,
                train_loss=self.train_loss,
                valid_loss=self.valid_loss,
                time=e_time,
            )
        )

    def get_train_dir(self):
        r"""Get the directory path where to save the training outputs."""
        if self.train_dir is not None:
            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)
        else:
            self.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.train_dir)

    def save_model(self, e: int):
        r"""
        Save the model parameters to a file.

            **Arguments**:
                **e** (int): The epoch number.
        """
        dir_path = os.path.join(self.train_dir, "checkpoints")
        # create checkpoint folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model
        torch.save(
            self.net.state_dict(), os.path.join(dir_path, "model_e" + str(e) + ".pt")
        )

    def early_stop(self):
        r"""
        Early stopping criterion.
        """
        if self.valid_loss[-1] < (self.min_val_loss - self.patience_delta):
            # update min validation loss
            self.min_val_loss = self.valid_loss[-1]
            self.counter = 0
        elif ((self.min_val_loss - self.patience_delta) < self.valid_loss[-1]) and (
            self.valid_loss[-1] < (self.min_val_loss + self.patience_delta)
        ):
            # no improvement, so update counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EagerTrainer:
    r"""
    Lightweight optimizer for a single differentiable system.

    Unlike :class:`Trainer`, which iterates a :class:`torch.utils.data.DataLoader`
    over epochs, :class:`EagerTrainer` fits one fixed ``(input, target)`` pair by
    running gradient descent directly — no ``Dataset``, ``DataLoader``, ``expand``,
    ``split``, ``shuffle``, or batch dimension. This matches the true nature of
    fitting one LTI system (e.g. an FDN): a pure optimization problem with no
    held-out data or generalization.

    Criteria are registered exactly as in :class:`Trainer` via
    :meth:`register_criterion`, so existing loss functions (and the
    ``requires_model`` flag) transfer unchanged.

        **Arguments / Attributes**:
            - **net** (nn.Module): The differentiable system to optimize.
            - **max_steps** (int): Maximum number of optimization steps. Default: 1000.
            - **lr** (float): Learning rate. Default: 1e-3.
            - **optimizer** (str): ``"adam"`` or ``"lbfgs"``. Default: ``"adam"``.
            - **step_size** (int): StepLR period (Adam only). Default: 50.
            - **step_factor** (float): StepLR gamma (Adam only). Default: 0.1.
            - **tol** (float): Relative-improvement threshold for plateau early stop. Default: 1e-6.
            - **patience** (int): Consecutive non-improving steps before stopping. Default: 10.
            - **log** (bool): Print timing / plateau messages. Default: True.
            - **train_dir** (str): Directory for checkpoints (required if save_checkpoints). Default: None.
            - **save_checkpoints** (bool): Save ``state_dict`` each step. Default: False.
            - **device** (str): Device for optimization. Default: 'cpu'.

        Examples::

            >>> opt = EagerTrainer(model, max_steps=1600, lr=1e-2)
            >>> opt.register_criterion(MultiResoSTFT(), 1)
            >>> opt.register_criterion(sparsity_loss(), 1, requires_model=True)
            >>> history = opt.optimize(input, target)
    """

    def __init__(
        self,
        net: nn.Module,
        max_steps: int = 1000,
        lr: float = 1e-3,
        optimizer: str = "adam",
        step_size: int = 2000,
        step_factor: float = 0.1,
        tol: float = 1e-6,
        patience: int = 10,
        log: bool = True,
        train_dir: str = None,
        save_checkpoints: bool = False,
        device: str = "cpu",
    ):
        self.device = device
        self.net = net.to(device)
        self.max_steps = max_steps
        self.lr = lr
        self.tol = tol
        self.patience = patience
        self.log = log
        self.train_dir = train_dir
        self.save_checkpoints = save_checkpoints
        self.n_loss = 0
        self.criterion, self.alpha, self.requires_model = [], [], []

        if self.save_checkpoints:
            assert train_dir is not None and os.path.isdir(train_dir), (
                "save_checkpoints=True requires an existing train_dir."
            )

        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=step_factor
            )
        elif self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.net.parameters(), lr=self.lr, line_search_fn="strong_wolfe"
            )
            self.scheduler = None
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Use 'adam' or 'lbfgs'.")

    def register_criterion(
        self, criterion: nn.Module, alpha: int = 1, requires_model: bool = False
    ):
        r"""Register a loss function and its weight. Mirrors :meth:`Trainer.register_criterion`."""
        self.criterion.append(criterion.to(self.device))
        self.alpha.append(alpha)
        self.requires_model.append(requires_model)
        self.n_loss += 1

    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def _compute_loss(self, estimations, targets, log_dict=None):
        r"""Weighted sum of registered criteria; optionally logs each into log_dict."""
        loss = 0
        for alpha, criterion, requires_model in zip(
            self.alpha, self.criterion, self.requires_model
        ):
            if requires_model:
                temp = criterion(estimations, targets, self.net)
            else:
                temp = criterion(estimations, targets)
            if log_dict is not None:
                log_dict[criterion.__class__.__name__].append(temp.item())
            loss = loss + alpha * temp
        return loss

    def optimize(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        r"""
        Optimize the system parameters on a single fixed ``(input, target)`` pair.

            **Returns**:
                - dict: Loss history with key ``"total"`` plus one key per criterion
                  class name. All lists have equal length (one entry per step run).
        """
        input = self.move_to_device(input)
        target = self.move_to_device(target)

        # L-BFGS assumes a deterministic objective; warn if a stochastic loss is used.
        if self.optimizer_name == "lbfgs":
            from flamo.optimize.loss import masked_mse_loss

            if any(isinstance(c, masked_mse_loss) for c in self.criterion):
                warnings.warn(
                    "L-BFGS assumes a deterministic objective, but a "
                    "masked_mse_loss (random per-call mask) is registered. The "
                    "mask changes within a line search, breaking L-BFGS "
                    "assumptions. Use optimizer='adam' for masked/colorless fits; "
                    "reserve L-BFGS for deterministic objectives.",
                    RuntimeWarning,
                )

        self.loss_history = {"total": []}
        for c in self.criterion:
            self.loss_history[c.__class__.__name__] = []

        best_loss = float("inf")
        counter = 0
        st = time.time()
        pbar = trange(self.max_steps, desc="Optimizing", disable=not self.log)
        for step in pbar:
            if self.optimizer_name == "adam":
                self.optimizer.zero_grad()
                est = self.net(input)
                loss = self._compute_loss(est, target, self.loss_history)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total = loss.item()
            else:  # lbfgs

                def closure():
                    self.optimizer.zero_grad()
                    est = self.net(input)
                    loss = self._compute_loss(est, target)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)
                with torch.no_grad():
                    est = self.net(input)
                    total = self._compute_loss(est, target, self.loss_history).item()

            self.loss_history["total"].append(total)
            pbar.set_postfix_str(f"loss: {total:.6f}")

            if self.save_checkpoints:
                self.save_model(step)

            # plateau early stopping on relative improvement of the total loss
            rel_improvement = (
                (best_loss - total) / (abs(best_loss) + 1e-12)
                if math.isfinite(best_loss)
                else float("inf")
            )
            if total < best_loss:
                best_loss = total
            if rel_improvement > self.tol:
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    if self.log:
                        print(f"Plateau reached at step {step}.")
                    break

        if self.log:
            print(f"Optimization time: {time.time() - st:.3f}s")
        return self.loss_history

    def save_model(self, step: int):
        r"""Save the model parameters to ``train_dir/checkpoints/model_e<step>.pt``."""
        dir_path = os.path.join(self.train_dir, "checkpoints")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(
            self.net.state_dict(),
            os.path.join(dir_path, "model_e" + str(step) + ".pt"),
        )


def get_str_results(
    epoch: Optional[int] = None,
    train_loss: Optional[list] = None,
    valid_loss: Optional[list] = None,
    time: Optional[int] = None,
):
    r"""
    Construct the string that has to be printed at the end of the epoch containing
    information relative to the training performance.

        **Arguments**:
            - **epoch** (int): The epoch number. Default: None.
            - **train_loss** (list): List of training loss values. Default: None.
            - **valid_loss** (list): List of validation loss values. Default: None.
            - **time** (float): The time taken for the epoch. Default: None.

        **Returns**:
            - str: The formatted string to be printed.
    """
    to_print = ""

    if epoch is not None:
        to_print += "epoch: {:3d} ".format(epoch)

    if train_loss is not None:
        to_print += "- train_loss: {:6.4f} ".format(train_loss[-1])

    if valid_loss is not None:
        to_print += "- test_loss: {:6.4f} ".format(valid_loss[-1])

    if time is not None:
        to_print += "- time: {:6.4f} s".format(time)

    return to_print
