import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange
from scipy.io import savemat
from flamo.processor.system import Shell
from pydantic import BaseModel
from typing import List, Callable, Optional


class ParameterConfig(BaseModel):
    key: str = None  # key of the parameter in the model
    param_map: Callable = (
        lambda x: x
    )  # mapping function for the parameters in param_dict
    lower_bound: float | List[float] | List[List[float]] = None
    upper_bound: float | List[float] | List[List[float]] = None
    target_value: float = None
    scale: str = "linear"
    n_steps: int = (
        None  # number of steps between lower and upper bound of the parameter
    )
    indx: tuple | int = tuple([slice(None)])  # index of the parameter in the module


class LossConfig(BaseModel):
    criteria: List[Callable] = None  # loss function to be used
    param_config: List[ParameterConfig] = None
    perturb_param: str = None  # key of the parameter to be perturbed
    perturb_map: Callable = (
        lambda x: x
    )  # mapping function for the perturbation parameter
    n_runs: int = None  # number of perturbation runs
    output_dir: str = None


class LossProfile:
    r"""
    Class for computing the loss profile of a model given the the optimizable parameters.
    This class allows to investigate how the loss profile changes when other parameters are being perturbed.

    **Arguments / Attributes**:
        - **net** (Shell): Model to be optimized.
        - **loss_config** (LossConfig): Configuration for the loss profile computation.

    **Attributes**:
        - **net** (Shell): Model to be optimized.
        - **loss_config** (LossConfig): Configuration for the loss profile computation.
        - **param_config** (ParameterConfig): Configuration for the parameter to be optimized.
        - **criteria** (List[Callable]): List of loss functions to be used.
        - **n_steps** (int): Number of steps between lower and upper bound of the parameter.
        - **n_runs** (int): Number of perturbation runs.
        - **output_dir** (str): Output directory for the loss profile plots.
        - **steps** (torch.Tensor): List of steps between the lower and upper bound of the parameter.

    """

    def __init__(self, net: Shell, loss_config: LossConfig, device: str = "cpu"):

        super().__init__()
        self.net = net
        self.loss_config = loss_config
        self.param_config = loss_config.param_config[0]
        self.criteria = loss_config.criteria
        self.n_runs = loss_config.n_runs
        self.output_dir = loss_config.output_dir
        self.device = device
        self.register_steps()

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        r"""
        Compute the loss profile of the model.
        """
        core = self.net.get_core()
        with torch.no_grad():

            # set steps for the parameter
            steps = self.steps

            for i_run in trange(self.n_runs, desc="Run"):
                # perturb the given parameter
                if self.loss_config.perturb_param:
                    new_value = self.sample_rand_param(
                        core,
                        self.get_nested_module(
                            core, self.loss_config.perturb_param
                        ).param,
                    )
                    self.set_raw_parameter(
                        self.loss_config.perturb_param,
                        new_value,
                        self.loss_config.perturb_map,
                    )

                if i_run == 0:
                    loss = np.empty((self.n_runs, len(steps), len(self.criteria)))
                for i_step in range(len(steps)):
                    if type(self.param_config.lower_bound) == list:
                        # interpolate between the lower and upper bound
                        new_value = (1 - steps[i_step]) * torch.tensor(
                            self.param_config.lower_bound, device=self.device
                        ) + steps[i_step] * torch.tensor(
                            self.param_config.upper_bound, device=self.device
                        )
                    else:
                        new_value = steps[i_step]
                    self.set_raw_parameter(
                        self.param_config.key,
                        new_value,
                        self.param_config.param_map,
                        self.param_config.indx,
                    )

                    for i_crit in range(len(self.criteria)):
                        pred = self.net(input)
                        loss[i_run, i_step, i_crit] = (
                            self.criteria[i_crit](pred, target).cpu().detach().numpy()
                        )
            # Save the partial loss for the current run
            partial_loss = loss[i_run, :, :]
            savemat(
                f"{self.output_dir}/partial_loss_run_{i_run + 1}.mat",
                {"loss": partial_loss, "steps": steps.cpu().numpy()},
            )

        return loss

    def plot_loss(self, loss: np.ndarray):
        r"""
        Plot the loss profile.
        """
        if len(self.criteria) == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig, ax = plt.subplots(
                3,
                (len(self.criteria) + 2) // 3,
                figsize=((len(self.criteria) + 2) // 3 * 5, 15),
            )
        steps = self.steps
        for i_crit in range(len(self.criteria)):
            mean_loss = loss[:, :, i_crit].mean(0)
            std_loss = loss[:, :, i_crit].std(0)
            criterion_name = self.criteria[i_crit].name
            if len(self.criteria) == 1:
                ax.plot(steps, mean_loss, label=criterion_name)
                ax.plot(
                    steps[mean_loss.argmin()],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                ax.set_xscale(self.param_config.scale)
                ax.fill_between(
                    steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2
                )
                try:
                    ax.axvline(
                        x=self.param_config.target_value,
                        color="r",
                        linestyle="--",
                        label="Target Value",
                    )
                except:
                    pass

                ax.set_xlabel(self.param_config.key)
                ax.set_ylabel("Loss")
                ax.legend()
                if criterion_name:
                    ax.set_title(criterion_name)
            else:
                if (len(self.criteria) + 2) // 3 == 1:
                    ax_i = (i_crit % 3,)
                else:
                    ax_i = (i_crit % 3, i_crit // 3)

                ax[ax_i].plot(steps, mean_loss, label=criterion_name)
                ax[ax_i].plot(
                    steps[mean_loss.argmin()],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                ax[ax_i].set_xscale(self.param_config.scale)
                ax[ax_i].fill_between(
                    steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2
                )
                try:
                    ax[ax_i].axvline(
                        x=self.param_config.target_value,
                        color="r",
                        linestyle="--",
                        label="Target Value",
                    )
                except:
                    pass
                ax[ax_i].set_xlabel(self.param_config.key)
                ax[ax_i].set_ylabel("Loss")
                ax[ax_i].legend()
                if criterion_name:
                    ax[ax_i].set_title(criterion_name)
        # delay empty axes
        try:
            for i_ax in range(i_crit + 1,math.prod(list(ax.shape))):
                if (len(self.criteria) + 2) // 3 == 1:
                    ax_i = (i_ax % 3,)
                else:
                    ax_i = (i_ax % 3, i_ax // 3)
                fig.delaxes(ax[ax_i])
        except:
            pass
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.param_config.key}.png")
        return fig, ax

    def get_steps(self, param_lower_bound, param_upper_bound, n_steps, scale):
        r"""
        Generate a list of steps between the lower and upper bound of the parameter.
        """
        if type(param_lower_bound) == list:
            # use linear interpolation
            lower_bound = 0
            upper_bound = 1
        else:
            lower_bound = param_lower_bound
            upper_bound = param_upper_bound

        if scale == "linear":
            steps = torch.linspace(lower_bound, upper_bound, n_steps)
        elif scale == "log":
            steps = torch.logspace(
                torch.log10(torch.tensor(lower_bound, device=self.device)),
                torch.log10(torch.tensor(upper_bound, device=self.device)),
                n_steps,
            )
        else:
            raise ValueError("Scale must be either 'linear' or 'log'")
        return steps

    def register_steps(self):
        self.steps = self.get_steps(
            self.param_config.lower_bound,
            self.param_config.upper_bound,
            self.param_config.n_steps,
            self.param_config.scale,
        )

    def set_raw_parameter(
        self,
        param_key: str,
        new_value: torch.Tensor,
        map: lambda x: x,
        indx: tuple = tuple([slice(None)]),
    ):

        core = self.net.get_core()
        self.keys = self.get_modules_keys(core)

        for key in self.keys:
            if key == param_key:
                module = self.get_nested_module(core, key)
                module.assign_value(map(new_value), indx)

    def get_modules_keys(self, module, prefix=""):
        r"""
        Get all the keys of the modules in the core.
        """
        keys = []
        for key, submodule in module._modules.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            keys.extend(self.get_modules_keys(submodule, full_key))
        return keys

    def get_nested_module(self, core, key):
        r"""
        Get the nested module from the core using the key.
        """
        keys = key.split(".")
        module = core
        for k in keys:
            module = module._modules[k]
        return module

    def sample_rand_param(self, core, ref_param: torch.Tensor):
        r"""
        Use the standard deviation and mean of the reference parameter to sample a random parameter value.

        **Arguments**:
            - **ref_param** (torch.Tensor): Reference parameter to sample from.
        """
        std = ref_param.std()
        mean = ref_param.mean()
        rand_param = torch.randn_like(ref_param) * std + mean
        return rand_param

    def compute_accuracy(self, loss):
        r"""
        Compute the accuracy of the losses according to
        Joseph Turian and Max Henry, “I'm sorry for your loss: Spectrally-based
        audio distances are bad at pitch,” arXiv preprint arXiv:2012.04572, 2020.
        """
        steps = self.steps
        # find the index in steps of the element closest to the target value
        target_indx = np.abs(steps - self.param_config.target_value).argmin()
        accuracy = np.empty(loss.shape)
        for i_crit in range(len(self.criteria)):
            for i_run in range(loss.shape[0]):
                for i_step in range(loss.shape[1]):
                    accuracy[i_run, i_step, i_crit] = int(
                        loss[i_run, i_step, i_crit] > loss[i_run, target_indx, i_crit]
                    )

        return accuracy.mean(axis=0)


class LossSurface(LossProfile):
    r"""
    Class for computing the loss surface of a model given two optimizable parameters.
    This class allows to investigate how the loss surface changes when other parameters are being perturbed.

    **Arguments / Attributes**:
        - **net** (Shell): Model to be optimized.
        - **loss_config** (LossConfig): Configuration for the loss surface computation.

    **Attributes**:
        - **param_config** (List[ParameterConfig]): Configuration for the parameters to be optimized.
        - **criteria** (List[Callable]): List of loss functions to be used.
        - **n_steps** (int): Number of steps between lower and upper bound of the parameters.
        - **n_runs** (int): Number of perturbation runs.
        - **output_dir** (str): Output directory for the loss surface plots.
        - **steps** (dict): Dictionary of steps between the lower and upper bound of the parameters.
    """

    def __init__(self, net: Shell, loss_config: LossConfig, device: str = "cpu"):

        super().__init__(net, loss_config, device)

        assert (
            len(loss_config.param_config) == 2
        ), "LossSurface supports only two optimizable parameters."
        self.param_config = loss_config.param_config

    def register_steps(self):
        self.steps_0 = self.get_steps(
            self.loss_config.param_config[0].lower_bound,
            self.loss_config.param_config[0].upper_bound,
            self.loss_config.param_config[0].n_steps,
            self.loss_config.param_config[0].scale,
        )
        self.steps_1 = self.get_steps(
            self.loss_config.param_config[1].lower_bound,
            self.loss_config.param_config[1].upper_bound,
            self.loss_config.param_config[1].n_steps,
            self.loss_config.param_config[1].scale,
        )

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        r"""
        Compute the loss surface of the model.
        """
        # first of compute the minimum loss for the given parameters
        # this assumes that the model is initialized with the target parameters
        for i_crit in range(len(self.criteria)):
            pred = self.net(input)
            current_loss = (
                self.criteria[i_crit](pred, target)
                .cpu()
                .detach()
                .numpy()
            )
            print(f"Loss for the criterion {self.criteria[i_crit].name}: {current_loss}")

        core = self.net.get_core()
        with torch.no_grad():
            steps_0 = self.steps_0
            steps_1 = self.steps_1
            for i_run in trange(self.n_runs, desc="Run"):
                # perturb the given parameter
                if self.loss_config.perturb_param:
                    new_value = self.sample_rand_param(
                        core,
                        self.get_nested_module(
                            core, self.loss_config.perturb_param
                        ).param,
                    )
                    self.set_raw_parameter(
                        self.loss_config.perturb_param,
                        new_value,
                        self.loss_config.perturb_map,
                    )

                if i_run == 0:
                    loss = np.empty(
                        (self.n_runs, len(steps_0), len(steps_1), len(self.criteria))
                    )

                for i_step_0 in range(len(steps_0)):
                    if type(self.param_config[0].lower_bound) == list:
                        # interpolate between the lower and upper bound
                        new_value = (1 - steps_0[i_step_0]) * torch.tensor(
                            self.param_config[0].lower_bound, device=self.device
                        ) + steps_0[i_step_0] * torch.tensor(
                            self.param_config[0].upper_bound, device=self.device
                        )
                    else:
                        new_value = steps_0[i_step_0]
                    self.set_raw_parameter(
                        self.param_config[0].key,
                        new_value,
                        self.param_config[0].param_map,
                        self.param_config[0].indx,
                    )
                    for i_step_1 in range(len(steps_1)):
                        if type(self.param_config[1].lower_bound) == list:
                            # interpolate between the lower and upper bound
                            new_value = (1 - steps_1[i_step_1]) * torch.tensor(
                                self.param_config[1].lower_bound, device=self.device
                            ) + steps_1[i_step_1] * torch.tensor(
                                self.param_config[1].upper_bound, device=self.device
                            )
                        else:
                            new_value = steps_1[i_step_1]
                        self.set_raw_parameter(
                            self.param_config[1].key,
                            new_value,
                            self.param_config[1].param_map,
                            self.param_config[1].indx,
                        )
                        for i_crit in range(len(self.criteria)):
                            pred = self.net(input)
                            current_loss = (
                                self.criteria[i_crit](pred, target)
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            loss[i_run, i_step_0, i_step_1, i_crit] = current_loss
                # Save the partial loss for the current run
                partial_loss = loss[i_run, ...]
                savemat(
                    f"{self.output_dir}/partial_loss_run_{i_run + 1}.mat",
                    {"loss": partial_loss, "steps_0": steps_0.cpu().numpy(), "steps_1": steps_1.cpu().numpy()},
                )
        return loss

    def plot_loss(self, loss: np.array):
        r"""
        Plot the loss surface.
        """
        if len(self.criteria) == 1:
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})
        else:
            fig, ax = plt.subplots(
                3,
                (len(self.criteria) + 2) // 3,
                figsize=(
                    (len(self.criteria) + 2) // 3 * 5,
                    15,
                ),
                subplot_kw={"projection": "3d"},
            )
        steps_0 = self.steps_0
        steps_1 = self.steps_1
        # create a meshgrid
        steps_0, steps_1 = np.meshgrid(steps_0, steps_1)
        title = ["mean", "std"]
        for i_plot in range(2):
            if len(self.criteria) == 1:
                fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})
            else:
                fig, ax = plt.subplots(
                    3,
                    (len(self.criteria) + 2) // 3,
                    figsize=(
                        (len(self.criteria) + 2) // 3 * 10,
                        30,
                    ),
                    subplot_kw={"projection": "3d"},
                )
            for i_crit in range(len(self.criteria)):
                if i_plot == 0:
                    x = loss[..., i_crit].mean(0)
                else:
                    x = loss[..., i_crit].std(0)
                criterion_name = self.criteria[i_crit].name
                if len(self.criteria) == 1:

                    ax.plot_surface(
                        steps_0,
                        steps_1,
                        x,
                        edgecolor=None,
                        alpha=0.5,
                        cmap=cm.PuBu,
                    )
                    X_min, Y_min = np.unravel_index(x.argmin(), x.shape, order="F")
                    ax.scatter(
                        steps_0[X_min, Y_min],
                        steps_1[X_min, Y_min],
                        x.argmin(),
                        marker="x",
                        label="Min Loss",
                    )
                    # ax.set_xscale(self.param_config.scale)
                    try:
                        ax.plot(
                            [
                                self.param_config[0].target_value,
                                self.param_config[0].target_value,
                            ],
                            [steps_1.min(), steps_1.max()],
                            [x.min(), x.min()],
                            color="r",
                            linestyle="--",
                            label="X Target Value",
                        )
                        ax.plot(
                            [steps_0.min(), steps_0.max()],
                            [
                                self.param_config[1].target_value,
                                self.param_config[1].target_value,
                            ],
                            [x.min(), x.min()],
                            color="r",
                            linestyle="--",
                            label="Y Target Value",
                        )
                    except:
                        pass
                    ax.set_xlabel(self.param_config[0].key)
                    ax.set_ylabel(self.param_config[1].key)
                    ax.set_zticks([])
                    ax.set_zlabel("Loss")
                    ax.legend()
                    if criterion_name:
                        ax.set_title(criterion_name)
                    fig.colorbar(
                        cm.ScalarMappable(cmap=cm.PuBu),
                        ax=ax,
                        orientation="vertical",
                        shrink=0.6,
                    )
                    ax.view_init(90, -90)
                else:
                    if (len(self.criteria) + 2) // 3 == 1:
                        ax_i = (i_crit % 3,)
                    else:
                        ax_i = (i_crit % 3, i_crit // 3)
                    ax[ax_i].plot_surface(
                        steps_0,
                        steps_1,
                        x,
                        edgecolor=None,
                        alpha=0.5,
                        cmap=cm.PuBu,
                    )
                    X_min, Y_min = np.unravel_index(x.argmin(), x.shape)
                    ax[ax_i].scatter(
                        steps_0[X_min, Y_min],
                        steps_1[X_min, Y_min],
                        x.argmin(),
                        marker="x",
                        label="Min Loss",
                    )
                    # ax.set_xscale(self.param_config.scale)
                    try:
                        ax[ax_i].plot(
                            [
                                self.param_config[0].target_value,
                                self.param_config[0].target_value,
                            ],
                            [steps_1.min(), steps_1.max()],
                            [x.min(), x.min()],
                            color="r",
                            linestyle="--",
                            label="X Target Value",
                        )
                        ax[ax_i].plot(
                            [steps_0.min(), steps_0.max()],
                            [
                                self.param_config[1].target_value,
                                self.param_config[1].target_value,
                            ],
                            [x.min(), x.min()],
                            color="r",
                            linestyle="--",
                            label="Y Target Value",
                        )
                    except:
                        pass
                    ax[ax_i].set_xlabel(self.param_config[0].key)
                    ax[ax_i].set_ylabel(self.param_config[1].key)
                    ax[ax_i].set_zlabel("Loss")
                    ax[ax_i].set_zticks([])
                    ax[ax_i].legend()
                    if criterion_name:
                        ax[ax_i].set_title(criterion_name)

                    ax[ax_i].view_init(90, -90)
            try:
                for i_ax in range(i_crit + 1, math.prod(list(ax.shape))):
                    if (len(self.criteria) + 2) // 3 == 1:
                        ax_i = (i_ax % 3,)
                    else:
                        ax_i = (i_ax % 3, i_ax // 3)
                    fig.delaxes(ax[ax_i])
            except:
                pass
            plt.tight_layout()
            fig.colorbar(
                cm.ScalarMappable(cmap=cm.PuBu),
                ax=ax,
                orientation="vertical",
                shrink=0.5,
            )
            plt.savefig(
                f"{self.output_dir}/{title[i_plot]}_{self.param_config[0].key}_{self.param_config[1].key}.png"
            )
            plt.clf()
        return fig, ax

    def compute_accuracy(self, loss):
        r"""
        Compute the accuracy of the losses according to
        Joseph Turian and Max Henry, “I'm sorry for your loss: Spectrally-based
        audio distances are bad at pitch,” arXiv preprint arXiv:2012.04572, 2020.
        """
        steps_0 = self.steps_0
        steps_1 = self.steps_1
        # find the index in steps of the element closest to the target value
        target_indx_0 = np.abs(steps_0 - self.param_config[0].target_value).argmin()
        target_indx_1 = np.abs(steps_1 - self.param_config[1].target_value).argmin()

        accuracy = np.empty(loss.shape)
        for i_crit in range(len(self.criteria)):
            for i_run in range(loss.shape[0]):
                for i_step_0 in range(len(steps_0)):
                    for i_step_1 in range(len(steps_1)):
                        accuracy[i_run, i_step_0, i_step_1, i_crit] = int(
                            loss[i_run, i_step_0, i_step_1, i_crit]
                            > loss[i_run, target_indx_0, target_indx_1, i_crit]
                        )

        return accuracy.mean(axis=0)
