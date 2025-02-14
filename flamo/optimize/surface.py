import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange
from flamo.processor.system import Shell
from pydantic import BaseModel
from typing import List, Callable, Optional


class ParameterConfig(BaseModel):
    key: str = None  # key of the parameter in the model
    param_map: Callable = None  # mapping function for the parameters in param_dict
    lower_bound: float | List[float] | List[List[float]] = None
    upper_bound: float | List[float] | List[List[float]] = None
    target_value: float = None
    scale: str = "linear"
    n_steps: int = (
        None  # number of steps between lower and upper bound of the parameter
    )


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

    def __init__(self, net: Shell, loss_config: LossConfig):

        super().__init__()
        self.net = net
        self.loss_config = loss_config
        self.param_config = loss_config.param_config[0]
        self.criteria = loss_config.criteria
        self.n_runs = loss_config.n_runs
        self.output_dir = loss_config.output_dir

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        r"""
        Compute the loss profile of the model.
        """
        self.steps = dict.fromkeys([self.param_config.key])
        core = self.net.get_core()
        with torch.no_grad():

            # set steps for the parameter
            steps = self.get_steps(
                self.param_config.lower_bound,
                self.param_config.upper_bound,
                self.param_config.n_steps,
                self.param_config.scale,
            )
            self.steps[self.param_config.key] = steps

            for i_run in trange(self.n_runs, desc="Run"):
                # perturb the given parameter
                if self.loss_config.perturb_param:
                    new_value = self.sample_rand_param(
                        core,
                        self.get_nested_module(core, self.loss_config.perturb_param).param,
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
                            self.param_config.lower_bound
                        ) + steps[i_step] * torch.tensor(self.param_config.upper_bound)
                    else:
                        new_value = steps[i_step]
                    self.set_raw_parameter(
                        self.param_config.key, new_value, self.param_config.param_map
                    )

                    for i_crit in range(len(self.criteria)):
                        pred = self.net(input)
                        loss[i_run, i_step, i_crit] = (
                            self.criteria[i_crit](pred, target).detach().numpy()
                        )

        return loss

    def plot_loss(self, loss: np.ndarray):
        r"""
        Plot the loss profile.
        """
        fig, ax = plt.subplots(
            3, (len(self.criteria) + 2) // 3, figsize=((len(self.criteria) + 2) // 3 * 5, 15)
        )
        steps = self.steps[self.param_config.key]
        for i_crit in range(len(self.criteria)):
            criterion_name = self.criteria[i_crit].name
            if len(self.criteria) == 1:
                mean_loss = loss[:, :, i_crit].mean(0)
                std_loss = loss[:, :, i_crit].std(0)

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
                mean_loss = loss[:, :, i_crit].mean(0)
                std_loss = loss[:, :, i_crit].std(0)

                ax[i_crit].plot(steps, mean_loss, label=criterion_name)
                ax[i_crit].plot(
                    steps[mean_loss.argmin()],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                ax[i_crit].set_xscale(self.param_config.scale)
                ax[i_crit].fill_between(
                    steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2
                )
                try:
                    ax[i_crit].axvline(
                        x=self.param_config.target_value,
                        color="r",
                        linestyle="--",
                        label="Target Value",
                    )
                except:
                    pass
                ax[i_crit].set_xlabel(self.param_config.key)
                ax[i_crit].set_ylabel("Loss")
                ax[i_crit].legend()
                if criterion_name:
                    ax[i_crit].set_title(criterion_name)

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
                torch.log10(torch.tensor(lower_bound)),
                torch.log10(torch.tensor(upper_bound)),
                n_steps,
            )
        else:
            raise ValueError("Scale must be either 'linear' or 'log'")
        return steps

    def set_raw_parameter(
        self, param_key: str, new_value: torch.Tensor, map: lambda x: x
    ):

        core = self.net.get_core()
        self.keys = self.get_modules_keys(core)

        for key in self.keys:
            if key == param_key:
                module = self.get_nested_module(core, key)
                module.assign_value(map(new_value))

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

    def __init__(self, net: Shell, loss_config: LossConfig):

        super().__init__(net, loss_config)

        assert (
            len(loss_config.param_config) == 2
        ), "LossSurface supports only two optimizable parameters."
        self.param_config = loss_config.param_config

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        r"""
        Compute the loss surface of the model.
        """
        self.steps = dict.fromkeys([self.param_config[0].key, self.param_config[1].key])
        core = self.net.get_core()
        with torch.no_grad():

            # set steps for the parameter
            steps_0 = self.get_steps(
                self.param_config[0].lower_bound,
                self.param_config[0].upper_bound,
                self.param_config[0].n_steps,
                self.param_config[0].scale,
            )
            steps_1 = self.get_steps(
                self.param_config[1].lower_bound,
                self.param_config[1].upper_bound,
                self.param_config[1].n_steps,
                self.param_config[1].scale,
            )
            self.steps[self.param_config[0].key] = steps_0
            self.steps[self.param_config[1].key] = steps_1

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
                            self.param_config[0].lower_bound
                        ) + steps_0[i_step_0] * torch.tensor(
                            self.param_config[0].upper_bound
                        )
                    else:
                        new_value = steps_0[i_step_0]
                    self.set_raw_parameter(
                        self.param_config[0].key,
                        new_value,
                        self.param_config[0].param_map,
                    )
                    for i_step_1 in range(len(steps_1)):
                        if type(self.param_config[1].lower_bound) == list:
                            # interpolate between the lower and upper bound
                            new_value = (1 - steps_1[i_step_1]) * torch.tensor(
                                self.param_config[1].lower_bound
                            ) + steps_1[i_step_1] * torch.tensor(
                                self.param_config[1].upper_bound
                            )
                        else:
                            new_value = steps_1[i_step_1]
                        self.set_raw_parameter(
                            self.param_config[1].key,
                            new_value,
                            self.param_config[1].param_map,
                        )
                        for i_crit in range(len(self.criteria)):
                            pred = self.net(input)
                            current_loss = (
                                self.criteria[i_crit](pred, target).detach().numpy()
                            )
                            loss[i_run, i_step_0, i_step_1, i_crit] = current_loss

        return loss

    def plot_loss(self, loss: dict):
        r"""
        Plot the loss surface.
        """
        for i_crit in range(len(self.criteria)):
            criterion_name = self.criteria[i_crit].name
            fig, ax = plt.subplots(
                1,
                2,
                figsize=(10, 5),
            )
            mean_loss = loss[..., i_crit].mean(0)
            std_loss = loss[..., i_crit].std(0)

            X = self.steps[self.param_config[0].key]
            Y = self.steps[self.param_config[1].key]
            # X, Y = np.meshgrid(X, Y, indexing="ij")

            # mean
            ax[0].set_title(f"Mean Loss - {criterion_name[i_crit]}")
            ax[0].imshow(
                mean_loss,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin="lower",
                aspect="auto",
                cmap=cm.PuBu,
            )
            # ax[0].set_xscale(self.param_config[0].scale)
            # ax[0].set_yscale(self.param_config[1].scale)
            X_min, Y_min = np.unravel_index(
                mean_loss.argmin(), mean_loss.shape, order="F"
            )
            ax[0].plot(
                X[X_min],
                Y[Y_min],
                marker="x",
                label="Min Loss",
            )
            ax[0].axvline(
                x=self.param_config[0].target_value,
                color="r",
                linestyle="--",
                label="X Target Value",
            )
            ax[0].axhline(
                y=self.param_config[1].target_value,
                color="r",
                linestyle="--",
                label="Y Target Value",
            )
            ax[0].set_xlabel(self.param_config[0].key)
            ax[0].set_ylabel(self.param_config[1].key)
            fig.colorbar(
                cm.ScalarMappable(cmap=cm.PuBu),
                ax=ax[0],
                orientation="vertical",
            )

            # standard deviation
            ax[1].set_title(f"Std Loss - {criterion_name[i_crit]}")
            ax[1].imshow(
                std_loss,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin="lower",
                aspect="auto",
                cmap=cm.PuBu,
            )
            # ax[1].set_xscale(self.param_config[0].scale)
            # ax[1].set_yscale(self.param_config[1].scale)
            ax[1].set_xlabel(self.param_config[0].key)
            ax[1].set_ylabel(self.param_config[1].key)
            fig.colorbar(
                cm.ScalarMappable(cmap=cm.PuBu),
                ax=ax[1],
                orientation="vertical",
            )
            plt.savefig(
                f"{self.output_dir}/{criterion_name[i_crit]}_{self.param_config[0].key}_{self.param_config[1].key}.png"
            )
        return fig, ax