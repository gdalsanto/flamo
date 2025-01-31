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
    target_value: float | List[float] | List[List[float]] = None
    scale: str = "linear"
    n_steps: int = (
        None  # number of steps between lower and upper bound of the parameter
    )


class LossConfig(BaseModel):
    criteria: List[Callable] = None  # loss function to be used
    param_config: List[ParameterConfig] = None
    perturb_dict: str = None  # key of the parameter to be perturbed
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
                new_value = self.sample_rand_param(
                    core,
                    self.get_nested_module(core, self.loss_config.perturb_dict).param,
                )
                self.set_raw_parameter(
                    self.loss_config.perturb_dict,
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
                        loss[i_run, i_step, i_crit] = self.criteria[i_crit](
                            pred, target
                        ).detach().numpy()

        return loss

    def plot_loss(self, loss: np.ndarray, criterion_name: List[str] = None):
        r"""
        Plot the loss profile.
        """
        fig, ax = plt.subplots(
            1, len(self.criteria), figsize=(len(self.criteria) * 5, 5)
        )
        steps = self.steps[self.param_config.key]
        for i_crit in range(len(self.criteria)):
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

                ax[i_crit].plot(steps, mean_loss, label=criterion_name[i_crit])
                ax[i_crit].plot(
                    steps[mean_loss.argmin()],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                ax.fill_between(
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
                    ax[i_crit].set_title(criterion_name[i_crit])

        plt.savefig(f"{self.output_dir}/{self.param_config.key}.png")

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
            steps = torch.logspace(torch.log10(torch.tensor(lower_bound)), torch.log10(torch.tensor(upper_bound)), n_steps)
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
                new_value = self.sample_rand_param(
                    core,
                    self.get_nested_module(core, self.loss_config.perturb_dict).param,
                )
                self.set_raw_parameter(
                    self.loss_config.perturb_dict,
                    new_value,
                    self.loss_config.perturb_map,
                )

                if i_run == 0: 
                    loss = np.empty((self.n_runs, len(steps_0), len(steps_1), len(self.criteria)))

                for i_step_0 in range(len(steps_0)):
                    if type(self.param_config[0].lower_bound) == list:
                        # interpolate between the lower and upper bound
                        new_value = (1 - steps_0[i_step_0]) * torch.tensor(
                            self.param_config[0].lower_bound
                        ) + steps_0[i_step_0] * torch.tensor(self.param_config[0].upper_bound)
                    else:
                        new_value = steps_0[i_step_0]
                    self.set_raw_parameter(
                        self.param_config[0].key, new_value, self.param_config[0].param_map
                    )
                    for i_step_1 in range(len(steps_1)):
                        if type(self.param_config[1].lower_bound) == list:
                            # interpolate between the lower and upper bound
                            new_value = (1 - steps_1[i_step_1]) * torch.tensor(
                                self.param_config[1].lower_bound
                            ) + steps_1[i_step_1] * torch.tensor(self.param_config[1].upper_bound)
                        else:
                            new_value = steps_1[i_step_1]
                        self.set_raw_parameter(
                            self.param_config[1].key, new_value, self.param_config[1].param_map
                        )
                        for i_crit in range(len(self.criteria)):
                            pred = self.net(input)
                            current_loss = self.criteria[i_crit](pred, target).detach().numpy()
                            loss[i_run, i_step_0, i_step_1, i_crit] = current_loss

        return loss

    def plot_loss(self, loss: dict, criterion_name: List[str] = None):
        r"""
        Plot the loss surface.
        """
        fig, ax = plt.subplots(
            1, len(self.criteria), figsize=(len(self.criteria) * 5, 5), 
            subplot_kw={"projection": "3d"}
        )
        for i_crit in range(len(self.criteria)):
            
            mean_loss = loss[..., i_crit].mean(0)
            std_loss = loss[..., i_crit].std(0)
            
            X = self.steps[self.param_config[0].key]
            Y = self.steps[self.param_config[1].key]
            X, Y = np.meshgrid(X, Y, indexing='ij')
            if len(self.criteria) == 1:
                
                ax.plot_surface(X, Y, mean_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
                ax.plot_surface(X, Y, mean_loss - std_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.2)
                ax.plot_surface(X, Y, mean_loss + std_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.2)
                X_min, Y_min = np.unravel_index(mean_loss.argmin(), mean_loss.shape)
                ax.plot(
                    X[X_min, Y_min],
                    Y[X_min, Y_min],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                try:
                    ax.plot_surface(
                        self.param_config[0].target_value*np.ones_like(X),
                        Y, 
                        mean_loss + std_loss,
                        color="k",
                        alpha=0.2,
                        label="Target Value",
                    )
                    ax.plot_surface(
                        X, 
                        self.param_config[1].target_value*np.ones_like(Y),
                        mean_loss + std_loss,
                        color="k",
                        alpha=0.2,
                        label="Target Value",
                    )
                except:
                    pass
                ax.set_xlabel(self.param_config[0].key)
                ax.set_ylabel(self.param_config[1].key)
                ax.set_zlabel("Loss")
                ax.legend()
                if criterion_name:
                    ax.set_title(criterion_name)
            else:

                ax[i_crit].plot_surface(X, Y, mean_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
                ax[i_crit].plot_surface(X, Y, mean_loss - std_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.2)
                ax[i_crit].plot_surface(X, Y, mean_loss + std_loss, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.2)

                X_min, Y_min = np.unravel_index(mean_loss.argmin(), mean_loss.shape)
                ax.plot(
                    X[X_min, Y_min],
                    Y[X_min, Y_min],
                    mean_loss.min(),
                    marker="x",
                    label="Min Loss",
                )
                try:
                    ax[i_crit].plot_surface(
                        self.param_config[0].target_value*np.ones_like(X),
                        Y, 
                        mean_loss + std_loss,
                        color="k",
                        alpha=0.2,
                        label="Target Value",
                    )
                    ax[i_crit].plot_surface(
                        X, 
                        self.param_config[1].target_value*np.ones_like(Y),
                        mean_loss + std_loss,
                        color="k",
                        alpha=0.2,
                        label="Target Value",
                    )
                except:
                    pass
                ax[i_crit].set_xlabel(self.param_config[0].key)
                ax[i_crit].set_ylabel(self.param_config[1].key)
                ax[i_crit].set_zlabel("Loss")
                ax[i_crit].legend()
                if criterion_name:
                    ax[i_crit].set_title(criterion_name[i_crit])

        plt.savefig(f"{self.output_dir}/{self.param_config[0].key}-{self.param_config[1].key}.png")