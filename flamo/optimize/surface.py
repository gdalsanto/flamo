import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
from flamo.processor.system import Shell
from pydantic import BaseModel 
from typing import List, Callable, Optional

class ParameterConfig(BaseModel):
    key: str = None               # key of the parameter in the model
    param_map: Callable = None    # mapping function for the parameters in param_dict
    lower_bound: float = None     
    upper_bound: float = None     
    target_value: float = None   
    scale: str = 'linear'

class LossConfig(BaseModel):
    criteria: List[Callable] = None     # loss function to be used
    param_config: ParameterConfig = None     
    perturb_dict: str = None            # key of the parameter to be perturbed
    perturb_map: Callable = lambda x: x        # mapping function for the perturbation parameter
    n_steps: int = None                 # number of steps between lower and upper bound of the parameter
    n_runs: int = None                  # number of perturbation runs
    output_dir: str = None


class LossProfile():
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
    def __init__(
            self, 
            net: Shell, 
            loss_config: LossConfig):
        
        self.net = net
        self.loss_config = loss_config  
        self.param_config = loss_config.param_config
        self.criteria = loss_config.criteria
        self.n_steps = loss_config.n_steps
        self.n_runs = loss_config.n_runs
        self.output_dir = loss_config.output_dir

        self.set_steps()

    def compute_loss_profile(
            self,
            input: torch.Tensor, 
            target: torch.Tensor):
        r"""
        Compute the loss profile of the model.
        """
        loss = torch.empty(self.n_runs, self.n_steps, len(self.criteria))
        core = self.net.get_core()
        with torch.no_grad():
            for i_run in trange(self.n_runs, desc='Run'):
                # perturb the given parameter
                new_value = self.sample_rand_param(core, self.get_nested_module(core, self.loss_config.perturb_dict).param)
                self.set_raw_parameter(self.loss_config.perturb_dict, new_value, self.loss_config.perturb_map)

                for i_step in range(self.n_steps):
                    self.set_raw_parameter(self.param_config.key, self.steps[i_step], self.param_config.param_map)

                    for i_crit in range(len(self.criteria)):
                        pred = self.net(input)
                        loss[i_run, i_step, i_crit] = self.criteria[i_crit](pred, target)
            
        return loss

    def plot_loss_profile(self, loss: np.ndarray, criterion_name: List[str] = None):
        r"""
        Plot the loss profile.
        """
        fig, ax = plt.subplots(1, len(self.criteria), figsize=(len(self.criteria)*5, 5))
        for i_crit in range(len(self.criteria)):
            if len(self.criteria) == 1:
                mean_loss = loss[:, :, i_crit].mean(0)
                std_loss = loss[:, :, i_crit].std(0)

                ax.plot(self.steps, mean_loss, label=criterion_name)
                ax.plot(self.steps[mean_loss.argmin()], mean_loss.min(), marker='x', label='Min Loss')
                ax.fill_between(self.steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
                ax.axvline(x=self.param_config.target_value, color='r', linestyle='--', label='Target Value')

                ax.set_xlabel(self.param_config.key)
                ax.set_ylabel('Loss')
                ax.legend()
                if criterion_name:
                    ax.set_title(criterion_name)
            else:
                mean_loss = loss[:, :, i_crit].mean(0)
                std_loss = loss[:, :, i_crit].std(0)

                ax[i_crit].plot(self.steps, mean_loss, label=criterion_name[i_crit])
                ax[i_crit].plot(self.steps[mean_loss.argmin()], mean_loss.min(), marker='x', label='Min Loss')
                ax.fill_between(self.steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
                ax[i_crit].axvline(x=self.param_config.target_value, color='r', linestyle='--', label='Target Value')

                ax[i_crit].set_xlabel(self.param_config.key)
                ax[i_crit].set_ylabel('Loss')
                ax[i_crit].legend()
                if criterion_name:
                    ax[i_crit].set_title(criterion_name[i_crit])
            
        plt.savefig(f"{self.output_dir}/{self.param_config.key}.png")

    def set_steps(self):
        r"""
        Generate a list of steps between the lower and upper bound of the parameter.
        """
        if self.param_config.scale == 'linear':
            self.steps = torch.linspace(
                self.param_config.lower_bound,
                self.param_config.upper_bound, 
                self.n_steps)
        elif self.param_config.scale == 'log':
            self.steps = torch.logspace(
                np.log10(self.param_config.lower_bound),
                np.log10(self.param_config.upper_bound), 
                self.n_steps)
        else:
            raise ValueError("Scale must be either 'linear' or 'log'")

    def set_raw_parameter(self, param_key: str, new_value: torch.Tensor, map: lambda x: x):

        core = self.net.get_core()
        self.keys = self.get_modules_keys(core)

        for key in self.keys:
            if key == param_key:    
                module = self.get_nested_module(core, key)
                module.assign_value(map(new_value))

    def get_modules_keys(self, module, prefix=''):
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
        keys = key.split('.')
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
        rand_param = torch.randn_like(ref_param)*std + mean
        return rand_param