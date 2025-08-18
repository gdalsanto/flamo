import torch
import os
import time
import numpy as np
import argparse
import yaml
from flamo.auxiliary.reverb import HomogeneousFDN, map_gamma, inverse_map_gamma
from flamo.auxiliary.config.config import HomogeneousFDNConfig
from flamo.optimize.loss import mse_loss, mel_mss_loss
from flamo.optimize.surface import LossProfile, LossConfig, ParameterConfig, LossSurface
from flamo.functional import signal_gallery, get_magnitude

def example_loss_profile(args):
    """
    Investigate the loss profile at different values of the attenuation
    """
    # create a homogeneous FDN model
    fdn_config = HomogeneousFDNConfig()
    FDN = HomogeneousFDN(fdn_config)
    FDN.set_model(output_layer=get_magnitude)
    inverse_gamma = inverse_map_gamma(torch.tensor(FDN.delays))
    # assign the target value to the attenuation parameter
    target_gamma = 0.995
    FDN.model.get_core().feedback_loop.feedforward.attenuation.assign_value(
        inverse_gamma(target_gamma)
    )

    # homogenous attenuation parameter 
    attenuation_config = ParameterConfig(
        key="feedback_loop.feedforward.attenuation",
        param_map=lambda x: inverse_gamma(x),
        lower_bound=0.9,
        upper_bound=0.999,
        target_value=target_gamma,
        scale="log",
        n_steps=100,
    )

    # define config structures
    loss_config = LossConfig(
        criteria=[mse_loss()],
        param_config=[attenuation_config],
        perturb_param="output_gain",
        perturb_map=lambda x: x,
        n_runs=10,
        output_dir=args.output_dir,
    )

    # save config file
    output_file = os.path.join(args.output_dir, "config.yml")
    # write the configuration to a YAML file
    with open(output_file, "w") as file:
        yaml.dump(loss_config, file)

    input_signal = signal_gallery(
        signal_type="impulse", batch_size=1, n=1, n_samples=fdn_config.nfft
    )
    target_signal = FDN.model(input_signal)

    loss_profile = LossProfile(FDN.model, loss_config)
    loss = loss_profile.compute_loss(input_signal, target_signal)
    loss_profile.plot_loss(loss)


def example_loss_surface(args):

    # create a homogeneous FDN model
    fdn_config = HomogeneousFDNConfig()
    FDN = HomogeneousFDN(fdn_config)
    FDN.set_model(output_layer=get_magnitude)
    inverse_gamma = inverse_map_gamma(torch.tensor(FDN.delays))

    # assign the target value to the attenuation parameter
    target_gamma = 0.92
    FDN.model.get_core().feedback_loop.feedforward.attenuation.assign_value(
        inverse_gamma(target_gamma)
    )

    # homogenous attenuation parameter 
    attenuation_config = ParameterConfig(
        key="feedback_loop.feedforward.attenuation",
        param_map=lambda x: inverse_gamma(x),
        lower_bound=0.9,
        upper_bound=0.999,
        target_value=target_gamma,
        scale="log",
        n_steps=25,
    )

    # assign the target value to the input gain parameter
    target_input_gain = torch.randn((FDN.N, 1))
    FDN.model.get_core().input_gain.assign_value(target_input_gain)

    input_gain_config = ParameterConfig(
        key="input_gain",
        param_map=lambda x: x,
        lower_bound=(target_input_gain - 0.5*target_input_gain).tolist(),
        upper_bound=(target_input_gain + 0.5*target_input_gain).tolist(),
        target_value=0.5,
        scale="linear",
        n_steps=25,
    )

    # define config structures
    loss_config = LossConfig(
        criteria=[mse_loss(), mel_mss_loss()],
        param_config=[attenuation_config, input_gain_config],
        # perturb_param=None,#"output_gain",
        perturb_map=lambda x: x,
        n_runs=1,
        output_dir=args.output_dir,
    )

    # save config file
    output_file = os.path.join(args.output_dir, "config.yml")
    # write the configuration to a YAML file
    with open(output_file, "w") as file:
        yaml.dump(loss_config, file)

    input_signal = signal_gallery(
        signal_type="impulse", batch_size=1, n=1, n_samples=fdn_config.nfft
    )
    target_signal = FDN.model(input_signal)

    loss_profile = LossSurface(FDN.model, loss_config)
    loss = loss_profile.compute_loss(input_signal, target_signal)
    loss_profile.plot_loss(loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument("--output_dir", type=str, help="directory to save loss plots")

    args = parser.parse_args()

    # check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # make output directory if it doesn't exist
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    else:
        args.output_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.output_dir)

    # example_loss_profile(args)
    example_loss_surface(args)
