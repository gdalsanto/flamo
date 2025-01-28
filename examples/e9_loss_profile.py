import torch
import os
import yaml 
from flamo.auxiliary.reverb import HomogeneousFDN, map_gamma, inverse_map_gamma
from flamo.auxiliary.config.config import HomogeneousFDNConfig
from flamo.optimize.loss import mse_loss
from flamo.optimize.surface import LossProfile, LossConfig, ParameterConfig
from flamo.functional import signal_gallery, get_magnitude

target_gamma = 0.99
fdn_config = HomogeneousFDNConfig()
FDN = HomogeneousFDN(fdn_config)
FDN.set_model(output_layer=get_magnitude)
inverse_gamma = inverse_map_gamma(torch.tensor(FDN.delays))
FDN.model.get_core().feedback_loop.feedforward.attenuation.assign_value(inverse_gamma(target_gamma))



loss_config = LossConfig(
    criteria=[mse_loss()],
    param_config=ParameterConfig(
            key="feedback_loop.feedforward.attenuation",
            param_map=lambda x: inverse_gamma(x),
            lower_bound=0.9,
            upper_bound=0.999,
            target_value=target_gamma,
            scale='log'
        ),
    perturb_dict='input_gain',
    perturb_map=lambda x: x,
    n_steps=100,
    n_runs=10,
    output_dir="output/loss-surface-test"
)

# make output directory if it doesn't exist
if not os.path.exists(loss_config.output_dir):
    os.makedirs(loss_config.output_dir)


output_file = os.path.join(loss_config.output_dir, 'config.yml')
# write the configuration to a YAML file
with open(output_file, 'w') as file:
    yaml.dump(loss_config, file)

loss_profile = LossProfile(FDN.model, loss_config)
input_signal = signal_gallery(signal_type="impulse", batch_size=1, n=1, n_samples=fdn_config.nfft)
target_signal = FDN.model(input_signal)

loss = loss_profile.compute_loss_profile(input_signal, target_signal)

loss_profile.plot_loss_profile(loss.detach().numpy(), criterion_name=["MSE"])