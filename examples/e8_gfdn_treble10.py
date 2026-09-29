from typing import Optional
import torch
import torch.nn as nn
import argparse
import os
import time
import auraloss
import numpy as np
import multislope
import pyfar as pf
import pyrato
import matplotlib.pyplot as plt

from multislope.plotting import edc_mse, plot_fit
from collections import OrderedDict

from flamo.auxiliary.reverb import (
    parallelGFDNFirstOrderShelving, 
    parallelGFDNGEQ)
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.loss import sparsity_loss, edr_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.utils import save_audio
from flamo.functional import signal_gallery, find_onset

torch.manual_seed(130798)


def estimate_band_decay_times(
    rir: torch.Tensor, fs: int, n_slopes: int
) -> tuple[np.ndarray, np.ndarray, multislope.DecayFit]:
    """
    Estimate per-octave-band multi-exponential decay times of a RIR with the
    multislope library's DecayFitNet.

    Args:
        rir: 1D RIR tensor, onset-aligned (no leading silence).
        fs: sample rate in Hz.
        n_slopes: number of decay slopes to fit per band.
    Returns:
        ``(band_freqs, t, fit)`` where ``band_freqs`` are the octave-band centre
        frequencies (Hz), ``t`` has shape ``(n_bands, n_slopes)`` -- the
        fitted decay times (T60, in seconds) per band, sorted ascending
        along the slopes axis -- and ``fit`` is the full DecayFitNet result.
    """
    x = rir.detach().cpu().numpy().astype(np.float64).flatten()
    net = multislope.DecayFitNet(n_slopes=n_slopes, sample_rate=fs)
    fit = net.estimate(x, analyse_full_rir=True)
    return np.array(fit.frequencies), np.sort(fit.t, axis=-1), fit


def compare_fitted_edcs(
    rir: torch.Tensor, fit: multislope.DecayFit, fs: int, plot_path: Optional[str] = None
) -> np.ndarray:
    """
    Compare the EDCs reconstructed from DecayFitNet's estimated parameters
    against the per-band Schroeder EDCs computed directly from the RIR.
    The last 5% of each EDC is discarded, where octave-filtering edge
    effects dominate.

    Args:
        rir: 1D RIR tensor, the same one ``fit`` was estimated from.
        fit: DecayFitNet result from :func:`estimate_band_decay_times`.
        fs: sample rate in Hz.
        plot_path: if given, save a plot of the measured vs. fitted EDCs there.
    Returns:
        Array of shape ``(n_bands,)`` -- the MSE (dB) between the measured
        and fitted EDC in each band.
    """
    x = rir.detach().cpu().numpy().astype(np.float64).flatten()
    preprocess = multislope.PreprocessRIR(sample_rate=fs, filter_frequencies=fit.frequencies)
    true_edc = preprocess.schroeder(x, analyse_full_rir=True)[0][0]  # (n_bands, n_samples)
    time_axis = np.arange(true_edc.shape[-1]) / fs
    fitted_edc = fit.edc(time_axis)

    mse = edc_mse(
        multislope.discard_last_n_percent(true_edc, 5),
        multislope.discard_last_n_percent(fitted_edc, 5),
    )

    if plot_path is not None:
        ax = plot_fit(fit, measured_edc=true_edc, sample_rate=fs, title="Target RIR: measured vs. fitted EDC")
        ax.figure.savefig(plot_path, bbox_inches="tight")
        plt.close(ax.figure)
    return mse


def shelving_group_rt60(attenuation: parallelGFDNFirstOrderShelving, freqs_hz: np.ndarray) -> np.ndarray:
    """
    Evaluate the frequency-dependent RT60 implied by each group of a trained
    :class:`parallelGFDNFirstOrderShelving` filter.

    Args:
        attenuation: a ``parallelGFDNFirstOrderShelving`` instance.
        freqs_hz: frequencies (Hz) at which to evaluate the RT60.
    Returns:
        Array of shape ``(n_groups, len(freqs_hz))``.
    """
    with torch.no_grad():
        H = attenuation.freq_response(attenuation.param)[0]  # (n_bins, n_delays)
    mag_db = (20 * torch.log10(torch.abs(H))).cpu().numpy()
    bin_freqs = np.linspace(0, attenuation.fs / 2, mag_db.shape[0])

    delays = attenuation.delays.cpu().numpy().reshape(attenuation.n_groups, attenuation.group_size)

    rt60 = np.zeros((attenuation.n_groups, len(freqs_hz)))
    for g in range(attenuation.n_groups):
        delay_len = delays[g, 0]
        gain_db = np.interp(freqs_hz, bin_freqs, mag_db[:, g * attenuation.group_size])
        slope_db_per_sample = gain_db / delay_len
        rt60[g] = -60 / (slope_db_per_sample * attenuation.fs)
    return rt60


def extend_noise(noise: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Extend a noise segment to ``n_samples`` via phase randomization: the
    segment's magnitude spectrum is interpolated onto the frequency grid of
    the target length and combined with uniformly random phases. The result
    is rescaled to the segment's mean power.

    Args:
        noise: 1D noise segment.
        n_samples: length of the extended noise.
    Returns:
        1D array of length ``n_samples``.
    """
    mag = np.abs(np.fft.rfft(noise))
    freqs = np.fft.rfftfreq(len(noise))
    new_freqs = np.fft.rfftfreq(n_samples)
    new_mag = np.interp(new_freqs, freqs, mag)
    phase = np.random.uniform(-np.pi, np.pi, len(new_freqs))
    phase[0] = 0
    if n_samples % 2 == 0:
        phase[-1] = 0  # Nyquist bin must be real
    out = np.fft.irfft(new_mag * np.exp(1j * phase), n=n_samples)
    return out * np.sqrt(np.mean(noise**2) / np.mean(out**2))


def load_treble10_rir(path: str, mic_idx: int = 0) -> tuple[np.ndarray, int]:
    """
    Load the omnidirectional RIR of one receiver from a Treble10 ``.npz`` file
    produced by treble10-analyzer's ``rir_hoa.py``. The file stores ``rir`` as
    ``(n_mics * n_samples, n_hoa)``, with the receivers stacked along time and
    the channels in ACN/SN3D order, so the omnidirectional (W) channel is
    channel 0.

    Args:
        path: path to a ``.npz`` file, or to a directory, in which case the
            first ``.npz`` file (in sorted order) is used.
        mic_idx: index of the receiver to take.
    Returns:
        ``(rir, fs)`` where ``rir`` is the 1D omnidirectional RIR.
    """
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
        assert len(files) > 0, f"no .npz files found in {path}"
        path = os.path.join(path, files[0])
    print(f"Loading target RIR from {path} (receiver {mic_idx})")
    data = np.load(path)
    fs = int(data["fs"])
    n_mics = data["posMic"].shape[0]
    assert 0 <= mic_idx < n_mics, f"mic_idx must be in [0, {n_mics}), got {mic_idx}"
    rir = data["rir"].reshape(n_mics, -1, data["rir"].shape[-1])
    return rir[mic_idx, :, 0], fs


def extract_background_noise(rir: torch.Tensor, fs: int) -> tuple[torch.Tensor, int]:
    """
    Estimate the background noise of a RIR: the intersection time between
    the decay and the noise floor is detected with pyrato's Lundeby method,
    the RIR tail from that point on is taken as the noise segment, and the
    segment is extended to the full RIR length via :func:`extend_noise`.

    Args:
        rir: 1D RIR tensor, onset-aligned (no leading silence).
        fs: sample rate in Hz.
    Returns:
        ``(noise, start)`` where ``noise`` is a 1D tensor with the same length,
        dtype and device as ``rir``, and ``start`` is the sample index where
        the noise segment begins.
    """
    x = rir.detach().cpu().numpy().astype(np.float64).flatten()
    intersection_time = pyrato.intersection_time_lundeby(pf.Signal(x, fs), freq="broadband")[0]
    start = int(np.round(np.squeeze(intersection_time) * fs))
    if start >= int(len(x) * 0.99):
        start = int(len(x) * 0.99)  # noise floor beyond the RIR: take the last 1%
    noise = extend_noise(x[start:], len(x))
    return torch.as_tensor(noise, dtype=rir.dtype, device=rir.device), start


class NoisyLoss(nn.Module):
    """
    Wrap a criterion so that a fixed noise term is summed to the model output
    before the loss is computed. This lets a noise-free model (e.g. a GFDN)
    be fit to a target RIR with a background noise floor, without the loss
    penalizing the missing noise in the tail of the EDC/EDR.

    Args:
        criterion: the loss to wrap, called as ``criterion(y_pred, y_true)``.
        noise: noise term of shape ``(1, n_samples, n_channels)``, broadcast
            over the batch.
    """

    def __init__(self, criterion: nn.Module, noise: torch.Tensor):
        super().__init__()
        self.criterion = criterion
        self.register_buffer("noise", noise)

    def forward(self, y_pred, y_true):
        return self.criterion(y_pred + self.noise.to(y_pred.dtype), y_true)


class MultiResoSTFT(nn.Module):
    """compute the mean absolute error between the auraloss of two RIRs"""

    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir1, rir2):
        return self.MRstft(rir1.permute(0, 2, 1), rir2.permute(0, 2, 1))


class GroupedFDN(system.Shell):
    """
    Grouped Feedback Delay Network (FDN), after Das & Abel [1]_.

    The ``n_groups * group_size`` delay lines are split into ``n_groups``
    groups. Within a group the lines are mixed by an orthogonal matrix built
    from that group's entry in ``mixing_angles``; the groups are then coupled
    to one another according to ``coupling_angles``. Delay lengths and the
    feedback (mixing + coupling) matrix are fixed. Only the input/output
    gains and the two parameters of the shared first-order shelving
    attenuation filter (the DC-frequency reverberation time and the shelving
    crossover) are learnable.

    ``group_size`` must be a power of two: the intra-group mixing matrix is
    built by repeated Kronecker self-products of a 2x2 rotation.

    .. [1] Das, O., & Abel, J. S. (2021). Grouped feedback delay networks for
       modeling of coupled spaces. J. Audio Eng. Soc, 69(7/8), 486-496.
    """

    def __init__(
        self,
        nfft: int,
        fs: int,
        in_ch: int,
        out_ch: int,
        group_size: int,
        n_groups: int,
        delay_lengths: torch.Tensor | list[int],
        rt_dc: Optional[float] = 1.0,
        rt_nyquist: Optional[float] = 0.2,
        crossover_freq: Optional[float] = 4000.0,
        alias_decay_db: Optional[float] = 0.0,
        device: Optional[str] = 'cuda',
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        assert group_size >= 2 and group_size & (group_size - 1) == 0, (
            f"group_size must be a power of two >= 2, got {group_size}"
        )
        filter_type = "geq"
        n_delays = group_size * n_groups
        delay_lengths = torch.as_tensor(delay_lengths, device=device, dtype=torch.int64)
        assert delay_lengths.numel() == n_delays, (
            f"delay_lengths must have {n_delays} entries (group_size * n_groups), got {delay_lengths.numel()}"
        )

        input_gain = dsp.Gain(
            size=(n_delays, in_ch),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        output_gain = dsp.Gain(
            size=(out_ch, n_delays),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        delays = dsp.parallelDelay(
            size=(n_delays,),
            max_len=delay_lengths.max(),
            nfft=nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        delays.assign_value(delays.sample2s(delay_lengths))

        mixing_matrix = dsp.Matrix(
            size=(n_delays, n_delays),
            nfft=nfft,
            matrix_type="random_block_diagonal",
            n_blocks=n_groups,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        if filter_type == 'shelf':
            attenuation = parallelGFDNFirstOrderShelving(
                nfft=nfft,
                fs=fs,
                rt_nyquist=rt_nyquist,
                n_groups=n_groups,
                delays=delay_lengths,
                alias_decay_db=alias_decay_db,
                requires_grad=True,
                device=device,
                dtype=dtype,
            )
            rt_dc = torch.as_tensor(rt_dc, device=device, dtype=dtype)
            omega_c = 2 * torch.pi * torch.as_tensor(
                crossover_freq, device=device, dtype=dtype
            ) / fs
            attenuation.assign_value(
                torch.stack((rt_dc, omega_c), dim=-1)
            )
        elif filter_type == 'geq':
            attenuation = parallelGFDNGEQ(octave_interval=1,
                                          n_groups=n_groups,
                                          nfft=nfft,
                                          fs=fs,
                                          delays=delay_lengths,
                                          alias_decay_db=alias_decay_db,
                                          requires_grad=True,
                                          device=device,
                                          dtype=dtype)
        else:
            raise ValueError('Filter type must be shelf or geq')
            
        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        core = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

        input_layer = dsp.FFT(nfft, dtype=dtype)
        output_layer = dsp.iFFTAntiAlias(
            nfft=nfft, alias_decay_db=alias_decay_db, device=device, dtype=dtype
        )
        super().__init__(core=core, input_layer=input_layer, output_layer=output_layer)


def example_gfdn(args):
    """
    Example function that demonstrates the construction and training of a
    Grouped Feedback Delay Network (GFDN) model via simple gradient descent
    directly on the GFDN's own parameters (input/output gains and attenuation
    filter), analogous to `example_fdn` in e8_fdn.py.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """

    # read target (omnidirectional channel of one Treble10 receiver)
    target_rir, fs = load_treble10_rir(args.target_rir, args.mic_idx)
    if fs != args.samplerate:
        print(f"Using the target RIR sample rate ({fs} Hz) instead of {args.samplerate} Hz")
        args.samplerate = fs
    target_rir = torch.tensor(target_rir, dtype=torch.float32)
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset : (rir_onset + args.nfft)].view(1, -1, 1)
    target_rir = target_rir / torch.max(torch.abs(target_rir))
    target_rir_1d = target_rir.flatten()

    # zero pad to nfft
    target_rir = torch.nn.functional.pad(target_rir, (0, 0, 0, args.nfft - target_rir.shape[1]))

    # background noise of the target, summed to the GFDN output in the loss
    # (zero-padded like the target)
    if args.model_noise:
        noise, noise_start = extract_background_noise(target_rir_1d, args.samplerate)
        print(f"Target RIR noise floor starts at {noise_start / args.samplerate:.3f} s")
        noise = torch.nn.functional.pad(noise, (0, args.nfft - noise.shape[0]))
        noise = noise.view(1, -1, 1).to(device=args.device, dtype=args.dtype)

    # GFDN parameters
    delays = [997, 1153, 1327, 1559]
    group_size = len(delays)
    n_groups = 2
    alias_decay_db = 30

    # analyze the target RIR's per-band multi-slope decay directly from the
    # RIR (DecayFitNet applies its own octave-band filterbank), for later
    # comparison against the GFDN's shelving-filter frequency response
    band_freqs, target_t, target_fit = estimate_band_decay_times(target_rir_1d, args.samplerate, n_groups)
    print(f"Target RIR octave bands (Hz): {band_freqs}")
    print(f"Target RIR RT60 per band, per slope (s):\n{target_t}")

    # check how well the fitted multi-slope EDCs match the RIR's own EDCs
    compare_fitted_edcs(
        target_rir_1d, target_fit, args.samplerate,
        plot_path=os.path.join(args.train_dir, "target_edc_fit.png"),
    )

    ## ---------------- CONSTRUCT GFDN ---------------- ##

    model = GroupedFDN(
        group_size=group_size,
        n_groups=n_groups,
        delay_lengths=delays * n_groups,
        nfft=args.nfft,
        fs=args.samplerate,
        in_ch=1,
        out_ch=1,
        rt_dc=[0.5, 1.5],
        rt_nyquist=0.2,
        crossover_freq=[4000.0, 4000.0],
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )

    # Get initial impulse response
    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    # read target RIR
    input = signal_gallery(
        1,
        n_samples=args.nfft,
        n=1,
        signal_type="impulse",
        fs=args.samplerate,
        device=args.device,
        dtype=args.dtype,
    )

    dataset = Dataset(
        input=input,
        target=target_rir,
        expand=args.num,
        device=args.device,
        dtype=args.dtype,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        patience=20,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device,
    )
    if args.model_noise:
        trainer.register_criterion(NoisyLoss(edr_loss(), noise), 1)
    else:
        trainer.register_criterion(edr_loss(), 1)
    trainer.register_criterion(sparsity_loss(), 1, requires_model=True)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get optimized impulse response
    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)),
            fs=args.samplerate,
        )
        if args.model_noise:
            ir_optim_noisy = ir_optim + noise.squeeze()
            save_audio(
                os.path.join(args.train_dir, "ir_optim_noisy.wav"),
                ir_optim_noisy / torch.max(torch.abs(ir_optim_noisy)),
                fs=args.samplerate,
            )

    # Compare the shelving filter's frequency-dependent RT60 (one curve per
    # group) against the target's per-band, per-slope DecayFitNet estimates
    attenuation = model.get_core().feedback_loop.feedback.attenuation
    model_rt60 = shelving_group_rt60(attenuation, band_freqs)  # (n_groups, n_bands)

    header = f"{'Band (Hz)':>10} | {'Target slopes (s)':>22} | {'Model groups (s)':>22}"
    print(header)
    print("-" * len(header))
    for b, f in enumerate(band_freqs):
        target_str = ", ".join(f"{t:.3f}" for t in target_t[b])
        model_str = ", ".join(f"{t:.3f}" for t in model_rt60[:, b])
        print(f"{f:>10.0f} | {target_str:>22} | {model_str:>22}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=64000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=32000, help="sampling rate")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"], help="data type for tensors")
    parser.add_argument("--num", type=int, default=100, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--masked_loss", type=bool, default=False, help="use masked loss"
    )
    parser.add_argument(
        "--target_rir",
        type=str,
        default="/Users/dalsag1/Documents/datasets/treble10-npz/rir-hoa8/bathroom1",
        help="Treble10 .npz file, or a directory (the first .npz file in it is used)",
    )
    parser.add_argument(
        "--mic_idx", type=int, default=0, help="receiver index within the .npz file"
    )
    parser.add_argument(
        "--model_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="estimate the target's background noise and add it to the GFDN output in the loss (disable with --no-model_noise)",
    )
    args = parser.parse_args()

    # Check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # Save arguments
    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    # Run the example GFDN training
    example_gfdn(args)
