from typing import Optional
import torch
import torch.nn as nn
import argparse
import os
import time
import auraloss
import numpy as np
import soundfile as sf
import multislope

from collections import OrderedDict

from flamo.auxiliary.reverb import parallelGFDNFirstOrderShelving
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.loss import sparsity_loss, edr_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.utils import save_audio
from flamo.functional import signal_gallery, find_onset

torch.manual_seed(130798)


def estimate_band_decay_times(rir: torch.Tensor, fs: int, n_slopes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-octave-band multi-exponential decay times of a RIR with the
    multislope library's DecayFitNet.

    Args:
        rir: 1D RIR tensor, onset-aligned (no leading silence).
        fs: sample rate in Hz.
        n_slopes: number of decay slopes to fit per band.
    Returns:
        ``(band_freqs, t)`` where ``band_freqs`` are the octave-band centre
        frequencies (Hz), and ``t`` has shape ``(n_bands, n_slopes)`` -- the
        fitted decay times (T60, in seconds) per band, sorted ascending
        along the slopes axis.
    """
    x = rir.detach().cpu().numpy().astype(np.float64).flatten()
    net = multislope.DecayFitNet(n_slopes=n_slopes, sample_rate=fs)
    fit = net.estimate(x, analyse_full_rir=True)
    return np.array(fit.frequencies), np.sort(fit.t, axis=-1)


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
        omega_c = 2 * torch.pi * crossover_freq / fs
        attenuation.assign_value(
            torch.tensor([rt_dc, omega_c], device=device, dtype=dtype)
        )

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

    # read target 
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=torch.float32)
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset : (rir_onset + args.nfft)].view(1, -1, 1)
    target_rir = target_rir / torch.max(torch.abs(target_rir))
    target_rir_1d = target_rir.flatten()

    # zero pad to nfft
    target_rir = torch.nn.functional.pad(target_rir, (0, 0, 0, args.nfft - target_rir.shape[1]))

    # GFDN parameters
    delays = [997, 1153, 1327, 1559]
    group_size = len(delays)
    n_groups = 2
    alias_decay_db = 30

    # analyze the target RIR's per-band multi-slope decay directly from the
    # RIR (DecayFitNet applies its own octave-band filterbank), for later
    # comparison against the GFDN's shelving-filter frequency response
    band_freqs, target_t = estimate_band_decay_times(target_rir_1d, args.samplerate, n_groups)
    print(f"Target RIR octave bands (Hz): {band_freqs}")
    print(f"Target RIR RT60 per band, per slope (s):\n{target_t}")

    ## ---------------- CONSTRUCT GFDN ---------------- ##

    model = GroupedFDN(
        group_size=group_size,
        n_groups=n_groups,
        delay_lengths=delays * n_groups,
        nfft=args.nfft,
        fs=args.samplerate,
        in_ch=1,
        out_ch=1,
        rt_dc=1.0,
        rt_nyquist=0.2,
        crossover_freq=4000.0,
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

    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
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
        default="rirs/multi-slope/ir_r2.wav",
        help="filepath to target RIR",
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
