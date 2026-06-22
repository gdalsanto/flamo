"""FDN optimization using EagerTrainer (no Dataset/DataLoader).

This mirrors examples/e8_fdn.py::example_fdn but replaces the ML training stack
(Dataset + load_dataset + Trainer over epochs) with a direct gradient-descent
loop over the single fixed (impulse, target RIR) pair. The FDN construction and
losses are identical.
"""
import torch
import torch.nn as nn
import argparse
import os
import time
import auraloss
import soundfile as sf
from collections import OrderedDict
from flamo.optimize.trainer import EagerTrainer
from flamo.processor import dsp, system
from flamo.optimize.loss import sparsity_loss
from flamo.utils import save_audio
from flamo.functional import signal_gallery, find_onset

torch.manual_seed(130799)


class MultiResoSTFT(nn.Module):
    """Mean absolute error between the auraloss MR-STFT of two RIRs."""

    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir1, rir2):
        return self.MRstft(rir1.permute(0, 2, 1), rir2.permute(0, 2, 1))


def example_fdn_optimize(args):
    # FDN parameters
    N = 6  # number of delays
    alias_decay_db = 30  # alias decay in dB
    delay_lengths = torch.tensor([593, 743, 929, 1153, 1399, 1699])

    ## ---------------- CONSTRUCT FDN ---------------- ##

    input_gain = dsp.Gain(
        size=(N, 1), nfft=args.nfft, requires_grad=True,
        alias_decay_db=alias_decay_db, device=args.device, dtype=args.dtype,
    )
    output_gain = dsp.Gain(
        size=(1, N), nfft=args.nfft, requires_grad=True,
        alias_decay_db=alias_decay_db, device=args.device, dtype=args.dtype,
    )
    delays = dsp.parallelDelay(
        size=(N,), max_len=delay_lengths.max(), nfft=args.nfft, isint=True,
        requires_grad=False, alias_decay_db=alias_decay_db,
        device=args.device, dtype=args.dtype,
    )
    delays.assign_value(delays.sample2s(delay_lengths))
    mixing_matrix = dsp.Matrix(
        size=(N, N), nfft=args.nfft, matrix_type="orthogonal", requires_grad=True,
        alias_decay_db=alias_decay_db, device=args.device, dtype=args.dtype,
    )
    attenuation = dsp.parallelGEQ(
        size=(N,), octave_interval=1, nfft=args.nfft, fs=args.samplerate,
        requires_grad=True, alias_decay_db=alias_decay_db,
        device=args.device, dtype=args.dtype,
    )
    attenuation.map = lambda x: 20 * torch.log10(torch.sigmoid(x))
    feedback = system.Series(
        OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
    )
    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    FDN = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(args.nfft, dtype=args.dtype)
    output_layer = dsp.iFFTAntiAlias(
        nfft=args.nfft, alias_decay_db=alias_decay_db,
        device=args.device, dtype=args.dtype,
    )
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)), fs=args.samplerate,
        )

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    input = signal_gallery(
        1, n_samples=args.nfft, n=1, signal_type="impulse", fs=args.samplerate,
        device=args.device, dtype=args.dtype,
    )
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=torch.float32)
    target_rir = target_rir / torch.max(torch.abs(target_rir))
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset : (rir_onset + args.nfft)].view(1, -1, 1)

    # Direct optimization: one fixed (impulse, target) pair, no Dataset/DataLoader.
    opt = EagerTrainer(
        model,
        max_steps=args.max_steps,
        lr=args.lr,
        optimizer=args.optimizer,
        train_dir=args.train_dir,
        device=args.device,
    )
    opt.register_criterion(MultiResoSTFT(), 1)
    opt.register_criterion(sparsity_loss(), 1, requires_model=True)

    ## ---------------- TRAIN ---------------- ##

    opt.optimize(input, target_rir)

    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim_eager.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)), fs=args.samplerate,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=["float32", "float64"], help="tensor data type")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device to use for computation")
    parser.add_argument("--max_steps", type=int, default=1600,
                        help="number of optimization steps")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "lbfgs"], help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--train_dir", type=str, help="directory to save results")
    parser.add_argument("--target_rir", type=str,
                        default="rirs/arni_35_3541_4_2.wav",
                        help="filepath to target RIR")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [str(k) + "," + str(v)
                 for k, v in sorted(vars(args).items(), key=lambda x: x[0])]
            )
        )

    example_fdn_optimize(args)
