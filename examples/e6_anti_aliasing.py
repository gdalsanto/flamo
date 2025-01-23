import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp, system

in_ch = 1
n_chs = 6


def get_system(args, alias_decay_db=0):
    torch.manual_seed(130799)  # needed to generate the same paramer values
    input_gain = dsp.Gain(
        size=(n_chs, 1),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    input_gain.assign_value(torch.ones(input_gain.param.shape))
    # output gains
    output_gain = dsp.Gain(
        size=(1, n_chs),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    output_gain.assign_value(torch.ones(output_gain.param.shape, device=args.device))

    # Feedback loop with delays
    delays = dsp.parallelDelay(
        size=(n_chs,),
        max_len=1000,
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    # Feedback path with orthogonal matrix
    feedback = dsp.Matrix(
        size=(n_chs, n_chs),
        nfft=args.nfft,
        matrix_type="orthogonal",
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    attenuation = dsp.parallelGain(
        size=(n_chs,), nfft=args.nfft, alias_decay_db=alias_decay_db, device=args.device
    )
    get_delays = delays.get_delays()
    attenuation.assign_value(0.99999 ** (get_delays(delays.param)))
    # Recursion
    feedback_loop = system.Recursion(fF=delays, fB=system.Series(feedback, attenuation))

    # Full FDN
    my_dsp = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    return my_dsp


def example_anti_aliasing(args):
    """
    In this example, we will see how to use the anti-aliasing feature to reduce
    the time aliasing in the system.
    """

    # ------------------ Model Definition -------------------
    # generate a IIR with long ringing modes that cause aliasing
    my_dsp = get_system(args, alias_decay_db=0)

    # Shell instance
    model = system.Shell(core=my_dsp)

    # Get the initial response for the comparison
    imp_resp = model.get_time_response(fs=args.samplerate).squeeze()
    mag_resp = torch.abs(model.get_freq_response(fs=args.samplerate).squeeze())

    # generate the same IIR filter but with time-aliasing reduction
    my_dsp_aa = get_system(args, alias_decay_db=30)
    model_aa = system.Shell(core=my_dsp_aa)

    # Get the initial response for the comparison
    imp_resp_aa = model_aa.get_time_response(fs=args.samplerate).squeeze()
    mag_resp_aa = torch.abs(model_aa.get_freq_response(fs=args.samplerate).squeeze())

    plt.subplot(2, in_ch, 1)
    plt.plot(imp_resp.squeeze().cpu().numpy(), label="Original")
    plt.plot(imp_resp_aa.squeeze().cpu().numpy(), label="Anti-aliasing")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim([0, 1000])
    plt.grid()
    plt.subplot(2, in_ch, 2)
    plt.plot(mag_resp.squeeze().cpu().numpy(), label="Original")
    plt.plot(mag_resp_aa.squeeze().cpu().numpy(), label="Anti-aliasing")
    plt.xlabel("Frequency bins")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    return


###########################################################################################

if __name__ == "__main__":

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()

    # ---------------------- Processing -------------------
    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    # ----------------------- Dataset ----------------------
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--num", type=int, default=2**8, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="split ratio for training and validation",
    )
    # ---------------------- Training ----------------------
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="maximum number of epochs"
    )
    parser.add_argument(
        "--patience_delta",
        type=float,
        default=0.001,
        help="Minimum improvement in validation loss to be considered as an improvement",
    )
    # ---------------------- Optimizer ---------------------
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # ----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments
    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    # Run examples
    example_anti_aliasing(args)
    # TODO add exampe of training with antialiasing
