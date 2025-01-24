import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch

from flamo.processor import dsp, system
from flamo.functional import signal_gallery

torch.manual_seed(1)


def example_recursion(args):
    """
    In this section we introduce the Recursion class.
    It implements a close-loop system with a feedforward path and a feedback path.
    The Recursion class, just as the Series class, is a container for modules.
    It does not change any attribute in the modules it contains, but it checks compatibility.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 3
    # Feedforward path
    delays = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=5000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    attenuation = dsp.parallelGain(size=(out_ch,), nfft=args.nfft, device=args.device)
    rand_vector = torch.rand(attenuation.param.shape)
    attenuation.assign_value(0.3 * rand_vector / torch.norm(rand_vector, p=2))
    feedforward_path = OrderedDict({"delays": delays, "attenuation": attenuation})

    # Feedback path
    feedback_matrix = dsp.Matrix(
        size=(in_ch, out_ch),
        matrix_type="orthogonal",
        nfft=args.nfft,
        device=args.device,
    )

    feedback_path = OrderedDict({"feedback_matrix": feedback_matrix})

    # Recursion
    recursion = system.Recursion(fF=feedforward_path, fB=feedback_path)

    # Input and output layers
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    my_dsp = system.Series(
        OrderedDict(
            {
                "input_layer": input_layer,
                "recursion": recursion,
                "output_layer": output_layer,
            }
        )
    )

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(
        signal_type="impulse",
        batch_size=1,
        n_samples=args.nfft,
        n=in_ch,
        fs=args.samplerate,
        device=args.device,
    )

    # Apply filter
    output_sig = my_dsp(input_sig)

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i + 1)
        plt.plot(output_sig.squeeze().cpu().numpy()[:, i])
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.xlim([0, 10000])
        plt.title(f"Output channel {i+1}")
    plt.tight_layout()
    plt.show()

    return None


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
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle dataset")
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
    example_recursion(args)
