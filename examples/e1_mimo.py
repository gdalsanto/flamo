import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp
from flamo.functional import (
    mag2db,
    signal_gallery,
    lowpass_filter,
    biquad2tf,
)
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer


torch.manual_seed(1)

def example_matrix(args) -> None:
    """
    Test all the different types of matrices available in dsp.Matrix.
    We will create different matrix types and apply them to a multi-channel input signal.
    """
    # ------------------- Signal Definition --------------------
    in_ch = 4
    out_ch = 4
    
    # Input signal - multi-channel noise
    input_sig = signal_gallery(
        signal_type="noise",
        batch_size=1,
        n_samples=args.nfft,
        n=in_ch,
        fs=args.samplerate,
        device=args.device,
    )
    
    # Test different matrix types
    matrix_types = ["random", "identity", "orthogonal", "hadamard", "rotation"]
    
    fig, axes = plt.subplots(len(matrix_types), out_ch, figsize=(12, 2*len(matrix_types)))
    if len(matrix_types) == 1:
        axes = axes.reshape(1, -1)
    
    for i, matrix_type in enumerate(matrix_types):
        print(f"Testing matrix type: {matrix_type}")
        
        # ------------------- DSP Definition --------------------
        matrix_filter = dsp.Matrix(
            size=(out_ch, in_ch),
            matrix_type=matrix_type,
            nfft=args.nfft,
            device=args.device,
        )
        
        input_layer = dsp.FFT(nfft=args.nfft)
        output_layer = dsp.iFFT(nfft=args.nfft)
        
        my_dsp = nn.Sequential(input_layer, matrix_filter, output_layer)
        
        # -------------- Apply signal to DSP --------------
        output_sig = my_dsp(input_sig)
        
        # ----------------------- Plot ------------------------
        for j in range(out_ch):
            ax = axes[i, j] if out_ch > 1 else axes[i]
            ax.plot(output_sig.squeeze()[:1000, j].cpu().numpy())  # Plot first 1000 samples
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            ax.set_title(f"{matrix_type.capitalize()} Matrix - Output Ch {j+1}")
            
    plt.tight_layout()
    plt.show()

    return None


def example_delays(args) -> None:
    """
    Let's create a multichannel Delay instance with three input channels and two output channels.
    We will give a unit impulse as input. Each output channel will contain three delays, one for each input channel.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 2
    filter = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=700,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    my_dsp = nn.Sequential(input_layer, filter, output_layer)

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

    # ----------------------- Plot ------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i + 1)
        plt.plot(output_sig.squeeze()[:, i].cpu().numpy())
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.title(f"Output channel {i+1}")
    plt.tight_layout()
    plt.show()

    return None


def example_biquads(args) -> None:
    """
    Let's train a multichannel Biquad instance with one input channel and two output channels.
    We want the biquad filter magnitude responses to match a given target.
    We will give a unit impulse as input. The output will be the magnitude responses of the two biquad filters.
    """
    # ------------------ Model Definition -------------------
    in_ch = 1
    out_ch = 2
    filter = dsp.Biquad(
        size=(out_ch, in_ch),
        n_sections=1,
        filter_type="lowpass",
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=True,
        device=args.device,
    )
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.Transform(lambda x: torch.abs(x))

    model = nn.Sequential(input_layer, filter, output_layer)

    # ----------------- Initialize dataset ------------------

    # Input unit impulse
    unit_imp = signal_gallery(
        signal_type="impulse",
        batch_size=args.batch_size,
        n_samples=args.samplerate,
        n=in_ch,
        fs=args.samplerate,
        device=args.device,
    )

    # Target frequency responses
    f_cut_1 = 500  # Cut-off frequency for the first lowpass filter
    g_1 = 10  # Bandpass gain for the first lowpass filter
    b_lp_1, a_lp_1 = lowpass_filter(f_cut_1, g_1, args.samplerate, device=args.device)
    H_lp_1 = biquad2tf(b=b_lp_1, a=a_lp_1, nfft=args.nfft)

    f_cut_2 = 5000  # Cut-off frequency for the second lowpass filter
    g_2 = 0.7  # Bandpass gain for the second lowpass filter
    b_lp_2, a_lp_2 = lowpass_filter(f_cut_2, g_2, args.samplerate, device=args.device)
    H_lp_2 = biquad2tf(b=b_lp_2, a=a_lp_2, nfft=args.nfft)

    target = torch.stack([torch.abs(H_lp_1), torch.abs(H_lp_2)], dim=1).unsqueeze(0)

    # Dataset
    dataset = Dataset(
        input=unit_imp, target=target, expand=args.num, device=args.device
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=args.batch_size, split=args.split
    )

    # ------------ Initialize training process ------------
    criterion = nn.MSELoss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device,
    )
    trainer.register_criterion(criterion, 1)

    # ------------------ Train the model ------------------

    # Filter impulse response at initialization
    mag_resp_init = model(unit_imp).detach().clone()

    # Train stage
    trainer.train(train_loader, valid_loader)

    # Filter impulse response after training
    mag_resp_optim = model(unit_imp).detach().clone()

    # ----------------------- Plot ------------------------

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(mag2db(mag_resp_init[0, :, 0]).squeeze().cpu().numpy(), label="Initial")
    plt.plot(mag2db(mag_resp_optim[0, :, 0]).squeeze().cpu().numpy(), label="Optimized")
    plt.plot(mag2db(target[0, :, 0]).cpu().numpy(), "-.", label="Target")
    plt.xlabel("Frequency bins")
    plt.ylabel("Magnitude in dB")
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(mag2db(mag_resp_init[0, :, 1]).squeeze().cpu().numpy(), label="Initial")
    plt.plot(mag2db(mag_resp_optim[0, :, 1]).squeeze().cpu().numpy(), label="Optimized")
    plt.plot(mag2db(target[0, :, 1]).cpu().numpy(), "--", label="Target")
    plt.xlabel("Frequency bins")
    plt.ylabel("Magnitude in dB")
    plt.grid()
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
        default=0.002,
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
    example_matrix(args)
    example_delays(args)
    example_biquads(args)
