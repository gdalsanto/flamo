import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp
from flamo.functional import signal_gallery
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer


torch.manual_seed(1)


def s0_e0() -> None:
    """
    Use of FFT and iFFT modules.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------ Module Definition ------------------
    fft = dsp.FFT(nfft=nfft)
    ifft = dsp.iFFT(nfft=nfft)

    # ------------------ Signal Definition ------------------
    x = signal_gallery(signal_type='sine', batch_size=1, n_samples=nfft, n=1, fs=samplerate)

    # ------------------ Apply FFT and iFFT -----------------
    X = fft(x)
    y = ifft(X)

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.plot(x.squeeze().cpu().numpy(), label='Input', linewidth=2)
    plt.plot(y.squeeze().cpu().numpy(), '--', label='Output', linewidth=2)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def s0_e1() -> None:
    """
    Simple filter creation and application.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    channels = 1
    filter = dsp.parallelGain(size=(channels,), nfft=nfft)
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, filter, output_layer)

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(signal_type='sine', batch_size=1, n_samples=nfft, n=channels, fs=samplerate)

    # Apply filter
    output_sig = my_dsp(input_sig)

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.plot(input_sig.squeeze().cpu().numpy(), label='Input')
    plt.plot(output_sig.squeeze().cpu().numpy(), label=f'Output - filter gain = {filter.param.item():.2f}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def s0_e2() -> None:
    """
    Filter parameter change.
    """
    # ------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition -------------------
    in_ch = 1
    out_ch = 1
    filter = dsp.Gain(size=(out_ch, in_ch), nfft=nfft)
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, filter, output_layer)

    # -------------- Change filter parameters --------------

    # Input signal
    input_sig = signal_gallery(signal_type='sine', batch_size=1, n_samples=nfft, n=in_ch, fs=samplerate)

    # Apply filter before changes
    output_init = my_dsp(input_sig)

    # Change filter parameters
    prev_value = filter.param.item()
    new_value = torch.tensor(-2.0)
    filter.assign_value(new_value.view(1,1).expand(out_ch, in_ch))

    # New filter parameters
    output_after = my_dsp(input_sig)

    # ------------------------ Plot ------------------------
    plt.figure()
    plt.plot(output_init.squeeze().cpu().numpy(), label=f'With original gain value = {prev_value:.2f}')
    plt.plot(output_after.squeeze().cpu().numpy(), label=f'With new gain value = {new_value.item():.2f}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def s0_e3(args) -> None:
    """
    Filter training.
    The filter coefficients will match a sine wave.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------ Model Definition -------------------
    FIR_order = nfft
    in_ch = 1
    out_ch = 1
    filter = dsp.Filter(size=(FIR_order, out_ch, in_ch), nfft=nfft, requires_grad=True)
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    model = nn.Sequential(input_layer, filter, output_layer)

    # ----------------- Initialize dataset ------------------
    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=args.batch_size, n_samples=samplerate, n=in_ch, fs=samplerate)

    # Target impulse response
    target = signal_gallery(signal_type='sine',  batch_size=args.batch_size, n_samples=nfft, n=out_ch, fs=samplerate)

    # Dataset
    dataset = Dataset(
        input=unit_imp,
        target=target,
        expand=args.num,
        device=args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split)

    # ------------- Initialize training process ------------
    criterion = nn.L1Loss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(criterion, 1)

    # ------------------ Train the model -------------------

    # Filter impulse response at initialization
    ir_init = model(unit_imp).detach().clone()

    # Train stage
    trainer.train(train_loader, valid_loader)

    # Filter impulse response after training
    ir_optim = model(unit_imp).detach().clone()

    # ----------------------- Plot -------------------------

    plt.figure()
    plt.plot(ir_init.squeeze().numpy(), label='Initial', linewidth=0.5)
    plt.plot(ir_optim.squeeze().numpy(), label='Optimized', linewidth=2)
    plt.plot(target.squeeze().numpy(), '--', label='Target', linewidth=2)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    #----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run examples
    # s0_e0()
    # s0_e1()
    # s0_e2()
    s0_e3(args)