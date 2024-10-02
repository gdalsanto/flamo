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


def s2_e0():
    """
    Let's now create a sequence of two SISO filters.
    The first filter will be a parallelGain module and the second one will be a Delay module.
    We will give a unit impulse as input. The output will be the impulse response of the series of the two filters.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 1
    out_ch = 1
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=700,
        isint=True,
        nfft=nfft,
        fs=samplerate,
    )
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, filter1, filter2, output_layer)

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(signal_type='impulse', batch_size=1, n_samples=nfft, n=in_ch, fs=samplerate)

    # Apply filter
    output_sig = my_dsp(input_sig)

    # ----------------------- Plot --------------------------
    plt.figure()
    plt.plot(output_sig.squeeze().numpy())
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.title(f'parallelGain = {filter1.param.item():.2f} - Delay = {filter2.s2sample(filter2.param.item()):.2f} samples')
    plt.tight_layout()
    plt.show()

    return None

def s2_e1():
    """
    We will now create a MIMO version of the previous example. Let's set 2 input channels and 3 output channels.
    The parallelGain class acts in a channel-wise manner. The Delay class, instead, applies a mixing to its input channels.
    It will be possible to distinguish two delays in each output channel. In all three output channels, one delay
    is scaled by the first channel of the parallelGain module and the other delay is scaled by the second channel.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate
    )
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, filter1, filter2, output_layer)

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(signal_type='impulse', batch_size=1, n_samples=nfft, n=in_ch, fs=samplerate)

    # Apply filter
    output_sig = my_dsp(input_sig)

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(output_sig.squeeze()[:,i].numpy())
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.tight_layout()
    plt.show()

    return None

def s2_e2():
    """
    Each time we instantiate a Gain class, its parameters are drawn from a normal distribution.
    Each time we instantiate a Delay class, its parameters are drawn from a uniform distribution.
    Different classes have different default initialization methods.
    We can easily take control of their parameters as we did in the example s0_e2.
    It is important to provide the new parameters with the correct shape.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate
    )
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, filter1, filter2, output_layer)

    # ----------------- Change parameters -------------------
    new_gains = torch.tensor([0.5,
                              -1.0])
    filter1.assign_value(new_gains)

    print(filter2.s2sample(filter2.param))

    new_delays_in_samples = torch.tensor([[100, 200],
                                          [300, 400],
                                          [500, 600]])
    new_delays_in_seconds = filter2.sample2s(new_delays_in_samples)
    filter2.assign_value(new_delays_in_seconds)

    print(filter2.s2sample(filter2.param))

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(signal_type='impulse', batch_size=1, n_samples=nfft, n=in_ch, fs=samplerate)

    # Apply filter
    output_sig = my_dsp(input_sig)

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(output_sig.squeeze()[:,i].numpy())
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.tight_layout()
    plt.show()

    return None

def s2_e3(args):
    """
    Thanks to the requires_grad attribute, we can decide which filters to train and which not to.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=True
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
    )
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    model = nn.Sequential(input_layer, filter1, filter2, output_layer)

    # ----------------- Initialize dataset ------------------

    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=args.batch_size, n_samples=samplerate, n=in_ch, fs=samplerate)

    # Target
    target_gains = [0.5, -1.0]
    target_delays = filter2.s2sample(filter2.param)
    target = torch.zeros(nfft, out_ch)
    for i in range(out_ch):
        for j in range(in_ch):
            target[int(target_delays[i,j].item()), i] = target_gains[j]
    

    # Dataset
    dataset = Dataset(
        input=unit_imp,
        target=target.unsqueeze(0),
        expand=args.num,
        device=args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split)

    # ------------ Initialize training process ------------
    criterion = nn.L1Loss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(criterion, 1)

    # ------------------ Train the model ------------------

    # Filter impulse response at initialization
    with torch.no_grad():
        ir_init = model(unit_imp).detach().clone()

    # Train stage
    trainer.train(train_loader, valid_loader)

    # Filter impulse response after training
    with torch.no_grad():
        ir_optim = model(unit_imp).detach().clone()

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(ir_init.squeeze()[:,i].numpy(), label='Initial')
        plt.plot(ir_optim.squeeze()[:,i].numpy(), label='Optimized')
        plt.plot(target.squeeze()[:,i].numpy(), '--', label='Target')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.subplot(out_ch, 1, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def erroneous_test(args):
    raise NotImplementedError
    # NOTE: Currently, Delay class does not learn as expected.
    """
    Thanks to the requires_grad attribute, we can decide which filters to train and which not to.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=1000,
        isint=False,
        nfft=nfft,
        fs=samplerate,
        requires_grad=True
    )
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    model = nn.Sequential(input_layer, filter1, filter2, output_layer)

    # ----------------- Change parameters -------------------
    new_gains = torch.tensor([2.0,
                              0.7])
    filter1.assign_value(new_gains)

    # ----------------- Initialize dataset ------------------

    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=args.batch_size, n_samples=samplerate, n=in_ch, fs=samplerate)

    # Target
    target_delays = torch.tensor([[100, 200],
                                  [300, 400], 
                                  [500, 600]])
    target = torch.zeros(nfft, out_ch)
    for i in range(3):
        target[target_delays[i,:],i] = new_gains

    # Dataset
    dataset = Dataset(
        input=unit_imp,
        target=target.unsqueeze(0),
        expand=args.num,
        device=args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split)

    # ------------ Initialize training process ------------
    criterion = nn.L1Loss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(criterion, 1)

    # ------------------ Train the model ------------------

    # Filter impulse response at initialization
    with torch.no_grad():
        ir_init = model(unit_imp).detach().clone()

    # Train stage
    trainer.train(train_loader, valid_loader)

    # Filter impulse response after training
    with torch.no_grad():
        ir_optim = model(unit_imp).detach().clone()

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(ir_init.squeeze()[:,i].numpy(), label='Initial')
        plt.plot(ir_optim.squeeze()[:,i].numpy(), label='Optimized')
        plt.plot(target.squeeze()[:,i].numpy(), '-.', label='Target')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.subplot(out_ch, 1, 1)
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
    # s2_e0()
    # s2_e1()
    # s2_e2()
    s2_e3(args)