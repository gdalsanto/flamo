import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp, system
from flamo.functional import signal_gallery
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer


torch.manual_seed(1)

def s3_e0():
    """
    Now, we introduce the Series class.
    You might have noticed that in order to create a sequence of MIMO filters, we have to take care
    of designing the single modules so that their input and output channels match. The same applies
    to some of their attributes (e.g. nfft). The Series class takes care of checking that the modules
    are compatible with each other.
    NOTE: The Series class does not create the number of channels for us, as we want complete control over it.
          It only checks that we made no errors in the connections or in the attributes of the modules.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch),
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    # Series class
    my_dsp = system.Series(input_layer, filter1, filter2, filter3, output_layer)

def s3_e1():
    """
    Let's purposefully make an error in the attributes/connections of the modules of the Series class.
    The Series class will tell us what the error is and where the error is located.
    It will through an error at the first erroneuos test found.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=2**8,                  # NOTE:Error here
        requires_grad=False
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, 10),          # NOTE:Error here
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    # Series class
    my_dsp = system.Series(input_layer, filter1, filter2, filter3, output_layer)

    return None

def s3_e2():
    """
    Just as the torch.nn.Sequential class, the Series class accepts an OrderedDict in input.
    This allows us to give a name to each module in the sequence.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch),
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    # Series class
    my_dsp = system.Series(
        OrderedDict({
            'input_layer': input_layer,
            'Gains': filter1,
            'Delays': filter2,
            'Eqs': filter3,
            'output_layer': output_layer
        })
    )
    print(my_dsp)

    return None


def s3_e3():
    """
    The Series class is designed to un-nest nested torch.nn.Sequential instances,
    OrderedDict instances, and/or other Series instances.
    This makes checking attribute matching i/o compatibility and between modules easier.
    If nested OrderedDict instances are used to generate the dsp, the custom key names will be maintained.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch),
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    # Series class
    filters = OrderedDict({
        'Gains': filter1,
        'Delays': filter2,
        'Eqs': filter3
    })

    my_dsp = system.Series(
        OrderedDict({
            'input_layer': input_layer,
            'filters': filters,
            'output_layer': output_layer
        })
    )
    print(my_dsp)

    return None

def s3_e4(args):
    """
    The Series class allows us to train a chain of filters (or just one filter in the chain) as
    we did in the example s2_e3.
    Let's reproduce the example with the Series class.
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

    # Series of filters
    filters = OrderedDict({
        'Gains': filter1,
        'Delays': filter2,
    })
    model = system.Series(
        OrderedDict({
            'input_layer': input_layer,
            'filters': filters,
            'output_layer': output_layer
        })
    )

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
        patience_delta=args.patience_delta,
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
    parser.add_argument('--patience_delta', type=float, default=0.0001, help='Minimum improvement in validation loss to be considered as an improvement')
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
    s3_e0()
    # s3_e1()
    # s3_e2()
    # s3_e3()
    # s3_e4(args)