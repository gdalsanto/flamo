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

    # First filter
    filter1 = parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = Delay(
        size=(5, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = GEQ(
        size=(10, out_ch, 5),
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = FFT(nfft=nfft)
    output_layer = iFFT(nfft=nfft)

    # Series of filters
    dsp = Series(input_layer, filter1, filter2, filter3, output_layer)

    # NOTE: Just as the torch.nn.Sequential class, the Series class accepts an OrderedDict in input.
    #       This allows us to give a name to each module in the sequence. This is useful for debugging.
    dsp = Series(
        OrderedDict({
            'input_layer': input_layer,
            'Gains': filter1,
            'Delays': filter2,
            'Eqs': filter3,
            'output_layer': output_layer
        })
    )
    print(dsp)

    # NOTE: The Series class is designed to un-nest nested torch.nn.Sequential instances,
    #       OrderedDict instances, and/or other Series instances.
    #       This makes checking i/o compatibility and attribute matching between modules easier.
    #       If OrderedDict instances were used to generate the dsp, the key names will be maintained.

    filters = OrderedDict({
        'Gains': filter1,
        'Delays': filter2,
        'Eqs': filter3
    })
    dsp = Series(
        OrderedDict({
            'input_layer': input_layer,
            'filters': filters,
            'output_layer': output_layer
        })
    )
    print(dsp)

    return None

def s3_e1():
    """
    The Series class does not create the number of channels for us, as we want complete control over it.
    It only checks that we made no errors in the connections or in the attributes of the modules.
    Let's try to connect the filters with non-compatible channels on purpose.
    The Series class will tell us where the error is located.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3

    # First filter
    filter1 = parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = Delay(
        size=(5, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = GEQ(
        size=(10, out_ch, 4),   # Wrong number of input channels
        nfft=nfft,
        requires_grad=False
    )
    # Input and output layers
    input_layer = FFT(nfft=nfft)
    output_layer = iFFT(nfft=nfft)

    # Series of filters
    dsp = Series(input_layer, filter1, filter2, filter3, output_layer)

    return None

def s3_e2(args):
    """
    Just as we trained a single filter, we can train a sequence of filters.
    Thanks to the requires_grad attribute, we can decide which filters to train and which not to.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**11

    # ------------------ Model Definition -------------------
    in_ch = 2
    out_ch = 3

    # First filter
    filter1 = parallelGain(
        size=(in_ch,),
        nfft=nfft,
        requires_grad=False
    )
    # Second filter
    filter2 = Delay(
        size=(5, in_ch),
        max_len=1000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    # Third filter
    filter3 = GEQ(
        size=(out_ch, 5),
        octave_interval=1,
        nfft=nfft,
        fs=samplerate,
        requires_grad=True
    )
    # Input and output layers
    input_layer = FFT(nfft=nfft)
    output_layer = iFFT(nfft=nfft)

    # Series of filters
    filters = OrderedDict({
        'Gains': filter1,
        'Delays': filter2,
        'Eqs': filter3
    })
    model = Series(
        OrderedDict({
            'input_layer': input_layer,
            'filters': filters,
            'output_layer': output_layer
        })
    )

    # ------------------- Dataset Definition --------------------
    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=args.batch_size, n_samples=samplerate, n=in_ch, fs=samplerate)

    # Target impulse response
    target_resp = signal_gallery(signal_type='sine',  batch_size=args.batch_size, n_samples=nfft, n=out_ch, fs=samplerate).squeeze(0)

    # Dataset
    dataset = Dataset(
        input = unit_imp[0,:,:].squeeze(0),
        target = target_resp,
        ds_len = args.num,
        device = args.device
        )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=args.shuffle)

    # ------------------- Trainer Definition --------------------
    criterion = nn.MSELoss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(criterion, 1)

    # ------------------- Training --------------------
    trainer.train(train_loader, valid_loader)

    return None

###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--num', type=int, default=2**10,
                        help = 'dataset size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--split', type=float, default=0.8,
                        help='training / validation split')
    parser.add_argument('--shuffle', action='store_false',
                        help='if true, shuffle the data in the dataset at every epoch')
    
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, default=None,
                        help ='path to output directory')
    parser.add_argument('--device', default='cpu',
                        help='training device')
    parser.add_argument('--max_epochs', type=int, default=10, 
                        help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true',
                        help='Store met parameters at every epoch')
    
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    
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
    s2_e0()
    # s2_e1()
    # s2_e2()
    # s2_e3()
    # s2_e4()
    # s2_e5(args)