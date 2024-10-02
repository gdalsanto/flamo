import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from fddsp.modules import (
    Transform,
    FFT,
    iFFT,
    parallelGain,
    Filter,
    Delay,
    Series,
    Recursion,
)
from fddsp.dsp import (
    Matrix,
)
from fddsp.filters.functional import (
    mag2db,
)
from fddsp.functional import (
    get_magnitude,
    signal_gallery,
)
from fddsp.utils.dataset import (
    Dataset_Colorless,
    load_dataset
)
from fddsp.utils.trainer import Trainer


torch.manual_seed(1)


def s3_e0():
    """
    In this section we introduce the Recursion class.
    It implements a close-loop system with a feedforward path and a feedback path.
    The Recursion class, just as the Series class, is a container for modules. It does not create change any attribute
    in the modules it contains, but it checks that they are compatible.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 3
    # Feedforward path
    delays = Delay(
        size=(out_ch, in_ch),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    attenuation = parallelGain(
        size=(out_ch,),
        nfft=nfft,
        requires_grad=False
    )
    rand_vector = torch.rand(attenuation.param.shape)
    attenuation.assign_value(0.3*rand_vector/torch.norm(rand_vector, p=2))
    feedforward_path = OrderedDict({
        'delays': delays,
        'attenuation': attenuation
    })

    # Feedback path
    feedback_matrix = Matrix(
        size=(in_ch, out_ch),
        matrix_type='orthogonal',
        nfft=nfft,
        requires_grad=False
    )

    feedback_path = OrderedDict({
        'attenuation': feedback_matrix
    })

    # Recursion
    recursion = Recursion(fF=feedforward_path, fB=feedback_path)

    # Input and output layers
    input_layer = FFT(nfft=nfft)
    output_layer = iFFT(nfft=nfft)

    dsp = Series(
        OrderedDict({
            'input_layer': input_layer,
            'recursion': recursion,
            'output_layer': output_layer
        })
    )

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(signal_type='impulse', batch_size=1, n_samples=nfft, n=in_ch, fs=samplerate)

    # Apply filter
    output_sig = dsp(input_sig)

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(output_sig.squeeze().numpy()[:,i])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.tight_layout()
    plt.show()

    return None

def s3_e1():
    """
    Once again, we can train individual modules as the Recursion class keeps the modules separate.
    We will train the feedback orthogonal matrix to provide system magnitude responses as spectrally flat as possible.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------ Model Definition -------------------
    in_ch = 3
    out_ch = 3
    # Feedforward path
    delays = Delay(
        size=(out_ch, in_ch),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    attenuation = parallelGain(
        size=(out_ch,),
        nfft=nfft,
        requires_grad=False
    )
    rand_vector = torch.rand(attenuation.param.shape)
    attenuation.assign_value(0.3*rand_vector/torch.norm(rand_vector, p=2))
    feedforward_path = OrderedDict({
        'delays': delays,
        'attenuation': attenuation
    })

    # Feedback path
    feedback_matrix = Filter(
        size=(100, in_ch, out_ch),
        nfft=nfft,
        requires_grad=True # NOTE: The Filter class instance is set to learnable
    )

    feedback_path = OrderedDict({
        'attenuation': feedback_matrix
    })

    # Recursion
    recursion = Recursion(fF=feedforward_path, fB=feedback_path)

    # Input and output layers
    input_layer = FFT(nfft=nfft)
    output_layer = Transform(get_magnitude)

    model = Series(
        OrderedDict({
            'input_layer': input_layer,
            'recursion': recursion,
            'output_layer': output_layer
        })
    )

    # ------------------- Dataset Definition --------------------

    # Dataset
    dataset = Dataset_Colorless(
        in_shape=(nfft//2+1, in_ch),
        target_shape=(nfft//2+1, out_ch),
        ds_len=args.num,
        device=args.device
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

    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=args.batch_size, n_samples=samplerate, n=in_ch, fs=samplerate)
   
    # Filter impulse response at initialization
    mag_resp_init = model(unit_imp).detach().clone()

    # Train stage
    trainer.train(train_loader, valid_loader)

    # Filter impulse response after training
    mag_resp_optim = model(unit_imp).detach().clone()

    # ----------------------- Plot --------------------------
    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i+1)
        plt.plot(mag2db(mag_resp_init).squeeze().numpy()[:,i], label='Initial')
        plt.plot(mag2db(mag_resp_optim).squeeze().numpy()[:,i], label='Optimized')
        plt.xlabel('Frequency bin')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.title(f'Output channel {i+1}')
    plt.tight_layout()
    plt.show()

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
    s3_e0()
    # s3_e1()