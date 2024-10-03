import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp, system
from flamo.functional import mag2db, signal_gallery
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.optimize.loss import mse_loss, sparsity_loss



torch.manual_seed(1)


def s4_e0():
    """
    In this section we introduce the Recursion class.
    It implements a close-loop system with a feedforward path and a feedback path.
    The Recursion class, just as the Series class, is a container for modules.
    It does not change any attribute in the modules it contains, but it checks compatibility.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 48000

    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 3
    # Feedforward path
    delays = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate
    )
    attenuation = dsp.parallelGain(
        size=(out_ch,),
        nfft=nfft
    )
    rand_vector = torch.rand(attenuation.param.shape)
    attenuation.assign_value(0.3*rand_vector/torch.norm(rand_vector, p=2))
    feedforward_path = OrderedDict({
        'delays': delays,
        'attenuation': attenuation
    })

    # Feedback path
    feedback_matrix = dsp.Matrix(
        size=(in_ch, out_ch),
        matrix_type='orthogonal',
        nfft=nfft
    )

    feedback_path = OrderedDict({
        'feedback_matrix': feedback_matrix
    })

    # Recursion
    recursion = system.Recursion(fF=feedforward_path, fB=feedback_path)

    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = system.Series(
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
    output_sig = my_dsp(input_sig)

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

def s4_e1(args):
    raise NotImplementedError
    # NOTE: It does not behave as I was expencting. Maybe it is correct, and I was just expecting better results that I can obtain.
    """
    Once again, we can train individual modules as the Recursion class keeps the modules separate.
    We will train the feedback path to provide system magnitude responses as spectrally flat as possible.
    We will use an instance of the Filter class instead of the Matrix class, as it provides a more evident result.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 2**10

    # ------------------ Model Definition -------------------
    in_ch = 6
    out_ch = 6
    # Feedforward path
    delays = dsp.parallelDelay(
        size=(in_ch,),
        max_len=3000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    delay_lengths = torch.tensor([887, 911, 941, 1699, 1951, 2053])
    delays.assign_value(delays.sample2s(delay_lengths))
    attenuation = dsp.parallelGain(
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
    feedback_matrix = dsp.Matrix(
        size=(in_ch, out_ch),
        matrix_type='orthogonal',
        nfft=nfft,
        requires_grad=True
    )

    feedback_path = OrderedDict({
        'feedback_matrix': feedback_matrix
    })

    # Recursion
    recursion = system.Recursion(fF=feedforward_path, fB=feedback_path)

    # Input and output layers
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.Transform(lambda x: torch.abs(x))

    model = system.Series(
        OrderedDict({
            'input_layer': input_layer,
            'recursion': recursion,
            'output_layer': output_layer
        })
    )

    # ------------------- Dataset Definition --------------------

    # Dataset
    dataset = DatasetColorless(
        input_shape=(args.batch_size, nfft//2+1, in_ch),
        target_shape=(args.batch_size, nfft//2+1, out_ch),
        expand=args.num,
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
    trainer.register_criterion(mse_loss(n_sections=args.num, nfft=nfft), 1)
    trainer.register_criterion(sparsity_loss(), 0.2, requires_model=True)

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
    for i in range(2):
        plt.subplot(2, 1, i+1)
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
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.001, help='Minimum improvement in validation loss to be considered as an improvement')
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
    s4_e0()
    # s4_e1(args)