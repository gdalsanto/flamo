import torch
import torch.nn as nn
import argparse
import os
import time
import matplotlib.pyplot as plt
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.auxiliary.eq import accurate_geq, eq_freqs
from flamo.functional import signal_gallery

torch.manual_seed(130710)


def example_geq(args):
    """
    Example function that demonstrates the training of GEQ command gains.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    in_ch, out_ch = 1, 2
    octave_interval = 1
    ## ---------------- TARGET ---------------- ##
    center_freq, shelving_crossover = eq_freqs(interval=octave_interval)
    target_gains = 10 ** (-12 / 20) + (10 ** (12 / 20) - 10 ** (-12 / 20)) * torch.rand(
        size=(len(center_freq) + 2, out_ch, in_ch)
    )
    b, a = torch.empty((3, len(center_freq) + 3, out_ch, in_ch)), torch.empty(
        (3, len(center_freq) + 3, out_ch, in_ch)
    )
    for m_i in range(out_ch):
        for n_i in range(in_ch):
            b[:, :, m_i, n_i], a[:, :, m_i, n_i] = accurate_geq(
                20 * torch.log10(target_gains[:, m_i, n_i]),
                center_freq=center_freq,
                shelving_crossover=shelving_crossover,
                fs=args.samplerate,
            )
    B = torch.fft.rfft(b, args.nfft, dim=0)
    A = torch.fft.rfft(a, args.nfft, dim=0)
    A[A == 0 + 1j * 0] = torch.tensor(1e-12)
    target_filter = (
        torch.prod(B, dim=1) / (torch.prod(A, dim=1))
    ).detach()  # there's an optimizations tep inside design_geq -> detach required
    ## ---------------- CONSTRUCT GEQ ---------------- ##

    # create another instance of the model
    filt = dsp.GEQ(
        size=(out_ch, in_ch),
        octave_interval=1,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=True,
        alias_decay_db=30,
        device=args.device,
    )
    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
    model = system.Shell(core=filt, input_layer=input_layer, output_layer=output_layer)
    estimation_init = model.get_freq_response()

    ## ---------------- OPTIMIZATION SET UP ---------------- ##
    input = signal_gallery(
        1, n_samples=args.nfft, n=in_ch, signal_type="impulse", fs=args.samplerate
    )
    target = torch.einsum("...ji,...i->...j", target_filter, input_layer(input))

    dataset = Dataset(
        input=input,
        target=torch.abs(target),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        step_size=25,
        train_dir=args.train_dir,
        patience_delta=1e-5,
        device=args.device,
    )
    trainer.register_criterion(nn.MSELoss(), 1)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    estimation = model.get_freq_response()

    # plot magniture response of target and estimated filter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(torch.abs(target[0, :, 0]).cpu().detach().numpy(), label="Target")
    ax1.plot(
        torch.abs(estimation_init[0, :, 0]).cpu().detach().numpy(), label="Estimation Init"
    )
    ax1.plot(torch.abs(estimation[0, :, 0]).cpu().detach().numpy(), "--", label="Estimation")
    ax1.set_title("Magnitude Response")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Magnitude")
    ax1.legend()

    ax2.plot(torch.abs(target[0, :, 1]).cpu().detach().numpy(), label="Target")
    ax2.plot(
        torch.abs(estimation_init[0, :, 1]).cpu().detach().numpy(), label="Estimation Init"
    )
    ax2.plot(torch.abs(estimation[0, :, 1]).cpu().detach().numpy(), "--", label="Estimation")
    ax2.set_title("Magnitude Response")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Magnitude")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.train_dir, "magnitude_response.png"))


def example_parallel_geq(args):
    """
    Example function that demonstrates the training of GEQ command gains.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    ch = 2
    octave_interval = 1
    ## ---------------- TARGET ---------------- ##
    center_freq, shelving_crossover = eq_freqs(interval=octave_interval)
    target_gains = 10 ** (-12 / 20) + (10 ** (12 / 20) - 10 ** (-12 / 20)) * torch.rand(
        size=(len(center_freq) + 2, ch)
    )
    b, a = torch.empty((3, len(center_freq) + 3, ch)), torch.empty(
        (3, len(center_freq) + 3, ch)
    )
    for n in range(ch):
        b[:, :, n], a[:, :, n] = accurate_geq(
            20 * torch.log10(target_gains[:, n]),
            center_freq=center_freq,
            shelving_crossover=shelving_crossover,
            fs=args.samplerate,
        )
    B = torch.fft.rfft(b, args.nfft, dim=0)
    A = torch.fft.rfft(a, args.nfft, dim=0)
    A[A == 0 + 1j * 0] = torch.tensor(1e-12)
    target_filter = (
        torch.prod(B, dim=1) / (torch.prod(A, dim=1))
    ).detach()  # there's an optimizations tep inside accurate_geq -> detach required
    ## ---------------- CONSTRUCT GEQ ---------------- ##

    # create another instance of the model
    filt = dsp.parallelGEQ(
        size=(ch,),
        octave_interval=1,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=True,
        alias_decay_db=30,
        device=args.device,
    )
    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
    model = system.Shell(core=filt, input_layer=input_layer, output_layer=output_layer)
    estimation_init = model.get_freq_response()

    ## ---------------- OPTIMIZATION SET UP ---------------- ##
    input = signal_gallery(
        1, n_samples=args.nfft, n=ch, signal_type="impulse", fs=args.samplerate
    )
    target = torch.einsum("...i,...i->...i", target_filter, input_layer(input))

    dataset = Dataset(
        input=input,
        target=torch.abs(target),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        step_size=25,
        train_dir=args.train_dir,
        patience_delta=1e-5,
        device=args.device,
    )
    trainer.register_criterion(nn.MSELoss(), 1)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    estimation = model.get_freq_response()

    # plot magniture response of target and estimated filter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(torch.abs(target[0, :, 0]).cpu().detach().numpy(), label="Target")
    ax1.plot(
        torch.abs(estimation_init[0, :, 0]).cpu().detach().numpy(), label="Estimation Init"
    )
    ax1.plot(torch.abs(estimation[0, :, 0]).cpu().detach().numpy(), "--", label="Estimation")
    ax1.set_title("Magnitude Response")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Magnitude")
    ax1.legend()

    ax2.plot(torch.abs(target[0, :, 1]).cpu().detach().numpy(), label="Target")
    ax2.plot(
        torch.abs(estimation_init[0, :, 1]).cpu().detach().numpy(), label="Estimation Init"
    )
    ax2.plot(torch.abs(estimation[0, :, 1]).cpu().detach().numpy(), "--", label="Estimation")
    ax2.set_title("Magnitude Response")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Magnitude")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.train_dir, "magnitude_response.png"))

def example_accurate_geq(args):
    """
    Example function that demonstrates the use of the Accurate GEQ.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    in_ch, out_ch = 1, 2
    octave_interval = 1
    ## ---------------- TARGET ---------------- ##
    center_freq, shelving_crossover = eq_freqs(interval=octave_interval)
    target_gains = 10**(-12/20) + (10**(12/20)-10**(-12/20))*torch.rand(size=(len(center_freq) + 2, out_ch, in_ch))
    b, a = torch.empty((3, len(center_freq) + 3, out_ch, in_ch)), torch.empty((3, len(center_freq) + 3, out_ch, in_ch))
    for m_i in range(out_ch):
        for n_i in range(in_ch):
            b[:, :, m_i, n_i], a[:, :, m_i, n_i] = accurate_geq( 
                20*torch.log10(target_gains[:, m_i, n_i]),
                center_freq=center_freq,
                shelving_crossover=shelving_crossover,
                fs=args.samplerate,
                )
    B = torch.fft.rfft(b, args.nfft, dim=0)
    A = torch.fft.rfft(a, args.nfft, dim=0)
    A[A == 0+1j*0] = torch.tensor(1e-12)
    target_filter = (torch.prod(B, dim=1) / (torch.prod(A, dim=1))).detach()    # there's an optimizations tep inside accurate_geq -> detach required 
    ## ---------------- CONSTRUCT GEQ ---------------- ##

    # create another instance of the model 
    filt = dsp.AccurateGEQ(
        size=(out_ch, in_ch), 
        octave_interval=1,
        nfft=args.nfft, 
        fs=args.samplerate,
        alias_decay_db=0,
        device=args.device
    )   
    filt.assign_value(target_gains)
    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x : torch.abs(x))
    model = system.Shell(core=filt, input_layer=input_layer, output_layer=output_layer)    
    filter_response = model.get_freq_response()

    input = signal_gallery(1, n_samples=args.nfft, n=in_ch, signal_type='impulse', fs=args.samplerate)
    target = torch.einsum('...ji,...i->...j', target_filter, input_layer(input))

    # plot magniture response of target and estimated filter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(torch.abs(target[0, :, 0]).detach().numpy(), label='Target')
    ax1.plot(torch.abs(filter_response[0, :, 0]).detach().numpy(), '--', label='Filter response')
    ax1.set_title('Magnitude Response')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    ax1.legend()

    ax2.plot(torch.abs(target[0, :, 1]).detach().numpy(), label='Target')
    ax2.plot(torch.abs(filter_response[0, :, 1]).detach().numpy(), '--', label='Filter response')
    ax2.set_title('Magnitude Response')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.train_dir, 'magnitude_response.png'))

def example_accurate_parallel_geq(args):
    """
    Example function that demonstrates the use of the parallel Accurate GEQ.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    ch = 2
    octave_interval = 1
    ## ---------------- TARGET ---------------- ##
    center_freq, shelving_crossover = eq_freqs(interval=octave_interval)
    target_gains = 10**(-12/20) + (10**(12/20)-10**(-12/20))*torch.rand(size=(len(center_freq) + 2, ch))
    b, a = torch.empty((3, len(center_freq) + 3, ch)), torch.empty((3, len(center_freq) + 3, ch))
    for n in range(ch):
        b[:, :, n], a[:, :, n] = accurate_geq( 
            20*torch.log10(target_gains[:, n]),
            center_freq=center_freq,
            shelving_crossover=shelving_crossover,
            fs=args.samplerate,
            )
    B = torch.fft.rfft(b, args.nfft, dim=0)
    A = torch.fft.rfft(a, args.nfft, dim=0)
    A[A == 0+1j*0] = torch.tensor(1e-12)
    target_filter = (torch.prod(B, dim=1) / (torch.prod(A, dim=1))).detach()    # there's an optimizations tep inside accurate_geq -> detach required 
    ## ---------------- CONSTRUCT GEQ ---------------- ##

    # create another instance of the model 
    filt = dsp.parallelAccurateGEQ(
        size=(ch, ), 
        octave_interval=1,
        nfft=args.nfft, 
        fs=args.samplerate,
        alias_decay_db=0,
        device=args.device
    )   
    filt.assign_value(target_gains)
    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x : torch.abs(x))
    model = system.Shell(core=filt, input_layer=input_layer, output_layer=output_layer)    
    filter_response = model.get_freq_response()

    input = signal_gallery(1, n_samples=args.nfft, n=ch, signal_type='impulse', fs=args.samplerate)
    target = torch.einsum('...i,...i->...i', target_filter, input_layer(input))

    # plot magnitude response of target and estimated filter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(torch.abs(target[0, :, 0]).detach().numpy(), label='Target')
    ax1.plot(torch.abs(filter_response[0, :, 0]).detach().numpy(), label='Filter response')
    ax1.set_title('Magnitude Response')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    ax1.legend()

    ax2.plot(torch.abs(target[0, :, 1]).detach().numpy(), label='Target')
    ax2.plot(torch.abs(filter_response[0, :, 1]).detach().numpy(), '--', label='Filter response')
    ax2.set_title('Magnitude Response')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.train_dir, 'magnitude_response.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument("--num", type=int, default=2**8, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=25, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--masked_loss", type=bool, default=False, help="use masked loss"
    )

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

    example_geq(args)
    example_parallel_geq(args)
    example_accurate_geq(args)
    example_accurate_parallel_geq(args)
    