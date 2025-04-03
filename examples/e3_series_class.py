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


def example_series(args):
    """
    Now, we introduce the Series class.
    You might have noticed that in order to create a sequence of MIMO filters, we have to take care
    of designing the single modules so that their input and output channels match. The same applies
    to some of their attributes (e.g. nfft). The Series class takes care of checking that the modules
    are compatible with each other.
    NOTE: The Series class does not create the number of channels for us, as we want complete control over it.
          It only checks that we made no errors in the connections or in the attributes of the modules.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=False,
        device=args.device,
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    # Series class
    my_dsp = system.Series(input_layer, filter1, filter2, filter3, output_layer)


def example_series_with_error(args):
    """
    Let's purposefully make an error in the attributes/connections of the modules of the Series class.
    The Series class will tell us what the error is and where the error is located.
    It will through an error at the first erroneuos test found.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,),
        nfft=args.nfft * 2,  # NOTE: This will raise an error
        requires_grad=False,
        device=args.device,
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=False,
        device=args.device,
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, 10),  # NOTE: This will raise an error
        nfft=args.nfft,
        requires_grad=False,
        device=args.device,
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    # Series class
    my_dsp = system.Series(input_layer, filter1, filter2, filter3, output_layer)

    return None


def example_series_OrderedDict(args):
    """
    Just as the torch.nn.Sequential class, the Series class accepts an OrderedDict in input.
    This allows us to give a name to each module in the sequence.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=False,
        device=args.device,
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    # Series class
    my_dsp = system.Series(
        OrderedDict(
            {
                "input_layer": input_layer,
                "Gains": filter1,
                "Delays": filter2,
                "Eqs": filter3,
                "output_layer": output_layer,
            }
        )
    )
    print(my_dsp)

    return None


def example_series_nesting(args):
    """
    The Series class is designed to un-nest nested torch.nn.Sequential instances,
    OrderedDict instances, and/or other Series instances.
    This makes checking attribute matching i/o compatibility and between modules easier.
    If nested OrderedDict instances are used to generate the dsp, the custom key names will be maintained.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    btw_ch = 5

    # First filter
    filter1 = dsp.parallelGain(
        size=(in_ch,), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Second filter
    filter2 = dsp.Delay(
        size=(btw_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=False,
        device=args.device,
    )
    # Third filter
    filter3 = dsp.GEQ(
        size=(out_ch, btw_ch), nfft=args.nfft, requires_grad=False, device=args.device
    )
    # Input and output layers
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    # Series class
    filters = OrderedDict({"Gains": filter1, "Delays": filter2, "Eqs": filter3})

    my_dsp = system.Series(
        OrderedDict(
            {
                "input_layer": input_layer,
                "filters": filters,
                "output_layer": output_layer,
            }
        )
    )
    print(my_dsp)

    return None


def example_series_training(args):
    """
    The Series class allows us to train a chain of filters (or just one filter in the chain) as
    we did in the example example_chaining_filters.py's example_requires_grad.
    Let's reproduce the example with the Series class.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 3
    filter1 = dsp.parallelGain(
        size=(in_ch,), nfft=args.nfft, requires_grad=True, device=args.device
    )
    filter2 = dsp.Delay(
        size=(out_ch, in_ch),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    input_layer = dsp.FFT(nfft=args.nfft)
    output_layer = dsp.iFFT(nfft=args.nfft)

    # Series of filters
    filters = OrderedDict(
        {
            "Gains": filter1,
            "Delays": filter2,
        }
    )
    model = system.Series(
        OrderedDict(
            {
                "input_layer": input_layer,
                "filters": filters,
                "output_layer": output_layer,
            }
        )
    )

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

    # Target
    target_gains = [0.5, -1.0]
    target_delays = filter2.s2sample(filter2.param)
    target = torch.zeros((args.nfft, out_ch), device=args.device)
    for i in range(out_ch):
        for j in range(in_ch):
            target[int(torch.round(target_delays[i, j]).item()), i] = target_gains[j]

    # Dataset
    dataset = Dataset(
        input=unit_imp, target=target.unsqueeze(0), expand=args.num, device=args.device
    )
    train_loader, valid_loader = load_dataset(
        dataset, batch_size=args.batch_size, split=args.split
    )

    # ------------ Initialize training process ------------
    criterion = nn.L1Loss()
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=0,
        train_dir=args.train_dir,
        device=args.device,
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
        plt.subplot(out_ch, 1, i + 1)
        plt.plot(ir_init.squeeze()[:, i].cpu().numpy(), label="Initial")
        plt.plot(ir_optim.squeeze()[:, i].cpu().numpy(), label="Optimized")
        plt.plot(target.squeeze()[:, i].cpu().numpy(), "--", label="Target")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.xlim(0, 1200)
        plt.grid()
        plt.title(f"Output channel {i+1}")
    plt.subplot(out_ch, 1, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None


def example_series_utils(args):
    """
    The Series class comes with three utility functions: prepend, append, and insert.
    These functions allow the user to add a modules to the beginning, to the end, or in the middle of the chain.
    All three functions can accept a single module, an OrderedDict, or another Series instance.
    In the case of an OrderedDict and of a Series, the functions will unpack the modules, check that the new keys
    are not already present in the Series, and that the new modules are compatible with the existing ones in terms
    of input/output channels and attributes.
    """
    # ------------------- DSP Definition --------------------
    channel_n = 2
    alias_decay_db = 60
    # Initial filters
    feedforward = dsp.parallelDelay(
        size=(channel_n,),
        max_len=1000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
        alias_decay_db=alias_decay_db,
    )
    feedback = dsp.Matrix(
        size=(channel_n, channel_n),
        nfft=args.nfft,
        matrix_type="orthogonal",
        device=args.device,
        alias_decay_db=alias_decay_db,
    )
    feedback_loop = system.Recursion(fF=feedforward, fB=feedback)

    # Series class instantiation
    my_dsp = system.Series(OrderedDict({"Recursion": feedback_loop}))

    # New filters to add an the beginning
    input_layer = dsp.FFT(nfft=args.nfft)
    input_gains = dsp.parallelGain(
        size=(channel_n,),
        nfft=args.nfft,
        requires_grad=False,
        device=args.device,
        alias_decay_db=alias_decay_db,
    )
    # New filter to add in the middle
    output_gains = dsp.Gain(
        size=(channel_n, channel_n),
        nfft=args.nfft,
        requires_grad=False,
        device=args.device,
        alias_decay_db=alias_decay_db,
    )
    # New filters to add at the end
    equalization = dsp.GEQ(
        size=(channel_n, channel_n),
        nfft=args.nfft,
        requires_grad=False,
        device=args.device,
        alias_decay_db=alias_decay_db,
    )
    output_layer = dsp.iFFTAntiAlias(nfft=args.nfft, alias_decay_db=alias_decay_db)

    # DSP so far
    print(my_dsp)

    # Prepend
    my_dsp.prepend(system.Series(input_layer, input_gains))
    print(my_dsp)

    # Append
    my_dsp.append(
        OrderedDict(
            {
                #           'Eqs': equalization,
                "output_layer": output_layer
            }
        )
    )
    print(my_dsp)

    # Insert
    my_dsp.insert(
        index=3,
        new_module=output_gains,
    )
    print(my_dsp)

    # input signal
    input_signal = signal_gallery(
        signal_type="impulse",
        batch_size=args.batch_size,
        n_samples=args.nfft,
        n=channel_n,
        fs=args.samplerate,
        device=args.device,
    )
    y = my_dsp(input_signal)

    # plot output signal
    plt.figure()
    for i in range(channel_n):
        plt.subplot(channel_n, 1, i + 1)
        plt.plot(y.squeeze()[:, i].cpu().numpy())
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.title(f"Output channel {i+1}")
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
        "--max_epochs", type=int, default=25, help="maximum number of epochs"
    )
    parser.add_argument(
        "--patience_delta",
        type=float,
        default=0.0001,
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
    example_series(args)
    # example_series_with_error(args)
    example_series_OrderedDict(args)
    example_series_nesting(args)
    example_series_training(args)
    example_series_utils(args)
