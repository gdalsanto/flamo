import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp, system
from flamo.functional import mag2db, get_magnitude, get_eigenvalues, signal_gallery
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer


def example_shell(args):
    """
    In all previous examples, we always defined input and output layers for the DSP.
    This is because the DSP needs to interface with the dataset input and to provide the output in
    a format that is compatible with the loss function.
    In this section we introduce the Shell class.
    The Shell class possess three attributes: input_layer, core and output_layer.
    The core is the DSP or the model that we want to train.
    The input_layer and the output_layer are the same ones we so far defined.
    The important property of the Shell class is that is keeps the three components separate instead
    of merging them into a single torch.nn.Sequential object, torch.nn.Model, or Series class instance.
    This means we can easily keep the dsp fixed and change the input and output layers according to
    the dataset and the loss function that we want to use.
    The Shell function provides dedicated methods to change the input and output layers.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(size=(out_ch, in_ch), nfft=args.nfft, device=args.device)
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filter3 = dsp.parallelFilter(
        size=(
            50,
            out_ch,
        ),
        nfft=args.nfft,
        device=args.device,
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=1,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filters = OrderedDict(
        {"Gain": filter1, "Delay": filter2, "FIR": filter3, "SVF": filter4}
    )

    # Shell
    my_dsp = system.Shell(core=filters)

    # ---------- DSP time and frequency responses -----------

    # Input unit impulse
    unit_imp = signal_gallery(
        signal_type="impulse",
        batch_size=1,
        n_samples=args.samplerate,
        n=in_ch,
        fs=args.samplerate,
        device=args.device,
    )

    # Time response
    my_dsp.set_inputLayer(dsp.FFT(nfft=args.nfft))
    my_dsp.set_outputLayer(dsp.iFFT(nfft=args.nfft))
    imp_resp = my_dsp(unit_imp)

    # Magnitude response
    my_dsp.set_outputLayer(
        nn.Sequential(dsp.Transform(get_magnitude), dsp.Transform(mag2db))
    )
    mag_resp = my_dsp(unit_imp)

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.subplot(211)
    plt.plot(imp_resp.squeeze().cpu().numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.xlim(0, 5500)
    plt.subplot(212)
    plt.plot(mag_resp.squeeze().cpu().numpy())
    plt.xlabel("Frequency bins")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    return None


def example_shell_error(args):
    """
    In addition to that, the Shell checks that all three components
    are comparitible in terms of attributes and input/output channels.
    Let's insert a filter in the input layer and set the wrong number of channels on purpose.
    The Shell class will tell us what the error is and where the error is located.
    It will through an error at the first erroneuos test found.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 3
    # Filters
    filter1 = dsp.Gain(size=(out_ch, in_ch), nfft=args.nfft, device=args.device)
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filter3 = dsp.parallelFilter(
        size=(
            50,
            out_ch,
        ),
        nfft=args.nfft,
        device=args.device,
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=2,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filters = OrderedDict(
        {"Gain": filter1, "Delay": filter2, "FIR": filter3, "SVF": filter4}
    )

    # Layers
    in_layer1 = dsp.FFT(nfft=args.nfft)
    in_layer2 = dsp.Gain(
        size=(in_ch, in_ch), nfft=args.nfft, requires_grad=False, device=args.device
    )  # NOTE: Error here
    in_layer = nn.Sequential(in_layer1, in_layer2)

    out_layer = dsp.iFFT(nfft=2**8)  # NOTE: Error here

    # Shell
    my_dsp = system.Shell(input_layer=in_layer, core=filters, output_layer=out_layer)

    return None


def example_shell_gets(args):
    """
    Finally, the Shell class provides useful functionalities to retrieve the time and frequency responses
    directly, without having to modify input and output layers manually.
    We can repeat the first example of the section using the get_time_response() and get_freq_response() methods.
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(size=(out_ch, in_ch), nfft=args.nfft, device=args.device)
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filter3 = dsp.parallelFilter(
        size=(
            50,
            out_ch,
        ),
        nfft=args.nfft,
        device=args.device,
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=1,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filters = OrderedDict(
        {"Gain": filter1, "Delay": filter2, "FIR": filter3, "SVF": filter4}
    )

    # Shell
    my_dsp = system.Shell(core=filters)

    # ----------- DSP time a frequency responses ------------

    # Time response
    imp_resp = my_dsp.get_time_response(fs=args.samplerate)

    # Magnitude response
    freq_resp = my_dsp.get_freq_response(fs=args.samplerate)
    mag_resp = mag2db(get_magnitude(freq_resp))

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.subplot(211)
    plt.plot(imp_resp.squeeze().cpu().numpy())
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.xlim(0, 5500)
    plt.subplot(212)
    plt.plot(mag_resp.squeeze().cpu().numpy())
    plt.xlabel("Frequency bins")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    return None


def example_shell_gets_2(args):
    """
    Thanks to the Shell class it is very easy to check what is happening inside the core.
    We can retrieve the time and frequency responses of:
        - the full dsp considering the final mixing (previous example)
        - the single modules considering the final mixing NOTE: not implemented yet
        - the full dsp without the final mixing (see Shell.get_time_response() documentation for more details)
        - the single modules without the final mixing NOTE: not implemented yet
    """
    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(size=(out_ch, in_ch), nfft=args.nfft, device=args.device)
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filter3 = dsp.parallelFilter(
        size=(
            50,
            out_ch,
        ),
        nfft=args.nfft,
        device=args.device,
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=1,
        nfft=args.nfft,
        fs=args.samplerate,
        device=args.device,
    )
    filters = OrderedDict(
        {"Gain": filter1, "Delay": filter2, "FIR": filter3, "SVF": filter4}
    )

    # Shell
    my_dsp = system.Shell(core=filters)

    # ----------- DSP time a frequency responses ------------

    # Time response
    imp_resp = my_dsp.get_time_response(fs=args.samplerate, identity=True)

    # Magnitude response
    freq_resp = my_dsp.get_freq_response(fs=args.samplerate, identity=True)
    mag_resp = mag2db(get_magnitude(freq_resp))

    # ------------------------ Plot -------------------------
    plt.figure()
    for i in range(in_ch):
        plt.subplot(2, in_ch, i + 1)
        plt.plot(imp_resp.squeeze().cpu().numpy()[:, i])
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.xlim(0, 5500)
        plt.title(f"Input channel {i+1}")
        plt.subplot(2, in_ch, i + 3)
        plt.plot(mag_resp.squeeze().cpu().numpy()[:, i])
        plt.xlabel("Frequency bins")
        plt.ylabel("Magnitude")
        plt.grid()
    plt.tight_layout()
    plt.show()

    return None


def example_shell_training(args):
    """
    In this example, we will see how flexible the Shell class can be.
    We will define the DSP just one time. We will istantiate the Shell class only once.
    We will then change input and output layers of the Shell accordingly to the type of
    training we want to perform.
    """
    # ------------------ Model Definition -------------------
    FIR_order = 1000
    in_ch = 2
    out_ch = 2
    my_dsp = dsp.Filter(
        size=(FIR_order, out_ch, in_ch),
        nfft=args.nfft,
        requires_grad=True,
        device=args.device,
    )

    # Shell instance
    model = system.Shell(core=my_dsp)

    # Get the initial response for the comparison
    fr_init = model.get_freq_response(fs=args.samplerate, identity=False)
    all_fr_init = model.get_freq_response(fs=args.samplerate, identity=True)
    evs_init = get_magnitude(
        get_eigenvalues(model.get_freq_response(fs=args.samplerate, identity=True))
    )

    # ========================================================================================================
    # Case 1: Train the DSP to have all flat magnitude responses
    # NOTE: We need the Shell.input_layer to modify the input in order to obtain the system's frequency
    #       responses without the final mixing (see Shell.get_time_response() documentation for more details).
    #       We will use the torchTensor.diag_embed() method to create a diagonal matrix from the input tensor.
    #       We will also need the Shell.output_layer to compute the magnitude of the output frequency responses.

    # Initialize dataset
    dataset = DatasetColorless(
        input_shape=(args.batch_size, args.nfft // 2 + 1, in_ch),
        target_shape=(args.batch_size, args.nfft // 2 + 1, out_ch, in_ch),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize loss function
    criterion = torch.nn.MSELoss()

    # Interface DSP with dataset and loss function
    model.set_inputLayer(
        nn.Sequential(dsp.Transform(lambda x: x.diag_embed()), dsp.FFT(args.nfft))
    )
    model.set_outputLayer(dsp.Transform(get_magnitude))

    # Initialize training process
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device,
    )
    trainer.register_criterion(criterion, 1)
    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    all_fr_optim = model.get_freq_response(fs=args.samplerate, identity=True)

    plt.figure()
    for i in range(out_ch):
        for j in range(in_ch):
            plt.subplot(out_ch, in_ch, i * in_ch + j + 1)
            plt.plot(
                mag2db(get_magnitude(all_fr_init[0, :, i, j]).detach()).cpu().numpy(),
                label="Init",
            )
            plt.plot(
                mag2db(get_magnitude(all_fr_optim[0, :, i, j]).detach()).cpu().numpy(),
                label="Optim",
            )
            plt.xlabel("Frequency bins")
            plt.ylabel("Magnitude [dB]")
            plt.grid()
    plt.legend()
    plt.suptitle("Filter's interior magnitude responses")
    plt.tight_layout()
    plt.show()

    # ========================================================================================================
    # Case 2: Train the DSP to have all eigenvalues equal to 1 in magnitude
    # NOTE: We will keep the same Shell.input_layer as before.
    #       We will also need the Shell.output_layer to compute first the eigenvalues of the complex output
    #       frequency responses, and then the magnitude of the eigenvalues.

    # Change the dataset
    dataset = DatasetColorless(
        input_shape=(args.batch_size, args.nfft // 2 + 1, in_ch),
        target_shape=(
            args.batch_size,
            args.nfft // 2 + 1,
            out_ch,
        ),  # The target has the a different shape now
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Interface DSP with new loss function
    model.set_outputLayer(
        nn.Sequential(dsp.Transform(get_eigenvalues), dsp.Transform(get_magnitude))
    )

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    evs_optim = get_magnitude(
        get_eigenvalues(model.get_freq_response(fs=args.samplerate, identity=True))
    )

    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i + 1)
        plt.plot(
            mag2db(torch.abs(evs_init[0, :, i])).detach().cpu().numpy(), label="Init"
        )
        plt.plot(
            mag2db(torch.abs(evs_optim[0, :, i])).detach().cpu().numpy(), label="Optim"
        )
        plt.xlabel("Frequency bins")
        plt.ylabel("Magnitude [dB]")
        plt.grid()
    plt.legend()
    plt.suptitle("Filters eigenvalues")
    plt.tight_layout()
    plt.show()

    # ========================================================================================================
    # Case 3: Train the DSP to have flat output responses
    # NOTE: Finally, we want to include the final mixing in the optimization.
    #       We will modify the Shell.input_layer accordingly.
    #       We will also remove the eigenvalue computation from the Shell.output_layer.
    #       In this case we can also keep the last DatasetColorless instance.

    # Interface DSP with dataset and loss function
    model.set_inputLayer(dsp.FFT(args.nfft))
    model.set_outputLayer(dsp.Transform(get_magnitude))

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    fr_optim = model.get_freq_response(fs=args.samplerate, identity=False)

    plt.figure()
    for i in range(out_ch):
        plt.subplot(out_ch, 1, i + 1)
        plt.plot(
            mag2db(get_magnitude(fr_init[0, :, i]).detach()).cpu().numpy(), label="Init"
        )
        plt.plot(
            mag2db(get_magnitude(fr_optim[0, :, i]).detach()).cpu().numpy(),
            label="Optim",
        )
        plt.xlabel("Frequency bins")
        plt.ylabel("Magnitude [dB]")
        plt.grid()
    plt.legend()
    plt.suptitle("Filters magnitude responses")
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
    example_shell(args)
    # example_shell_error(args)
    example_shell_gets(args)
    example_shell_gets_2(args)
    example_shell_training(args)
