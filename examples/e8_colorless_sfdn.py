import torch
import argparse
import os
import time
import scipy

from collections import OrderedDict

from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import sparsity_loss, masked_mse_loss
from flamo.utils import save_audio

torch.manual_seed(130709)


def example_fdn(args):
    """
    Example function that demonstrates the construction and training of a Feedback Delay Network (FDN) model
    with scattering feedback matrix and sparse marsking of the loss.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """

    # FDN parameters
    N = 6  # number of delays
    alias_decay_db = 30  # alias decay in dB
    delay_lengths = torch.tensor([997, 1153, 1327, 1559, 1801, 2099])
    args.num = (args.nfft // 2 + 1) // 2000

    ## ---------------- CONSTRUCT FDN ---------------- ##

    # Input and output gains
    input_gain = dsp.Gain(
        size=(N, 1),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    output_gain = dsp.Gain(
        size=(1, N),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    # Feedback loop with delays
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=delay_lengths.max(),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    # Feedback path with scattering matrix
    m_L = torch.randint(
        low=1,
        high=int(torch.floor(min(delay_lengths) / 2)),
        size=[N],
        device=args.device,
    )
    m_R = torch.randint(
        low=1,
        high=int(torch.floor(min(delay_lengths) / 2)),
        size=[N],
        device=args.device,
    )
    feedback = dsp.ScatteringMatrix(
        size=(4, N, N),
        nfft=args.nfft,
        gain_per_sample=1,
        sparsity=3,
        m_L=m_L,
        m_R=m_R,
        alias_decay_db=alias_decay_db,
        requires_grad=True,
        device=args.device,
    )

    # Recursion
    feedback_loop = system.Recursion(fF=delays, fB=feedback)

    # Full FDN
    FDN = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    # Get initial impulse response
    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )
        save_fdn_params(model, filename="parameters_init")

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device,
    )
    trainer.register_criterion(
        masked_mse_loss(
            nfft=args.nfft,
            n_samples=2000,
            n_sets=1,
            regenerate_mask=True,
            device=args.device,
        ),
        1,
    )
    trainer.register_criterion(sparsity_loss(), 0.2, requires_model=True)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get optimized impulse response
    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)),
            fs=args.samplerate,
        )
        save_fdn_params(model, filename="parameters_optim")


def save_fdn_params(net, filename="parameters"):
    r"""
    Retrieves the parameters of a feedback delay network (FDN) from a given network and saves them in .mat format.

    **Parameters**:
        net (Shell): The Shell class containing the FDN.
        filename (str): The name of the file to save the parameters without file extension.
    **Returns**:
        dict: A dictionary containing the FDN parameters.
            - 'A' (ndarray): The feedback loop parameter A.
            - 'B' (ndarray): The input gain parameter B.
            - 'C' (ndarray): The output gain parameter C.
            - 'm' (ndarray): The feedforward parameter m.
    """

    core = net.get_core()
    param = {}
    param["A"] = core.feedback_loop.feedback.param.squeeze().detach().cpu().numpy()
    param["B"] = core.input_gain.param.squeeze().detach().cpu().numpy()
    param["C"] = core.output_gain.param.squeeze().detach().cpu().numpy()
    param["m"] = (
        core.feedback_loop.feedforward.s2sample(
            core.feedback_loop.feedforward.map(core.feedback_loop.feedforward.param)
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    scipy.io.savemat(os.path.join(args.train_dir, filename + ".mat"), param)

    return param


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=48000 * 2, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=40, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument("--masked_loss", action="store_true", help="use masked loss")

    args = parser.parse_args()

    # check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, will use cpu")

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

    example_fdn(args)
