import torch
import torch.nn as nn
import argparse
import os
import time
from typing import Optional
import matplotlib.pyplot as plt
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.functional import lowpass_filter


class MSELoss(nn.Module):
    """
    Custom loss function that computes the mean squared error.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        # Compute the mean squared error in dB scale
        loss = torch.mean((torch.abs(pred) - torch.abs(target)) ** 2)
        return loss


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for generating inputs and target.
    The targets are frequency responses of the system for different instances of the biquad filter.
    The input to the MLP is the target response, while the input to the DDSP system is an impulse.
    """

    def __init__(self, args, ch, num, n_sections, delay_lengths):
        # Create the input to the ddsp
        input_biquad = torch.zeros((1, args.nfft, ch, ch), device=args.device)
        input_biquad[:, 0, ...] = 1
        input_biquad = input_biquad.expand(
            tuple([num] + [d for d in input_biquad.shape[1:]])
        )
        self.input_biquad = input_biquad

        # Create many instances of recursive filter as target
        target = torch.ones((num, args.nfft // 2 + 1, ch, ch)) + 1j * torch.zeros(
            (num, args.nfft // 2 + 1, ch, ch),
            device=args.device,
        )
        z = torch.polar(torch.ones(args.nfft // 2 + 1), torch.fft.rfftfreq(args.nfft))
        d = torch.diag_embed(torch.unsqueeze(z, dim=-1) ** torch.tensor(delay_lengths))
        for i in range(num):
            target_filter = generate_biquad_filter(args, ch, ch, n_sections)
            open_loop = torch.inverse(
                torch.eye(ch) - torch.matmul(d, target_filter.to(d.dtype))
            )

            target[i] = torch.matmul(open_loop, d)

        self.target = torch.abs(target)
        self.input = []
        for i in range(num):
            self.input.append((target[i], input_biquad[i]))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.input[index], self.target[index]


class LossyBiquad(dsp.Biquad):
    """
    Subclass of Biquad that forces the gain of the filter to be less than 0 dB.
    This is also an example on how you can customize the different dsp modules in
    flamo to achieve the exact behavior you want.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        n_sections: int = 1,
        filter_type: str = "lowpass",
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(
            size=size,
            n_sections=n_sections,
            filter_type=filter_type,
            nfft=nfft,
            fs=fs,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device,
        )

    def get_map(self):

        match self.filter_type:
            case "lowpass" | "highpass":
                self.map = lambda x: torch.clamp(
                    torch.stack(
                        (
                            torch.sigmoid(x[:, 0, :, :] * 0.25),
                            20 * torch.log10(torch.sigmoid(x[:, 1, :, :] * 0.25)),
                        ),
                        dim=1,
                    ),
                    min=torch.tensor([0, -60], device=self.device)
                    .view(-1, 1, 1)
                    .expand_as(x),
                    max=torch.tensor([1, -0.1], device=self.device)
                    .view(-1, 1, 1)
                    .expand_as(x),
                )
            case "bandpass":
                self.map = lambda x: torch.clamp(
                    torch.stack(
                        (
                            x[:, 0, :, :],
                            x[:, 1, :, :],
                            20 * torch.log10(torch.sigmoid(torch.abs(x[:, -1, :, :]))),
                        ),
                        dim=1,
                    ),
                    min=torch.tensor(
                        [
                            0 + torch.finfo(torch.float).eps,
                            0 + torch.finfo(torch.float).eps,
                            -60,
                        ],
                        device=self.device,
                    )
                    .view(-1, 1, 1)
                    .expand_as(x),
                    max=torch.tensor(
                        [
                            1 - torch.finfo(torch.float).eps,
                            1 - torch.finfo(torch.float).eps,
                            0,
                        ],
                        device=self.device,
                    )
                    .view(-1, 1, 1)
                    .expand_as(x),
                )


class nnComb(nn.Module):
    """
    Neural network model composed of a stack of MLPs and a recursive system
    with delay lines in the feedforward path and biquad filters in the feedback path.
    The system is multichannel. An MLP hypeconditions the parameters of the biquads,
    namely the cutoff frequency and the gain.
    """

    def __init__(self, n_sect, n_param, ch, delay_lengths, args):
        super(nnComb, self).__init__()

        self.n_sect = n_sect
        self.ch = ch
        self.n_param = n_param

        # Stack of MLPs
        self.stack = nn.Sequential(
            nn.Linear(args.nfft // 2 + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Final dense layer to ensure output shape matches that of the LossyBiquad class
        self.final_dense = nn.Linear(64, n_sect * n_param)

        # Create another instance of the model
        filt = LossyBiquad(
            size=(ch, ch),
            n_sections=n_sect,
            filter_type="lowpass",
            nfft=args.nfft,
            fs=args.samplerate,
            requires_grad=True,
            alias_decay_db=args.alias_decay_db,
            device=args.device,
        )

        delays = dsp.parallelDelay(
            size=(ch,),
            max_len=delay_lengths.max(),
            nfft=args.nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=args.alias_decay_db,
            device=args.device,
        )
        delays.assign_value(delays.sample2s(delay_lengths))
        # To condition the biquad filter, we will need to pass the parameters estimated
        # by the neural network as a directory. For this to work, we need to use OrderedDict.
        # The Recursion class will automatically create an instance of OrderedDict in which
        # the keys are "feedforward" and "feedback".

        # Recursion
        comb = system.Recursion(fF=delays, fB=filt)

        # Create the model with Shell
        input_layer = dsp.FFT(args.nfft)
        output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
        self.comb = system.Shell(
            core=comb, input_layer=input_layer, output_layer=output_layer
        )

    def forward(self, data):

        # Data consists in a tuple of input to the MLPs and input to the DDSP
        x = torch.abs(data[0])  # input to the MLP
        z = data[1]  # input to the DDSP

        # Reshape the tensor so that the frequencies are in the last dimension
        x = x.permute(0, 2, 3, 1)
        # Pass through the three stacks
        x = self.stack(x)

        # Pass through the final dense layer
        x = self.final_dense(x)

        # Reshape the output
        x = x.view(-1, self.n_sect, self.n_param, self.ch, self.ch)

        # As of now, this is the only way to process batches larger than 1:
        y = []
        for i in range(x.size(0)):
            # Create a dictionary whose key is the name of the module whose parameters are to be estimated
            param_dict = {"feedback": x[i]}
            y.append(self.comb(z[0].unsqueeze(0), param_dict))
        y = torch.vstack(y)
        return y


def example_comb_nn(args):
    """
    Example function that demonstrates the training of a recursive system in with
    delay lines in the feedforward bath and biquad filters in the feedback path.
    The system is multichannel. An MLP hypeconditions the parameters of the biquads, namely the cutoff frequency and the gain.
    We want the system to be stable, thus the gain of the biquad filters should be smaller than 0dB.
    To achieve that we need to overwrite the  get_map method of the Biquad class (check LossyBiquad class above).
    NOTE: the model and the parameterizations should be fine tuned for better results. This example serves just as a demo of the API.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    delay_lengths = [151, 211, 239, 317]
    ch = len(delay_lengths)
    n_sections = 1
    n_param = 2  # highpass filter

    model = nnComb(
        n_sect=n_sections,
        n_param=n_param,
        ch=ch,
        delay_lengths=torch.tensor(delay_lengths),
        args=args,
    )

    # Create a dataset
    dataset = Dataset(args, ch, args.num, n_sections, delay_lengths)
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        train_dir=args.train_dir,
        device=args.device,
        step_size=10,
        patience_delta=1e-5,
    )

    trainer.register_criterion(MSELoss(), 1)

    trainer.train(train_loader, valid_loader)

    data, target = next(iter(valid_loader))
    estimation = model(data)
    freq_axis = torch.fft.rfftfreq(args.nfft, 1 / args.samplerate)
    plt.plot(
        freq_axis,
        20 * torch.log10(torch.abs(target[0, :, 0, 0])).cpu().detach().numpy(),
        label="target",
    )
    plt.plot(
        freq_axis,
        20 * torch.log10(torch.abs(estimation[0, :, 0, 0])).cpu().detach().numpy(),
        label="estimation",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    # save the fig
    plt.savefig(os.path.join(args.train_dir, "estimation.png"))


def generate_biquad_filter(args, in_ch, out_ch, n_sections):
    """
    Generate biquad filter response.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        n_sections: Number of sections in the biquad filter.
    Returns:
        target_filter: Generated biquad filter coefficients.
    """
    target_filter = torch.zeros(args.nfft // 2 + 1, out_ch, in_ch) / torch.zeros(
        args.nfft // 2 + 1, out_ch, in_ch
    )
    while torch.isnan(torch.abs(target_filter)).any():
        b, a = lowpass_filter(
            fc=torch.tensor(args.samplerate // 4)
            * torch.rand(size=(n_sections, out_ch, in_ch)),
            gain=torch.tensor(-5.1)
            + torch.tensor(5) * torch.rand(size=(n_sections, out_ch, in_ch)),
            fs=args.samplerate,
        )
        B = torch.fft.rfft(b.to(torch.double), args.nfft, dim=0)
        A = torch.fft.rfft(a.to(torch.double), args.nfft, dim=0)
        target_filter = torch.prod(B, dim=1) / torch.prod(A, dim=1)
    return target_filter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=32000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=16000, help="sampling rate")
    parser.add_argument("--num", type=int, default=2**10, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size for training"
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
    parser.add_argument(
        "--alias_decay_db",
        type=int,
        default=0,
        help="maximum attenuation of the anti time-aliasing envelope in dB",
    )
    args = parser.parse_args()

    # Check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # Save arguments
    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    example_comb_nn(args)
