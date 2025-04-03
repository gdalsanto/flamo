import torch
import torch.nn as nn
import argparse
import os
import time
from collections import OrderedDict
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.functional import signal_gallery, highpass_filter

torch.manual_seed(130798)


class dBMSELoss(nn.Module):
    """
    Custom loss function that computes the mean squared error in dB scale.
    """

    def __init__(self):
        super(dBMSELoss, self).__init__()

    def forward(self, pred, target):
        # Compute the mean squared error in dB scale
        loss = torch.mean(
            (torch.abs(pred) - torch.abs(target))
            ** 2
        )
        return loss


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for generating biquad filter data.
    """

    def __init__(self, args, in_ch, out_ch, num, n_sections):
        # Create the input to the ddsp
        input_biquad = torch.zeros((1, args.nfft, in_ch), device=args.device)
        input_biquad[:, 0, :] = 1
        input_biquad = input_biquad.expand(
            tuple([num] + [d for d in input_biquad.shape[1:]])
        )
        self.input_biquad = input_biquad

        # Create many instances of biquad filters as target
        target = torch.ones((num, args.nfft // 2 + 1, out_ch)) + 1j * torch.zeros(
            (num, args.nfft // 2 + 1, out_ch),
            device=args.device,
        )
        input_layer = dsp.FFT(args.nfft)
        imp = signal_gallery(
            1, n_samples=args.nfft, n=in_ch, signal_type="impulse", fs=args.samplerate
        )
        for i in range(num):
            target_filter = generate_biquad_filter(args, in_ch, out_ch, n_sections)
            target[i] = torch.einsum(
                "...ji,...i->...j", target_filter, input_layer(imp)
            )
        self.target = torch.abs(target)
        self.input = []
        for i in range(num):
            self.input.append((target[i], input_biquad[i]))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.input[index], self.target[index]


class nnBiquad(nn.Module):
    """
    Neural network model for biquad filter coefficient prediction.
    """

    def __init__(self, n_sect, n_param, in_ch, out_ch, args):
        super(nnBiquad, self).__init__()

        self.n_sect = n_sect
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_param = n_param

        # Stack of MLPs
        self.stack = nn.Sequential(
            nn.Linear(args.nfft // 2 + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Final dense layer to ensure output shape (3, n_sections, in_channels, out_channels)
        self.final_dense = nn.Linear(256, n_sect * n_param * in_ch)

        # Create another instance of the model
        filt = dsp.Biquad(
            size=(out_ch, in_ch),
            n_sections=n_sect,
            filter_type="highpass",
            nfft=args.nfft,
            fs=args.samplerate,
            requires_grad=False,
            alias_decay_db=30,
            device=args.device,
        )

        # To condition the biquad filter, we will need to pass the parameters estimated
        # by the neural network as a directory. For this to work, we need to use OrderedDict.
        # OrderedDict allows us to indentify the modules whose parameters are to be estimated by
        # the neural network via their keys.
        core = OrderedDict(
            {
                "biquad": filt,
            }
        )

        # Create the model with Shell
        input_layer = dsp.FFT(args.nfft)
        output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
        self.biquad = system.Shell(
            core=core, input_layer=input_layer, output_layer=output_layer
        )

    def forward(self, data):

        # Data consists in a tuple of input to the MLPs and input to the DDSP
        x = torch.abs(data[0])  # input to the MLP
        z = data[1]  # input to the DDSP

        # Reshape the tensor so that the frequencies are in the last dimension
        x = x.permute(0, 2, 1)
        # Pass through the three stacks
        x = self.stack(x)

        # Pass through the final dense layer
        x = self.final_dense(x)

        # Reshape the output to (3, n_sections, in_channels, out_channels)
        x = x.view(-1, self.n_sect, self.n_param, self.out_ch, self.in_ch)
        x[:, :, 0, :, :] = torch.sigmoid(x[:, :, 0, :, :] * 0.25)

        # As of now, this is the only way to process batches larger than 1:
        y = []
        for i in range(x.size(0)):
            # Create a dictionary whose key is the name of the module whose parameters are to be estimated
            param_dict = {"biquad": x[i]}
            y.append(self.biquad(z[0].unsqueeze(0), param_dict))

        y = torch.vstack(y)
        return y


def example_biquad_nn(args):
    """
    Example function that demonstrates the training of biquad coefficients via MLPs.
    This example shows how to use flamo's modules within a neural network model.
    NOTE: the model and the parameterizations should be fine tuned for better results. This example serves just as a demo of the API.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    in_ch, out_ch = 1, 4
    n_sections = 1
    n_param = 2  # highpass filter
    model = nnBiquad(
        n_sect=n_sections, n_param=n_param, in_ch=in_ch, out_ch=out_ch, args=args
    )

    # Create a dataset
    dataset = Dataset(args, in_ch, out_ch, args.num, n_sections)
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        train_dir=args.train_dir,
        device=args.device,
        step_size=10,
        patience_delta=1e-5,
    )

    trainer.register_criterion(dBMSELoss(), 1)

    trainer.train(train_loader, valid_loader)


def generate_biquad_filter(args, in_ch, out_ch, n_sections):
    """
    Generate biquad filter coefficients.
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
        b, a = highpass_filter(
            fc=torch.tensor(args.samplerate // 2)
            * torch.rand(size=(n_sections, out_ch, in_ch)),
            gain=torch.tensor(-1)
            + (torch.tensor(2)) * torch.rand(size=(n_sections, out_ch, in_ch)),
            fs=args.samplerate,
        )
        B = torch.fft.rfft(b.to(torch.double), args.nfft, dim=0)
        A = torch.fft.rfft(a.to(torch.double), args.nfft, dim=0)
        target_filter = torch.prod(B, dim=1) / torch.prod(A, dim=1)
    return target_filter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=4096, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument("--num", type=int, default=2**10, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--masked_loss", type=bool, default=False, help="use masked loss"
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

    # Run the example biquad neural network training
    example_biquad_nn(args)
