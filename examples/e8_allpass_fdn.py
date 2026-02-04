import argparse
import math
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from flamo.functional import skew_matrix
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.loss import masked_mse_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.utils import save_audio, to_complex

torch.manual_seed(130799)


def _block_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    blocks = []
    for theta in angles:
        c = torch.cos(theta)
        s = torch.sin(theta)
        blocks.append(torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
    return torch.block_diag(*blocks)


class AllpassFDNMatrix(nn.Module):
    """
    Feedback matrix for all-pass FDNs using
        A_AP = [[-QG, Q],
                [I - G^2, G]]
    where Q is unitary and G is diagonal with entries in (-1, 1).
    """

    def __init__(
        self,
        N: int,
        nfft: int,
        alias_decay_db: float = 0.0,
        q_type: str = "block_rotation",
        requires_grad: bool = True,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if q_type == "block_rotation" and (N % 2 != 0):
            raise ValueError("block_rotation requires an even N")

        self.N = N
        self.nfft = nfft
        self.dtype = dtype
        self.device = device
        self.alias_decay_db = torch.tensor(
            alias_decay_db, device=self.device, dtype=self.dtype
        )
        self.input_channels = 2 * N
        self.output_channels = 2 * N
        self.q_type = q_type

        if q_type == "block_rotation":
            self.q_param = nn.Parameter(
                torch.randn(N // 2, device=self.device, dtype=self.dtype),
                requires_grad=requires_grad,
            )
        elif q_type == "orthogonal":
            self.q_param = nn.Parameter(
                torch.randn(N, N, device=self.device, dtype=self.dtype),
                requires_grad=requires_grad,
            )
        else:
            raise ValueError(f"Unknown q_type: {q_type}")

        self.g_param = nn.Parameter(
            torch.randn(N, device=self.device, dtype=self.dtype),
            requires_grad=requires_grad,
        )

    def _map_q(self) -> torch.Tensor:
        if self.q_type == "block_rotation":
            angles = math.pi * torch.tanh(self.q_param)
            return _block_rotation_matrix(angles)
        return torch.matrix_exp(skew_matrix(self.q_param))

    def _map_g(self) -> torch.Tensor:
        return 0.99 * torch.tanh(self.g_param)

    def get_matrix(self) -> torch.Tensor:
        Q = self._map_q()
        g = self._map_g()
        I = torch.eye(self.N, device=Q.device, dtype=Q.dtype)
        G = torch.diag(g)
        A_top = torch.cat([-(Q * g), Q], dim=1)
        A_bottom = torch.cat([I - torch.diag(g**2), G], dim=1)
        return torch.cat([A_top, A_bottom], dim=0)

    def forward(self, x: torch.Tensor, ext_param=None) -> torch.Tensor:
        if ext_param is not None:
            raise ValueError("External parameters are not supported for AllpassFDNMatrix.")
        A = self.get_matrix()
        return torch.einsum("mn,bfn...->bfm...", to_complex(A), x)


def example_allpass_fdn(args):
    """
    Example function that demonstrates a trainable all-pass FDN feedback matrix.
    For N=4, Q is built as two 2x2 rotation blocks ("two N=2 all-pass" sections).
    """

    # All-pass FDN parameters
    N = args.N  # all-pass order
    alias_decay_db = args.alias_decay_db
    n_delays = 2 * N  # standard FDN size from the all-pass transformation

    # Delay lengths: [m1...mN, m1'...mN]
    fdn_delays = (593 + 150 * torch.arange(N, device=args.device)).to(torch.int64)
    ap_delays = (431 + 120 * torch.arange(N, device=args.device)).to(torch.int64)
    delay_lengths = torch.cat([fdn_delays, ap_delays])

    # Input and output gains (fixed by default)
    input_gain = dsp.Gain(
        size=(n_delays, 1),
        nfft=args.nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    output_gain = dsp.Gain(
        size=(1, n_delays),
        nfft=args.nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    init_in = torch.ones((n_delays, 1), device=args.device, dtype=args.dtype) / math.sqrt(
        n_delays
    )
    init_out = torch.ones((1, n_delays), device=args.device, dtype=args.dtype) / math.sqrt(
        n_delays
    )
    input_gain.assign_value(init_in)
    output_gain.assign_value(init_out)

    # Feedforward delays
    delays = dsp.parallelDelay(
        size=(n_delays,),
        max_len=delay_lengths.max(),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    # All-pass feedback matrix with trainable Q and G
    feedback = AllpassFDNMatrix(
        N=N,
        nfft=args.nfft,
        alias_decay_db=alias_decay_db,
        q_type=args.q_type,
        requires_grad=True,
        device=args.device,
        dtype=args.dtype,
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
    input_layer = dsp.FFT(args.nfft, dtype=args.dtype)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x), dtype=args.dtype)
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    # Save initial impulse response (time domain)
    with torch.no_grad():
        ir_init = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )

    # ---------------- OPTIMIZATION SET UP ----------------
    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=args.num,
        device=args.device,
        dtype=args.dtype,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

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
            n_samples=12000,
            n_sets=1,
            regenerate_mask=True,
            device=args.device,
        ),
        1,
    )

    # ---------------- TRAIN ----------------
    trainer.train(train_loader, valid_loader)

    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)),
            fs=args.samplerate,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=4, help="all-pass order")
    parser.add_argument("--q_type", type=str, default="block_rotation", choices=["block_rotation", "orthogonal"])
    parser.add_argument("--alias_decay_db", type=float, default=30.0, help="alias decay in dB")
    parser.add_argument("--nfft", type=int, default=48000 * 4, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="data type for tensors",
    )
    parser.add_argument("--num", type=int, default=2**8, help="dataset size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, will use cpu")

    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    example_allpass_fdn(args)
