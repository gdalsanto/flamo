import math
from typing import Optional
import torch
import torch.nn as nn
import argparse
import os
import time
import auraloss
import soundfile as sf

from collections import OrderedDict

from flamo.auxiliary.reverb import parallelFirstOrderShelving
from flamo.optimize.dataset import load_dataset
from flamo.optimize.loss import sparsity_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.functional import find_onset

torch.manual_seed(130798)


class MultiResoSTFT(nn.Module):
    """compute the mean absolute error between the auraloss of two RIRs"""

    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir1, rir2):
        return self.MRstft(rir1.permute(0, 2, 1), rir2.permute(0, 2, 1))
    

def count_learnable_params(module: nn.Module) -> tuple[OrderedDict, OrderedDict, int]:
    """
    Recursively walk every named submodule inside `module` and record how
    many learnable parameters each *leaf* DSP module owns, together with the
    shape those parameters must be reshaped to before being handed back to
    that module as `ext_param`.

    A submodule is a leaf if it exposes a `.param` tensor (flamo's
    convention for the raw learnable tensor of a DSP module). Only leaves
    with `requires_grad=True` are included: those are the only ones meant to
    be driven externally, and a zero-count entry would otherwise pass an
    empty tensor as `ext_param` to a module expecting its full (fixed) shape.
    Each entry's count is `numel(param)` and its shape is `param.shape[1:]`
    -- the parameter shape without the leading (size-1) batch dimension,
    since `ext_param` is reshaped with the runtime batch size substituted in
    its place. Keys are the full dotted path (e.g.
    "feedback_loop.feedback.attenuation"), so containers like
    Series/Recursion/Shell are resolved all the way down to the actual
    parameter-holding modules instead of being collapsed into one entry.
    """
    counts = OrderedDict()
    shapes = OrderedDict()
    total = 0
    for name, submodule in module.named_modules():
        if (
            hasattr(submodule, "param")
            and isinstance(submodule.param, nn.Parameter)
            and submodule.param.requires_grad
        ):
            # remove the shell level from the name if present
            if name.startswith("_Shell__core."):
                name = name[len("_Shell__core.") :]
            counts[name] = submodule.param.numel()
            shapes[name] = tuple(submodule.param.shape[1:])
            total = total + counts[name]
    return counts, shapes, total


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for generating biquad filter data.
    """

    def __init__(self, args, time_target, in_ch, num, dtype=torch.float64):
        # Create the input to the ddsp
        in_sig = torch.zeros((1, args.nfft, in_ch), device=args.device, dtype=dtype)
        in_sig[:, 0, :] = 1
        in_sig = in_sig.expand(
            tuple([num] + [d for d in in_sig.shape[1:]])
        )
        self.in_sig = in_sig

        # compute the stft of the target sigal 
        self.time_target = time_target.expand(tuple([num] + [d for d in time_target.shape]))
        target = torch.abs(torch.stft(time_target, n_fft=256, return_complex=True)) 
        # repeat the target signal to match the number of samples in the dataset
        self.target = target.expand(tuple([num] + [d for d in target.shape]))

        self.input = []
        for i in range(num):
            self.input.append((self.target[i], in_sig[i]))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.input[index], self.time_target[index].unsqueeze(-1)
    
class GroupedFDN(system.Shell):
    """
    Grouped Feedback Delay Network (FDN), after Das & Abel [1]_.

    The ``n_groups * group_size`` delay lines are split into ``n_groups``
    groups. Within a group the lines are mixed by an orthogonal matrix built 
    from that group's entry in ``mixing_angles``; the groups are then coupled
    to one another according to ``coupling_angles``. Delay lengths and the
    feedback (mixing + coupling) matrix are fixed. Only the input/output
    gains and the two parameters of the shared first-order shelving
    attenuation filter (the DC-frequency reverberation time and the shelving
    crossover) are learnable.

    ``group_size`` must be a power of two: the intra-group mixing matrix is
    built by repeated Kronecker self-products of a 2x2 rotation.

    .. [1] Das, O., & Abel, J. S. (2021). Grouped feedback delay networks for
       modeling of coupled spaces. J. Audio Eng. Soc, 69(7/8), 486-496.
    """

    def __init__(
        self,
        nfft: int,
        fs: int,
        in_ch: int,
        out_ch: int,
        group_size: int,
        n_groups: int,
        delay_lengths: torch.Tensor | list[int],
        rt_dc: Optional[float] = 1.0,
        rt_nyquist: Optional[float] = 0.2,
        crossover_freq: Optional[float] = 4000.0,
        alias_decay_db: Optional[float] = 0.0,
        device: Optional[str] = 'cuda',
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        assert group_size >= 2 and group_size & (group_size - 1) == 0, (
            f"group_size must be a power of two >= 2, got {group_size}"
        )

        n_delays = group_size * n_groups
        delay_lengths = torch.as_tensor(delay_lengths, device=device, dtype=torch.int64)
        assert delay_lengths.numel() == n_delays, (
            f"delay_lengths must have {n_delays} entries (group_size * n_groups), got {delay_lengths.numel()}"
        )

        input_gain = dsp.Gain(
            size=(n_delays, in_ch),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        output_gain = dsp.Gain(
            size=(out_ch, n_delays),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        delays = dsp.parallelDelay(
            size=(n_delays,),
            max_len=delay_lengths.max(),
            nfft=nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        delays.assign_value(delays.sample2s(delay_lengths))

        mixing_matrix = dsp.Matrix(
            size=(n_delays, n_delays),
            nfft=nfft,
            matrix_type="random_block_diagonal",
            n_blocks=n_groups,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        attenuation = parallelFirstOrderShelving(
            nfft=nfft,
            fs=fs,
            rt_nyquist=rt_nyquist,
            delays=delay_lengths,
            alias_decay_db=alias_decay_db,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        omega_c = 2 * torch.pi * crossover_freq / fs
        attenuation.assign_value(
            torch.tensor([rt_dc, omega_c], device=device, dtype=dtype)
        )

        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        core = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

        input_layer = dsp.FFT(nfft, dtype=dtype)
        output_layer = dsp.iFFTAntiAlias(
            nfft=nfft, alias_decay_db=alias_decay_db, device=device, dtype=dtype
        )
        super().__init__(core=core, input_layer=input_layer, output_layer=output_layer)


class nnGFDN(nn.Module):
    """
    Neural network model for biquad filter coefficient prediction.
    """

    def __init__(self, in_ch, out_ch, args):
        super(nnGFDN, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dtype = args.dtype
        # Stack of MLPs
        self.stack = nn.Sequential(
            nn.Linear(751 , 256, dtype=args.dtype),
            nn.LayerNorm(256, dtype=args.dtype),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=args.dtype),
            nn.LayerNorm(128, dtype=args.dtype),
            nn.ReLU(),
            nn.Linear(128, 64, dtype=args.dtype),
            nn.LayerNorm(64, dtype=args.dtype),
            nn.ReLU(),
            nn.Flatten(), 
        )

        delays = [997, 1153, 1327, 1559]
        N = len(delays)
        n_groups = 2
        # Create another instance of the model
        self.gfdn = GroupedFDN(
            group_size=N,
            n_groups=2,
            delay_lengths=delays * n_groups,
            nfft=args.nfft,
            fs=args.samplerate,
            in_ch=in_ch,
            out_ch=out_ch,
            rt_dc=1.0,
            rt_nyquist=0.2,
            crossover_freq=4000.0,
            alias_decay_db=0.0,
            device=args.device,
            dtype=args.dtype,
        )

        self.param_dict, self.param_shapes, total_params = count_learnable_params(self.gfdn)
        # Final dense layer to ensure output shape (3, n_sections, in_channels, out_channels)
        self.final_dense = nn.Linear(8256, total_params, dtype=args.dtype)

        # Simple profiling counters (aggregate over forwards)
        self.profile_enabled = True
        self._profile = {"calls": 0, "total_time": 0.0, "loop_time": 0.0}

    def forward(self, data):

        # Data consists in a tuple of input to the MLPs and input to the DDSP
        x = torch.abs(data[0])  # input to the MLP
        z = data[1]  # input to the DDSP
        
        # Pass through the stack of layers 
        x = self.stack(x)

        # Pass through the final dense layer
        x = self.final_dense(x)

        # divide the output of the MLPs into the GFDN param, reshaping each
        # slice to the shape its target module's own .param tensor expects
        # (batch dimension aside)
        map_dict = {}
        for name, count in self.param_dict.items():
            shape = self.param_shapes[name]
            map_dict[name] = torch.real(x[:, :count]).contiguous().view(-1, *shape)
            x = x[:, count:]

        y = self.gfdn(z[0].unsqueeze(0), map_dict)

        return y

    def get_core(self):
        """Return the core DSP module (the GFDN)."""
        return self.gfdn.get_core()
    
    def report_profile(self, reset: bool = False) -> dict:
        """Return and optionally reset aggregated profiling stats.

        Returns a dict with `calls`, `total_time`, `loop_time`, `rest_time`,
        and averages per call.
        """
        stats = self._profile.copy()
        calls = stats.get("calls", 0)
        total_time = stats.get("total_time", 0.0)
        loop_time = stats.get("loop_time", 0.0)
        rest_time = total_time - loop_time
        out = {
            "calls": calls,
            "total_time": total_time,
            "loop_time": loop_time,
            "rest_time": rest_time,
            "avg_total": (total_time / calls) if calls else 0.0,
            "avg_loop": (loop_time / calls) if calls else 0.0,
            "avg_rest": (rest_time / calls) if calls else 0.0,
        }
        if reset:
            self._profile = {"calls": 0, "total_time": 0.0, "loop_time": 0.0}
        return out


def example_gfdn_nn(args):
    """
    Example function that demonstrates the training of GFDN coefficients via MLPs.
    This example shows how to use flamo's modules within a neural network model.
    NOTE: the model and the parameterizations should be fine tuned for better results. This example serves just as a demo of the API.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """
    # read target RIR
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=args.dtype, device=args.device)
    target_rir = target_rir / torch.max(torch.abs(target_rir))
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset : (rir_onset + args.nfft)]

    dataset = Dataset(
        args=args,
        time_target=target_rir,
        in_ch=1,
        num=args.num,
        dtype=args.dtype
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    model = nnGFDN(
        in_ch=1,
        out_ch=1,
        args=args,
    ).to(args.device)
    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device,
    )
    trainer.register_criterion(MultiResoSTFT(), 1)
    trainer.register_criterion(sparsity_loss(), 1, requires_model=True)

    trainer.train(train_loader, valid_loader)

    out = model.report_profile()
    #print the profiling report 
    print(f"Profiling report: {out}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=48000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"], help="data type for tensors")
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
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--masked_loss", type=bool, default=False, help="use masked loss"
    )
    parser.add_argument(
        "--target_rir",
        type=str,
        default="rirs/arni_35_3541_4_2.wav",
        help="filepath to target RIR",
    )

    args = parser.parse_args()

    # Check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

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

    # Run the example GFDN neural network training
    example_gfdn_nn(args)
