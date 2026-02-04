import argparse
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch

from flamo.optimize.dataset import DatasetColorless
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import sparsity_loss, masked_mse_loss
from flamo.functional import skew_matrix


DEFAULT_SEED = 130709


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def primes_in_range(low: int, high: int) -> list[int]:
    if high < 2 or high < low:
        return []
    low = max(low, 2)
    sieve = [True] * (high + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(high**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : high + 1 : step] = [False] * len(range(start, high + 1, step))
    return [p for p in range(low, high + 1) if sieve[p]]


def sample_coprime_delays(n: int, low: int, high: int) -> torch.Tensor:
    primes = primes_in_range(low, high)
    if len(primes) < n:
        raise ValueError(
            f"Not enough coprime candidates in [{low}, {high}] to sample {n} delays."
        )
    delays = random.sample(primes, n)
    return torch.tensor(delays, dtype=torch.int64)


def hadamard_power2(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    U = torch.tensor([[1.0]], device=device, dtype=dtype)
    while U.shape[0] < n:
        U = torch.kron(
            U, torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
        ) / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return U


def build_fdn(
    args: argparse.Namespace,
    N: int,
    delay_lengths: torch.Tensor,
    feedback_kind: str,
) -> tuple[system.Shell, dict]:
    alias_decay_db = args.alias_decay_db
    delay_lengths = delay_lengths.to(torch.int64)

    input_gain = dsp.Gain(
        size=(N, 1),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    output_gain = dsp.Gain(
        size=(1, N),
        nfft=args.nfft,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )

    delays = dsp.parallelDelay(
        size=(N,),
        max_len=int(delay_lengths.max().item()),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
        dtype=args.dtype,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    metadata: dict = {}
    if feedback_kind == "orthogonal":
        feedback = dsp.Matrix(
            size=(N, N),
            nfft=args.nfft,
            matrix_type="orthogonal",
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind == "householder":
        feedback = dsp.HouseholderMatrix(
            size=(N, N),
            nfft=args.nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=args.device,
            dtype=args.dtype,
        )
    elif feedback_kind in ("scattering", "scattering_hadamard"):
        min_delay = int(delay_lengths.min().item())
        high = max(2, int(np.floor(min_delay / 2)))
        m_L = torch.randint(
            low=1,
            high=high,
            size=[N],
            device=args.device,
            dtype=args.dtype,
        )
        m_R = torch.randint(
            low=1,
            high=high,
            size=[N],
            device=args.device,
            dtype=args.dtype,
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
            dtype=args.dtype,
        )
        metadata["m_L"] = m_L
        metadata["m_R"] = m_R

        if feedback_kind == "scattering_hadamard":
            if (N & (N - 1)) != 0:
                raise ValueError(
                    f"Hadamard scattering requires power-of-two N, got N={N}."
                )
            fixed_h = hadamard_power2(N, device=args.device, dtype=args.dtype)

            def map_with_fixed(param: torch.Tensor) -> torch.Tensor:
                U0 = torch.matrix_exp(skew_matrix(param[0]))
                fixed = fixed_h.to(param.device, param.dtype)
                U = [U0] + [fixed] * (param.shape[0] - 1)
                return torch.stack(U, dim=0)

            feedback.map = map_with_fixed
    else:
        raise ValueError(f"Unsupported feedback kind: {feedback_kind}")

    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    FDN = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(args.nfft, dtype=args.dtype)
    output_layer = dsp.Transform(transform=lambda x: torch.abs(x), dtype=args.dtype)
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    return model, metadata


def collect_params(model: system.Shell) -> dict:
    core = model.get_core()
    params: dict[str, np.ndarray] = {}
    params["input_gain"] = core.input_gain.param.detach().cpu().numpy()
    params["output_gain"] = core.output_gain.param.detach().cpu().numpy()
    params["delay_param"] = core.feedback_loop.feedforward.param.detach().cpu().numpy()
    params["delay_samples"] = (
        core.feedback_loop.feedforward.s2sample(
            core.feedback_loop.feedforward.map(core.feedback_loop.feedforward.param)
        )
        .detach()
        .cpu()
        .numpy()
    )
    feedback = core.feedback_loop.feedback
    params["feedback_param"] = feedback.param.detach().cpu().numpy()
    try:
        params["feedback_mapped"] = (
            feedback.map(feedback.param).detach().cpu().numpy()
        )
    except Exception:
        pass
    if hasattr(feedback, "map_filter"):
        params["scattering_shifts"] = (
            feedback.map_filter.shifts.detach().cpu().numpy()
        )
        if getattr(feedback, "m_L", None) is not None:
            params["m_L"] = feedback.m_L.detach().cpu().numpy()
        if getattr(feedback, "m_R", None) is not None:
            params["m_R"] = feedback.m_R.detach().cpu().numpy()
    return params


def save_config(path: str, config: dict) -> None:
    def to_jsonable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    config = {k: to_jsonable(v) for k, v in config.items()}
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def run_training(
    args: argparse.Namespace,
    run_dir: str,
    model: system.Shell,
    dataset_expand: int,
    extra_config: dict,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=dataset_expand,
        device=args.device,
        dtype=args.dtype,
    )
    train_loader, valid_loader = load_dataset_seeded(
        dataset,
        batch_size=args.batch_size,
        split=args.split,
        shuffle=True,
        seed=args.seed,
        device=args.device,
    )

    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=run_dir,
        device=args.device,
        log=False,
    )
    trainer.register_criterion(
        masked_mse_loss(
            nfft=args.nfft,
            n_samples=args.mask_samples,
            n_sets=1,
            regenerate_mask=True,
            device=args.device,
        ),
        1,
    )
    trainer.register_criterion(sparsity_loss(), args.sparsity_weight, requires_model=True)

    start_time = time.time()
    trainer.train(train_loader, valid_loader)
    train_time = time.time() - start_time

    params = collect_params(model)
    np.savez(os.path.join(run_dir, "params.npz"), **params)
    torch.save(model.state_dict(), os.path.join(run_dir, "model_state.pt"))

    loss_log = {
        "train_loss": trainer.train_loss,
        "valid_loss": trainer.valid_loss,
        "train_time_sec": train_time,
    }
    save_config(os.path.join(run_dir, "loss.json"), loss_log)

    save_config(os.path.join(run_dir, "config.json"), extra_config)


def load_dataset_seeded(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    split: float,
    shuffle: bool,
    seed: int,
    device: torch.device,
):
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    train_set, valid_set = torch.utils.data.random_split(
        dataset, [train_set_size, valid_set_size], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return train_loader, valid_loader


def main(args: argparse.Namespace) -> None:
    # check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, will use cpu")

    # convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    set_global_seed(args.seed)

    if args.train_dir is None:
        args.train_dir = os.path.join(
            "output", "benchmarking_" + time.strftime("%Y%m%d-%H%M%S")
        )
    os.makedirs(args.train_dir, exist_ok=True)

    root_args = {k: v for k, v in vars(args).items()}
    save_config(os.path.join(args.train_dir, "args.json"), root_args)

    orthogonal_N = [8, 16, 64]
    scattering_N = [4, 6, 8]
    scattering_hadamard_N = [4, 8]

    worklist = [
        ("orthogonal", orthogonal_N),
        ("householder", orthogonal_N),
        ("scattering", scattering_N),
        ("scattering_hadamard", scattering_hadamard_N),
    ]

    unique_sizes = sorted({n for _, n_list in worklist for n in n_list})
    delays_by_size: dict[int, list[torch.Tensor]] = {}
    for N in unique_sizes:
        if N == 64:
            delay_low = args.delay_min_64
            delay_high = args.delay_max_64
        else:
            delay_low = args.delay_min
            delay_high = args.delay_max
        delays_by_size[N] = [
            sample_coprime_delays(N, delay_low, delay_high)
            for _ in range(args.sets_per_config)
        ]

    for feedback_kind, n_list in worklist:
        for N in n_list:
            if N == 64:
                delay_low = args.delay_min_64
                delay_high = args.delay_max_64
            else:
                delay_low = args.delay_min
                delay_high = args.delay_max

            for set_idx in range(args.sets_per_config):
                delay_lengths = delays_by_size[N][set_idx]
                model, metadata = build_fdn(args, N, delay_lengths, feedback_kind)

                run_dir = os.path.join(
                    args.train_dir,
                    feedback_kind,
                    f"N{N}",
                    f"set_{set_idx:02d}",
                )

                dataset_expand = (
                    args.num_scattering
                    if feedback_kind in ("scattering", "scattering_hadamard")
                    else args.num
                )
                if dataset_expand is None:
                    dataset_expand = max(1, (args.nfft // 2 + 1) // 2000)

                config = {
                    "feedback_kind": feedback_kind,
                    "N": N,
                    "set_index": set_idx,
                    "delay_lengths_samples": delay_lengths,
                    "delay_range": [delay_low, delay_high],
                    "samplerate": args.samplerate,
                    "nfft": args.nfft,
                    "max_epochs": args.max_epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "dataset_expand": dataset_expand,
                    "mask_samples": args.mask_samples,
                    "sparsity_weight": args.sparsity_weight,
                    "alias_decay_db": args.alias_decay_db,
                    "seed": args.seed,
                    **metadata,
                }

                run_training(args, run_dir, model, dataset_expand, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--num_scattering",
        type=int,
        default=None,
        help="dataset size for scattering runs (defaults to nfft//2+1 // 2000)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for computation"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="maximum number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--train_dir", type=str, help="directory to save training results"
    )
    parser.add_argument(
        "--sets_per_config",
        type=int,
        default=4,
        help="number of random sets per configuration",
    )
    parser.add_argument(
        "--delay_min",
        type=int,
        default=500,
        help="minimum delay length (samples) for standard runs",
    )
    parser.add_argument(
        "--delay_max",
        type=int,
        default=2500,
        help="maximum delay length (samples) for standard runs",
    )
    parser.add_argument(
        "--delay_min_64",
        type=int,
        default=500,
        help="minimum delay length (samples) for N=64 runs",
    )
    parser.add_argument(
        "--delay_max_64",
        type=int,
        default=5000,
        help="maximum delay length (samples) for N=64 runs",
    )
    parser.add_argument(
        "--mask_samples",
        type=int,
        default=12000,
        help="number of bins used for masked MSE loss",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=0.2,
        help="weight for sparsity loss",
    )
    parser.add_argument(
        "--alias_decay_db",
        type=float,
        default=30.0,
        help="alias decay in dB",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="train/valid split",
    )

    main(parser.parse_args())
