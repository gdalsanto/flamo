import torch
import numpy as np
import scipy.signal

def skew_matrix(X):
    r"""
    Generate a skew symmetric matrix from a given matrix X.
    """
    A = X.triu(1)
    return A - A.transpose(-1, -2)

def signal_gallery(
    batch_size: int,
    n_samples: int,
    n: int,
    signal_type: str = "impulse",
    fs: int = 48000,
    reference=None,
):
    # TODO adapt docu string
    """
    Generate a gallery of signals based on the specified signal type.

    Args:
        batch_size (int): The number of signal batches to generate.
        n_samples (int): The number of samples in each signal.
        n (int): The number of channels in each signal.
        signal_type (str, optional): The type of signal to generate. Defaults to 'impulse'.
        fs (int, optional): The sampling frequency of the signals. Defaults to 48000.
        reference (torch.Tensor, optional): A reference signal to use. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, n_samples, n) containing the generated signals.
    """

    signal_types = {
        "impulse",
        "sine",
        "sweep",
        "wgn",
        "reference",
    }

    if signal_type not in signal_types:
        raise ValueError(f"Matrix type {signal_type} not recognized.")
    match signal_type:
        case "impulse":
            x = torch.zeros(batch_size, n_samples, n)
            x[:, 0, :] = 1
            return x
        case "sine":
            return torch.sin(
                torch.linspace(0, 2 * np.pi, n_samples)
                .unsqueeze(-1)
                .expand(batch_size, n_samples, n)
            )
        case "sweep":
            t = torch.linspace(0, n_samples / fs - 1 / fs, n_samples)
            x = torch.tensor(
                scipy.signal.chirp(t, f0=20, f1=20000, t1=t[-1], method="linear")
            ).unsqueeze(-1)
            return x.expand(batch_size, n_samples, n)
        case "wgn":
            return torch.randn((batch_size, n_samples, n))
        case "reference":
            return reference.expand(batch_size, n_samples, n)
        