import math
import torch
import torch.nn as nn
import numpy as np
import scipy.signal
from typing import Optional
from flamo.utils import RegularGridInterpolator


def get_magnitude(x: torch.Tensor):
    r"""
    Gets the magnitude of a complex tensor.

        **Arguments**:
            **x** (torch.tensor): The input tensor.

        **Returns**:
            torch.tensor: The absolute value of **x**.
    """
    # get the magnitude of a complex tensor
    return torch.abs(x)


def get_eigenvalues(x: torch.Tensor):
    r"""
    Gets the eigenvalues of a complex tensor.
    The last two dimensions of the input tensor must be identical.

        **Arguments**:
            **x** (torch.tensor): The input tensor.

        **Returns**:
            torch.tensor: The eigenvalues of **x**.
    """
    assert x.shape[-1] == x.shape[-2]
    if x.shape[-1] == 1:
        return x

    return torch.linalg.eigvals(x)


def skew_matrix(X):
    r"""
    Generate a skew symmetric matrix from a given matrix :math:`\mathbf{X}_{\textrm{Tr}}` as follows

    .. math::

        \mathbf{X} = \mathbf{X}_{\textrm{Tr}} - \mathbf{X}_{\textrm{Tr}}^\top

    where :math:`\mathbf{X_{\textrm{Tr}}}` is the upper triangular part of :math:`\mathbf{X}`, and :math:`\cdot^\top` denotes the transpose operation.

    **Arguments**:
        **X** (torch.Tensor): The input matrix.
    """
    A = X.triu(1)
    return A - A.transpose(-1, -2)


def get_frequency_samples(num: int, device: str | torch.device = None):
    r"""
    Get frequency samples (in radians) sampled at linearly spaced points along the unit circle.

    **Arguments**
        - **num** (int): number of frequency samples
        - **device** (torch.device, str): The device of constructed tensors. Default: None.

    **Returns**
        - frequency samples in radians between [0, pi]
    """
    angle = torch.linspace(0, 1, num, device=device)
    abs = torch.ones(num, device=device)
    return torch.polar(abs, angle * np.pi)


class HadamardMatrix(nn.Module):
    """
    Generate a Hadamard matrix of size N as a nn.Module.
    """

    def __init__(self, N, device: Optional[str] = None):
        super().__init__()
        self.N = N
        self.device = device

    def forward(self, x):
        U = torch.tensor([[1.0]], device=self.device)
        while U.shape[0] < self.N:
            U = torch.kron(
                U, torch.tensor([[1, 1], [1, -1]], dtype=U.dtype, device=U.device)
            ) / torch.sqrt(torch.tensor(2.0, device=U.device))
        return U


class RotationMatrix(nn.Module):
    """
    Generate a rotation matrix of size N as a nn.Module from a given angle.
    """

    def __init__(
        self,
        N: int,
        min_angle: float = 0,
        max_angle: float = torch.pi / 4,
        iter: Optional[int] = None,
        device: Optional[str] = None,
    ):

        super().__init__()
        self.N = N
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.iter = iter
        self.device = device

    def create_submatrix(self, angles: torch.Tensor, iters: int = 1):
        """Create a submatrix for each group."""
        X = torch.zeros(2, 2, device=self.device)
        angles[0] = torch.clamp(angles[0], self.min_angle, self.max_angle)
        X.fill_diagonal_(torch.cos(angles[0]))
        X[1, 0] = -torch.sin(torch.tensor(angles[0], device=self.device))
        X[0, 1] = torch.sin(torch.tensor(angles[0], device=self.device))

        if iters is None:
            iters = torch.log2(torch.tensor(self.N)).int().item() - 1
        for i in range(iters):
            if len(angles) > 1:
                X = torch.kron(X, self.create_submatrix([angles[i]]))
            else:
                X = torch.kron(X, X)
        return X

    def forward(self, theta):

        return self.create_submatrix(theta, self.iter)


def biquad2tf(b: torch.Tensor, a: torch.Tensor, nfft: int):
    r"""
    Converts a biquad filter representation to a transfer function.
    Shape of :attr:`b` and :attr:`a` is (3, n_sections)

    **Arguments**:
        - **b** (torch.Tensor): Coefficients of the numerator polynomial of the biquad filter.
        - **a** (torch.Tensor): Coefficients of the denominator polynomial of the biquad filter.
        - **nfft** (int): The number of FFT points to compute teh transfer function.

    **Returns**:
        - torch.Tensor: Transfer function of the biquad filter evaluated at x.
    """
    if len(b.shape) < 2:
        b = b.unsqueeze(-1)
    if len(a.shape) < 2:
        a = a.unsqueeze(-1)
    B = torch.fft.rfft(b, nfft, dim=0)
    A = torch.fft.rfft(a, nfft, dim=0)
    H = torch.prod(B, dim=1) / torch.prod(A, dim=1)
    return H


def signal_gallery(
    batch_size: int,
    n_samples: int,
    n: int,
    signal_type: str = "impulse",
    fs: int = 48000,
    rate: float = 1.0,
    reference: torch.Tensor = None,
    device: str | torch.device = None,
):
    r"""
    Generate a tensor containing a signal based on the specified signal type.

    Supported signal types are:
        - ``impulse``: A single impulse at the first sample, followed by :attr:`n_samples-1` zeros.
        - ``sine``: A sine wave of frequency :attr:`rate` Hz, if given. Otherwise, a sine wave of frequency 1 Hz.
        - ``sweep``: A linear sweep from 20 Hz to 20 kHz.
        - ``wgn``: White Gaussian noise.
        - ``exp``: An exponential decay signal.
        - ``velvet``: A velvet noise signal with density :attr:`rate` impulses per second.
        - ``reference``: A reference signal provided as argument :attr:`reference`.

        **Arguments**:
            - **batch_size** (int): The number of batches to generate.
            - **n_samples** (int): The signal length in samples.
            - **n_channel** (int): The number of channels in each signal.
            - **signal_type** (str, optional): The type of signal to generate. Defaults to 'impulse'.
            - **fs** (int, optional): The sampling frequency of the signals. Defaults to 48000.
            - **reference** (torch.Tensor, optional): A reference signal to use. Defaults to None.
            - **device** (torch.device, optional): The device of constructed tensors. Defaults to None.

        **Returns**:
            - torch.Tensor: A tensor of shape (batch_size, n_samples, n) containing the generated signals.
    """

    signal_types = {
        "impulse",
        "sine",
        "sweep",
        "wgn",
        "exp",
        "reference",
        "noise",
        "velvet",
    }

    if signal_type not in signal_types:
        raise ValueError(f"Signal type {signal_type} not recognized.")
    match signal_type:
        case "impulse":
            x = torch.zeros(batch_size, n_samples, n)
            x[:, 0, :] = 1
            return x.to(device)
        case "sine":
            if rate is not None:
                return (
                    torch.sin(
                        2
                        * np.pi
                        * rate
                        / fs
                        * torch.linspace(0, n_samples / fs, n_samples)
                    )
                    .unsqueeze(-1)
                    .expand(batch_size, n_samples, n)
                    .to(device)
                )
            else:
                return torch.sin(
                    torch.linspace(0, 2 * np.pi, n_samples)
                    .unsqueeze(-1)
                    .expand(batch_size, n_samples, n)
                ).to(device)
        case "sweep":
            t = torch.linspace(0, n_samples / fs - 1 / fs, n_samples)
            x = torch.tensor(
                scipy.signal.chirp(t, f0=20, f1=20000, t1=t[-1], method="linear"),
                device=device,
            ).unsqueeze(-1)
            return x.expand(batch_size, n_samples, n)
        case "wgn":
            return torch.randn((batch_size, n_samples, n), device=device)
        case "exp":
            return (
                torch.exp(-rate * torch.arange(n_samples) / fs)
                .unsqueeze(-1)
                .expand(batch_size, n_samples, n)
                .to(device)
            )
        case "velvet":
            x = torch.empty((batch_size, n_samples, n), device=device)
            for i_batch in range(batch_size):
                for i_ch in range(n):
                    x[i_batch, :, i_ch] = gen_velvet_noise(n_samples, fs, rate, device)
            return x
        case "reference":
            if isinstance(reference, torch.Tensor):
                return reference.expand(batch_size, n_samples, n).to(device)
            else:
                return torch.tensor(reference, device=device).expand(
                    batch_size, n_samples, n
                )
        case "noise":
            return torch.randn((batch_size, n_samples, n), device=device)


def gen_velvet_noise(n_samples: int, fs: int, density: float, device: str | torch.device = None) -> torch.Tensor:
    r"""
    Generate a velvet noise sequence.
    **Arguments**:
        - **n_samples** (int): The length of the signal in samples.
        - **fs** (int): The sampling frequency of the signal in Hz.
        - **density** (float): The density of impulses in impulses per second.
        - **device** (str | torch.device): The device of constructed tensors.
    **Returns**:
        - torch.Tensor: A tensor of shape (n_samples,) containing the velvet noise sequence.
    """
    Td = fs / density # average distance between impulses
    num_impulses = n_samples / Td # expected number of impulses
    floor_impulses = math.floor(num_impulses)
    grid = torch.arange(floor_impulses) * Td

    jitter_factors = torch.rand(floor_impulses)
    impulse_indices = torch.ceil(grid + jitter_factors * (Td - 1)).long()

    # first impulse is at position 0 and all indices are within bounds
    impulse_indices[0] = 0
    impulse_indices = torch.clamp(impulse_indices, max=n_samples - 1)
            
    # Generate random signs (+1 or -1)
    signs = 2 * torch.randint(0, 2, (floor_impulses,)) - 1
            
    # Construct sparse signal
    sequence = torch.zeros(n_samples, device=device)
    sequence[impulse_indices] = signs.float()

    return sequence

def hertz2rad(hertz: torch.Tensor, fs: int):
    r"""
    Convert frequency from Hz to rad.

    .. math::
        \omega = \frac{2\pi f}{f_s}

    where :math:`f` is the frequency in Hz and :math:`f_s` is the sampling frequency in Hz.

    **Arguments**:
        - **hertz** (torch.Tensor): The frequency in Hz.
        - **fs** (int): The sampling frequency in Hz.
    """
    return torch.divide(hertz, fs) * 2 * torch.pi


def rad2hertz(rad: torch.Tensor, fs: int):
    r"""
    Convert frequency from rad to Hz

    .. math::
        f = \frac{\omega f_s}{2\pi}

    where :math:`\omega` is the frequency in rad and :math:`f_s` is the sampling frequency in Hz.

    **Arguments**:
        - rad (torch.Tensor): The frequency in rad.
        - fs (int): The sampling frequency in Hz.
    """
    return torch.divide(rad * fs, 2 * torch.pi)


def db2mag(dB: torch.Tensor | float):
    r"""
    Convert a value from decibels (dB) to magnitude.

    .. math::
        \text{magnitude} = 10^{dB/20}

    where :math:`dB` is the input value in decibels.

    **Arguments**:
        **dB** (torch.tensor | float): The value in decibels.
    **Returns**:
        - float: The corresponding magnitude value.
    """

    return 10 ** (dB / 20)


def mag2db(mag: torch.Tensor):
    r"""
    Convert a value from magnitude to decibels (dB).

    .. math::
        \text{dB} = 20\log_{10}(\text{magnitude})

    where :math:`\text{magnitude}` is the input value in magnitude.

    **Arguments**:

        - **mag** (torch.tensor): The value in magnitude.

    **Returns**:
        - float: The corresponding value in decibels.
    """

    return 20 * torch.log10(torch.abs(mag))


def lowpass_filter(
    fc: float = 500.0,
    gain: float = 0.0,
    fs: int = 48000,
    device: str | torch.device = None,
) -> tuple:
    r"""
    Lowpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map
    the cutoff frequency and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.
    The transfer function of the filter is given by

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \frac{1 - \cos(\omega_c)}{2},\;\; b_1 = 1 - \cos(\omega_c),\;\; b_2 = \frac{1 - \cos(\omega_c)}{2}

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where :math:`\omega_c = 2\pi f_c / f_s`, :math:`\alpha = \sin(\omega_c)/2 \cdot \sqrt{2}` and :math:`\cos(\omega_c)` is the cosine of the cutoff frequency.
    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

    **Arguments**:
        - **fc** (float, optional): The cutoff frequency of the filter in Hz. Default: 500 Hz.
        - **gain** (float, optional): The gain of the filter in dB. Default: 0 dB.
        - **fs** (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.
        - **device** (str | torch.device, optional): The device of constructed tensors. Default: None.

    **Returns**:
        - **b** (torch.tensor): The numerator coefficients of the filter transfer function.
        - **a** (torch.tensor): The denominator coefficients of the filter transfer function.
    """

    omegaC = hertz2rad(fc, fs).to(device=device)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) / 2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = (1 - cosOC) / 2
    b[1] = 1 - cosOC
    b[2] = (1 - cosOC) / 2
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10 ** (gain / 20) * b, a


def highpass_filter(
    fc: float = 10000.0,
    gain: float = 0.0,
    fs: int = 48000,
    device: str | torch.device = None,
) -> tuple:
    r"""
    Highpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map
    the cutoff frequency and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \frac{1 + \cos(\omega_c)}{2},\;\; b_1 = - 1 - \cos(\omega_c),\;\; b_2 = \frac{1 + \cos(\omega_c)}{2}

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where :math:`\omega_c = 2\pi f_c / f_s`, :math:`\alpha = \sin(\omega_c)/2 \cdot \sqrt{2}` and :math:`\cos(\omega_c)` is the cosine of the cutoff frequency.
    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

        **Arguments**:
            - **fc** (float, optional): The cutoff frequency of the filter in Hz. Default: 10000 Hz.
            - **gain** (float, optional): The gain of the filter in dB. Default: 0 dB.
            - **fs** (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.
            - **device** (str | torch.device), optional: The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.tensor): The denominator coefficients of the filter transfer function.
    """

    omegaC = hertz2rad(fc, fs)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) / 2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = (1 + cosOC) / 2
    b[1] = -(1 + cosOC)
    b[2] = (1 + cosOC) / 2
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10 ** (gain / 20) * b, a


def bandpass_filter(
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    gain: float = 0.0,
    fs: int = 48000,
    device: str | torch.device = None,
) -> tuple:
    r"""
    Bandpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map
    the cutoff frequencies and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \alpha,\;\; b_1 = 0,\;\; b_2 = - \alpha

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where

    .. math::
        \omega_c = \frac{2\pi f_{c1} + 2\pi f_{c2}}{2 f_s}`,

    .. math::
        \text{ BW } = \log_2\left(\frac{f_{c2}}{f_{c1}}\right),

    .. math::
        \alpha = \sin(\omega_c) \sinh\left(\frac{\log(2)}{2} \text{ BW } \frac{\omega_c}{\sin(\omega_c)}\right)

    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

        **Arguments**:
            - **fc1** (float): The left cutoff frequency of the filter in Hz.
            - **fc2** (float): The right cutoff frequency of the filter in Hz.
            - **gain** (float, optional): The gain of the filter in dB. Default: 0 dB.
            - **fs** (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.
            - **device** (torch.device | str, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.Tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    omegaC = (hertz2rad(fc1, fs) + hertz2rad(fc2, fs)) / 2
    BW = torch.log2(fc2 / fc1)
    two = torch.tensor(2, device=device)
    alpha = torch.sin(omegaC) * torch.sinh(
        torch.log(two) / two * BW * (omegaC / torch.sin(omegaC))
    )

    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape, device=device)
    b = torch.ones(3, *omegaC.shape, device=device)

    b[0] = alpha
    b[1] = 0
    b[2] = -alpha
    a[0] = 1 + alpha
    a[1] = -2 * cosOC
    a[2] = 1 - alpha

    return 10 ** (gain / 20) * b, a


def shelving_filter(
    fc: torch.Tensor,
    gain: torch.Tensor,
    type: str = "low",
    fs: int = 48000,
    device: torch.device | str = None,
):
    r"""
    Shelving filter coefficients.
    Maps the cutoff frequencies and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for low shelving filter:

    .. math::
        b_0 = \sqrt{g} ( \sqrt{g} \tau^2 + \sqrt{2} \tau g^{1/4} + 1 ),\;\; b_1 = \sqrt{g} (2 \sqrt{g} \tau^2 - 2 ),\;\; b_2 = \sqrt{g} ( \sqrt{g} \tau^2 - \sqrt{2} \tau g^{1/4} + 1 )

        a_0 = \sqrt{g} + \sqrt{2} \tau g^{1/4} + \tau^2,\;\; a_1 = 2 \tau^{2} - 2 \sqrt{g} ,\;\; a_2 = \sqrt{g} - \sqrt{2} \tau g^{1/4} + \tau^2

    for high shelving filter:

    .. math::
        b_0 = g ( \sqrt{g} + \sqrt{2} \tau g^{1/4} + \tau^2 ),\;\; a_1 = g ( 2 \tau^{2} - 2 \sqrt{g} ),\;\; a_2 = g (\sqrt{g} - \sqrt{2} \tau g^{1/4} + \tau^2)

        a_0 = \sqrt{g} ( \sqrt{g} \tau^2 + \sqrt{2} \tau g^{1/4} + 1 ),\;\; a_1 = \sqrt{g} (2 \sqrt{g} \tau^2 - 2 ),\;\; a_2 = \sqrt{g} ( \sqrt{g} \tau^2 - \sqrt{2} \tau g^{1/4} + 1 )

    where :math:`\tau = \tan(2 \pi f_c / (2 f_s))`, :math:`f_c` is the cutoff frequency, :math:`f_s` is the sampling frequency, and :math:`g` is the linear gain.

        **Arguments**:
            - **fc** (torch.Tensor): The cutoff frequency of the filter in Hz.
            - **gain** (torch.Tensor): The linear gain of the filter.
            - **type** (str, optional): The type of shelving filter. Can be 'low' or 'high'. Default: 'low'.
            - **fs** (int, optional): The sampling frequency of the signal in Hz.
            - **device** (torch.device | str, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.Tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.Tensor): The denominator coefficients of the filter transfer function.
    """
    b = torch.ones(3, device=device)
    a = torch.ones(3, device=device)

    omegaC = hertz2rad(fc, fs)
    t = torch.tan(omegaC / 2)
    t2 = t**2
    g2 = gain**0.5
    g4 = gain**0.25

    two = torch.tensor(2, device=device)
    b[0] = g2 * t2 + torch.sqrt(two) * t * g4 + 1
    b[1] = 2 * g2 * t2 - 2
    b[2] = g2 * t2 - torch.sqrt(two) * t * g4 + 1

    a[0] = g2 + torch.sqrt(two) * t * g4 + t2
    a[1] = 2 * t2 - 2 * g2
    a[2] = g2 - torch.sqrt(two) * t * g4 + t2

    b = g2 * b

    if type == "high":
        tmp = torch.clone(b)
        b = a * gain
        a = tmp

    return b, a


def peak_filter(
    fc: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
    fs: int = 48000,
    device: str | torch.device = None,
) -> tuple:
    r"""
    Peak filter coefficients.
    Outputs the cutoff frequencies and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for peak filter:

    .. math::
        b_0 = \sqrt{g} + g \tau,\;\; b_1 = -2 \sqrt{g} \cos(\omega_c),\;\; b_2 = \sqrt{g} - g \tau

        a_0 = \sqrt{g} + \tau,\;\; a_1 = -2 \sqrt{g} \cos(\omega_c),\;\; a_2 = \sqrt{g} - \tau

    where :math:`\tau = \tan(\text{BW}/2)`, :math:`BW = \omega_c / Q`, :math:`\omega_c = 2\pi f_c / f_s`, :math:`g`is the linear gain, and :math:`Q` is the quality factor.

        **Arguments**:
            - **fc** (torch.Tensor): The cutoff frequency of the filter in Hz.
            - **gain** (torch.Tensor): The linear gain of the filter.
            - **Q** (torch.Tensor): The quality factor of the filter.
            - **fs** (int, optional): The sampling frequency of the signal in Hz. Default: 48000.
            - **device** (torch.device | str, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.Tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.Tensor): The denominator coefficients of the filter transfer function
    """
    b = torch.ones(3, device=device)
    a = torch.ones(3, device=device)

    omegaC = hertz2rad(fc, fs)
    bandWidth = omegaC / Q
    t = torch.tan(bandWidth / 2)

    b[0] = torch.sqrt(gain) + gain * t
    b[1] = -2 * torch.sqrt(gain) * torch.cos(omegaC)
    b[2] = torch.sqrt(gain) - gain * t

    a[0] = torch.sqrt(gain) + t
    a[1] = -2 * torch.sqrt(gain) * torch.cos(omegaC)
    a[2] = torch.sqrt(gain) - t

    return b, a


def prop_shelving_filter(
    fc: torch.Tensor,
    gain: torch.Tensor,
    type: str = "low",
    fs: int = 48000,
    device="cpu",
):
    r"""
    Proportional first order Shelving filter coefficients.
    Maps the cutoff frequencies and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` first order coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1}}{a_0 + a_1 z^{-1}}

        t = \text{tan}(\pi f_c / f_s),\;\; k = 10^{g_{\textrm{dB}}/20}

    for low shelving filter:

    .. math::
        b_0 = t \sqrt{k} + 1,\;\; b_1 =  t \sqrt{k} - 1

        a_0 = t / \sqrt{k} + 1,\;\; a_1 =t / \sqrt{k} - 1

    for high shelving filter:

    .. math::
        b_0 = k * (t \sqrt{1/k} + 1),\;\; b_1 =  k * (t \sqrt{1/k} - 1)

        a_0 = t / \sqrt{1/k} + 1,\;\; a_1 =t / \sqrt{1/k} - 1

    where :math:`f_c` is the cutoff frequency, :math:`f_s` is the sampling frequency, and :math:`g_{\textrm{dB}}` is the gain in dB.

        **Arguments**:
            - **fc** (torch.Tensor): The cutoff frequency of the filter in Hz.
            - **gain** (torch.Tensor): The gain in dB of the filter.
            - **type** (str, optional): The type of shelving filter. Can be 'low' or 'high'. Default: 'low'.
            - **fs** (int, optional): The sampling frequency of the signal in Hz.
            - **device** (torch.device | str, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.Tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.Tensor): The denominator coefficients of the filter transfer function.

    Reference:
    - Jot, J. M. (2015, October). Proportional parametric equalizers—Application to digital reverberation and environmental audio processing. In Proceedings of the 139th Convention of the Audio Engineering Society, New York, NY, USA (Vol. 29).
    """

    t = torch.tan(torch.pi * fc / fs)
    k = 10 ** (gain / 20)

    a = torch.zeros((2, *fc.shape), device=device)
    b = torch.zeros_like(a)

    if type == "low":
        b[0] = t * torch.sqrt(k) + 1
        b[1] = t * torch.sqrt(k) - 1
        a[0] = t / torch.sqrt(k) + 1
        a[1] = t / torch.sqrt(k) - 1
    elif type == "high":
        k = 1 / k
        b[0] = (t * torch.sqrt(k) + 1) / k
        b[1] = (t * torch.sqrt(k) - 1) / k
        a[0] = t / torch.sqrt(k) + 1
        a[1] = t / torch.sqrt(k) - 1

    return b, a


def prop_peak_filter(
    fc: torch.Tensor,
    bw: torch.Tensor,
    gain: torch.Tensor,
    fs: int = 48000,
    device="cpu",
):
    r"""
    Proportional Peak (Presence) filter coefficients.
    Maps the cutoff frequencies and gain to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

        t = \text{tan}(\pi BW / f_s),\;\; c = \text{cos}(2 \pi f_c / f_s),\;\;k = 10^{g_{\textrm{dB}}/20}

    Biquad coefficients:

    .. math::
        b_0 = 1 + t \sqrt{k} ,\;\; b_1 = -2 c,\;\; b_2 = 1 - t \sqrt{k}

        a_0 = 1 + t / \sqrt{k},\;\; a_1 = - 2 c ,\;\; 1 - t / \sqrt{k}

    where :math:`f_c` is the cutoff frequency, :math:`f_s` is the sampling frequency, and :math:`g_{\textrm{dB}}` is the gain in dB.

        **Arguments**:
            - **fc** (torch.Tensor): The cutoff frequency of the filter in Hz.
            - **bw** (torch.Tensor): The bandwidth of the filter in Hz.
            - **gain** (torch.Tensor): The gain in dB of the filter.
            - **fs** (int, optional): The sampling frequency of the signal in Hz.
            - **device** (torch.device | str, optional): The device of constructed tensors. Default: None.

        **Returns**:
            - **b** (torch.Tensor): The numerator coefficients of the filter transfer function.
            - **a** (torch.Tensor): The denominator coefficients of the filter transfer function.

    Reference:
    - Jot, J. M. (2015, October). Proportional parametric equalizers—Application to digital reverberation and environmental audio processing. In Proceedings of the 139th Convention of the Audio Engineering Society, New York, NY, USA (Vol. 29).
    """

    t = torch.tan(torch.pi * bw / fs)
    c = torch.cos(2 * np.pi * fc / fs)
    k = 10 ** (gain / 20)

    a = torch.zeros((3, *fc.shape), device=device)
    b = torch.zeros_like(a)

    b[0] = 1 + torch.sqrt(k) * t
    b[1] = -2 * c
    b[2] = 1 - torch.sqrt(k) * t
    a[0] = 1 + t / torch.sqrt(k)
    a[1] = -2 * c
    a[2] = 1 - t / torch.sqrt(k)

    return b, a


def sosfreqz(sos: torch.Tensor, nfft: int = 512):
    r"""
    Compute the complex frequency response via FFT of cascade of second order filter sections (SOS).

        **Arguments**:
            - **sos** (torch.Tensor): Second order filter sections with shape (n_sections, 6)
            - **nfft** (int): FFT size. Default: 512

        **Returns**:
            - **H** (torch.Tensor): Overall complex frequency response.
    """
    n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order

    B = torch.fft.rfft(sos[:, :3], nfft, dim=-1)
    A = torch.fft.rfft(sos[:, 3:], nfft, dim=-1)
    H = torch.prod(B, dim=0) / (torch.prod(A, dim=0))
    return H


def svf(
    fc: torch.Tensor,
    R: torch.Tensor,
    m: torch.Tensor = torch.ones((3,)),
    G: torch.Tensor = None,
    filter_type: str = None,
    fs: int = 48000,
    device: str | torch.device = None,
):
    r"""
    Implements a State Variable Filter (SVF) with various filter types.

        **Arguments**:
            - **fc** (torch.Tensor): The cutoff frequency of the filter.
            - **R** (torch.Tensor): The resonance parameter of the filter.
            - **m** (torch.Tensor, optional): The mixing coefficients of the filter. Default: torch.ones((3,)).
            - **G** (torch.Tensor, optional): The gain coefficients of the filter. Default: None.
            - **filter_type** (str, optional): The type of filter to be applied. Can be one of "lowpass", "highpass", "bandpass", "lowshelf", "highshelf", "peaking", "notch", or None. Default: None.
            - **fs** (int, optional): The sampling frequency. Default: 48000.
            - **device** (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            Tuple[torch.Tensor, torch.Tensor]: The numerator and denominator coefficients of the filter transfer function.

    """

    f = torch.tan(torch.pi * fc / fs)
    assert (R > 0).any(), "Resonance must be positive to ensure stability"

    if G is None:
        G = torch.ones_like(f)

    match filter_type:
        case "lowpass":
            m = torch.cat(
                (
                    (torch.ones_like(G)).unsqueeze(-1),
                    (torch.zeros_like(G)).unsqueeze(-1),
                    torch.zeros_like(G).unsqueeze(-1),
                ),
                dim=-1,
            )
        case "highpass":
            m = torch.cat(
                (
                    (torch.zeros_like(G)).unsqueeze(-1),
                    (torch.zeros_like(G)).unsqueeze(-1),
                    torch.ones_like(G).unsqueeze(-1),
                ),
                dim=-1,
            )
        case "bandpass":
            m = torch.cat(
                (
                    (torch.zeros_like(G)).unsqueeze(-1),
                    (torch.ones_like(G)).unsqueeze(-1),
                    torch.zeros_like(G).unsqueeze(-1),
                ),
                dim=-1,
            )
        case "lowshelf":
            m = torch.cat(
                (
                    (torch.ones_like(G)).unsqueeze(-1),
                    (2 * R * torch.sqrt(G)).unsqueeze(-1),
                    (G * torch.ones_like(G)).unsqueeze(-1),
                ),
                dim=-1,
            )
        case "highshelf":
            m = torch.cat(
                (
                    (G * torch.ones_like(G)).unsqueeze(-1),
                    (2 * R * torch.sqrt(G)).unsqueeze(-1),
                    (torch.ones_like(G)).unsqueeze(-1),
                ),
                dim=-1,
            )
        case "peaking" | "notch":
            m = torch.cat(
                (
                    (torch.ones_like(G)).unsqueeze(-1),
                    (2 * R * torch.sqrt(G)).unsqueeze(-1),
                    (torch.ones_like(G)).unsqueeze(-1),
                ),
                dim=-1,
            )
        case None:
            print("No filter type specified. Using the given mixing coefficents.")

    b = torch.zeros((3, *f.shape), device=device)
    a = torch.zeros((3, *f.shape), device=device)

    b[0] = (f**2) * m[..., 0] + f * m[..., 1] + m[..., 2]
    b[1] = 2 * (f**2) * m[..., 0] - 2 * m[..., 2]
    b[2] = (f**2) * m[..., 0] - f * m[..., 1] + m[..., 2]

    a[0] = (f**2) + 2 * R * f + 1
    a[1] = 2 * (f**2) - 2
    a[2] = (f**2) - 2 * R * f + 1

    return b, a


def probe_sos(
    sos: torch.Tensor,
    control_freqs: list | torch.Tensor,
    nfft: int,
    fs: int,
    device: str | torch.device = None,
):
    r"""
    Probe the frequency / magnitude response of a cascaded SOS filter at the points
    specified by the control frequencies.

        **Arguments**:
            - **sos** (torch.Tensor): Cascaded second-order sections (SOS) filter coefficients.
            - **control_freqs** (list or torch.Tensor): Frequencies at which to probe the filter response.
            - **nfft** (int): Length of the FFT used for frequency analysis.
            - **fs** (float): Sampling frequency in Hz.
            - **device** (torch.device, optional): The device of constructed tensors. Default: None.

        **Returns**:
            tuple: A tuple containing the following:
                - **G** (torch.Tensor): Magnitude response of the filter at the control frequencies.
                - **H** (torch.Tensor): Frequency response of the filter.
                - **W** (torch.Tensor): Phase response of the filter.
    """
    n_freqs = sos.shape[-1]

    H = torch.zeros((nfft // 2 + 1, n_freqs), dtype=torch.cdouble, device=device)
    W = torch.zeros((nfft // 2 + 1, n_freqs), device=device)
    G = torch.zeros((len(control_freqs), n_freqs), device=device)

    for band in range(n_freqs):
        sos[:, band] = sos[:, band] / sos[3, band]

        B = torch.fft.rfft(sos[:3, band], nfft, dim=0)
        A = torch.fft.rfft(sos[3:, band], nfft, dim=0)
        h = B / (A + torch.tensor(1e-10, device=device))
        f = torch.fft.rfftfreq(nfft, 1 / fs)
        interp = RegularGridInterpolator([f], 20 * torch.log10(torch.abs(h)))
        g = interp([control_freqs])

        G[:, band] = g
        H[:, band] = h
        W[:, band] = 2 * torch.pi * f / fs

    return G, H, W


def find_onset(rir: torch.Tensor):
    #
    r"""
    Find onset in input RIR by extracting a local energy envelope of the
    RIR then finding its maximum point

        **Arguments**:
            - **rir** (torch.Tensor): Room impulse response.

    """
    # extract local energy envelope
    win_len = 64
    overlap = 0.75
    win = torch.hann_window(win_len)

    # pad rir
    rir = torch.nn.functional.pad(rir, (int(win_len * overlap), int(win_len * overlap)))
    hop = 1 - overlap
    n_wins = np.floor(rir.shape[0] / (win_len * hop) - 1 / 2 / hop)

    local_energy = []
    for i in range(1, int(n_wins - 1)):
        local_energy.append(
            torch.sum(
                (
                    rir[
                        (i - 1) * int(win_len * hop) : (i - 1) * int(win_len * hop)
                        + win_len
                    ]
                    ** 2
                )
                * win
            ).item()
        )
    # discard trailing points
    # remove (1/2/hop) to avoid map to negative time (center of window)
    n_win_discard = (overlap / hop) - (1 / 2 / hop)

    local_energy = local_energy[int(n_win_discard) :]
    return int(
        win_len * hop * (np.argmax(local_energy) - 1)
    )  # one hopsize as safety margin


def WGN_reverb(
    matrix_size: tuple = (1, 1), t60: float = 1.0, samplerate: int = 48000, device=None
) -> torch.Tensor:
    r"""
    Generates White-Gaussian-Noise-reverb impulse responses.

        **Arguments**:
            - **matrix_size** (tuple, optional): (output_channels, input_channels). Defaults to (1,1).
            - **t60** (float, optional): Reverberation time. Defaults to 1.0.
            - **samplerate** (int, optional): Sampling frequency. Defaults to 48000.
            - **nfft** (int, optional): Number of frequency bins. Defaults to 2**11.

        **Returns**:
            torch.Tensor: Matrix of WGN-reverb impulse responses.
    """
    # Number of samples
    n_samples = int(1.5 * t60 * samplerate)
    # White Guassian Noise
    noise = torch.randn(n_samples, *matrix_size, device=device)
    # Decay
    dr = t60 / torch.log(torch.tensor(1000, dtype=torch.float32, device=device))
    decay = torch.exp(-1 / dr * torch.linspace(0, t60, n_samples))
    decay = decay.view(-1, *(1,) * (len(matrix_size))).expand(-1, *matrix_size)
    # Decaying WGN
    IRs = torch.mul(noise, decay)
    # Go to frequency domain
    TFs = torch.fft.rfft(input=IRs, n=n_samples, dim=0)

    # Generate bandpass filter
    fc_left = torch.tensor([20], dtype=torch.float32, device=device)
    fc_right = torch.tensor([20000], dtype=torch.float32, device=device)
    g = torch.tensor([1], dtype=torch.float32, device=device)
    b, a = bandpass_filter(
        fc1=fc_left, fc2=fc_right, gain=g, fs=samplerate, device=device
    )
    sos = torch.cat((b.reshape(1, 3), a.reshape(1, 3)), dim=1)
    bp_H = sosfreqz(sos=sos, nfft=n_samples).squeeze()
    bp_H = bp_H.view(*bp_H.shape, *(1,) * (len(TFs.shape) - 1)).expand(*TFs.shape)

    # Apply bandpass filter
    TFs = torch.mul(TFs, bp_H)

    # Return to time domain
    IRs = torch.fft.irfft(input=TFs, n=n_samples, dim=0)

    # Normalize
    vec_norms = torch.linalg.vector_norm(IRs, ord=2, dim=(0))
    return IRs / vec_norms
