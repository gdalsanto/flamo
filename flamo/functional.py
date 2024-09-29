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
        
def hertz2rad(hertz: torch.Tensor, fs):
    r'''
    Convert frequency from cycles per second to rad
    .. math::
        \omega = \frac{2\pi f}{f_s}
    where :math:`f` is the frequency in Hz and :math:`f_s` is the sampling frequency in Hz.

    **Args**:
        - hertz (torch.Tensor): The frequency in Hz.
        - fs (int): The sampling frequency in Hz.
    '''
    return torch.divide(hertz, fs)*2*torch.pi

def rad2hertz(rad: torch.Tensor, fs):
    r'''
    Convert frequency from rad to cycles per second
    .. math::
        f = \frac{\omega f_s}{2\pi}
    where :math:`\omega` is the frequency in rad and :math:`f_s` is the sampling frequency in Hz.

    **Args**:
        - rad (torch.Tensor): The frequency in rad.
        - fs (int): The sampling frequency in Hz.
    '''
    return torch.divide(rad*fs, 2*torch.pi)

def lowpass_filter(fc: float=500.0, gain:float=0.0, fs: int=48000) -> tuple:
    r"""
    Lowpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequency and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.
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

    **Args**:
        - fc (float): The cutoff frequency of the filter in Hz. Default: 500 Hz.
        - gain (float): The gain of the filter in dB. Default: 0 dB.
        - fs (int): The sampling frequency of the signal in Hz. Default: 48000 Hz.

    **Returns**:
        - b (ndarray): The numerator coefficients of the filter transfer function.
        - a (ndarray): The denominator coefficients of the filter transfer function.
    """

    omegaC = hertz2rad(fc, fs)
    two = torch.tensor(2)
    alpha = torch.sin(omegaC)/2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape)
    b = torch.ones(3, *omegaC.shape)

    b[0] = (1 - cosOC) / 2
    b[1] = 1 - cosOC
    b[2] = (1 - cosOC) / 2
    a[0] = 1 + alpha
    a[1] = - 2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain/20)*b, a

def highpass_filter(fc: float=10000.0, gain:float=0.0, fs: int=48000) -> tuple:
    r"""
    Highpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequency and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    for

    .. math::
        b_0 = \frac{1 + \cos(\omega_c)}{2},\;\; b_1 = - 1 - \cos(\omega_c),\;\; b_2 = \frac{1 + \cos(\omega_c)}{2}

    .. math::
        a_0 = 1 + \alpha,\;\; a_1 = -2 \cos(\omega_c),\;\; a_2 = 1 - \alpha

    where :math:`\omega_c = 2\pi f_c / f_s`, :math:`\alpha = \sin(\omega_c)/2 \cdot \sqrt{2}` and :math:`\cos(\omega_c)` is the cosine of the cutoff frequency.
    The gain is applied to the filter coefficients as :math:`b = 10^{g_{\textrm{dB}}/20} b`.

        **Args**:
            - fc (float, optional): The cutoff frequency of the filter in Hz. Default: 10000 Hz.
            - gain (float, optional): The gain of the filter in dB. Default: 0 dB.
            - fs (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.

        **Returns**:
            - b (ndarray): The numerator coefficients of the filter transfer function.
            - a (ndarray): The denominator coefficients of the filter transfer function.
    """

    omegaC = hertz2rad(fc, fs)
    two = torch.tensor(2)
    alpha = torch.sin(omegaC)/2 * torch.sqrt(two)
    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape)
    b = torch.ones(3, *omegaC.shape)

    b[0] = (1 + cosOC) / 2
    b[1] = - (1 + cosOC)
    b[2] = (1 + cosOC) / 2
    a[0] = 1 + alpha
    a[1] = - 2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain/20)*b, a

def bandpass_filter(fc1:torch.Tensor, fc2:torch.Tensor, gain:float=0.0, fs: int=48000) -> tuple:
    r"""
    Bandpass filter coefficients. It uses the `RBJ cookbook formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_ to map 
    the cutoff frequencies and gain to the filter coefficients to the to the :math:`\mathbf{b}` and :math:`\mathbf{a}` biquad coefficients.

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

        **Args**:
            - fc1 (float): The left cutoff frequency of the filter in Hz. 
            - fc2 (float): The right cutoff frequency of the filter in Hz. 
            - gain (float, optional): The gain of the filter in dB. Default: 0 dB.
            - fs (int, optional): The sampling frequency of the signal in Hz. Default: 48000 Hz.

        **Returns**:
            - b (ndarray): The numerator coefficients of the filter transfer function.
            - a (ndarray): The denominator coefficients of the filter transfer function.
    """

    omegaC = (hertz2rad(fc1, fs) + hertz2rad(fc2, fs)) / 2
    BW = torch.log2(fc2/fc1)
    two = torch.tensor(2)
    alpha = torch.sin(omegaC) * torch.sinh(torch.log(two) / two * BW * (omegaC / torch.sin(omegaC)))

    cosOC = torch.cos(omegaC)

    a = torch.ones(3, *omegaC.shape)
    b = torch.ones(3, *omegaC.shape)

    b[0] = alpha
    b[1] = 0
    b[2] = - alpha
    a[0] = 1 + alpha
    a[1] = - 2 * cosOC
    a[2] = 1 - alpha

    return 10**(gain/20)*b, a

def sosfreqz(sos: torch.Tensor, nfft: int = 512):
    """
    Compute the complex frequency response via FFT of cascade of biquads

        **Args**:
            - sos (torch.Tensor): Second order filter sections with shape (n_sections, 6)
            - nfft (int): FFT size. Default: 512

        **Returns**:
            - H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order

    B = torch.fft.rfft(sos[:, :3], nfft, dim=-1)
    A = torch.fft.rfft(sos[:, 3:], nfft, dim=-1)
    H = torch.prod(B, dim=0) / (torch.prod(A, dim=0) + torch.tensor(1e-12))
    return H

def svf(fc: torch.Tensor, R: torch.Tensor, m: torch.Tensor = torch.ones((3,)), G: torch.Tensor=None, filter_type=None, fs: int=48000):
    r"""
    Implements a State Variable Filter (SVF) with various filter types.

        **Args**:
            fc (torch.Tensor): The cutoff frequency of the filter.
            R (torch.Tensor): The resonance parameter of the filter.
            m (torch.Tensor, optional): The mixing coefficients of the filter. Default: torch.ones((3,)).
            G (torch.Tensor, optional): The gain coefficients of the filter. Default: None.
            filter_type (str, optional): The type of filter to be applied. Can be one of "lowpass", "highpass", "bandpass", "lowshelf", "highshelf", "peaking", "notch", or None. Default: None.
            fs (int, optional): The sampling frequency. Default: 48000.
        
        **Returns**:
            Tuple[torch.Tensor, torch.Tensor]: The numerator and denominator coefficients of the filter transfer function.

    """
    
    f = torch.tan(torch.pi * fc/fs)
    assert (R > 0).any(), "Resonance must be positive to ensure stability"

    if G is None:
        G = torch.ones_like(f)

    match filter_type:
        case "lowpass":
            m = torch.cat(
                ((torch.ones_like(G)).unsqueeze(-1),
                 (torch.zeros_like(G)).unsqueeze(-1),
                 torch.zeros_like(G).unsqueeze(-1),),dim=-1,)
        case "highpass":
            m = torch.cat(
                ((torch.zeros_like(G)).unsqueeze(-1),
                 (torch.zeros_like(G)).unsqueeze(-1),
                 torch.ones_like(G).unsqueeze(-1),),dim=-1,)
        case "bandpass":
            m = torch.cat(
                ((torch.zeros_like(G)).unsqueeze(-1),
                 (torch.ones_like(G)).unsqueeze(-1),
                 torch.zeros_like(G).unsqueeze(-1),),dim=-1,)
        case "lowshelf":
            m = torch.cat(
                ((torch.ones_like(G)).unsqueeze(-1),
                 (2 * R * torch.sqrt(G)).unsqueeze(-1),
                 (G * torch.ones_like(G)).unsqueeze(-1),),dim=-1,)
        case "highshelf":
            m = torch.cat(
                ((G * torch.ones_like(G)).unsqueeze(-1),
                 (2 * R * torch.sqrt(G)).unsqueeze(-1),
                 (torch.ones_like(G)).unsqueeze(-1),),dim=-1,)
        case "peaking" | "notch":
            m = torch.cat(
                ((torch.ones_like(G)).unsqueeze(-1),
                 (2 * R * torch.sqrt(G)).unsqueeze(-1),
                 (torch.ones_like(G)).unsqueeze(-1),),dim=-1,)
        case None:
            print("No filter type specified. Using the given mixing coefficents.")

    b = torch.zeros((3, *f.shape))
    a = torch.zeros((3, *f.shape))

    b[0] = (f**2) * m[...,0] + f * m[...,1] + m[...,2]
    b[1] = 2 * (f**2) * m[...,0] - 2 * m[...,2]
    b[2] = (f**2) * m[...,0] - f * m[...,1] + m[...,2]

    a[0] = (f**2) + 2 * R * f + 1
    a[1] = 2 * (f**2) - 2
    a[2] = (f**2) - 2 * R * f + 1

    return b, a