import torch
import numpy as np
import scipy.signal as signal
from scipy.io import loadmat 
from flamo.utils import RegularGridInterpolator, freq2rad, rad2freq
from flamo.functional import db2mag, shelving_filter, peak_filter, probe_sos
from flamo.auxiliary.minimize import minimize_LBFGS


def eq_freqs(interval: int = 1, start_freq: float = 31.25, end_freq: float = 16000.0):
    r"""
    Calculate the center frequencies and shelving crossover frequencies for an equalizer.

    **Arguments**:
        - **interval** (int, optional): The fraction of octave to use in one interval. Default: 1.
        - **start_freq** (float, optional): The starting frequency in Hz. Default: 31.25.
        - **end_freq** (float, optional): The ending frequency in Hz. Default: 16000.

    **Returns**:
        - tuple: A tuple containing the center frequencies and shelving crossover frequencies in Hz.

    """
    center_freq = torch.tensor(
        octave_bands(interval=interval, start_freq=start_freq, end_freq=end_freq)
    )
    shelving_crossover = torch.tensor(
        [
            center_freq[0] / np.power(2, 1 / interval / 2),
            center_freq[-1] * np.power(2, 1 / interval / 2),
        ]
    )

    return center_freq, shelving_crossover


def octave_bands(
    interval: int = 1, start_freq: float = 31.25, end_freq: float = 16000.0
):
    r"""
    Generate a list of octave band central frequencies.

    **Arguments**:
        - **interval** (int, optional): The fraction of octave to use in one interval. Default: 1.
        - **start_freq** (float, optional): The starting frequency in Hz. Default: 31.25.
        - **end_freq** (float, optional): The ending frequency in Hz. Default: 16000.

    **Returns**:
        - central_freq (list): A list of octave band central frequencies.

    """
    central_freq = []
    c_freq = start_freq
    while c_freq < end_freq:
        central_freq.append(c_freq * np.power(2, 1 / interval))
        c_freq = central_freq[-1]
    return central_freq


def geq(
    center_freq: torch.Tensor,
    shelving_freq: torch.Tensor,
    R: torch.Tensor,
    gain_db: torch.Tensor,
    fs: int = 48000,
    device: str = "cpu",
):
    r"""
    Computes the second-order sections coefficients of a graphic equalizer.
    The EQ is implemented as a series of shelving peak filters, using the :meth:`flamo.functional.shelving_filter` and :meth:`flamo.functional.peak_filter` methods.

    **Arguments**:
        - **center_freq** (torch.Tensor): Tensor containing the center frequencies of the bandpass filters in Hz.
        - **shelving_freq** (torch.Tensor): Tensor containing the corner frequencies of the shelving filters in Hz.
        - **R** (torch.Tensor): Tensor containing the resonance factor for the bandpass filters.
        - **gain_db** (torch.Tensor): Tensor containing the gain values in decibels for each frequency band.
        - **fs** (int, optional): Sampling frequency. Default: 48000 Hz.
        - **device** (str, optional): Device to use for constructing tensors. Default: cpu.

    **Returns**:
        - tuple: A tuple containing the numerator and denominator coefficients of the GEQ filter.

    """
    num_bands = len(center_freq) + len(shelving_freq) + 1
    assert (
        len(gain_db) == num_bands
    ), "The number of gains must be equal to the number of frequencies."
    sos = torch.zeros((6, num_bands), device=device)

    for band in range(num_bands):
        if band == 0:
            b = torch.tensor([db2mag(gain_db[band]), 0, 0], device=device)
            a = torch.tensor([1, 0, 0], device=device)
        elif band == 1:
            b, a = shelving_filter(
                shelving_freq[0], db2mag(gain_db[band]), "low", fs=fs, device=device
            )
        elif band == num_bands - 1:
            b, a = shelving_filter(
                shelving_freq[1], db2mag(gain_db[band]), "high", fs=fs, device=device
            )
        else:
            Q = torch.sqrt(R) / (R - 1)
            b, a = peak_filter(
                center_freq[band - 2], db2mag(gain_db[band]), Q, fs=fs, device=device
            )

        sos_band = torch.hstack((b, a))
        sos[:, band] = sos_band

    return sos[:3], sos[3:]


def design_geq(
    target_gain: torch.Tensor,
    center_freq: torch.Tensor,
    shelving_crossover: torch.Tensor,
    fs=48000,
    device: str = "cpu",
):
    r"""
    Design a Graphic Equalizer (GEQ) filter.
    Based on the method presented in

        Schlecht, S., Habets, E. (2017). Accurate reverberation time control in
        feedback delay networks Proc. Int. Conf. Digital Audio Effects (DAFx)

    Adapted to python by: Dal Santo G.

        **Args**:
            - target_gain (torch.Tensor): Target gain values in dB for each frequency band.
            - center_freq (torch.Tensor): Center frequencies of each band.
            - shelving_crossover (torch.Tensor): Crossover frequencies for shelving filters.
            - fs (int, optional): Sampling frequency. Default: 48000 Hz.
            - device (str, optional): Device to use for constructing tensors. Default: 'cpu'.

        **Returns**:
            - tuple: A tuple containing the numerator and denominator coefficients of the GEQ filter.
    """

    # Initialization

    assert (
        len(target_gain) == len(center_freq) + 2
    ), "The number of target gains must be equal to the number of center frequencies + 2."

    nfft = 2**16
    num_freq = len(center_freq) + len(shelving_crossover)
    R = torch.tensor(2.7, device=device)
    # Control frequencies are spaced logarithmically
    num_control = 100
    control_freq = torch.round(
        torch.logspace(
            torch.log10(torch.tensor(1, device=device)),
            np.log10(fs / 2.1),
            num_control + 1,
        )
    )
    # interpolate the target gain values at control frequencies
    target_freq = torch.cat(
        (
            torch.tensor([1], device=device),
            center_freq,
            torch.tensor([fs / 2.1], device=device),
        )
    )
    # targetInterp = torch.tensor(np.interp(control_freq, target_freq, target_gain.squeeze()))
    interp = RegularGridInterpolator([target_freq], target_gain)
    targetInterp = interp([control_freq])

    # Design prototype of the biquad sections
    prototype_gain = 10  # dB
    prototype_gain_array = torch.full((num_freq + 1, 1), prototype_gain)
    prototype_b, prototype_a = geq(
        center_freq, shelving_crossover, R, prototype_gain_array, fs
    )
    prototype_sos = torch.vstack((prototype_b, prototype_a))
    G, _, _ = probe_sos(prototype_sos, control_freq, nfft, fs)
    G = G / prototype_gain  # dB vs control frequencies

    # Define the optimization bounds
    upperBound = torch.tensor(
        [torch.inf] + [2 * prototype_gain] * num_freq, device=device
    )
    lowerBound = torch.tensor([-val for val in upperBound], device=device)

    # Optimization
    opt_gains = minimize_LBFGS(G, targetInterp, lowerBound, upperBound, num_freq)

    # Generate the SOS coefficients
    b, a = geq(center_freq, shelving_crossover, R, opt_gains, fs, device=device)

    return b, a


def low_shelf(fc: torch.Tensor, fs: int, GL: torch.Tensor, GH: torch.Tensor, device="cpu"):
    r"""
    Implementation of first-order low-shelf filter.

    **Arguments**:
        - **fc** (torch.Tensor): Crossover frequency in Hz.
        - **fs** (int): Sampling rate in Hz.
        - **GL** (torch.Tensor): Gain in the low frequencies (dB).
        - **GH** (torch.Tensor): Gain in the high frequencies (dB).
        - **device** (str, optional): Device to use for constructing tensors. Default: 'cpu'.

    **Returns**:
        - num (torch.Tensor): Numerator coefficients
        - den (torch.Tensor): Denominator coefficients
    """
    wH = freq2rad(fc, torch.tensor(fs)) # crossover frequency in radians
    gl = 10 ** (GL / 20) # low frequency gain
    gh = 10 ** (GH / 20) # high frequency gain
    g = gl / gh

    # compute filter coefficients
    aH0 = torch.tan(wH / 2) + torch.sqrt(g)
    aH1 = torch.tan(wH / 2) - torch.sqrt(g)
    bH0 = g * torch.tan(wH / 2) + torch.sqrt(g)
    bH1 = g * torch.tan(wH / 2) - torch.sqrt(g)

    num = gh * torch.stack([bH0, bH1, torch.tensor(0, device=device)])
    den = torch.stack([aH0, aH1, torch.tensor(0, device=device)])

    num = num / den[0]
    den = den / den[0]

    return num, den

def design_geq_liski(
    target_gain: torch.Tensor,
    fs: int = 48000,
    device: str = "cpu",
):
    r"""
    Design a two-stage Graphic Equalizer (GEQ) filter.
    Based on the method presented in

        V. Välimäki, K. Prawda, and S. J. Schlecht, “Two-stage attenuation
        filter for artificial reverberation,” IEEE Signal Process. Lett., Jan. 2024.

    Adapted to python by: Dal Santo G.

        **Args**:
            - target_gain (torch.Tensor): Target gain values in dB for each frequency band.
            - fs (int, optional): Sampling frequency. Default: 48000 Hz.
            - device (str, optional): Device to use for constructing tensors. Default: 'cpu'.

        **Returns**:
            - tuple: A tuple containing the numerator and denominator coefficients of the GEQ filter.

    References:
        - J. Liski and V. Välimäki, “The quest for the best graphic equalizer,”
        in Proc. DAFx, Edinburgh, UK, Sep. 2017, pp. 95-102.
        - V. Välimäki, K. Prawda, and S. J. Schlecht, “Two-stage attenuation
        filter for artificial reverberation,” IEEE Signal Process. Lett., Jan. 2024.

    """

    fc1 = 10**3 * (2 ** (torch.arange(-17, 14, device=device) / 3))
    fc2 = torch.zeros(61, device=device)
    fc2[0::2] = fc1
    for k in range(1, 61, 2):
        # add geometric mean of adjacent frequencies
        fc2[k] = torch.sqrt(fc2[k - 1] * fc2[k + 1])

    # convert to radians
    wg = freq2rad(fc1, torch.tensor(fs, device=device))
    wc = freq2rad(fc2, torch.tensor(fs, device=device))

    Gdb = target_gain 

    gw = 0.4  # gain factor at bandwidth
    # hardcoded EQ filter bandwidths
    bw = (
        2
        * torch.pi
        / fs
        * torch.tensor(
            [
                9.178,
                11.56,
                14.57,
                18.36,
                23.13,
                29.14,
                36.71,
                46.25,
                58.28,
                73.43,
                92.51,
                116.6,
                146.9,
                185.0,
                233.1,
                293.7,
                370.0,
                466.2,
                587.4,
                740.1,
                932.4,
                1175,
                1480,
                1865,
                2350,
                2846,
                3502,
                4253,
                5038,
                5689,
                5570,
            ],
            device=device,
        )
    )

    # estimate leakage between bands
    leak = interaction_matrix(10 ** (17 / 20) * torch.ones(31), gw, wg, wc, bw)
    Gdb2 = torch.zeros(61, device=device)
    Gdb2[0::2] = Gdb
    for k in range(1, 61, 2):
        # interpolate target gains linearly
        Gdb2[k] = (Gdb2[k - 1] + Gdb2[k + 1]) / 2

    Goptdb = torch.linalg.lstsq(leak.transpose(1, 0), Gdb2)[0]
    Gopt = 10 ** (Goptdb / 20)

    # iterate once
    leak2 = interaction_matrix(Gopt, gw, wg, wc, bw)
    G2optdb = torch.linalg.lstsq(leak2.transpose(1, 0), Gdb2)[0]
    G2opt = 10 ** (G2optdb / 20)
    G2woptdb = gw * G2optdb
    G2wopt = 10 ** (G2woptdb / 20)

    numsopt = torch.zeros((3, 31), device=device)
    densopt = torch.zeros((3, 31), device=device)
    for k in range(31):
        num, den = pareq(G2opt[k], G2wopt[k], wg[k], bw[k])
        numsopt[:, k] = num
        densopt[:, k] = den
    numsopt = numsopt / densopt[0, :]
    densopt = densopt / densopt[0, :]
    return numsopt, densopt


def interaction_matrix(G, gw, wg, wc, bw, device="cpu"):
    """
    Compute the interaction matrix of a cascade graphic equalizer
    to account for band interaction when assigning filter gains.

    Parameters:
    G  : array-like, Linear gain at which the leakage is determined
    gw : float, Gain factor at bandwidth (e.g., 0.5 refers to db(G)/2)
    wg : array-like, Command frequencies i.e. filter center frequencies (rad/sample)
    wc : array-like, Design frequencies (rad/sample) at which leakage is computed
    bw : array-like, Bandwidth of filters in radians

    Returns:
    leak : ndarray, (M, N) matrix showing the magnitude responses of
           band filters leaking to the design frequencies.
    """

    M = len(wg)  # number of filters
    N = len(wc)  # number of design frequencies
    leak = torch.zeros((M, N), device=device)  # initialize interaction matrix

    Gdb = 20 * torch.log10(G)
    Gdbw = gw * Gdb  # dB gain at bandwidth
    Gw = 10 ** (Gdbw / 20)

    exp_term = torch.exp(-1j * wc[:, None] * torch.arange(3, device=device))
    # estimate leak factors of peak/notch filters
    if torch.sum(torch.abs(Gdb)) != 0:
        for m in range(M):
            b, a = pareq(G[m], Gw[m], wg[m], bw[m])
            num = torch.sum(b * exp_term[:, : len(b)], dim=1)
            den = torch.sum(a * exp_term[:, : len(a)], dim=1)
            H = num / den
            Gain = 20 * torch.log10(torch.abs(H)) / Gdb[m]
            leak[m, :] = torch.abs(Gain)
    else:
        leak = torch.vstack(
            [torch.zeros((M,), device=device), torch.eye(M, device=device)]
        )[
            1:
        ]  # Avoid warning for zero-dB gains

    return leak

def pareq(G, GB, w0, B):
    """
    Second-order parametric equalizing filter design with adjustable bandwidth gain.

    Parameters:
    G  : float, Peak gain (linear)
    GB : float, Bandwidth gain (linear)
    w0 : float, Center frequency (radians/sample)
    B  : float, Bandwidth (radians/sample)x

    Returns:
    num : ndarray, [b0, b1, b2] = numerator coefficients
    den : ndarray, [1, a1, a2] = denominator coefficients

    Written by Vesa Valimaki, August 24, 2016
    Adapted to python by G. Dal Santo, 2025


    Ref. S. Orfanidis, Introduction to Signal Processing, p. 594
    We have set the dc gain to G0 = 1.

    """

    if G == 1:
        beta = torch.tan(B / 2)  # Avoid division by zero when G = 1
    else:
        beta = torch.sqrt(
            torch.abs(GB**2 - 1) / torch.abs(G**2 - GB**2)
        ) * torch.tan(B / 2)

    num = torch.tensor([(1 + G * beta), -2 * torch.cos(w0), (1 - G * beta)]) / (
        1 + beta
    )
    den = torch.tensor([1, -2 * torch.cos(w0) / (1 + beta), (1 - beta) / (1 + beta)])

    return num, den


if __name__ == "__main__":
    fs = 44100  # sampling frequency
    rt = loadmat("rirs/two-stage-RT-values.mat")["rt_"][:, 0]
    d_len = 593  # delay line length
    n_bands = 31  # number of bands
    device = "cpu"
    ## ## ## GET THE TARGET MAGNITUDE RESPONSE AT DESIRED FREQS ## ## ##

    if n_bands == 10:  # Octave bands
        f_band = 16000.0 / (2.0 ** torch.arange(9, -1, -1))
    elif n_bands == 31 or n_bands == 30:  # Third octave bands
        f_band = 10**3 * (2.0 ** (torch.arange(-17, 14) / 3.0))
        if n_bands == 30:
            f_band = f_band[:-1]  # Remove the highest band if 30 bands
    else:
        raise ValueError("Number of bands out of range")

    # step one - prepare the target filter gains
    gdB = -60.0 / (rt)
    # reverberation time converted to target gains in dB
    gains = 10 ** (gdB / 20)
    # frequency-dependent linear gain
    gLin_dl = torch.tensor(gains ** (d_len / fs))
    # linear gains adjusted for the delay-line length
    gdB_dl = 20 * torch.log10(gLin_dl)  # delay-adjusted gains in dB

    # step two - interpolate the shape of the target attenuation

    n_freq = fs//2+1  # Number of frequency points
    freq = torch.cat(
        (
            torch.logspace(
                torch.log10(torch.tensor(1.0)),
                torch.log10(torch.tensor(fs / 2 - 1.0)),
                n_freq - 1,
            ),
            torch.tensor([fs / 2]),
        )
    )  # Frequency points
    freq = torch.fft.rfftfreq(n=fs, d=1/fs)
    ind = torch.zeros(n_bands, dtype=torch.long)  # Locations of the band frequencies
    for i in range(len(f_band)):
        ind[i] = torch.argmin(torch.abs(freq - f_band[i]))

    target_mag = torch.zeros(n_freq)  # Target magnitude response
    target_mag[ind[0] : ind[-1] + 1] = torch.from_numpy(
        np.interp(
            freq[ind[0] : ind[-1] + 1].numpy(),
            f_band.numpy(),
            gdB_dl.numpy(),
            left=gdB_dl[0],
            right=gdB_dl[-1],
        )
    )

    target_mag = torch.from_numpy(
        np.interp(
            freq.numpy(),
            freq[ind[0] : ind[-1] + 1].numpy(),
            target_mag[ind[0] : ind[-1] + 1].numpy(),
            left=target_mag[ind[0]].item(),
            right=target_mag[ind[-1]].item(),
        )
    )
    ## ## ## DESIGN THE LOW SHELF FILTER ## ## ##

    # G_low = 10 ** (target_mag[0] / 20)  # low shelf gain
    # G_high = 10 ** (target_mag[-1] / 20)  # low shelf gain
    G_low = gdB_dl[0]
    G_high = gdB_dl[-1]
    fc = torch.tensor(300, device=device) # crossover frequency in Hz
    [b, a] = low_shelf(fc, fs, G_low, G_high)
    b = b / a[0]
    a = a / a[0]
    exp_term = torch.exp(-1j * (2 * torch.pi * freq[:, None] / fs) * torch.arange(3, device=device))
    num = torch.sum(b * exp_term, dim=1)
    den = torch.sum(a * exp_term, dim=1)
    H_shelf = num / den
    G0 = 20 * torch.log10(torch.abs(H_shelf))

    # Gdb = target_mag - G0
    Gdb = gdB_dl - G0[ind]
    # densopt, numsopt = design_geq_liski(Gdb[ind], fs=torch.tensor(fs))
    numsopt, densopt = design_geq_liski(Gdb, fs=torch.tensor(fs))

    Hopttot = torch.ones(n_freq, dtype=complex, device=device)
    exp_term = torch.exp(-1j * ( 2 * torch.pi * freq[:, None] / fs ) * torch.arange(3, device=device))
    for k in range(31):
        num = torch.sum(numsopt[:, k] * exp_term, dim=1)
        den = torch.sum(densopt[:, k] * exp_term, dim=1)
        Hopttot *= num / den

    H_final = Hopttot * H_shelf

    a_tot = torch.hstack((a.unsqueeze(-1), densopt))
    b_tot = torch.hstack((b.unsqueeze(-1), numsopt))

    A = torch.prod(torch.fft.rfft(a_tot, n=int(len(freq)*2-1), dim=0), dim=-1)
    B = torch.prod(torch.fft.rfft(b_tot, n=int(len(freq)*2-1), dim=0), dim=-1)

    H = B / A
    
    import matplotlib.pyplot as plt


    # plt.plot(freq, 20 * torch.log10(torch.abs(H_final)))
    # plt.plot(freq, 20 * torch.log10(torch.abs(H_final_2)))
    plt.plot(freq, 20 * torch.log10(torch.abs(H)))
    plt.plot(f_band, gdB_dl, "ro")
    plt.xscale("log")
    plt.grid()
    plt.show()

