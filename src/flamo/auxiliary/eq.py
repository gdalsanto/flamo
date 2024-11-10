import torch
import numpy as np
from flamo.utils import RegularGridInterpolator
from flamo.functional import (
    db2mag,
    shelving_filter,
    peak_filter,
    probe_sos
)
from flamo.auxiliary.minimize import minimize_LBFGS

def eq_freqs(interval=1, start_freq=31.25, end_freq=16000):
    r"""
    Calculate the center frequencies and shelving crossover frequencies for an equalizer.

    **Args**:
        - interval (int, optional): The interval between center frequencies in octaves. Default: 1.
        - start_freq (float, optional): The starting frequency for the equalizer in Hz. Default: 31.25 Hz.
        - end_freq (float, optional): The ending frequency for the equalizer in Hz. Default: 16000 Hz.

    **Returns**:
        - tuple: A tuple containing the center frequencies and shelving crossover frequencies in Hz.

    """
    center_freq = torch.tensor(octave_bands(interval=interval, start_freq=start_freq, end_freq=end_freq))
    shelving_crossover = torch.tensor([center_freq[0]/np.power(2, 1/interval/2), center_freq[-1]*np.power(2, 1/interval/2)])  

    return center_freq, shelving_crossover

def octave_bands(interval=1, start_freq=31.25, end_freq=16000):
    r"""
    Generate a list of octave band central frequencies.

        **Args**:
            - interval (int, optional): The interval between the central frequencies. Default: 1.
            - start_freq (float, optional): The starting frequency of the octave bands. Default: 31.25 Hz.
            - end_freq (float, optional): The ending frequency of the octave bands. Default: 16000 Hz.

        **Returns**:
            - central_freq (list): A list of octave band central frequencies.

    """
    central_freq = []
    c_freq = start_freq
    while c_freq < end_freq:
        central_freq.append(c_freq*np.power(2, 1/interval))
        c_freq = central_freq[-1]       
    return central_freq

def geq(center_freq: torch.Tensor, shelving_freq: torch.Tensor, R: torch.Tensor, gain_db: torch.Tensor, fs: int = 48000, device=None):
    r"""
    Computes the second-order sections coefficents of a graphic equalizer.

    **Args**:
        - center_freq (torch.Tensor): Tensor containing the center frequencies of the bandpass filters in Hz.
        - shelving_freq (torch.Tensor): Tensor containing the corner frequencies of the shelving filters in Hz.
        - R (torch.Tensor): Tensor containing the resonance factor for the bandpass filters.
        - gain_db (torch.Tensor): Tensor containing the gain values in decibels for each frequency band.
        - fs (int, optional): Sampling frequency. Default: 48000 Hz.
        - device (str, optional): Device to use for constructing tensors. Default: None.

    **Returns**:
        - tuple: A tuple containing the numerator and denominator coefficients of the GEQ filter.
    """
    num_bands = len(center_freq) + len(shelving_freq) + 1
    assert len(gain_db) == num_bands, 'The number of gains must be equal to the number of frequencies.'
    sos = torch.zeros((6, num_bands), device=device)

    for band in range(num_bands):
        if band == 0:
            b = torch.tensor([db2mag(gain_db[band]), 0, 0], device=device)
            a = torch.tensor([1, 0, 0], device=device)
        elif band == 1:
            b, a = shelving_filter(shelving_freq[0], db2mag(gain_db[band]), 'low', fs=fs, device=device)
        elif band == num_bands - 1:
            b, a = shelving_filter(shelving_freq[1], db2mag(gain_db[band]), 'high', fs=fs, device=device)
        else:
            Q = torch.sqrt(R) / (R - 1)
            b, a = peak_filter(center_freq[band-2], db2mag(gain_db[band]), Q , fs=fs, device=device)
            
        sos_band = torch.hstack((b, a))
        sos[:, band] = sos_band

    return  sos[:3] ,  sos[3:] 

def design_geq(target_gain: torch.Tensor, center_freq: torch.Tensor, shelving_crossover: torch.Tensor, fs=48000):
    r"""
    Design a Graphic Equalizer (GEQ) filter.

        **Args**:
            - target_gain (torch.Tensor): Target gain values in dB for each frequency band.
            - center_freq (torch.Tensor): Center frequencies of each band.
            - shelving_crossover (torch.Tensor): Crossover frequencies for shelving filters.
            - fs (int, optional): Sampling frequency. Default: 48000 Hz.

        **Returns**:
            - tuple: A tuple containing the numerator and denominator coefficients of the GEQ filter.
    """

    # Initialization
    
    assert len(target_gain) == len(center_freq) + 2, 'The number of target gains must be equal to the number of center frequencies + 2.'

    nfft = 2**16
    num_freq = len(center_freq) + len(shelving_crossover) 
    R = torch.tensor(2.7)
    # Control frequencies are spaced logarithmically
    num_control = 100
    control_freq = torch.round(torch.logspace(np.log10(1), np.log10(fs/2.1), num_control+1))
    # interpolate the target gain values at control frequencies
    target_freq = torch.cat((torch.tensor([1]), center_freq, torch.tensor([fs/2.1])))
    # targetInterp = torch.tensor(np.interp(control_freq, target_freq, target_gain.squeeze()))
    interp = RegularGridInterpolator([target_freq], target_gain)
    targetInterp = interp([control_freq])

    # Design prototype of the biquad sections
    prototype_gain = 10  # dB
    prototype_gain_array = torch.full((num_freq + 1 ,1), prototype_gain)
    prototype_b, prototype_a = geq(center_freq, shelving_crossover, R, prototype_gain_array, fs)
    prototype_sos = torch.vstack((prototype_b, prototype_a))
    G, _, _ = probe_sos(prototype_sos, control_freq, nfft, fs)
    G = G / prototype_gain  # dB vs control frequencies

    # Define the optimization bounds
    upperBound = torch.tensor([torch.inf] + [2 * prototype_gain] * num_freq)
    lowerBound = torch.tensor([-val for val in upperBound])

    # Optimization
    opt_gains = minimize_LBFGS(G, targetInterp, lowerBound, upperBound, num_freq)

    # Generate the SOS coefficients
    b, a = geq(center_freq, shelving_crossover, R, opt_gains, fs)

    return b, a


