"""
Example usage of velvet noise filters integrated with FLAMO Filter system.

This example shows how to use the new Filter-based velvet noise classes
that are differentiable and can be used within the FLAMO processing chain.
"""

import torch.nn as nn   

import matplotlib.pyplot as plt
from flamo.auxiliary.velvet import (
    VelvetNoiseFilter,
)
from flamo.functional import signal_gallery
from flamo.processor import dsp

def example_velvet_noise_filter():
    """Example using VelvetNoiseFilter."""
    print("=== VelvetNoiseFilter Example ===")
    
    in_ch, out_ch = 1, 1
    nfft = 2048  # FFT size for the filter
    length = nfft # Length of the filter

    # Create a velvet noise filter
    # size = (length, output_channels, input_channels)
    velvet_filter = VelvetNoiseFilter(
        size=(length, out_ch, in_ch),  # 1024 samples, 1x1 matrix
        density=1000.0,     # 1000 impulses per second
        sample_rate=48000,
        requires_grad=True  # Make it differentiable
    )
    
    input_layer = dsp.FFT(nfft=nfft)
    output_layer = dsp.iFFT(nfft=nfft)

    my_dsp = nn.Sequential(input_layer, velvet_filter, output_layer)

    # -------------- Apply unit impulse to DSP --------------

    # Input signal
    input_sig = signal_gallery(
        signal_type="impulse",
        batch_size=1,
        n_samples=nfft,  
        n=in_ch,
        fs=48000,
        device="cpu",
    )

    output_vn = my_dsp(input_sig)
    # plot the sequence 
    plt.figure(figsize=(12, 4))
    plt.plot(output_vn[0, :, 0].squeeze().numpy())
    for pos in velvet_filter.grid:
        plt.axvline(x=pos.item(), color='r', linestyle='--', alpha=0.5)
    plt.title('Output Velvet Noise Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return 