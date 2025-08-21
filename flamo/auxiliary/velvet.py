"""
Velvet Noise implementations for FLAMO

Velvet noise is a sparse pseudo-random noise used in artificial reverberation.
It consists of sample values of +1, -1, and 0, with the non-zero samples 
occurring at pseudo-random locations.

References:
    Välimäki, V., & Prawda, K. (2021). Late-Reverberation Synthesis Using 
    Interleaved Velvet-Noise Sequences. IEEE/ACM Transactions on Audio, 
    Speech, and Language Processing, 29, 1149-1160.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from flamo.processor.dsp import Filter, parallelFilter


class VelvetNoiseFilter(Filter):
    """
    TODO 
    Args:
        size: Size of the filter parameters (length, output_channels, input_channels)
        density: Number of impulses per second
        delta: Scaling factor for impulse range (0 < delta <= 1)
               When delta=0.25, impulses only appear in first 25% of each grid
        sample_rate: Sample rate in Hz (default: 48000)
        nfft: Number of FFT points required to compute the frequency response
        requires_grad: Whether the filter parameters require gradients
        alias_decay_db: The decaying factor in dB for the time anti-aliasing envelope
        device: The device of the constructed tensors
    """
    
    def __init__(
        self,
        size: tuple = (1, 1, 1),
        density: float = 1000.0,
        delta: float = 1.0,
        sample_rate: int = 48000,
        nfft: int = 2**11,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: Optional[str] = None,
    ):
        self.density = density
        self.sample_rate = sample_rate
        self.Td = sample_rate / density  # Average distance between impulses
        if not 0 < delta <= 1:
            raise ValueError("Delta must be in range (0, 1]")
            
        self.delta = delta
        # Create mapping function that generates velvet noise
        map = lambda x: self._generate_velvet_impulse_response(x)
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device,
        )
    
    def _generate_velvet_impulse_response(self, param: torch.Tensor) -> torch.Tensor:
        """Generate velvet noise impulse response from parameters."""        
        # Calculate grid size (average distance between impulses)

        
        result = torch.zeros_like(param)
        
        for out_ch in range(self.param.shape[1]):
            for in_ch in range(self.param.shape[2]):
                # Extract parameters for this channel pair
                result[:, out_ch, in_ch] = self._generate_velvet_sequence()
                
        return result
    
    def _generate_velvet_sequence(
        self, 
    ) -> torch.Tensor:
        """Generate a single velvet noise sequence."""
        
        # Add random jitter to each position (uniform distribution)
        jitter_factors = torch.rand(self.floor_impulses)
        impulse_indices = torch.ceil(self.grid + self.delta *  jitter_factors * (self.Td - 1)).long()

        # first impulse is at position 0 and all indices are within bounds
        impulse_indices[0] = 0
        impulse_indices = torch.clamp(impulse_indices, max=self.param.shape[0] - 1)
        
        # Generate random signs (+1 or -1)
        signs = 2 * torch.randint(0, 2, (self.floor_impulses,)) - 1
        
        # Construct sparse signal
        sequence = torch.zeros(self.size[0], device=self.device)
        sequence[impulse_indices] = signs.float()

        return sequence

    def initialize_class(self):
        r"""
        Initializes the Filter module.

        This method checks the shape of the gain parameters, computes the frequency response of the filter,
        and computes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_io()
        num_impulses = self.param.shape[0] / self.Td
        self.floor_impulses = math.floor(num_impulses)
        self.grid = torch.arange(self.floor_impulses) * self.Td
        self.get_freq_response()
        self.get_freq_convolve()