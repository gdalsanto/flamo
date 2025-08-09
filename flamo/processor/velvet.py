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


class VelvetNoiseSequence(nn.Module):
    """
    Basic velvet noise sequence generator.
    
    Velvet noise is characterized by:
    - Sparse impulses (mostly zeros)
    - Random signs (+1 or -1) for each impulse
    - Jittered positioning on a quasi-regular grid
    
    Following Välimäki & Prawda (2021).
    
    Args:
        length: Length of the output sequence in samples
        density: Number of impulses per second
        sample_rate: Sample rate in Hz (default: 48000)
    """
    
    def __init__(
        self, 
        length: int, 
        density: float, 
        sample_rate: int = 48000
    ):
        super().__init__()
        self.length = length
        self.density = density
        self.sample_rate = sample_rate
        
        # Generate and store the velvet noise sequence
        self.register_buffer("sequence", self._generate())
        
    def _generate(self) -> torch.Tensor:
        """Generate the velvet noise sequence."""
        # Calculate grid size (average distance between impulses)
        Td = self.sample_rate / self.density
        num_impulses = self.length / Td
        floor_impulses = math.floor(num_impulses)
        
        # Generate fixed grid positions
        grid_positions = torch.arange(floor_impulses) * Td
        
        # Add random jitter to each position (uniform distribution)
        jitter_factors = torch.rand(floor_impulses)
        impulse_indices = torch.ceil(grid_positions + jitter_factors * (Td - 1)).long()
        
        # Ensure first impulse is at position 0 and all indices are within bounds
        impulse_indices[0] = 0
        impulse_indices = torch.clamp(impulse_indices, max=self.length - 1)
        
        # Generate random signs (+1 or -1)
        signs = 2 * torch.randint(0, 2, (floor_impulses,)) - 1
        
        # Construct sparse signal
        sequence = torch.zeros(self.length)
        sequence[impulse_indices] = signs.float()
        
        return sequence
    
    def forward(self) -> torch.Tensor:
        """Return the velvet noise sequence."""
        return self.sequence


class VelvetNoiseBank(nn.Module):
    """
    Bank of velvet noise sequences for multi-channel processing.
    
    This can be used for:
    - Feedback Delay Networks (FDN)
    - Multi-channel decorrelation
    - Parallel reverb processing
    
    Args:
        num_channels: Number of velvet noise sequences to generate
        length: Length of each sequence in samples
        density: Number of impulses per second
        sample_rate: Sample rate in Hz (default: 48000)
    """
    
    def __init__(
        self,
        num_channels: int,
        length: int,
        density: float,
        sample_rate: int = 48000
    ):
        super().__init__()
        self.num_channels = num_channels
        self.length = length
        self.density = density
        self.sample_rate = sample_rate
        
        # Generate bank of velvet noise sequences
        self.register_buffer("bank", self._generate_bank())
        
    def _generate_bank(self) -> torch.Tensor:
        """Generate multiple independent velvet noise sequences."""
        bank = torch.zeros((self.num_channels, self.length))
        
        for i in range(self.num_channels):
            vn = VelvetNoiseSequence(
                length=self.length,
                density=self.density,
                sample_rate=self.sample_rate
            )
            bank[i, :] = vn.sequence
            
        return bank
    
    def forward(self) -> torch.Tensor:
        """Return the bank of velvet noise sequences."""
        return self.bank


class ExtendedVelvetNoiseSequence(nn.Module):
    """
    Extended Velvet Noise (EVN) sequence with configurable impulse range.
    
    EVN limits where impulses can appear within each grid period using
    a scaling factor delta. This enables interleaving multiple sequences
    without collisions, as described in Välimäki & Prawda (2021).
    
    Args:
        length: Length of the output sequence in samples
        density: Number of impulses per second
        sample_rate: Sample rate in Hz (default: 48000)
        delta: Scaling factor for impulse range (0 < delta <= 1)
               When delta=0.25, impulses only appear in first 25% of each grid
    """
    
    def __init__(
        self,
        length: int,
        density: float,
        sample_rate: int = 48000,
        delta: float = 1.0
    ):
        super().__init__()
        
        if not 0 < delta <= 1:
            raise ValueError("Delta must be in range (0, 1]")
            
        self.length = length
        self.density = density
        self.sample_rate = sample_rate
        self.delta = delta
        
        # Generate and store the extended velvet noise sequence
        self.register_buffer("sequence", self._generate())
        
    def _generate(self) -> torch.Tensor:
        """Generate the extended velvet noise sequence."""
        # Calculate grid size
        Td = self.sample_rate / self.density
        num_impulses = self.length / Td
        floor_impulses = math.floor(num_impulses)
        
        # Generate fixed grid positions
        grid_positions = torch.arange(floor_impulses) * Td
        
        # Add LIMITED random jitter based on delta
        jitter_factors = torch.rand(floor_impulses)
        impulse_indices = torch.ceil(
            grid_positions + self.delta * jitter_factors * (Td - 1)
        ).long()
        
        # Ensure first impulse is at position 0 and all indices are within bounds
        impulse_indices[0] = 0
        impulse_indices = torch.clamp(impulse_indices, max=self.length - 1)
        
        # Generate random signs (+1 or -1)
        signs = 2 * torch.randint(0, 2, (floor_impulses,)) - 1
        
        # Construct sparse signal
        sequence = torch.zeros(self.length)
        sequence[impulse_indices] = signs.float()
        
        return sequence
    
    def forward(self) -> torch.Tensor:
        """Return the extended velvet noise sequence."""
        return self.sequence