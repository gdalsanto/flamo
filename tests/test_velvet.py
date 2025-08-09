"""
Simple tests for velvet noise classes
"""
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flamo.processor.velvet import (
    VelvetNoiseSequence, 
    VelvetNoiseBank,
    ExtendedVelvetNoiseSequence
)


def test_velvet_noise_sequence_basic():
    """Test basic VelvetNoiseSequence generation."""
    length = 48000  # 1 second at 48kHz
    density = 2205  # Standard density from paper
    
    vn = VelvetNoiseSequence(length=length, density=density)
    sequence = vn()
    
    # Check shape
    assert sequence.shape == (length,)
    
    # Check sparsity (should be mostly zeros)
    num_nonzero = torch.count_nonzero(sequence).item()
    expected_impulses = length / (48000 / density)
    assert 0.8 * expected_impulses <= num_nonzero <= 1.2 * expected_impulses
    
    # Check values are only -1, 0, or 1
    unique_values = torch.unique(sequence)
    assert all(v in [-1, 0, 1] for v in unique_values)
    
    # Check first impulse is at position 0
    assert sequence[0] != 0


def test_velvet_noise_sequence_density():
    """Test different density values."""
    length = 24000  # 0.5 seconds
    
    for density in [1000, 2205, 4000]:
        vn = VelvetNoiseSequence(length=length, density=density)
        sequence = vn()
        
        num_impulses = torch.count_nonzero(sequence).item()
        expected = length / (48000 / density)
        
        # Allow 20% variance due to randomness
        assert 0.8 * expected <= num_impulses <= 1.2 * expected


def test_velvet_noise_bank():
    """Test VelvetNoiseBank generation."""
    num_channels = 4
    length = 12000  # 0.25 seconds
    density = 2205
    
    bank = VelvetNoiseBank(
        num_channels=num_channels,
        length=length,
        density=density
    )
    
    sequences = bank()
    
    # Check shape
    assert sequences.shape == (num_channels, length)
    
    # Check each channel is different
    for i in range(num_channels):
        for j in range(i+1, num_channels):
            assert not torch.equal(sequences[i], sequences[j])
    
    # Check each channel has correct properties
    for i in range(num_channels):
        channel = sequences[i]
        num_nonzero = torch.count_nonzero(channel).item()
        expected = length / (48000 / density)
        assert 0.7 * expected <= num_nonzero <= 1.3 * expected


def test_extended_velvet_noise_sequence():
    """Test ExtendedVelvetNoiseSequence with delta parameter."""
    length = 48000
    density = 2205
    
    # Test different delta values
    for delta in [0.25, 0.5, 0.75, 1.0]:
        evn = ExtendedVelvetNoiseSequence(
            length=length,
            density=density,
            delta=delta
        )
        sequence = evn()
        
        # Check basic properties
        assert sequence.shape == (length,)
        unique_values = torch.unique(sequence)
        assert all(v in [-1, 0, 1] for v in unique_values)
        
        # Check density is maintained
        num_impulses = torch.count_nonzero(sequence).item()
        expected = length / (48000 / density)
        assert 0.8 * expected <= num_impulses <= 1.2 * expected


def test_extended_velvet_noise_delta_validation():
    """Test that invalid delta values raise errors."""
    with pytest.raises(ValueError):
        ExtendedVelvetNoiseSequence(
            length=48000,
            density=2205,
            delta=0.0  # Invalid: must be > 0
        )
    
    with pytest.raises(ValueError):
        ExtendedVelvetNoiseSequence(
            length=48000,
            density=2205,
            delta=1.5  # Invalid: must be <= 1
        )


def test_velvet_noise_deterministic():
    """Test that sequences are deterministic when using same seed."""
    torch.manual_seed(42)
    vn1 = VelvetNoiseSequence(length=1000, density=2205)
    seq1 = vn1()
    
    torch.manual_seed(42)
    vn2 = VelvetNoiseSequence(length=1000, density=2205)
    seq2 = vn2()
    
    assert torch.equal(seq1, seq2)


if __name__ == "__main__":
    print("Running velvet noise tests...")
    
    test_velvet_noise_sequence_basic()
    print("✓ Basic sequence test passed")
    
    test_velvet_noise_sequence_density()
    print("✓ Density test passed")
    
    test_velvet_noise_bank()
    print("✓ Bank test passed")
    
    test_extended_velvet_noise_sequence()
    print("✓ Extended sequence test passed")
    
    test_extended_velvet_noise_delta_validation()
    print("✓ Delta validation test passed")
    
    test_velvet_noise_deterministic()
    print("✓ Deterministic test passed")
    
    print("\nAll tests passed!")