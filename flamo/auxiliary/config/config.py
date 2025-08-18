import os
import time
import pickle
import torch 
import numpy as np
import sympy as sp
from pydantic import BaseModel, model_validator, field_validator, computed_field, ConfigDict
from typing import Dict, Optional, List, Tuple

class HomogeneousFDNConfig(BaseModel):
    """
    Configuration class for the HomogeneousFDN class.
    """
    # number of delay lines
    N: int = 6
    # alias decay in dB
    alias_decay_db: int = 0
    # reveberation time in seconds
    rt60: Optional[float] = None
    # sampling rate 
    sample_rate: int = 48000
    # number of fft points
    nfft: int = 96000
    # device to run the model
    device: str = 'cpu'
    # delays in samples
    delays: Optional[List[int]] = None
    # delay lengths range in ms
    delay_range_ms: List[float] = [20.0, 50.0]    
    # requires gradients for optimization 
    input_gain_grad: bool =  True
    output_gain_grad: bool = True
    delays_grad: bool = False
    mixing_matrix_grad: bool = True
    attenuation_grad: bool = True
    is_delay_int: bool = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.delays is None:
            self.delay_length_samps()

    def delay_length_samps(self) -> List[int]:
        """Co-prime delay line lenghts for a given range"""
        ms_to_samps = lambda ms, fs: np.round(ms * fs / 1000).astype(int)
        delay_range_samps = ms_to_samps(np.asarray(self.delay_range_ms),
                                        self.sample_rate)
        # generate prime numbers in specified range
        prime_nums = np.array(list(
            sp.primerange(delay_range_samps[0], delay_range_samps[1])),
                              dtype=np.int32)
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        self.delays = np.array(np.r_[rand_primes[:self.N - 1],
                                       sp.nextprime(delay_range_samps[1])],
                                 dtype=np.int32).tolist()
        return self.delays
        
    # check that N coincides with the length of delays or generate delays if 
    # they don't exists already
    @field_validator('delays', mode='after')
    @classmethod
    def check_delays_length(cls, v, values):
        if v is not None:
            if len(v) != values['N']:
                raise ValueError(f"Length of delays ({len(v)}) must match N ({values['N']})")
        return v
    
    # validator for training on GPU
    @field_validator('device', mode='after')
    @classmethod
    def validate_training_device(cls, value):
        """Validate GPU, if it is used for training"""
        if value == 'cuda':
            assert torch.cuda.is_available(
            ), "CUDA is not available for training"
    
    # forbid extra fields - adding this to help prevent errors in config file creation
    model_config = ConfigDict(extra="forbid")
