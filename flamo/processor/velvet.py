import torch
import torch.nn as nn

"""basicslly velvet noise matrix is a pytorch module that builds a 1D time-domain signal/tensor (multidim array) made of 
 -randomly placed impulses 
 -each impulse has a +1 or -1 sign
 -the impulse amplitudes decay exponentially in time
 -this sequence has sparse energy and sounds like a fuzzy hiss (?) that can be used in reverb tails


"""

class VelvetNoiseMatrix(nn.Module): #make it a pytorch module
    def __init__(self, length: int, density: float, sample_rate: int = 48000, decay_db: float = 60.0):
        super().__init__()
        self.length = length #how many samples long the output sequence is
        self.density = density #number of impulses per second
        self.sample_rate = sample_rate #sampes per second
        self.decay_db = decay_db #how much the pulse gains should decay in dB
        self.register_buffer("sequence", self._generate()) #store generated signal as a buffer in self.sequence
        #is this last line correct?


        #next write a generate method that generates the actual velvet noise seq 
        def _generate(self):
            Td = self.sample_rate / self.density