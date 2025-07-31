import torch
import torch.nn as nn
import math

"""basicslly velvet noise matrix is a pytorch module that builds a 1D time-domain signal/tensor (multidim array) made of 
 -randomly placed impulses 
 -each impulse has a +1 or -1 sign
 -the impulse amplitudes decay exponentially in time
 -this sequence has sparse energy and sounds like a fuzzy hiss (?) that can be used in reverb tails

done with the help of Gloria Dal Santo. formulas used come from the "Late-Reverberation Synthesis Using Interleaved
Velvet-Noise Sequences" by Vesa Välimäki and Karolina Prawda.

"""

class VelvetNoiseMatrix(nn.Module): #make it a pytorch module
        def __init__(self, length: int, density: float, sample_rate: int = 48000, decay_db: float = 60.0):
            super().__init__()
            self.length = length #how many samples long the output sequence is
            self.density = density #number of impulses per second
            self.sample_rate = sample_rate #sampes per second
            self.decay_db = decay_db #how much the pulse gains should decay in dB
            self.register_buffer("sequence", self._generate()) #store generated signal as a buffer in self.sequence
            #is this last line correct? buffers as they are named tensors that do not update gradients during training 


        #next write a generate method that generates the actual velvet noise seq 
        def _generate(self):
            Td = self.sample_rate / self.density  #td is grid size!!!!
            num_impulses = self.length / Td #number of impulses in the sequence, 
            floor_impulses=math.floor(num_impulses) #round down to the nearest integer

            
            #first generate fixed grid positions where impulses will be placed
            grid_positions = torch.arange(floor_impulses) * Td
            
            #next generate random ppositions for the impulses with jitter (como en el algoritmo q me mostro gloria)
            #generate random jitter factors between 0 and 1 for each impulse
            jitter_factors= torch.rand(floor_impulses) #how much jitter to apply to each impulse?
            #now compute the final positions of the impulses plus the jittter 
            impulse_indices= torch.ceil(grid_positions + jitter_factors * (Td - 1)).long() 
            impulse_indices[0] = 0 #make sure the first impulse always starts at 0
            impulse_indices = torch.clamp(impulse_indices, max=self.length - 1) #stay within the bounds of the length of the sequence
            print(impulse_indices)

            #now lets'g generate the random signs (), since each impulse can be either +1 or -1 and they should be generated randomly 
            #and should be uniformly distributed
            sign = 2 * torch.randint(0, 2, (floor_impulses,)) - 1
            print(sign)


            return torch.zeros(self.length)
if __name__ == "__main__":
    vnm = VelvetNoiseMatrix(length=48000, density=100)
    print("generated velvet noise sequence:")
    print(vnm.sequence)    