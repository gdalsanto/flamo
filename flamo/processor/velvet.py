import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

"""basicslly velvet noise matrix is a pytorch module that builds a 1D time-domain signal/tensor (multidim array) made of 
 -randomly placed impulses 
 -each impulse has a +1 or -1 sign
 -the impulse amplitudes decay exponentially in time
 -this sequence has sparse energy and sounds like a fuzzy hiss (?) that can be used in reverb tails

done with the help of Gloria Dal Santo. formulas used come from the "Late-Reverberation Synthesis Using Interleaved
Velvet-Noise Sequences" by Vesa Välimäki and Karolina Prawda.

"""

class VelvetNoiseSequence(nn.Module): #make it a pytorch module
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

            #now construct the actual sparse signal (1D tensor) with impulses
            sequence = torch.zeros(self.length)
            sequence[impulse_indices] = sign.float()

            return sequence
        
if __name__ == "__main__":
    vnm = VelvetNoiseSequence(length=48000, density=100)
    print("generated velvet noise sequence:")
    print(vnm.sequence)    
    #plot 
    plt.plot(vnm.sequence[:2000].numpy())
    plt.figure(figsize=(12, 4))
    plt.plot(vnm.sequence.numpy())
    plt.title("Velvet Noise Sequence")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




"""
what's done so far:
 VelvetNoiseSequence is done, a 1D generator
"""

#TODO!!!!!
"""next steps wld be 
generate multiple velvet noise sequences, one for each of the n^2 entries in the matrix, or for each filter path (more efficient???)
then filter each sequence (IIR or FIR??) to shape the spectral prof of each particular sequence 
then stack em into a 3d matrix (tensor). each filtered signal bwcomes a time domain "impulse response" for a corresponding matrix entry
tensor shld look like: NxNxT

the 3d matrix/tensor could be used in flamo 


in short:::::
now VelvetNoiseBank, Create NxN of them, shape (N, N, T)
then filtering step to make each 1D sequence smoother
last return a torch.Tensor of shape (N, N, T)


"""