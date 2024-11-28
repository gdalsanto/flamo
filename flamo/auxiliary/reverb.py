import torch
import numpy as np
import sympy as sp

from collections import OrderedDict

from flamo.processor import dsp, system

def rt2slope(rt60, fs):
    '''convert time in seconds of 60db decay to energy decay slope'''
    return -60/(rt60*fs)

def rt2absorption(rt60, fs, delays_len):
    '''convert time in seconds of 60db decay to energy decay slope relative to the delay line length'''
    slope = rt2slope(rt60, fs)
    return torch.einsum('i,j->ij', slope , delays_len)

class HomogeneousFDN:
    
    def __init__(self, config_dict):

        # get config parameters
        self.config_dict = config_dict

        # get number of delay lines
        self.N = self.config_dict.N

        # get delay line lengths
        self.delays = self.config_dict.delays

        # create a random isntance of the FDN 
        self.fdn = self.get_fdn_instance()
        
        self.set_model()

    def set_model(self, input_layer=None , output_layer=None):
        # set the input and output layers of the FDN model
        if input_layer is None:
            input_layer = dsp.FFT(self.config_dict.nfft)
        if output_layer is None:
            output_layer = dsp.iFFTAntiAlias(nfft=self.config_dict.nfft, alias_decay_db=self.config_dict.alias_decay_db)

        self.model = self.get_shell(input_layer, output_layer)
        
    def get_fdn_instance(self):

        # delay lines 
        delay_lines = torch.tensor(self.delays)

        # Input and output gains
        input_gain = dsp.Gain(
            size=(self.N, 1), 
            nfft=self.config_dict.nfft, 
            requires_grad=self.config_dict.input_gain_grad, 
            alias_decay_db=self.config_dict.alias_decay_db, 
            device=self.config_dict.device
        )
        output_gain = dsp.Gain(
            size=(1, self.N), 
            nfft=self.config_dict.nfft, 
            requires_grad=self.config_dict.output_gain_grad, 
            alias_decay_db=self.config_dict.alias_decay_db, 
            device=self.config_dict.device
        )

        # RECURSION
        # Feedback loop with delays
        delays = dsp.parallelDelay(
            size=(self.N,),
            max_len=delay_lines.max(),
            nfft=self.config_dict.nfft,
            isint=True,
            requires_grad=self.config_dict.delays_grad,
            alias_decay_db=self.config_dict.alias_decay_db,
            device=self.config_dict.device
        )
        # assign the required delay line lengths
        delays.assign_value(delays.sample2s(delay_lines))

        # feedback mixing matrix 
        mixing_matrix = dsp.Matrix(
            size=(self.N, self.N),
            nfft=self.config_dict.nfft,
            matrix_type="orthogonal",
            requires_grad=self.config_dict.mixing_matrix_grad,
            alias_decay_db=self.config_dict.alias_decay_db,
            device=self.config_dict.device
        )

        # homogeneous attenuation 
        attenuation = dsp.parallelGain(
            size=(self.N, ),
            nfft=self.config_dict.nfft,
            requires_grad=self.config_dict.attenuation_grad,
            alias_decay_db=self.config_dict.alias_decay_db,
            device=self.config_dict.device
        )
        attenuation.map = map_gamma(delay_lines)
        attenuation.assign_value(4*torch.ones((self.N, ),))
        
        feedforward = system.Series(OrderedDict({
            'delays': delays,
            'attenuation': attenuation
        }))
        # Build recursion
        feedback_loop = system.Recursion(fF=feedforward, fB=mixing_matrix)

        # Build the FDN
        FDN = system.Series(OrderedDict({
            'input_gain': input_gain,
            'feedback_loop': feedback_loop,
            'output_gain': output_gain,
        }))
        return FDN
    
    def get_shell(self, input_layer, output_layer):
        return system.Shell(core=self.fdn, 
                            input_layer=input_layer,
                            output_layer=output_layer)
    
    def get_delay_lines(self):
        """Co-prime delay line lenghts for a given range"""
        ms_to_samps = lambda ms, fs: np.round(ms * fs / 1000).astype(int)
        delay_range_samps = ms_to_samps(np.asarray(self.config_dict.delay_range_ms),
                                        self.config_dict.sample_rate)
        # generate prime numbers in specified range
        prime_nums = np.array(list(
            sp.primerange(delay_range_samps[0], delay_range_samps[1])),
                                dtype=np.int32)
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        delay_lengths = np.array(np.r_[rand_primes[:self.N - 1],
                                        sp.nextprime(delay_range_samps[1])],
                                    dtype=np.int32).tolist()
        return delay_lengths
    
    def get_raw_parameters(self):
        # get the raw parameters of the FDN
        with torch.no_grad():
            core = self.model.get_core()
            param = {}
            param['A'] = core.feedback_loop.feedback.param.cpu().numpy()
            param['attenuation'] = core.feedback_loop.feedforward.attenuation.param.cpu().numpy()
            param['B'] = core.input_gain.param.cpu().numpy()
            param['C'] = core.output_gain.param.cpu().numpy()
            param['m'] = core.feedback_loop.feedforward.delays.param.cpu().numpy()
            return param
    
    def set_raw_parameters(self, param: dict):
        # set the raw parameters of the FDN from a dictionary
        with torch.no_grad():
            core = self.model.get_core()
            for key, value in param.items():
                tensor_value = torch.tensor(value)
                if key == 'A':
                    core.feedback_loop.feedback.assign_value(tensor_value)
                elif key == 'attenuation':
                    core.feedback_loop.feedforward.attenuation.assign_value(tensor_value.squeeze())
                elif key == 'B':
                    core.input_gain.assign_value(tensor_value)
                elif key == 'C':
                    core.output_gain.assign_value(tensor_value)
                elif key == 'm':
                    core.feedback_loop.feedforward.delays.assign_value(tensor_value.squeeze())
            self.model.set_core(core)
               
    def normalize_energy(self, target_energy = 1, ):
        """ energy normalization done in the frequency domain
        Note that the energy computed from the frequency response is not the same as the energy of the impulse response
        Read more at https://pytorch.org/docs/stable/generated/torch.fft.rfft.html
        """

        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H),2))

        # apply energy normalization on input and output gains only
        with torch.no_grad():
            core = self.model.get_core()
            core.input_gain.assign_value(torch.div(core.input_gain.param, torch.pow( energy_H / target_energy, 1/4)))
            core.output_gain.assign_value(torch.div(core.output_gain.param, torch.pow( energy_H / target_energy, 1/4)))
            self.model.set_core(core)

        # recompute the energy of the FDN
        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H),2))
        assert abs(energy_H - target_energy)/target_energy < 0.0001, "Energy normalization failed"

    def rt2gain(self, rt60):
        # convert RT60 to gain
        gdB = rt2absorption(rt60, 
                            self.config_dict.sample_rate, 
                            torch.tensor(self.delays)).squeeze()
        return 10**(gdB/20)
    
class map_gamma(torch.nn.Module):

    def __init__(self, delays):
        super().__init__()
        self.delays = delays
        self.g_min = 0.99
        self.g_max = 1.0

    def forward(self, x):
        return (torch.sigmoid(x[0]) * (self.g_max - self.g_min) + self.g_min)**self.delays