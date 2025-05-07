import torch
import numpy as np
import sympy as sp
from typing import Optional

from collections import OrderedDict

from flamo.processor import dsp, system
from flamo.auxiliary.eq import design_geq
from flamo.functional import (
    prop_peak_filter,
    prop_shelving_filter,
)
def rt2slope(rt60: torch.Tensor, fs: int):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope.
    """
    return -60 / (rt60 * fs)


def rt2absorption(rt60: torch.Tensor, fs: int, delays_len: torch.Tensor):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope relative to the delay line length.
    """
    slope = rt2slope(rt60, fs)
    return torch.einsum("i,j->ij", slope, delays_len)

class map_gamma(torch.nn.Module):

    def __init__(self, delays, is_compressed=True):
        super().__init__()
        self.delays = delays
        self.is_compressed = is_compressed
        self.g_min = 0.99
        self.g_max = 1.0

    def forward(self, x):
        if self.is_compressed:
            return (
                torch.sigmoid(x[0]) * (self.g_max - self.g_min) + self.g_min
            ) ** self.delays
        else:
            return x[0] ** self.delays

class inverse_map_gamma(torch.nn.Module):

    def __init__(self, delays = None,  is_compressed=True):
        super().__init__()
        self.delays = delays
        self.is_compressed = is_compressed
        self.g_min = 0.99
        self.g_max = 1.0

    def forward(self, y):

        if self.is_compressed:
            if self.delays is None:
                sig = (y - self.g_min) / (self.g_max - self.g_min)
            else:
                sig = (y**(1/self.delays) - self.g_min) / (self.g_max - self.g_min)
            return torch.log(sig/(1-sig)) 
        else:
            if self.delays is None:
                return y
            else:
                return y**(1/self.delays)
            

class HomogeneousFDN:
    r"""
    Class for creating a Feedback Delay Network (FDN) model with homogeneous attenuation.
    """

    def __init__(self, config_dict: dict):

        # get config parameters
        self.config_dict = config_dict

        # get number of delay lines
        self.N = self.config_dict.N

        # get delay line lengths
        self.delays = self.config_dict.delays

        # create a random isntance of the FDN
        self.fdn = self.get_fdn_instance()

        self.set_model()

    def set_model(self, input_layer=None, output_layer=None):
        # set the input and output layers of the FDN model
        if input_layer is None:
            input_layer = dsp.FFT(self.config_dict.nfft)
        if output_layer is None:
            output_layer = dsp.iFFTAntiAlias(
                nfft=self.config_dict.nfft,
                alias_decay_db=self.config_dict.alias_decay_db,
            )

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
            device=self.config_dict.device,
        )
        output_gain = dsp.Gain(
            size=(1, self.N),
            nfft=self.config_dict.nfft,
            requires_grad=self.config_dict.output_gain_grad,
            alias_decay_db=self.config_dict.alias_decay_db,
            device=self.config_dict.device,
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
            device=self.config_dict.device,
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
            device=self.config_dict.device,
        )

        # homogeneous attenuation
        attenuation = dsp.parallelGain(
            size=(self.N,),
            nfft=self.config_dict.nfft,
            requires_grad=self.config_dict.attenuation_grad,
            alias_decay_db=self.config_dict.alias_decay_db,
            device=self.config_dict.device,
        )
        attenuation.map = map_gamma(delay_lines)
        attenuation.assign_value(
            6
            * torch.ones(
                (self.N,),
            )
        )

        feedforward = system.Series(
            OrderedDict({"delays": delays, "attenuation": attenuation})
        )
        # Build recursion
        feedback_loop = system.Recursion(fF=feedforward, fB=mixing_matrix)

        # Build the FDN
        FDN = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )
        return FDN

    def get_shell(self, input_layer, output_layer):
        return system.Shell(
            core=self.fdn, input_layer=input_layer, output_layer=output_layer
        )

    def get_delay_lines(self):
        """Co-prime delay line lenghts for a given range"""
        ms_to_samps = lambda ms, fs: np.round(ms * fs / 1000).astype(int)
        delay_range_samps = ms_to_samps(
            np.asarray(self.config_dict.delay_range_ms), self.config_dict.sample_rate
        )
        # generate prime numbers in specified range
        prime_nums = np.array(
            list(sp.primerange(delay_range_samps[0], delay_range_samps[1])),
            dtype=np.int32,
        )
        rand_primes = prime_nums[np.random.permutation(len(prime_nums))]
        # delay line lengths
        delay_lengths = np.array(
            np.r_[rand_primes[: self.N - 1], sp.nextprime(delay_range_samps[1])],
            dtype=np.int32,
        ).tolist()
        return delay_lengths

    def get_raw_parameters(self):
        # get the raw parameters of the FDN
        with torch.no_grad():
            core = self.model.get_core()
            param = {}
            param["A"] = core.feedback_loop.feedback.param.cpu().numpy()
            param["attenuation"] = (
                core.feedback_loop.feedforward.attenuation.param.cpu().numpy()
            )
            param["B"] = core.input_gain.param.cpu().numpy()
            param["C"] = core.output_gain.param.cpu().numpy()
            param["m"] = core.feedback_loop.feedforward.delays.param.cpu().numpy()
            return param

    def set_raw_parameters(self, param: dict):
        # set the raw parameters of the FDN from a dictionary
        with torch.no_grad():
            core = self.model.get_core()
            for key, value in param.items():
                tensor_value = torch.tensor(value)
                if key == "A":
                    core.feedback_loop.feedback.assign_value(tensor_value)
                elif key == "attenuation":
                    core.feedback_loop.feedforward.attenuation.assign_value(
                        tensor_value.squeeze()
                    )
                elif key == "B":
                    core.input_gain.assign_value(tensor_value)
                elif key == "C":
                    core.output_gain.assign_value(tensor_value)
                elif key == "m":
                    core.feedback_loop.feedforward.delays.assign_value(
                        tensor_value.squeeze()
                    )
            self.model.set_core(core)

    def normalize_energy(
        self,
        target_energy=1,
    ):
        """energy normalization done in the frequency domain
        Note that the energy computed from the frequency response is not the same as the energy of the impulse response
        Read more at https://pytorch.org/docs/stable/generated/torch.fft.rfft.html
        """

        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))

        # apply energy normalization on input and output gains only
        with torch.no_grad():
            core = self.model.get_core()
            core.input_gain.assign_value(
                torch.div(
                    core.input_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            core.output_gain.assign_value(
                torch.div(
                    core.output_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            self.model.set_core(core)

        # recompute the energy of the FDN
        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))
        assert (
            abs(energy_H - target_energy) / target_energy < 0.0001
        ), "Energy normalization failed"

    def rt2gain(self, rt60):
        # convert RT60 to gain
        gdB = rt2absorption(
            rt60, self.config_dict.sample_rate, torch.tensor(self.delays)
        ).squeeze()
        return 10 ** (gdB / 20)
    

class parallelFDNAccurateGEQ(dsp.parallelAccurateGEQ):
    r"""
    Class for creating a set of parallel attenuation filters that are scaled 
    according to the given delay lengths. This is meant to be used inside the 
    feedback path of an FDN.
    
    The parameters represent the reverberation time in seconds. 
    The command gains are computed from the reverberation time as follows:

    .. math::
        \gamma = 10^{(60 \cdot \text{rt60} / 20)}
        \gamma_i = \gamma^{d_i}
    
    where :math:`\gamma` is the command gain, :math:`\text{rt60}` is the reverberation time,
    and :math:`d_i` is the delay length of the :math:`i`-th delay line.

        **Arguments / Attributes**:
            - **octave_interval** (int): Interval of octaves for the equalizer bands. Default is 1.
            - **nfft** (int): Number of FFT points. Default is 2**11.
            - **fs** (int): Sampling frequency. Default is 48000.
            - **delays** (torch.Tensor): Tensor containing the delay lengths. Must be provided.
            - **alias_decay_db** (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - **device** (optional): Device to perform computations on. Default is None.

    """
    def __init__(
        self,
        octave_interval: int = 1,
        nfft: int = 2**11,
        fs: int = 48000,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        device=None
    ):
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        map = lambda x: torch.mul(rt2slope(x, fs).unsqueeze(-1), delays.unsqueeze(0))
        super().__init__(
            size=( ),
            octave_interval=octave_interval,
            nfft=nfft,
            fs=fs,
            map=map,
            alias_decay_db=alias_decay_db,
            device=device
        )


    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        b = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        for n_i in range(len(self.delays)):
                b[:, :, n_i], a[:, :, n_i] = design_geq(
                    target_gain=param[:, n_i],
                    center_freq=self.center_freq,
                    shelving_crossover=self.shelving_crossover,                    
                    fs=self.fs,
                    device=self.device
                )
         
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        A[A == 0+1j*0] = torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        return H, B, A
    
    def check_param_shape(self):
        assert (
            len(self.size) == 1
        ), 'The parameter should contain only the command gains'

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = len(self.delays)
        self.output_channels = len(self.delays)

class parallelFirstOrderShelving(dsp.parallelFilter):
    
    def __init__(
        self,
        nfft: int = 2**11,
        fs: int = 48000,
        rt_nyquist: float = 0.01,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        device: str = None,
    ):
        size = (2,)      # rt at DC and crossover frequency
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        self.rt_nyquist = torch.tensor(rt_nyquist)
        map = lambda x: self.map_param(x, fs)
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            alias_decay_db=alias_decay_db,
            device=device
        )
        gamma = 10 ** (
            -torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20
        )
        self.alias_envelope_dcy = gamma ** torch.arange(0, 2, 1, device=device)
        self.fs = fs

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        b, a = param
        b_aa = torch.einsum('p, pn -> pn', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pn -> pn', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H = B / A
    
        return H, B, A 

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert (
            len(self.size) == 1
        ), "Filter must be 1D, for 2D filters use Filter module."

    def map_param(self, param, fs):
        rt_DC = param[0]
        gain_DC = torch.mul(rt2slope(rt_DC, fs), self.delays.unsqueeze(0))
        gain_Nyq = torch.mul(rt2slope(self.rt_nyquist, fs), self.delays.unsqueeze(0))
        omega_c = torch.clamp(param[1], min=0, max=torch.pi*2/5)
        t = torch.tan(omega_c)
        k = 10**(gain_DC/20) / 10**(gain_Nyq/20)

        a = torch.ones(2, len(self.delays), device=self.device)
        b = torch.ones(2, len(self.delays), device=self.device)

        a[0] = t / torch.sqrt(k) + 1
        a[1] = t / torch.sqrt(k) - 1
        b[0] = t * torch.sqrt(k) + 1
        b[1] = t * torch.sqrt(k) - 1
        return b * 10**(gain_Nyq/20), a

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = len(self.delays)
        self.output_channels = len(self.delays)

class parallelPropFirstOrderShelving(dsp.parallelFilter):
    
    def __init__(
        self,
        nfft: int = 2**11,
        fs: int = 48000,
        rt_nyquist: float = 0.01,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        requires_grad: bool = True,
        device: str = None,
    ):
        size = (2,)      # rt at DC and crossover frequency
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        map = lambda x: self.map_param(x, fs)
        self.rt_nyquist = torch.tensor(rt_nyquist)
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            alias_decay_db=alias_decay_db,
            requires_grad=requires_grad,
            device=device
        )
        gamma = 10 ** (
            -torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20
        )
        self.alias_envelope_dcy = gamma ** torch.arange(0, 2, 1, device=device)
        self.fs = fs

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        b, a = param
        b = b * ( 10** ( self.gain_DC / 20))
        b_aa = torch.einsum('p, pn -> pn', self.alias_envelope_dcy, b.unsqueeze(-1))
        a_aa = torch.einsum('p, pn -> pn', self.alias_envelope_dcy, a.unsqueeze(-1))
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H = B / A
    
        return H ** self.delays, B, A 

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert (
            len(self.size) == 1
        ), "Filter must be 1D, for 2D filters use Filter module."

    def map_param(self, param, fs):
        rt = param[0]
        self.gain_DC = rt2slope(rt, fs)
        gain_Ny = rt2slope(self.rt_nyquist, fs)
        b, a = prop_shelving_filter(
            fc=param[1],
            gain=gain_Ny,
            fs=fs,
            type="high",
            device=self.device,
        )       
        return b, a

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = len(self.delays)
        self.output_channels = len(self.delays)
