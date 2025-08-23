import torch
import numpy as np
import sympy as sp
from typing import Optional

from collections import OrderedDict

from flamo.processor import dsp, system
from flamo.utils import to_complex
from flamo.auxiliary.eq import accurate_geq, geq
from flamo.functional import (
    prop_peak_filter,
    prop_shelving_filter,
)
from flamo.processor.dsp import Filter
from flamo.auxiliary.eq import eq_freqs
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

class map_gfdn_gamma(torch.nn.Module):
    def __init__(self, delays: torch.Tensor, n_groups: int, fs: int):
        super().__init__()
        self.delays = delays
        self.n_groups = n_groups
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to grouped RT values."""
        gamma = torch.mul(rt2slope(x, self.fs).unsqueeze(-1), self.delays.unsqueeze(0))
        return gamma

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
            isint=self.config_dict.is_delay_int,
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
        start_freq: float = 31.25,
        end_freq: float = 16000.0,
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
            start_freq=start_freq,
            end_freq=end_freq,
            device=device
        )


    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        b = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        for n_i in range(len(self.delays)):
                b[:, :, n_i], a[:, :, n_i] = accurate_geq(
                    target_gain=param[:, n_i],
                    center_freq=self.center_freq,
                    shelving_crossover=self.shelving_crossover,                    
                    fs=self.fs,
                    device=self.device
                )
         
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), b.to(torch.double))
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), a.to(torch.double))
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H_temp = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        H = torch.where(torch.abs(torch.prod(A, dim=1)) != 0, H_temp, torch.finfo(H_temp.dtype).eps*torch.ones_like(H_temp))
        H_type = torch.complex128 if param.dtype == torch.float64 else torch.complex64
        return H.to(H_type), B, A
    
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

class parallelGFDNAccurateGEQ(parallelFDNAccurateGEQ):
    # TODO
    def __init__(
        self,
        octave_interval: int = 1,
        n_groups: int = 2,
        nfft: int = 2**11,
        fs: int = 48000,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        start_freq: float = 31.25,
        end_freq: float = 16000.0,
        device=None
    ):
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        self.n_groups = n_groups
        super().__init__(
            octave_interval=octave_interval,
            nfft=nfft,
            delays=delays,
            fs=fs,
            alias_decay_db=alias_decay_db,
            start_freq=start_freq,
            end_freq=end_freq,
            device=device
        )
        self.n_gains = self.size[0]
        self.size = (self.n_groups * self.size[0],)
        self.param = torch.nn.Parameter(
            torch.empty(self.size, device=self.device), requires_grad=self.requires_grad
        )
        self.map = map_gfdn_gamma(self.delays, self.n_groups, self.fs)

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        b = torch.zeros((3, self.size[0]+1, len(self.delays)), device=self.device)
        for n_i in range(len(self.delays)):
            for i_group in range(self.n_groups):
                (
                    b[:, i_group * (self.n_gains) : (i_group + 1) * self.n_gains, n_i],
                    a[:, i_group * (self.n_gains) : (i_group + 1) * self.n_gains, n_i],
                ) = accurate_geq(
                    target_gain=param[
                        i_group * self.n_gains : (i_group + 1) * self.n_gains, n_i
                    ],
                    center_freq=self.center_freq,
                    shelving_crossover=self.shelving_crossover,
                    fs=self.fs,
                    device=self.device,
                )

        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), b.to(torch.double))
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), a.to(torch.double))
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H_temp = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        H = torch.where(torch.abs(torch.prod(A, dim=1)) != 0, H_temp, torch.finfo(H_temp.dtype).eps*torch.ones_like(H_temp))
        H_type = torch.complex128 if param.dtype == torch.float64 else torch.complex64
        return H.to(H_type), B, A

class parallelFDNGEQ(dsp.parallelGEQ):
    r"""
    Class for creating a set of parallel attenuation filters that are scaled 
    according to the given delay lengths. This is meant to be used inside the 
    feedback path of an FDN. This class differs from `parallelFDNAccurateGEQ` in 
    that it uses a nonoptimized version of the GEQ filter design. But this makes 
    it trainable. 
    
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
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: Optional[str] = None,
    ):
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        # convert from second to decay in dB
        # map = lambda x: torch.mul(rt2slope(x, fs).unsqueeze(-1), delays.unsqueeze(0))
        # map = lambda x: torch.mul(rt2slope(x, fs).unsqueeze(-1), delays.unsqueeze(0))
        map = lambda x: x
        super().__init__(
            size=( ),   # the size is (n_bands, n_delays)
            octave_interval=octave_interval,
            nfft=nfft,
            fs=fs,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, self.size[0], len(self.delays)), device=self.device)
        b = torch.zeros((3, self.size[0], len(self.delays)), device=self.device)
        R = torch.tensor(2.7, device=self.device)
        for n_i in range(len(self.delays)):
                b[:, :, n_i], a[:, :, n_i] = geq(
                    gain_db=torch.mul(rt2slope(param, self.fs).unsqueeze(-1), self.delays[n_i]),
                    center_freq=self.center_freq,
                    R=R,
                    shelving_freq=self.shelving_crossover,                    
                    fs=self.fs,
                    device=self.device
                )
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), b.to(torch.double))
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy.to(torch.double), a.to(torch.double))
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H_temp = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        H = torch.where(torch.abs(torch.prod(A, dim=1)) != 0, H_temp, torch.finfo(H_temp.dtype).eps*torch.ones_like(H_temp))
        H_type = torch.complex128 if param.dtype == torch.float64 else torch.complex64
        return H.to(H_type), B, A

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

    def init_param(self):
        return torch.nn.init.uniform_(self.param, a=1.0, b=3.0) 

class parallelFDNPEQ(Filter):
    r"""
    Parallel counterpart of the :class:`PEQ` class
    For information about **attributes** and **methods** see :class:`flamo.processor.dsp.PEQ`.
    """
    def __init__(
        self,
        n_bands: int = 10,
        f_min: float = 20,
        f_max: float = 20000,
        delays: torch.Tensor =  None,
        design: str = "biquad",
        is_twostage: bool = False,
        is_proportional: bool = False,
        nfft: int = 2**11,
        fs: int = 48000,
        map: callable = lambda x: x,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: Optional[str] = None,
    ):
        self.delays = delays
        self.is_twostage = is_twostage
        self.is_proportional = is_proportional
        self.n_bands = n_bands
        self.design = design
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        gamma = 10 ** (
            -torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20
        )
        k = torch.arange(1, self.n_bands + 1, dtype=torch.float32)
        self.center_freq_bias = f_min * (f_max / f_min) ** ((k - 1) / (self.n_bands - 1))
        self.alias_envelope_dcy = gamma ** torch.arange(0, 3, 1, device=device)
        super().__init__(
            size=(self.n_bands+1 if self.is_twostage else self.n_band, 3, 1 if self.is_proportional else len(delays)),
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device,
        )

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        if self.is_twostage:
            param_eq = self.map_eq(param[:-1, ...])
            param_ls = self.map_eq(param[-1, ...], is_twostage=True)
        else:
            param_eq = self.map_eq(param)

        a = torch.zeros((self.n_bands + 1 if self.is_twostage else self.n_bands, 3, len(self.delays)), device=self.device)
        b = torch.zeros((self.n_bands + 1 if self.is_twostage else self.n_bands, 3, len(self.delays)), device=self.device)
            
        for n_i in range(len(self.delays)):
            if self.is_proportional:
                f = param_eq[0, :, 0]
                R = param_eq[1, :, 0]
                G = param_eq[2, :, 0] * self.delays[n_i]
            else:
                f = param_eq[0, :, n_i] 
                R = param_eq[1, :, n_i]
                G = param_eq[2, :, n_i]
            if self.is_twostage:
                if self.is_proportional:
                    f = torch.cat((f, param_ls[0, 0].unsqueeze(0)), dim=0) 
                    R = torch.cat((R, param_ls[1, 0].unsqueeze(0)), dim=0)
                    G = torch.cat((G, param_ls[2, 0].unsqueeze(0) * self.delays[n_i]), dim=0) 
                else:
                    f = torch.cat((f, param_ls[0, n_i].unsqueeze(0)), dim=0) 
                    R = torch.cat((R, param_ls[1, n_i].unsqueeze(0)), dim=0)
                    G = torch.cat((G, param_ls[2, n_i].unsqueeze(0)), dim=0)                    
            # low shelf filter 
            a[0, :, n_i], b[0, :, n_i] = self.compute_biquad_coeff(
                f=f[0],
                R=R[0] if self.design == 'biquad' else R[0] + torch.sqrt(torch.tensor( 1 / 2)),
                G=G[0],
                type='highshelf'
            )
            # high shelf filter 
            a[self.n_bands-1, :, n_i], b[self.n_bands-1, :, n_i] = self.compute_biquad_coeff(
                f=f[self.n_bands-1],
                R=R[self.n_bands-1] if self.design == 'biquad' else R[self.n_bands-1] + torch.sqrt(torch.tensor( 1 / 2)),
                G=G[self.n_bands-1],
                type='lowshelf'
            )
            # peak filter 
            a[1:(self.n_bands-1), :, n_i], b[1:(self.n_bands-1), :, n_i] = self.compute_biquad_coeff(
                f=f[1:(self.n_bands-1)],
                R=R[1:(self.n_bands-1)],
                G=G[1:(self.n_bands-1)],
                type='peaking'
            )
            if self.is_twostage:
                a[-1, :, n_i], b[-1, :, n_i]= self.compute_biquad_coeff(
                f=f[-1],
                R=R[-1] if self.design == 'biquad' else R[-1] + torch.sqrt(torch.tensor( 1 / 2)),
                G=G[-1],
                type='highshelf'
            )

        b_aa = torch.einsum("p, opn -> opn", self.alias_envelope_dcy.to(torch.double), b.to(torch.double))
        a_aa = torch.einsum("p, opn -> opn", self.alias_envelope_dcy.to(torch.double), a.to(torch.double))
        B = torch.fft.rfft(b_aa, self.nfft, dim=1)
        A = torch.fft.rfft(a_aa, self.nfft, dim=1)
        H_temp = (torch.prod(B, dim=0) / (torch.prod(A, dim=0)))
        # H_temp = (torch.prod(B, dim=0) / (torch.prod(A, dim=0)))

        H = torch.where(torch.abs(torch.prod(A, dim=0)) != 0, H_temp, torch.finfo(H_temp.dtype).eps*torch.ones_like(H_temp))
        H_type = torch.complex128 if param.dtype == torch.float64 else torch.complex64
        return H.to(H_type), B, A

    def compute_biquad_coeff(self, f, R, G, type='peaking'):
        # f : freq, R : resonance, G : gain in dB
        b = torch.zeros(*f.shape, 3, device=self.device)     
        a = torch.zeros(*f.shape, 3, device=self.device)  

        if self.design == 'svf':
            G = 10 ** (G / 20)
            if type == 'peaking':
                mLP = torch.ones_like(G)
                mBP = 2 * R * torch.sqrt(G)
                mHP = torch.ones_like(G)
            elif type == 'lowshelf':
                mLP = G
                mBP = 2 * R * torch.sqrt(G)
                mHP = torch.ones_like(G)
            elif type == 'highshelf':
                mLP = torch.ones_like(G)
                mBP = 2 * R * torch.sqrt(G)
                mHP = G
            b[..., 0] = (f**2) * mLP + f * mBP + mHP
            b[..., 1] = 2*(f**2) * mLP - 2 * mHP
            b[..., 2] = (f**2) * mLP - f * mBP + mHP
            a[..., 0] = f**2 + 2*R*f + 1
            a[..., 1] = 2* (f**2) - 2
            a[..., 2] = f**2 - 2*R*f + 1  
        elif self.design == 'biquad':
            G = 10 ** (G / 40)
            if type == 'peaking':
                alpha = torch.sin(f) / (2 * R)
                b[..., 0] = 1 + alpha * G
                b[..., 1] = -2 * torch.cos(f)
                b[..., 2] = 1 - alpha * G
                a[..., 0] = 1 + alpha / G
                a[..., 1] = -2 * torch.cos(f)
                a[..., 2] = 1 - alpha / G
            elif type == 'lowshelf':
                alpha = torch.sin(f) * torch.sqrt((G**2 + 1) * (1/R - 1) + 2*G)
                b[..., 0] = G * ((G + 1) - (G - 1) * torch.cos(f) + alpha)
                b[..., 1] = 2 * G * ((G - 1) - (G + 1) * torch.cos(f))
                b[..., 2] = G * ((G + 1) - (G - 1) * torch.cos(f) - alpha)
                a[..., 0] = (G + 1) + (G - 1) * torch.cos(f) + alpha
                a[..., 1] = -2 * ((G - 1) + (G + 1) * torch.cos(f))
                a[..., 2] = (G + 1) + (G - 1) * torch.cos(f) - alpha
            elif type == 'highshelf':
                alpha = torch.sin(f) * torch.sqrt((G**2 + 1) * (1/R - 1) + 2*G)
                b[..., 0] = G * ((G + 1) + (G - 1) * torch.cos(f) + alpha)
                b[..., 1] = -2 * G * ((G - 1) + (G + 1) * torch.cos(f))
                b[..., 2] = G * ((G + 1) + (G - 1) * torch.cos(f) - alpha)
                a[..., 0] = (G + 1) - (G - 1) * torch.cos(f) + alpha
                a[..., 1] = 2 * ((G - 1) - (G + 1) * torch.cos(f))
                a[..., 2] = (G + 1) - (G - 1) * torch.cos(f) - alpha

        return a, b
    
    def map_eq(self, param, is_twostage=False):
        r"""
        Mapping function for the raw parameters to the SVF filter coefficients.
        """
        if self.design == 'biquad' and not is_twostage:
            # frequency mapping
            bias = torch.tensor(self.center_freq_bias / self.fs * 2 * torch.pi, device=self.device)
            min_f = torch.tensor(2 * torch.pi * self.f_min / self.fs, device=self.device)
            max_f = torch.tensor(2 * torch.pi * self.f_max / self.fs, device=self.device)
            f = torch.clamp(torch.sigmoid(param[:, 0, ...] - 1/2) / 2**(torch.linspace(self.n_bands, 0, self.n_bands, device=self.device)).unsqueeze(-1) + bias.unsqueeze(-1), min=min_f, max=max_f) 
            # Q factor mapping 
            R = torch.zeros_like(param[:, 1, ...])
            R[0, :] = 0.1 + torch.sigmoid(R[0, :]) * 0.9
            R[-1, :] = 0.1 + torch.sigmoid(R[-1, :]) * 0.9
            R[1:-1, :] = 0.1 + torch.sigmoid(R[1:-1, :] ) * 3
            # Gain mapping
            clip_min = torch.tensor(-1e-6, device=self.device)
            clip_max = torch.tensor(-5, device=self.device)
            G =  clip_min + torch.sigmoid(param[:, 2, ...] - 1/2) * clip_max
        elif self.design == 'svf' and not is_twostage:
            # frequency mapping
            bias = torch.log(2 * self.center_freq_bias / self.fs / (1 - 2 * self.center_freq_bias / self.fs)).to(device=self.device)
            f = torch.tan(torch.pi * torch.sigmoid(param[:, 0, ...] + bias.unsqueeze(-1)) * 0.5) 
            # Q factor mapping
            R =  torch.log(1+torch.exp(param[:, 1, ...]))  / torch.log(torch.tensor(2, device=self.device)) 
            # G 
            G = 10**(-torch.log(1+torch.exp(param[:, 2, ...] - 1/2)) / torch.log(torch.tensor(2, device=self.device))) - 10
        elif (self.design == 'svf' or self.design == 'biquad') and is_twostage:
            # frequency mapping
            bias = torch.tensor((torch.pi / 3), device=self.device)
            f = torch.sigmoid(param[0]) / self.n_bands  + bias
            # Q factor mapping 
            R = torch.zeros_like(param[1])
            R[:] = 0.1 + torch.sigmoid(R) * 0.9
            # Gain mapping
            clip_min = torch.tensor(-1e-6, device=self.device)
            clip_max = torch.tensor(-30, device=self.device)
            G = clip_min + torch.sigmoid(param[2] - 1/2) * clip_max

        param = torch.cat(
            (
                f.unsqueeze(0),
                R.unsqueeze(0),
                G.unsqueeze(0),
            ),
            dim=0,
        )
        return param
    
    def init_param(self):
        torch.nn.init.uniform_(self.param)

    def check_param_shape(self):
        assert (
            len(self.size) == 3
        ), "Filter must be 2D in the parallel configuration, for 3D filters use PEQ module."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )

    def initialize_class(self):
        self.check_param_shape()
        self.get_io()
        self.freq_response = to_complex(
            torch.empty((self.nfft // 2 + 1, *self.size[2:]))
        )
        self.get_freq_response()
        self.get_freq_convolve()

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]


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
        rt_nyquist: float = 0.2,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        device: str = None,
    ):
        size = (2,)      # rt at DC and crossover frequency
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        self.rt_nyquist = torch.tensor(rt_nyquist, device=device)
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
        omega_c = torch.clamp(param[1], min=0, max=torch.pi)
        t = torch.tan(omega_c / 2)
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
