import torch
import numpy as np
import sympy as sp

from collections import OrderedDict

from flamo.processor import dsp, system 
from flamo.auxiliary.eq import design_geq, low_shelf, design_geq_liski

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
                a[:, :, n_i], b[:, :, n_i] = design_geq(
                    target_gain=param[:, n_i],
                    center_freq=self.center_freq,
                    shelving_crossover=self.shelving_crossover,                    
                    fs=self.fs,
                    device=self.device
                )
         
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, a)
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, b)
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


class parallelFDNTwoStageGEQ(dsp.parallelFilter):
    r"""
    Class for creating a set of parallel attenuation filters that are scaled 
    according to the given delay lengths. The filter design is based on the 
    two stage attenuation filer by Välimaki, Prawda, and Schlecht. This is meant 
    to be used inside the feedback path of an FDN.
    Uses 1/3 octave bands from 20 Hz to 20 kHz (31 bands).
    
    The parameters represent the reverberation time in seconds. 
    The command gains are computed from the reverberation time as follows:

    .. math::
        \gamma = 10^{(60 \cdot \text{rt60} / 20)}
        \gamma_i = \gamma^{d_i}
    
    where :math:`\gamma` is the command gain, :math:`\text{rt60}` is the reverberation time,
    and :math:`d_i` is the delay length of the :math:`i`-th delay line.

        **Arguments / Attributes**:
            - **nfft** (int): Number of FFT points. Default is 2**11.
            - **fs** (int): Sampling frequency. Default is 48000.
            - **delays** (torch.Tensor): Tensor containing the delay lengths. Must be provided.
            - **device** (optional): Device to perform computations on. Default is None.

    Reference:
        - J. Liski and V. Välimäki, “The quest for the best graphic equalizer,”
        in Proc. DAFx, Edinburgh, UK, Sep. 2017, pp. 95-102.
        - V. Välimäki, K. Prawda, and S. J. Schlecht, “Two-stage attenuation
        filter for artificial reverberation,” IEEE Signal Process. Lett., Jan. 2024.
    """
    def __init__(
        self,
        nfft: int = 2**11,
        fs: int = 48000,
        fc_low_shelf: float = 300.0,
        delays: torch.Tensor =  None,
        alias_decay_db: float = 0.0,
        device=None
    ):
        assert (delays is not None), "Delays must be provided"
        self.delays = delays
        map = lambda x: torch.mul(rt2slope(x, fs).unsqueeze(-1), delays.unsqueeze(0))
        self.fs = fs
        self.fc_low_shelf = torch.tensor(fc_low_shelf, device = device)
        self.f_band = 10**3 * (2.0 ** (torch.arange(-17, 14) / 3.0))
        self.n_bands = len(self.f_band)
        freqs = torch.fft.rfftfreq(nfft, 1/fs)
        self.freq_ind = torch.zeros(self.n_bands, dtype=torch.long)  # Locations of the band frequencies
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20)
        self.alias_envelope_dcy = (gamma ** torch.arange(0, 3, 1, device=device))
        for i in range(self.n_bands):
            self.freq_ind[i] = torch.argmin(torch.abs(freqs - self.f_band[i]))

        super().__init__(
            size=(self.n_bands,),
            nfft=nfft,
            map=map,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_filters(self.map(param))[0]

    def get_filters(self, param):
        r"""
        Computes the filter coefficients for the attenuation.
        """
        # TODO: this unfortunately is not good enough and lead to nan valued magnitude response. 
        # one way is to follow the example at the bottom of eq.py to interpolate the target transfer function. 
        # that however cannot be backpropagated 
        b = torch.zeros((3, self.n_bands + 1, len(self.delays)), device=self.device, dtype=torch.float64)
        a = torch.zeros((3, self.n_bands + 1, len(self.delays)), device=self.device, dtype=torch.float64)
        
        for n_i in range(len(self.delays)):
            # compute the low shelf filter 
            b[:, 0, n_i], a[:, 0, n_i] = low_shelf(
                fc=self.fc_low_shelf,
                fs=torch.tensor(self.fs),
                GL=param[0, n_i],
                GH=param[-1, n_i],
                device=self.device
            )
            B_ls = torch.fft.rfft(b[:, 0, n_i], n=self.nfft, dim=0)
            A_ls = torch.fft.rfft(a[:, 0, n_i], n=self.nfft, dim=0)
            H_ls = B_ls / A_ls
            geq_target_gain = param[:, n_i] - 20*torch.log10(torch.abs(H_ls[self.freq_ind]))
            b[:, 1:, n_i], a[:, 1:, n_i] = design_geq_liski(
                target_gain=geq_target_gain,
                fs=self.fs,
                device=self.device)

        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, a)
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, b)
        B = torch.fft.rfft(b, n=self.nfft, dim=0)
        A = torch.fft.rfft(a, n=self.nfft, dim=0)
        A[torch.real(A) < 1e-12] = A[torch.real(A) < 1e-12] + torch.tensor(1e-12)
        A[torch.imag(A) < 1e-12] = A[torch.imag(A) < 1e-12] + torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        return H.to(torch.float32), B, A
    
    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = len(self.delays)
        self.output_channels = len(self.delays)

    def check_param_shape(self):
        assert (
            len(self.size) == 1
        ), 'The parameter should contain only the command gains'
