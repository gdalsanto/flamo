import torch
import torch.nn as nn
import warnings
from collections import OrderedDict
from flamo.functional import *

# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    def __init__(self, transform=lambda x: x):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)


class FFT(Transform):
    def __init__(self, nfft=2**11, norm="backward"):
        """
        Fast Fourier Transform (FFT) module.
        Args:
            nfft (int): Window length.
            norm (str or None): Normalization mode.
        """
        self.nfft = nfft
        self.norm = norm
        transform = lambda x: torch.fft.rfft(x, n=self.nfft, dim=1, norm=self.norm)
        super().__init__(transform=transform)


class iFFT(Transform):
    def __init__(self, nfft=2**11, norm="backward"):
        """
        Inverse Fast Fourier Transform (iFFT) module.
        Args:
            nfft (int): Signal length.
            norm (str or None): Normalization mode.
        """
        self.nfft = nfft
        self.norm = norm
        transform = lambda x: torch.fft.irfft(x, n=self.nfft, dim=1, norm=self.norm)
        super().__init__(transform=transform)


# ============================= CORE ================================


class DSP(nn.Module):
    """
    Processor core module consisting of learnable parameters used to derive the frequency response of a LTI system, then convolved with the input signal.

    Args:
        size (tuple): shape of the parameters.
        nfft (int, optional): Number of FFT points required to compute the frequency response. Defaults to 2 ** 11.
        map (function, optional): Mapping function applied to the raw parameters. Defaults to lambda x: x.
        requires_grad (bool, optional): Whether the parameters requires gradients. Defaults to False.
        alias_decay_db (float, optional): Decaying factor in dB for the time anti-aliasing envelope.
                                          The decay refers to the attenuation after nfft samples. Defaults to 0.
    """

    def __init__(
        self,
        size: tuple,
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__()
        assert isinstance(size, tuple), "Size must be a tuple."
        self.size = size  # size of the parameters
        self.nfft = nfft  # number of FFT points
        self.map = (
            map  # mapping function from raw parameters to their desired distribution
        )
        self.new_value = 0  # flag indicating if new values have been assigned
        self.requires_grad = requires_grad  # flag indicating if gradients are required
        self.param = nn.Parameter(
            torch.empty(self.size), requires_grad=self.requires_grad
        )  # parameters of the DSP module
        self.fft = lambda x: torch.fft.rfft(x, n=self.nfft, dim=0)  # FFT function
        self.ifft = lambda x: torch.fft.irfft(
            x, n=self.nfft, dim=0
        )  # Inverse FFT function
        # initialize time anti-aliasing envelope function
        self.alias_decay_db = torch.tensor(alias_decay_db)
        self.init_param()
        self.get_gamma()

    def forawrd(self, x):
        Warning("Forward method not implemented. Input is retruned")
        return x

    def init_param(self):
        torch.nn.init.normal_(self.param)

    def get_gamma(self):
        self.gamma = torch.tensor(
            10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)
        )

    def assign_value(self, new_value, indx: tuple = tuple([slice(None)])):
        """
        Assigns new values to the parameters.

        Args:
            new_value (torch.Tensor): New values to be assigned.
            indx (tuple, optional): Index to specify the subset of values to be assigned. Defaults to tuple([slice(None)]).
        """
        assert (
            self.param[indx].shape == new_value.shape
        ), f"New values shape {new_value.shape} is not compatible with the parameter shape {self.param[indx].shape}."

        Warning("Assigning new values. Gradient calculation is disabled.")
        with torch.no_grad():
            self.param[indx].copy_(new_value)
            self.new_value = 1  # flag indicating new values have been assigned


# ============================= GAINS ================================


class Gain(DSP):
    """
    A class representing a set of gains.

    Args:
        size (tuple): The size of the gain parameters. Default is (1, 1).
        map (function): A mapping function applied to the raw parameters. Default is lambda x: x.
        requires_grad (bool): Whether the parameters requires gradients. Default is False.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )
        self.initialize_class()

    def forward(self, x):
        self.check_input_shape(x)
        return self.freq_convolve(x)

    def check_input_shape(self, x):
        """
        Checks if the dimensions of the input tensor x are compatible with the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, n, m).
        """
        if (self.size[-1]) != (x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.size} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        assert len(self.size) == 2, "gains must be 2D."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "mn,bfn...->bfm...", to_complex(self.map(self.param)), x
        )

    def initialize_class(self):
        self.check_param_shape()
        self.get_freq_convolve()


class parallelGain(Gain):
    """
    A class representing a parallel gain module.

    Args:
        size (tuple, optional): The size of the module's input tensor. Defaults to (1,).
        map (function, optional): A mapping function applied to the raw parameter. Defaults to lambda x: x.
        requires_grad (bool, optional): Whether the module's parameter requires gradient computation. Defaults to False.
    """

    def __init__(
        self,
        size: tuple = (1,),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )

    def check_param_shape(self):
        assert len(self.size) == 1, "gains must be 1D, for 2D gains use Gain module."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "n,bfn...->bfn...", to_complex(self.map(self.param)), x
        )


# ============================= FILTERS ================================


class Filter(DSP):
    """
    Filter class representing a FIR filter module.

    Args:
        size (tuple): The size of the filter gains. Default is (1, 1, 1).
        nfft (int): The size of the FFT. Default is 2 ** 11.
        map (function): A mapping function applied to the raw parameters. Default is lambda x: x.
        requires_grad (bool): Whether the filter parameters require gradients. Default is False.
    """

    def __init__(
        self,
        size: tuple = (1, 1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )
        self.initialize_class()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The filtered output signal.
        """
        self.check_input_shape(x)
        if self.requires_grad or self.new_value:
            self.get_freq_response()
            self.new_value = 0
        return self.freq_convolve(x)

    def check_input_shape(self, x):
        """
        Checks if the input dimensions are compatible with the filter parameters.

        Args:
            x (torch.Tensor): The input signal.
        """
        if (int(self.nfft / 2 + 1), self.size[-1]) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        assert (
            len(self.size) == 3
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelFilter module."

    def get_freq_response(self):
        """
        Computes the frequency response of the filter.
        """
        self.ir = self.map(self.param)
        self.decaying_envelope = (self.gamma ** torch.arange(0, self.ir.shape[0])).view(
            -1, *tuple([1 for i in self.ir.shape[1:]])
        )
        self.freq_response = self.fft(self.ir * self.decaying_envelope)

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response, x
        )

    def initialize_class(self):
        self.check_param_shape()
        self.get_freq_response()
        self.get_freq_convolve()


class parallelFilter(Filter):
    """
    Filter class representing a set of parallel FIR filters.

    Args:
        size (tuple): The size of the filter gains. Default is (1,).
        nfft (int): The size of the FFT. Default is 2 ** 11.
        map (function): A mapping function applied to the raw parameters. Default is lambda x: x.
        requires_grad (bool): Whether the filter parameters require gradients. Default is False.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )

    def check_param_shape(self):
        assert (
            len(self.size) == 2
        ), "Filter must be 2D, for 1D filters use Filter module."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response, x
        )


# ============================= DELAYS ================================


class Delay(DSP):
    """
    Delay module that applies a time delay to the input signal.

    Args:
        size (tuple, optional): Size of the delay module. Defaults to (1, 1).
        max_len (int, optional): Maximum length of the delay in samples. Defaults to 2000.
        isint (bool, optional): Flag indicating whether the delay length should be rounded to the nearest integer. Defaults to False.
        nfft (int, optional): Number of FFT points. Defaults to 2 ** 11.
        fs (int, optional): Sampling frequency. Defaults to 48000.
        requires_grad (bool, optional): Flag indicating whether the module parameters require gradients. Defaults to False.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        max_len: int = 2000,
        isint: bool = False,
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        self.fs = fs  # sampling frequency
        self.max_len = max_len  # maximum length of the delay in samples
        self.unit = 100  # unit value used for second2sample conversion
        self.isint = isint  # flag indicating whether the delay length should be rounded to the nearest integer
        super().__init__(
            size=size,
            nfft=nfft,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )
        self.initialize_class()
        if self.alias_decay_db != 0 and (not self.isint):
            print(
                "Warning: Anti time-aliasiang might not work properly under these conditions. We need to debug it"
            )

    def forward(self, x):
        self.check_input_shape(x)
        if self.requires_grad or self.new_value:
            self.get_freq_response()
        return self.freq_convolve(x)

    def init_param(self):
        if self.isint:
            delay_len = torch.randint(1, self.max_len, self.size)
        else:
            delay_len = torch.rand(self.size) * self.max_len
        self.assign_value(self.sample2s(delay_len))
        self.order = (delay_len).max() + 1

    def s2sample(self, delay):
        """
        Converts a delay value from seconds to samples.

        Parameters:
            delay (float): The delay value in seconds.
            unit (int): The unit value used for conversion. Default is 100.
        """
        return delay * self.fs / self.unit

    def sample2s(self, delay):
        """
        Convert delay from sample to seconds.
        Args:
            delay (torch.Tensor): Delay in samples
            unit (int): Unit of the delay. Default is 100.
        """
        return delay / self.fs * self.unit

    def get_freq_response(self):
        m = self.s2sample(self.map(self.param))
        self.freq_response = (self.gamma**m) * torch.exp(
            -1j
            * torch.einsum(
                "fo, omn -> fmn",
                self.omega,
                m.unsqueeze(0),
            )
        )

    def check_input_shape(self, x):
        """
        Checks if the input dimensions are compatible with the filter parameters.

        Args:
            x (torch.Tensor): The input signal.
        """
        if (int(self.nfft / 2 + 1), self.size[-1]) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        assert (
            len(self.size) == 2
        ), "delay must be 2D, for 1D (parallel) delay use parallelDelay module."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response, x
        )

    def initialize_class(self):
        self.check_param_shape()
        if self.requires_grad:
            if self.isint:
                self.map = lambda x: nn.functional.softplus(x).round()
            else:
                self.map = lambda x: nn.functional.softplus(x)
        self.omega = (
            2 * torch.pi * torch.arange(0, self.nfft // 2 + 1) / self.nfft
        ).unsqueeze(1)
        self.get_freq_response()
        self.get_freq_convolve()


class parallelDelay(Delay):
    """
    Parallel delay module that applies a time delay to the input signal.

    Args:
        size (tuple, optional): Size of the delay module. Defaults to (1, ).
        max_len (int, optional): Maximum length of the delay in samples. Defaults to 2000.
        isint (bool, optional): Flag indicating whether the delay length should be rounded to the nearest integer. Defaults to False.
        nfft (int, optional): Number of FFT points. Defaults to 2 ** 11.
        fs (int, optional): Sampling frequency. Defaults to 48000.
        requires_grad (bool, optional): Flag indicating whether the module parameters require gradients. Defaults to False.
    """

    def __init__(
        self,
        size: tuple = (1,),
        max_len=2000,
        isint: bool = False,
        nfft=2**11,
        fs: int = 48000,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(
            size=size,
            max_len=max_len,
            isint=isint,
            nfft=nfft,
            fs=fs,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )

    def check_param_shape(self):
        assert len(self.size) == 1, "delays must be 1D, for 2D delays use Delay module."

    def get_freq_convolve(self):
        self.freq_convolve = lambda x: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response, x
        )

    def get_freq_response(self):
        m = self.s2sample(self.map(self.param))
        self.freq_response = (self.gamma**m) * torch.exp(
            -1j
            * torch.einsum(
                "fo, on -> fn",
                self.omega,
                m.unsqueeze(0),
            )
        )

# ============================= SERIES ================================

class Series(nn.Sequential):
    def __init__(self, *args):
        """
        Series module for cascading multiple DSP modules.

        Args:
            *args: A variable number of DSP modules.
        """

        super().__init__(self.__unpack_modules(modules=args, current_keys=[]))
        
        # Check nfft and alpha values
        self.nfft = self.__check_attribute('nfft')
        self.gamma = self.__check_attribute('gamma')

    def __unpack_modules(self, modules: tuple, current_keys: list) -> OrderedDict:
        """
        Generate an OrderedDict containing the modules given in the input tuple. Give a key to each module.
        If any module in the input tuple is a nn.Sequential (or OrderedDict), it is recognized as nested nn.Sequential (or OrderedDict),
        it is unpacked, and its modules are added to the OrderedDict.
        For each encountered module, key generation follows these rules:
            1 - if the analyzed module is nn.Sequential or OrderedDict, it is unpacked, thus, if it was nested inside
                another nn.Sequential or OrderedDict, the key given to itself is neglected. This until the analyzed module is
                a 'leaf' module (i.e. nn.Module and not nn.Sequential or OrderedDict), then rules 2-5 are applied
            2 - if the module has a unique custom key (e.g. 'my_module'), it is used as is
            3 - if the module has a custom key, but such key was previously used for another module (hence it is not unique). An error is raised
            4 - if the module has no key, a key is generated for it, equal to its position in the series. A warning is raised
            5 - if a custom key is found and can be converted to an integer (e.g. '3'), such key is considered missing
                * rule 4 is applied

        Args:
            modules (tuple): The input modules.
            current_keys (list): The current keys of the already unpacked modules.

        Returns:
            OrderedDict: The unpacked modules.
        """
        # initialize the unpacked modules as empty OrderedDict
        unpacked_modules = OrderedDict()
        # iterate over the modules
        for module in modules:
            if isinstance(module, nn.Sequential):
                # module is a nn.Sequential
                unpacked_modules.update(
                    self.__unpack_modules(
                        (module._modules,), [*current_keys, *unpacked_modules.keys()]
                    )
                )
            elif isinstance(module, OrderedDict):
                for k, v in module.items():
                    if isinstance(v, nn.Sequential):
                        # nested nn.Sequential
                        unpacked_modules.update(
                            self.__unpack_modules(
                                (v._modules,), [*current_keys, *unpacked_modules.keys()]
                            )
                        )
                    elif isinstance(v, OrderedDict):
                        # nested OrderedDict
                        unpacked_modules.update(
                            self.__unpack_modules(
                                (v,), [*current_keys, *unpacked_modules.keys()]
                            )
                        )
                    else:
                        try:
                            int(k)
                            # custom key is convertible to integer, key is overwritten
                            new_key = str(len(unpacked_modules) + len(current_keys))
                            unpacked_modules[new_key] = v
                            if k != new_key:
                                warnings.warn(f"Key {k} is an integer, it will be overwritten.")
                        except: # custom key found
                            if k in current_keys or k in unpacked_modules.keys():
                                # custom key is not unique
                                raise ValueError(f"Key {k} is already present in the Series.")
                            # custome key is unique
                            unpacked_modules[k] = v
            elif isinstance(module, nn.Module):
                # module without key
                unpacked_modules[str(len(unpacked_modules) + len(current_keys))] = (
                    module
                )
            else:
                raise ValueError("Modules must be nn.Module, nn.Sequential, or OrderedDict.")
            
        return unpacked_modules
    
    def __check_attribute(self, attr: str) -> int | float | None:
        """
        Checks if all modules have the same value of the requested attribute.

        Args:
            attr (str): The attribute to check.

        Returns:
            int | float: The attrubute value.

        Raises:
            ValueError: At least one module with incorrect attribute value was found.
        """
        value = None
        # First, store the value of the attribute if found in any of modules.
        for module in self:
            if hasattr(module, attr):
                value = getattr(module, attr)
                break
        if value is None:
            warnings.warn(f"Attribute {attr} not found in any of the modules.")
        # Then, check if all modules have the same value of the attribute. If any module has a different value, raise an error.
        for i,module in enumerate(self):
            if hasattr(module, attr):
                if getattr(module, attr) != value:
                    raise ValueError(f"All modules must have the same {attr} value. Module at index {i} is incoherent with the part of the Series preceeding it.")
        
        return value

# ============================= RECURSION ================================

class Recursion(nn.Module):
    def __init__(self,
                 fF: nn.Module | nn.Sequential | OrderedDict | Series,
                 fB: nn.Module | nn.Sequential | OrderedDict | Series,
                ):
        """
        Recursion module for computing closed-loop transfer function.

        Args:
            Ff (nn.Module): The feedforward path with size (nfft, m, n).
            Fb (nn.Module): The feedback path with size (nfft, n, m).
            alpha (float, optional): The alpha value. Defaults to None.

        References:
            - Closed-loop transfer function: https://en.wikipedia.org/wiki/Closed-loop_transfer_function
        """
        # Prepare the feedforward and feedback paths
        if isinstance(fF, (nn.Sequential, OrderedDict)) and not isinstance(fF, Series):
            fF = Series(fF)
            warnings.warn('Feedforward path has been converted to a Series class instance.')
        if isinstance(fB, (nn.Sequential, OrderedDict)) and not isinstance(fB, Series):
            fB = Series(fB)
            warnings.warn('Feedback path has been converted to a Series class instance.')

        super().__init__()
        self.feedforward = fF
        self.feedback = fB

        # Check nfft and alpha values
        self.nfft = self.__check_attribute('nfft')
        # self.alpha = self.__check_attribute('gamma')

        # Check I/O compatibility
        # self.input_channels, self.output_channels = self.__check_io()


    def forward(self, X):
        """
        Forward pass of the Recursion module.

        Args:
            X (torch.Tensor): The input tensor with shape (batch, nfft, n, ..).

        Returns:
            torch.Tensor: The output tensor with shape (batch, nfft, m, ..).
        """
        B = self.feedforward(X)
       
        # Identity matrix
        if not hasattr(self, "I"):
            self.I = torch.zeros(
                B.shape[1], B.shape[-1], B.shape[-1], dtype=torch.complex64
            )
            for i in range(B.shape[-1]):
                self.I[:, i, i] = 1

        # expand identity matrix to batch size
        expand_dim = [X.shape[0]] + [d for d in self.I.shape]
        I = self.I.expand(tuple(expand_dim))

        HH = self.feedback(I)
        A = I - self.feedforward(HH)
        return torch.linalg.solve(A, B)
    
    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str) -> int | float:
        """
        Checks if feedforward and feedback paths have the same value of the requested attribute

        Args:
            attr (str): The attribute to check.

        Returns:
            int | float: The attribute value.

        Raises:
            ValueError: The two paths have different values of the requested attribute.
        """
        # First, check that both feedforward and feedback paths possess the attribute.
        if getattr(self.feedforward, attr, None) is None:
            raise ValueError(f"The feedforward pass does not possess the attribute {attr}.")
        if getattr(self.feedback, attr, None) is None:
            raise ValueError(f"The feedback pass does not possess the attribute {attr}.")
        # Then, check that the two paths have the same value of the attribute.
        assert getattr(self.feedforward, attr) == getattr(self.feedback, attr), f"Feedforward and feedback paths must have the same {attr} value."

        return getattr(self.feedforward, attr)
    
    def __check_io(self) -> tuple:
        # NOTE: still work in progress
        # NOTE: does not work for SVF
        """
        Check if the feedforward and feedback paths have compatible input/output shapes.
        """
        # Get input channels of both feedforward and feedback
        if isinstance(self.feedforward, Series):
            ff_in_ch = self.feedforward[0].size[-1]
        else:
            ff_in_ch = self.feedforward.size[-1]
        if isinstance(self.feedback, Series):
            fb_in_ch = self.feedback[0].size[-1]
        else:
            fb_in_ch = self.feedback.size[-1]

        # Found out the number of output channels of both feedforward and feedback
        x = to_complex(torch.zeros(1, self.nfft//2+1, ff_in_ch))
        ff_out_ch = self.feedforward(x).shape[-1]
        x = to_complex(torch.zeros(1, self.nfft//2+1, fb_in_ch))
        fb_out_ch = self.feedback(x).shape[-1]

        # Check if the input/output channels are compatible
        assert(ff_out_ch == fb_in_ch), "Feedforward output channels and feedback input channels must have the same."
        assert(fb_out_ch == ff_in_ch), "Feedforward input channels and feedback output channels must have the same."

        return ff_in_ch, ff_out_ch


# =============================== SHELL =================================

class Shell(nn.Module):
    def __init__(self,
                 core: nn.Module | Recursion | nn.Sequential,
                 input_layer: Recursion | Series | nn.Module=nn.Identity(),
                 output_layer: Recursion | Series | nn.Module=nn.Identity(),
                 gamma: float=None,
                ):
        """
        DSP wrapper class. Interface between DSP and loss function.

        Args:
            core (nn.Module | nn.Sequential): DSP.
            input_layer (nn.Module, optional): between Dataset input and DSP. Defaults to Transform(lambda x: x).
            output_layer (nn.Module, optional): between DSP and Dataset target. Defaults to Transform(lambda x: x).
        """

        # Prepare the core, input layer, and output layer
        if isinstance(core, (nn.Sequential, OrderedDict)) and not isinstance(core, Series):
            core = Series(core)
            warnings.warn('Core has been converted to a Series class instance.')
        if isinstance(input_layer, (nn.Sequential, OrderedDict)) and not isinstance(input_layer, Series):
            input_layer = Series(input_layer)
            warnings.warn('Input layer has been converted to a Series class instance.')
        if isinstance(output_layer, (nn.Sequential, OrderedDict)) and not isinstance(output_layer, Series):
            output_layer = Series(output_layer)
            warnings.warn('Output layer has been converted to a Series class instance.')

        nn.Module.__init__(self)        
        self.__input_layer = input_layer
        self.__core = core
        self.__output_layer = output_layer

        # Check model nfft and alpha values
        self.nfft = self.__check_attribute('nfft')
        self.gamma = self.__check_attribute('gamma', gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the input layer, the core, and the output layer. Keeps the three components separated.

        Args:
            x (torch.Tensor): input Tensor

        Returns:
            torch.Tensor: output Tensor
        """
        x = self.__input_layer(x)
        x = self.__core(x)
        x = self.__output_layer(x)
        return x
    
    # ---------------------- Get and set methods ----------------------
    def get_inputLayer(self) -> nn.Module | nn.Sequential:
        return self.__input_layer
    
    def set_inputLayer(self, input_layer: nn.Module=None) -> None:
        self.__input_layer = input_layer

    def get_outputLayer(self) -> nn.Module | nn.Sequential:
        return self.__output_layer
    
    def set_outputLayer(self, output_layer: nn.Module=None) -> None:
        self.__output_layer = output_layer

    def get_core(self) -> nn.Module | nn.Sequential:
        return self.__core
    
    def set_core(self, core: nn.Module) -> None:
        self.__core = core

    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str, new_value: float=None) -> int | float:
        """
        Check if all the modules in model, input layer, and output layer have the same value for the requested attribute.
        If a new value is provided, it is assigned to all modules.

        Returns:
            int: The attribute value.

        Raises:
            ValueError: The core component does not possess the requested attribute.
            AssertionError: Core, input layer, and output layer must have the same value of the requested attribute
        """

        # Check that core, input layer, and output layer all possess the nfft attribute.
        if getattr(self.__core, attr, None) is None:
            raise ValueError(f"The core does not possess the attribute {attr}.")
        if getattr(self.__input_layer, attr, None) is not None:
            assert getattr(self.__core, attr) == getattr(self.__input_layer, attr), f"Core and input layer must have the same {attr} value."
        if getattr(self.__output_layer, attr, None) is not None:
            assert getattr(self.__core, attr) == getattr(self.__output_layer, attr), f"Core and output layer must have the same {attr} value."

        # Get current value
        current_value = getattr(self.__core, attr)

        # If new value is given, assign it to all modules that have a different value from it
        if new_value is not None and new_value != current_value:
            assert isinstance(new_value, (int, float)), "New value must be a int or float."
            self.__change_attr_value('core', self.__core, attr, new_value)
            if getattr(self.__input_layer, attr, None) is not None:
                self.__change_attr_value('input layer', self.__input_layer, attr, new_value)
            if getattr(self.__output_layer, attr, None) is not None:
                self.__change_attr_value('output layer', self.__output_layer, attr, new_value)
        
        return new_value if new_value is not None else current_value

    def __change_attr_value(self, layer_name: str, layer: nn.Module | Recursion | Series, attr: str, new_value: float) -> None:
        """
        Change the attribute value of the provided layer to the requested value.

        Args:
            layer (nn.Module | Recursion | Series): provided layer
            gamma (float): requested attribute value
        """
        warnings.warn(f"As of now, this method change only the given attribute, not the attributes and values that depends on it.")

        if isinstance(layer, Series):
            for module in layer:
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, new_value)
        elif isinstance(layer, Recursion):
            if isinstance(layer.feedforward, Series):
                for module in layer.feedforward:
                    if getattr(module, attr, None) is not None:
                        setattr(module, attr, new_value)
            else:
                if getattr(layer.feedforward, attr, None) is not None:
                    setattr(layer.feedforward, attr, new_value)
            if isinstance(layer.feedback, Series):
                for module in layer.feedback:
                    if getattr(module, attr, None) is not None:
                        setattr(module, attr, new_value)
            else:
                if getattr(layer.feedback, attr, None) is not None:
                    setattr(layer.feedback, attr, new_value)
                
        setattr(layer, attr, new_value)
        warnings.warn(f"The value of the attribute {attr} in the {layer_name} has been modified to {new_value}.")

    # ---------------------- Responses methods ----------------------
    def get_time_response(self, fs: int=48000, interior: bool=False) -> torch.Tensor:

        """
        Generate the impulse response of the DSP.

        Args:
            nfft (int, optional): Number of frequency points. Defaults to 2**11.
            fs (int, optional): Sampling frequency. Defaults to 48000.
            ir_len (int, optional): Number of samples of the returned impulse response. Defaults to 96000.
            interior (bool, optional): If False, return the input-to-output impulse responses of the DSP.
                                       If True, return the input-free impulse responses of the DSP.
                                       Defaults to False.
            
        NOTE: Definition of 'input-to-output' and 'input-free'
            Let A \in R^{t x n x m} be a time filter matrix. If x \in R^{t x m} is an m-dimensional time signal having
            a unit impulse at time t=0 for each element along m. Let I \in R^{t x m x m} be an diagonal matrix across
            second and third dimension, with unit impulse at time t=0 for each element along such diagonal.
            If * represent the signal-wise matrix convolution operator, then:
                - y = A * x is the 'input-to-output' impulse response of A.
                - A * I is the 'input-free' impulse response of A.

        Returns:
            torch.Tensor: generated DSP impulse response.
        """

        # get parameters from the model
        if isinstance(self.__core, nn.Sequential):
            nfft = self.__core[0].nfft
            input_channels = self.__core[0].size[-1]
        else:
            nfft = self.__core.nfft
            input_channels = self.__core.size[-1]

        # save input/output layers
        input_save = self.get_inputLayer()
        output_save = self.get_outputLayer()

        # update input/output layers
        self.set_inputLayer(FFT(nfft))
        self.set_outputLayer(iFFT(nfft))

        # generate input signal
        x = signal_gallery(
            batch_size=1, n_samples=nfft, n=input_channels, signal_type="impulse", fs=fs
        )
        if interior:
            x = x.diag_embed()

        # generate impulse response
        with torch.no_grad():
            y = self.forward(x)

        # restore input/output layers
        self.set_inputLayer(input_save)
        self.set_outputLayer(output_save)

        return y
    
    def get_freq_response(self, fs: int=48000, interior: bool=False) -> torch.Tensor:

        """
        Generate the frequency response of the DSP.

        Args:
            nfft (int, optional): Number of frequency points. Defaults to 2**11.
            fs (int, optional): Sampling frequency. Defaults to 48000.
            interior (bool, optional): If False, return the input-to-output frequency responses of the DSP.
                                       If True, return the input-free frequency responses of the DSP.
                                       Defaults to False.
            
        NOTE: Definition of 'input-to-output' and 'input-free'
            Let A \in R^{f x n x m} be a frequency filter matrix. If x \in R^{f x m} is an m-dimensional signal having
            a unit impulse at time t=0 spectrum for each element along m. Let I \in R^{f x m x m} be an diagonal matrix across
            second and third dimension, with unit impulse at time t=0 spectra for each element along such diagonal.
            If * represent the signal-wise matrix product operator, then:
                - y = A * x is the 'input-to-output' frequency response of A.
                - A * I is the 'input-free' frequency response of A.

        Returns:
            torch.Tensor: generated DSP frequency response.
        """
        # get parameters from the model
        if isinstance(self.__core, nn.Sequential):
            nfft = self.__core[0].nfft
            input_channels = self.__core[0].size[-1]
        else:
            nfft = self.__core.nfft
            input_channels = self.__core.size[-1]

        # save input/output layers
        input_save = self.get_inputLayer()
        output_save = self.get_outputLayer()

        # update input/output layers
        self.set_inputLayer(FFT(nfft))
        self.set_outputLayer(nn.Identity())

        # generate input signal
        x = signal_gallery(
            batch_size=1, n_samples=nfft, n=input_channels, signal_type="impulse", fs=fs
        )
        if interior:
            x = x.diag_embed()

        # generate frequency response
        with torch.no_grad():
            y = self.forward(x)

        # restore input/output layers
        self.set_inputLayer(input_save)
        self.set_outputLayer(output_save)

        return y
