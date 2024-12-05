import torch
import torch.nn as nn
import torch.nn.functional as F
from flamo.utils import to_complex
from flamo.functional import (
    skew_matrix, 
    lowpass_filter, 
    highpass_filter, 
    bandpass_filter,
    rad2hertz )
from flamo.auxiliary.eq import (
    eq_freqs,
    geq)
from flamo.auxiliary.scattering import (
    ScatteringMapping)
# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    r"""
    Base class for all transformations. 

    The transformation is a callable, e.g., :class:`lambda` expression, function, :class:`nn.Module`. 

        **Args**:
            - transform (callable): The transformation function to be applied to the input. Default: lambda x: x  
            - device (str): The device of the constructed tensor, if any. Default: None.
        **Attributes**:
            - transform (callable): The transformation function to be applied to the input.  
            - device (str): The device of constructed tensors.
        **Methods**:
            - forward(x): Applies the transformation function to the input.  

        Examples::

            >>> pow2 = Transform(lambda x: x**2)
            >>> input = torch.tensor([1, 2, 3])
            >>> pow2(input)
            tensor([1, 4, 9])
    """
    def __init__(self, transform: callable = lambda x: x, device=None):
        super().__init__()
        self.transform = transform
        self.device = device

    def forward(self, x):
        r"""
        Applies the transformation to the input tensor.

            **Args**:
                x (Tensor): The input tensor.

            **Returns**:
                Tensor: The transformed tensor.
        """
        return self.transform(x)


class FFT(Transform):
    r"""
    Real Fast Fourier Transform (FFT) class.
    
    The :class:`FFT` class is an instance of the :class:`Transform` class. The transformation function is the :func:`torch.fft.rfft` function.
    Computes the one dimensional Fourier transform of real-valued input. The input is interpreted as a real-valued signal in time domain. The output contains only the positive frequencies below the Nyquist frequency. 
    
        **Args**:
            - nfft (int): The number of points to compute the FFT.
            - norm (str): The normalization mode for the FFT.  

        **Attributes**:
            - nfft (int): The number of points to compute the FFT.
            - norm (str): The normalization mode for the FFT.
        **Methods**:
            - foward(x): Apply the FFT to the input tensor x and return the one sided FFT.

    For details on the FFT function, see `torch.fft.rfft documentation <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_.
    """

    def __init__(self, nfft=2**11, norm="backward"):
        self.nfft = nfft
        self.norm = norm
        transform = lambda x: torch.fft.rfft(x, n=self.nfft, dim=1, norm=self.norm)
        super().__init__(transform=transform)


class iFFT(Transform):
    r"""
    Inverse Fast Fourier Transform (iFFT) class.

    The :class:`iFFT` class is an instance of the :class:`Transform` class. The transformation function is the :func:`torch.fft.irfft` function.
    Computes the inverse of the Fourier transform of a real-valued tensor. The input is interpreted as a one-sided Hermitian signal in the Fourier domain. The output is a real-valued signal in the time domain.
    
        **Args**:
            - nfft (int): The size of the FFT. Default: 2**11.
            - norm (str): The normalization mode. Default: "backward".
        **Attributes**:
            - nfft (int): The size of the FFT.
            - norm (str): The normalization mode.
        **Methods**:
            - foward(x): Apply the inverse FFT to the input tensor x and returns its corresponding real valued tensor.

    For details on the inverse FFT function, see `torch.fft.irfft documentation <https://pytorch.org/docs/stable/generated/torch.fft.irfft.html>`_.
    """

    def __init__(self, nfft=2**11, norm="backward"):
        self.nfft = nfft
        self.norm = norm
        transform = lambda x: torch.fft.irfft(x, n=self.nfft, dim=1, norm=self.norm)
        super().__init__(transform=transform)

class FFTAntiAlias(Transform):
    r"""
    Real Fast Fourier Transform (FFT) class with time-aliasing mitigation enabled.
    Inherits from the :class:`Transform` class.

    Computes the one dimensional Fourier transform of real-valued input after mupltiplying it by an 
    by an exponentially decaying envelope to mitigate time aliasing.
    The input is interpreted as a real-valued signal in time domain. 
    The output contains only the positive frequencies below the Nyquist frequency. 
    
        **Args**:
            - nfft (int): The number of points to compute the FFT.
            - norm (str): The normalization mode for the FFT.  
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. Default: 0.0.
            - device (str): The device of the constructed tensors. Default: None.
        **Attributes**:
            - nfft (int): The number of points to compute the FFT.
            - norm (str): The normalization mode for the FFT.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            - device (str): The device of constructed tensors.
        **Methods**:
            - foward(x): Apply the FFT to the input tensor x and return the one sided FFT.

    For details on the FFT function, see `torch.fft.rfft documentation <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_.
    """
    def __init__(self, nfft=2**11, norm="backward", alias_decay_db=0.0, device=None):
        self.nfft = nfft
        self.norm = norm
        self.device = device
        
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=self.device)) / (self.nfft) / 20)
        self.alias_envelope = (gamma ** torch.arange(0, -self.nfft, -1, device=self.device))
        
        fft = lambda x: torch.fft.rfft(x, n=self.nfft, dim=1, norm=self.norm)
        transform = lambda x: fft(torch.einsum('btm, t->btm', x, self.alias_envelope))
        super().__init__(transform=transform)

class iFFTAntiAlias(Transform):
    r"""
    Inverse Fast Fourier Transform (iFFT) class with time-aliasing mitigation enabled.
    Inherits from the :class:`Transform` class.

    Computes the inverse of the Fourier transform of a real-valued tensor to which anti time aliasing has been applied. 
    The input is interpreted as a one-sided Hermitian signal in the Fourier domain. 
    The output is a real-valued signal in the time domain. The output is multiplied 
    by an exponentially decaying envelope to mitigate time aliasing.

        **Args**:
            - nfft (int): The size of the FFT. Default: 2**11.
            - norm (str): The normalization mode. Default: "backward".
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. Default: 0.0.
        **Attributes**:
            - nfft (int): The size of the FFT.
            - norm (str): The normalization mode.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
        **Methods**:
            - foward(x): Apply the inverse FFT to the input tensor x and returns its corresponding real valued tensor multiplied by an exponentially decaying envelope.

    For details on the inverse FFT function, see `torch.fft.irfft documentation <https://pytorch.org/docs/stable/generated/torch.fft.irfft.html>`_.
    """

    def __init__(self, nfft=2**11, norm="backward", alias_decay_db=0.0, device=None):
        self.nfft = nfft
        self.norm = norm
        self.device = device
        
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=self.device)) / (self.nfft) / 20)
        self.alias_envelope = (gamma ** torch.arange(0, -self.nfft, -1, device=self.device))
        
        ifft =  lambda x: torch.fft.irfft(x, n=self.nfft, dim=1, norm=self.norm)
        transform = lambda x: torch.einsum('btm, t->btm', ifft(x), self.alias_envelope)
        super().__init__(transform=transform)


# ============================= CORE ================================


class DSP(nn.Module):
    r"""
    Processor core module consisting of learnable parameters representing a Linear Time-Invariant (LTI) system, which is then convolved with the input signal.
       
    The parameters are stored in :attr:`param` tensor whose values at initialization 
    are drawn from the normal distribution :math:`\mathcal{N}(0, 1)` and can be 
    modified using the :meth:`assign_value` method. 

    The anti aliasing envelope is computed using the :meth:`get_gamma` method from 
    the :attr:`alias_decay_db` attribute which determines the decay in dB reached 
    by the exponentially decaying envelope :math:`\gamma(n)` after :attr:`nfft` samples. 
    The envelope :math:`\gamma(n)` is then applied to the time domain signal before computing the FFT

        **Args**:
            - size (tuple): The shape of the parameters before mapping.
            - nfft (int, optional): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - map (function, optional): The mapping function applied to the raw parameters. Default: lambda x: x.
            - requires_grad (bool, optional): Whether the parameters require gradients. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str): The device of the constructed tensor, if any. Default: None.

        **Attributes**:
            - size (tuple): The shape of the parameters.
            - nfft (int): The number of FFT points required to compute the frequency response.
            - map (function): The mapping function applied to the raw parameters.
            - requires_grad (bool): Whether the parameters require gradients.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            - param (nn.Parameter): The parameters of the DSP module.
            - fft (function): The FFT function. Calls the :func:`torch.fft.rfft` function.
            - ifft (function): The Inverse FFT function. Calls the :func:`torch.fft.irfft`.
            - gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.
            - new_value (int): Flag indicating if new values have been assigned.
            - device (str): The device of the constructed tensors.
            
        **Methods**:
            - forward(x): Applies the processor core module to the input tensor x by multiplication.
            - init_param(): Initializes the parameters of the DSP module.
            - get_gamma(): Computes the gamma value used for time anti-aliasing envelope.
            - assign_value(new_value, indx): Assigns new values to the parameters.
    """

    def __init__(
        self,
        size: tuple,
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__()
        assert isinstance(size, tuple), "Size must be a tuple."
        self.size = size 
        self.nfft = nfft
        self.map = map
        self.new_value = 0  # flag indicating if new values have been assigned
        self.requires_grad = requires_grad 
        self.device = device
        self.param = nn.Parameter(torch.empty(self.size, device=self.device), requires_grad=self.requires_grad)
        self.fft = lambda x: torch.fft.rfft(x, n=self.nfft, dim=0)  
        self.ifft = lambda x: torch.fft.irfft(x, n=self.nfft, dim=0)  
        # initialize time anti-aliasing envelope function
        self.alias_decay_db = torch.tensor(alias_decay_db, device=self.device)
        self.init_param()
        self.get_gamma()

    def forward(self, x, **kwargs):
        r"""
        Forward method.

        .. warning::
            Forward method not implemented. Input is returned.

        """
        Warning("Forward method not implemented. Input is retruned")
        return x

    def init_param(self):
        r"""
        Initializes the parameters of the model using a normal distribution :math:`\mathcal{N}(0, 1)`.
        It uses the :func:`torch.nn.init.normal_` function to set the values of :attr:`param`.
        """
        torch.nn.init.normal_(self.param)

    def get_gamma(self):
        r"""
        Calculate the gamma value based on the alias decay in dB and the number of FFT points.
        The gamma value is computed as follows and saved in the attribute :attr:`gamma`:

        .. math::

            \gamma = 10^{\frac{-|\alpha_{\text{dB}}|}{20 \cdot \texttt{nfft}}}\; \text{and}\; \gamma(n) = \gamma^{n}

        where :math:`\alpha_{\textrm{dB}}` is the alias decay in dB, :math:`\texttt{nfft}` is the number of FFT points, 
        and :math:`n` is the descrete time index :math:`0\\leq n < N`, where N is the length of the signal.
        """

        self.gamma = 10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)

    def assign_value(self, new_value, indx: tuple = tuple([slice(None)])):
        """
        Assigns new values to the parameters.

        **Args**:
            - new_value (torch.Tensor): New values to be assigned.
            - indx (tuple, optional): Index to specify the subset of values to be assigned. Default: tuple([slice(None)]).

        .. warning::
            the gradient calulcation is disable when assigning new values to :attr:`param`.
        
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
    r"""
    A class representing a set of gains. Inherits from :class:`DSP`.
    The input tensor is expected to be a complex-valued tensor representing the 
    frequency response of the input signal. The input tensor is then multiplied
    with the gain parameters to produce the output tensor. 

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - param: :math:`(N_{out}, N_{in})`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, and :math:`N_{out}` is the number of output channels.
    Ellipsis :math:`(...)` represents additional dimensions.

        **Args**:
            - size (tuple): The size of the gain parameters. Default: (1, 1).
            - nfft (int): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - map (function): A mapping function applied to the raw parameters. Default: lambda x: x.
            - requires_grad (bool): Whether the parameters requires gradients. Default: False.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - size (tuple): The size of the gain parameters.
            - nfft (int): The number of FFT points required to compute the frequency response.
            - map (function): A mapping function applied to the raw parameters.
            - requires_grad (bool): Whether the parameters requires gradients.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            - param (nn.Parameter): The parameters of the Gain module.
            - fft (function): The FFT function. Calls the torch.fft.rfft function.
            - ifft (function): The Inverse FFT function. Calls the torch.fft.irfft.
            - gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.
            - new_value (int): Flag indicating if new values have been assigned.
            - device (str): The device of the constructed tensors.
            
        **Methods**:
            - forward(x): Applies the Gain module to the input tensor x by multiplication.
            - check_input_shape(x): Checks if the dimensions of the input tensor x are compatible with the module.
            - check_param_shape(): Checks if the shape of the gain parameters is valid.
            - get_freq_convolve(): Computes the frequency convolution function.
            - initialize_class(): Initializes the Gain module.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )
        self.initialize_class()

    def forward(self, x, ext_param=None):
        r"""
        Applies the Gain module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
                ext_paran (torch.Tensor, optional): Parameter values from outer modules. Default: None.
            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else: 
            # log the parameters that are being passed 
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)

    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (self.input_channels) != (x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.size} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the gain parameters is valid.
        """
        assert len(self.size) == 2, "gains must be 2D. For 1D (parallel) gains use parallelGain module."

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "mn,bfn...->bfm...", to_complex(self.map(param)), x
        )

    def initialize_class(self):
        r"""
        Initializes the Gain module.

        This method checks the shape of the gain parameters and computes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_io()
        self.get_freq_convolve()

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]


class parallelGain(Gain):
    """
    Parallel counterpart of the :class:`Gain` class.
    For information about **attributes** and **methods** see :class:`Gain`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    and :math:`N` is the number of input channels.
    Ellipsis :math:`(...)` represents additional dimensions.
    """

    def __init__(
        self,
        size: tuple = (1,),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad=False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        r"""
        Checks if the shape of the gain parameters is valid.
        """
        assert len(self.size) == 1, "gains must be 1D, for 2D gains use Gain module."

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "n,bfn...->bfn...", to_complex(self.map(param)), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]

# ============================= MATRICES ================================


class Matrix(Gain):
    """
    A class representing a matrix. inherits from :class:`Gain`.

        **Args**:
            - size (tuple, optional): The size of the matrix. Default: (1, 1).
            - nfft (int, optional): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - map (function, optional): The mapping function to apply to the raw matrix elements. Default: lambda x: x.
            - matrix_type (str, optional): The type of matrix to generate. Default: "random".
            - requires_grad (bool, optional): Whether the matrix requires gradient computation. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str, optional): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - size (tuple): The size of the matrix.
            - nfft (int): The number of FFT points required to compute the frequency response.
            - map (function): The mapping function to apply to the raw matrix elements.
            - matrix_type (str): The type of matrix to generate.
            - requires_grad (bool): Whether the matrix requires gradient computation.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            - param (nn.Parameter): The parameters of the Matrix module.
            - fft (function): The FFT function. Calls the torch.fft.rfft function.
            - ifft (function): The Inverse FFT function. Calls the torch.fft.irfft.
            - gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.
            - device (str): The device of the constructed tensors.
            
        **Methods**:
            - forward(x): Applies the Matrix module to the input tensor x by multiplication.
            - check_input_shape(x): Checks if the dimensions of the input tensor x are compatible with the module.
            - check_param_shape(): Checks if the shape of the matrix parameters is valid.
            - get_freq_convolve(): Computes the frequency convolution function.
            - initialize_class(): Initializes the Matrix module.
            - matrix_gallery(): Generates the matrix based on the specified matrix type.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        matrix_type: str = "random",
        requires_grad=False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        self.matrix_type = matrix_type
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def matrix_gallery(self):
        r"""
        Generates the matrix based on the specified matrix type.
        The :attr:`map` attribute will be overwritten based on the matrix type.
        """
        Warning(
            f"you asked for {self.matrix_type} matrix type, map will be overwritten"
        )
        match self.matrix_type:
            case "random":
                self.map = lambda x: x
            case "orthogonal":
                assert (
                    self.size[0] == self.size[1]
                ), "Matrix must be square to be orthogonal"
                self.map = lambda x: torch.matrix_exp(skew_matrix(x))

    def initialize_class(self):
        r"""
        Initializes the Matrix module.

        This method checks the shape of the matrix parameters, sets the matrix type, generates the matrix, and computes the frequency convolution function.

        """
        self.check_param_shape()
        self.get_io()
        self.matrix_gallery()
        self.get_freq_convolve()


class HouseholderMatrix(Gain):
    r"""
    HouseholderMatrix is a class that generates a Householder matrix for signal processing.

        **Args**:
            size (tuple, optional): Size of the matrix. Must be a square matrix. Defaults to (1, 1).
            nfft (int, optional): Number of FFT points. Defaults to 2**11.
            requires_grad (bool, optional): If True, gradients will be computed for the parameters. Defaults to False.
            alias_decay_db (float, optional): Alias decay in decibels. Defaults to 0.0.
            device (optional): Device on which to perform computations. Defaults to None.
    """
    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        requires_grad=False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        assert size[0] == size[1], "Matrix must be square"
        size = (size[0], 1)
        map = lambda x: to_complex(x) / torch.norm(x, dim=0, keepdim=True)
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def forward(self, x, ext_param=None):
        self.check_input_shape(x)
        if ext_param is None:
            u = self.map(self.param)
        else: 
            # log the parameters that are being passed 
            with torch.no_grad():
                self.assign_value(ext_param)
            # generate householder matrix from unitary vector
            u = self.map(ext_param)
        uTx = torch.einsum("mn,bfn...->bfm...", u.transpose(1, 0), x)
        uuTx = torch.einsum("nm,bfm...->bfn...", u, uTx)
        return x - 2 * uuTx

    def check_input_shape(self, x):
        if (self.size[0]) != (x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.size} not compatible with input signal of shape = ({x.shape})."
            )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[0]
        self.output_channels = self.size[0]


# ============================= FILTERS ================================


class Filter(DSP):
    r"""
    A class representing a set of FIR filters. Inherits from :class:`DSP`.
    The input tensor is expected to be a complex-valued tensor representing the
    frequency response of the input signal. The input tensor is then convolved in
    frequency domain with the filter frequency responses to produce the output tensor.
    The filter parameters correspond to the filter impulse responses in case the mapping
    function is map=lambda x: x.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - param: :math:`(N_{taps}, N_{out}, N_{in})`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, :math:`N_{out}` is the number of output channels,
    and :math:`N_{taps}` is the number of filter parameters per input-output channel pair.
    Ellipsis :math:`(...)` represents additional dimensions.

        **Args**:
            - size (tuple): The size of the filter parameters. Default: (1, 1, 1).
            - nfft (int): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - map (function): A mapping function applied to the raw parameters. Default: lambda x: x.
            - requires_grad (bool): Whether the filter parameters require gradients. Default: False.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - size (tuple): The size of the filter parameters.
            - nfft (int): The number of FFT points required to compute the frequency response.
            - map (function): A mapping function applied to the raw parameters.
            - requires_grad (bool): Whether the filter parameters require gradients.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            - param (nn.Parameter): The parameters of the Filter module.
            - fft (function): The FFT function. Calls the torch.fft.rfft function.
            - ifft (function): The Inverse FFT function. Calls the torch.fft.irfft.
            - gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.
            - freq_response (torch.Tensor): The frequency response of the filter.
            - freq_convolve (function): The frequency convolution function.
            - device (str): The device of the constructed tensors.

        **Methods**:
            - forward(x): Applies the Filter module to the input tensor x by convolution in frequency domain.
            - check_input_shape(x): Checks if the dimensions of the input tensor x are compatible with the module.
            - check_param_shape(): Checks if the shape of the filter parameters is valid.
            - get_freq_response(): Computes the frequency response of the filter.
            - get_freq_convolve(): Computes the frequency convolution function.
            - initialize_class(): Initializes the Filter module.
    """

    def __init__(
        self,
        size: tuple = (1, 1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )
        self.initialize_class()

    def forward(self, x, ext_param=None):
        r"""
        Applies the Filter module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
                ext_paran (torch.Tensor, optional): Parameter values from outer modules. Default: None.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else: 
            # log the parameters that are being passed 
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)
    
    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert len(self.size) == 3, "Filter must be 3D, for 2D (parallel) filters use ParallelFilter module."

    def get_freq_response(self):
        r"""
        Computes the frequency response of the filter.

        The mapping function is applied to the filter parameters to obtain the filter impulse responses.
        Then, the time anti-aliasing envelope is computed and applied to the impulse responses. Finally,
        the frequency response is obtained by computing the FFT of the filter impulse responses.
        """
        self.ir = lambda x: self.map(x)  
        self.freq_response = lambda param: self.fft(
            self.ir(param) * (self.gamma ** torch.arange(0, self.ir(param).shape[0], device=self.device)).view(
            -1, *tuple([1 for i in self.map(param).shape[1:]])
            )
        )

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )

    def initialize_class(self):
        r"""
        Initializes the Gain module.

        This method checks the shape of the gain parameters, computes the frequency response of the filter, 
        and computes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_io()
        self.get_freq_response()
        self.get_freq_convolve()

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]

class ScatteringMatrix(Filter):

    def __init__(
        self,
        size: tuple = (1, 1, 1),
        nfft: int = 2**11,
        sparsity: int = 3, 
        gain_per_sample: float = 0.9999, 
        pulse_size: int = 1, 
        m_L: torch.tensor = None, 
        m_R: torch.tensor = None, 
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device=None
    ):  
        self.sparsity = sparsity
        self.gain_per_sample = gain_per_sample
        self.pulse_size = pulse_size
        self.m_L = m_L
        self.m_R = m_R
        map = lambda x: torch.matrix_exp(skew_matrix(x))
        assert size[1] == size[2], "Matrix must be square"
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )


    def get_freq_response(self):
        r"""
        Computes the frequency response of the filter.

        The mapping function is applied to the filter parameters to obtain the filter impulse responses.
        Then, the time anti-aliasing envelope is computed and applied to the impulse responses. Finally,
        the frequency response is obtained by computing the FFT of the filter impulse responses.
        """
        L = (sum(self.map_filter.shifts).max()+1).item() + self.m_L.max().item() + self.m_R.max().item()   
        self.freq_response = lambda param: self.fft(
            self.map_filter(self.map(param)) * (self.gamma ** torch.arange(0, L, device=self.device)).view(
            -1, *tuple([1 for i in self.size[1:]])
            )
        )

    def initialize_class(self):
        r"""
        Initializes the Gain module.

        This method checks the shape of the gain parameters, computes the frequency response of the filter, 
        and computes the frequency convolution function.
        """
        self.map_filter = ScatteringMapping(
            self.size[-1],
            n_stages=self.size[0]-1,
            sparsity=self.sparsity,
            gain_per_sample=self.gain_per_sample,
            pulse_size=self.pulse_size,
            m_L=self.m_L,
            m_R=self.m_R,
            device=self.device
        )
        self.check_param_shape()
        self.get_io()
        self.get_freq_response()
        self.get_freq_convolve()

class parallelFilter(Filter):
    """
    Parallel counterpart of the :class:`Filter` class.
    For information about **attributes** and **methods** see :class:`Filter`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(N_{taps}, N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N` is the number of input channels, and :math:`N_{taps}` is the number of
    filter parameters per input-output channel pair.
    Ellipsis :math:`(...)` represents additional dimensions.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        nfft: int = 2**11,
        map=lambda x: x,
        requires_grad: bool=False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        r"""
        Checks if the shape of the filter parameters is valid.
        """
        assert len(self.size) == 2, "Filter must be 1D, for 2D filters use Filter module."

    def get_freq_convolve(self):#NOTE: is it correct to say that there is an input argument in this case?
                                #      Same, is it correct to say that it returns something?
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]

class Biquad(Filter):
    r"""
    Biquad filter class. Inherits from the :class:`Filter` class.
    It supports class lowpass, highpass, and bandpass filters using `RBJ cookbook 
    formulas <https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt>`_, 
    which map the cut-off frequency :math:`f_{c}` and gain  :math:`g` parameters 
    to the :math:`\mathbf{b}` and :math:`\mathbf{a}` coefficients. 
    The transfer function of the filter is given by
    
    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

    The mapping from learnable parameters :math:`\mathbf{f_c}` and :math:`\mathbf{g}` are defined by :meth:`flamo.functional.lowpass_filter`, :meth:`flamo.functional.highpass_filter`, :meth:`flamo.functional.bandpass_filter`.

    Shape:
        - input: :math:`(B, M, N_{\text{in}}, ...)`
        - param: :math:`(K, P, N_{\text{out}}, N_{\text{in}})`  # TODO change so (P, K, ..)
        - freq_response: :math:`(M, N_{\text{out}}, N_{\text{in}})`
        - output: :math:`(B, M, N_{\text{out}}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    and :math:`N_{\text{in}}` is the number of input channels, and :math:`N_{\text{out}}` is the number of output channels.
    The :attr:'param' attribute represent the biquad filter coefficents. The first dimension of the :attr:'param' tensor corresponds to the number of filters :math:`K`, the second dimension corresponds to the number of filter parameters :math:`P`.
    Ellipsis :math:`(...)` represents additional dimensions (not tested).

        **Args**:
            - size (tuple, optional): Size of the filter. Default: (1, 1).
            - n_sections (int, optional): Number of filters. Default: 1.
            - filter_type (str, optional): Type of the filter. Must be one of "lowpass", "highpass", or "bandpass". Default: "lowpass".
            - nfft (int, optional): Number of points for FFT. Default: 2048.
            - fs (int, optional): Sampling frequency. Default: 48000.
            - requires_grad (bool, optional): Whether the filter parameters require gradient computation. Default: True.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str, optional): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - filter_type (str): Type of the filter.
            - fs (int): Sampling frequency.
            - freq_response (torch.Tensor): Frequency response of the filter.
            - param (nn.Parameter): Parameters of the Biquad filter.

        **Methods**:
            - get_size(): Get the size of the filter based on the filter type.
            - get_freq_response(): Compute the frequency response of the filter.
            - get_map(): Get the mapping function for parameter values based on the filter type.
            - init_param(): Initialize the filter parameters.
            - check_param_shape(): Check the shape of the filter parameters.
            - initialize_class(): Initialize the Biquad class.
    """
    def __init__(
        self,
        size: tuple = (1, 1),
        n_sections: int = 1,
        filter_type: str = "lowpass",
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad: bool=True,
        alias_decay_db: float = 0.0,
        device=None
    ):
        assert filter_type in ["lowpass", "highpass", "bandpass"], "Invalid filter type"
        self.filter_type = filter_type
        self.fs = fs
        self.device = device
        self.get_map()
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=self.device)) / (nfft) / 20)
        self.alias_envelope_dcy = (gamma ** torch.arange(0, 3, 1, device=self.device))
        super().__init__(
            size=(n_sections, *self.get_size(), *size),
            nfft=nfft,
            map=self.map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def get_size(self):
        r"""
        Get the leading dimensions of the parameters based on the filter type.
        These coincide with

            - 2 for lowpass and highpass filters (:math:`f_\textrm{c}`, gain)
            - 3 for bandpass filters (:math:`f_\textrm{c1}`, :math:`f_\textrm{c2}`, gain)

        **Returns**:
            - tuple: leading dimensions of the parameters.
        """
        match self.filter_type:
            case "lowpass":
                return (2,)  # fc, gain
            case "highpass":
                return (2,)  # fc, gain
            case "bandpass":
                return (3,)  # fc1, fc2, gain

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the specified filter type.
        It calls the :func:`flamo.functional.lowpass_filter`, :func:`flamo.functional.highpass_filter`, or :func:`flamo.functional.bandpass_filter` functions based on the filter type.

        **Args**:
            - param (torch.Tensor): A tensor containing the filter parameters. 
            The shape of the tensor should be (batch_size, num_params, height, width). 
            The parameters are interpreted differently based on the filter type:
            - For "lowpass" and "highpass" filters, param[:, 0, :, :] represents 
            the cutoff frequency and param[:, 1, :, :] represents the gain.
            - For "bandpass" filters, param[:, 0, :, :] represents the lower 
            cutoff frequency, param[:, 1, :, :] represents the upper cutoff 
            frequency, and param[:, 2, :, :] represents the gain.
        **Returns**:       
            - H (torch.Tensor): The frequency response of the filter.
            - B (torch.Tensor): The Fourier transformed numerator polynomial coefficients.
            - A (torch.Tensor): The Fourier transformed denominator polynomial coefficients.


        The method uses the filter type specified in `self.filter_type` to determine 
        which filter to apply. It applies an aliasing envelope decay to the filter coefficients.
        Zero values in the denominator polynomial coefficients are replaced with 
        a small constant to avoid division by zero.
        """
        match self.filter_type:
            case "lowpass":
                b, a = lowpass_filter(fc=rad2hertz(param[:, 0, :, :]*torch.pi, fs=self.fs), gain=param[:, 1, :, :], fs=self.fs, device=self.device)
            case "highpass":
                b, a = highpass_filter(fc=rad2hertz(param[:, 0, :, :]*torch.pi, fs=self.fs), gain=param[:, 1, :, :], fs=self.fs, device=self.device)
            case "bandpass":
                b, a = bandpass_filter(
                    fc1=rad2hertz(param[:, 0, :, :]*torch.pi, fs=self.fs), fc2=rad2hertz(param[:, 1, :, :]*torch.pi, fs=self.fs), gain=param[:, 2, :, :], fs=self.fs, device=self.device
                )
        b_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        A[A == 0+1j*0] = torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        if torch.isnan(H).any():
            print("Warning: NaN values in the frequency response. This is a common issue with high order, we are working on it. But please rise an issue on github if you encounter it. One thing that can help is to reduce the learning rate.")
        return H, B, A

    def get_map(self):
        r"""
        Get the mapping function :attr:`map` to parameter values that ensure stability.
        The type of mapping is based on the filter type.
        """
        match self.filter_type:
            case "lowpass" | "highpass":
                self.map = lambda x: torch.clamp(
                    torch.stack((x[:, 0, :, :], 20*torch.log10(torch.abs(x[:, 1, :, :]))), dim=1),
                    min=torch.tensor([0, -60], device=self.device).view(-1, 1, 1).expand_as(x),
                    max=torch.tensor([1, 60], device=self.device).view(-1, 1, 1).expand_as(x),
                )
            case "bandpass":   
                self.map = lambda x: torch.clamp(
                    torch.stack((x[:, 0, :, :], x[:, 1, :, :], 20*torch.log10(torch.abs(x[:, -1, :, :]))), dim=1),
                    min=torch.tensor([0, 0, -60], device=self.device).view(-1, 1, 1).expand_as(x),
                    max=torch.tensor([1, 1, 60], device=self.device).view(-1, 1, 1).expand_as(x),
                )

    def init_param(self):
        r"""
        Initialize the filter parameters.
        """
        torch.nn.init.uniform_(self.param[:, 0, :], a=0, b=1)
        if self.filter_type == "bandpass":
            torch.nn.init.uniform_(self.param[:, 1, :], a=self.param[:, 0, :, :].max().item(), b=1)
        torch.nn.init.uniform_(self.param[:, -1, :], a=-1, b=1)
    
    def check_param_shape(self):
        r"""
        Check the shape of the filter parameters.
        """
        assert (
            len(self.size) == 4
        ), "Parameter size must be 4D, for 3D (parallel) biquads use parallelBiquad module."

    def initialize_class(self):
        r"""
        Initialize the :class:`Biquad` class.
        """
        self.check_param_shape()
        self.get_io()
        self.freq_response = to_complex(
            torch.empty((self.nfft // 2 + 1, *self.size[1:]))
        )
        self.get_freq_response()
        self.get_freq_convolve()

    def get_io(self): # NOTE: This method does not need to be reimplemented here, it is inherited from Filter.
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]


class parallelBiquad(Biquad):
    r"""
    Parallel counterpart of the :class:`Biquad` class.
    For information about **attributes** and **methods** see :class:`flamo.processor.dsp.Biquad`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(K, P, N, N)`
        - freq_response: :math:`(M, N, N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    and :math:`N` is the number of input/output channels.
    The :attr:'param' attribute represent the biquad filter coefficents. The first dimension of the :attr:'param' tensor corresponds to the number of filters :math:`K`, the second dimension corresponds to the number of filter parameters :math:`P` (3).
    Ellipsis :math:`(...)` represents additional dimensions (not tested).
    """
    def __init__(
        self,
        size: tuple = (1,),
        n_sections: int = 1,
        filter_type: str = "lowpass",
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad=True,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            n_sections=n_sections,
            filter_type=filter_type,
            nfft=nfft,
            fs=fs,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        assert (
            len(self.size) == 3
        ), "Parameter size must be 3D, for 3D sapce use Biquad module."

    def get_map(self):
        r"""
        Get the mapping function :attr:`map` to parameter values that ensure stability.
        The type of mapping is based on the filter type.
        """
        match self.filter_type:
            case "lowpass" | "highpass":
                self.map = lambda x: torch.clamp(
                    torch.stack((x[:, 0, :], 20*torch.log10(torch.abs(x[:, -1, :]))), dim=1),
                    min=torch.tensor([0, -60], device=self.device).view(-1, 1).expand_as(x),
                    max=torch.tensor([1, 60], device=self.device).view(-1, 1).expand_as(x),
                )
            case "bandpass":
                self.map = lambda x: torch.clamp(
                    torch.stack((x[:, 0, :], x[:, 1, :], 20*torch.log10(torch.abs(x[:, -1, :]))), dim=1),
                    min=torch.tensor([0, 0, -60], device=self.device).view(-1, 1).expand_as(x),
                    max=torch.tensor([1, 1, 60], device=self.device).view(-1, 1).expand_as(x),
                )

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the specified filter type.
        It calls the :func:`flamo.functional.lowpass_filter`, :func:`flamo.functional.highpass_filter`, or :func:`flamo.functional.bandpass_filter` functions based on the filter type.

        **Args**:
            - param (torch.Tensor): A tensor containing the filter parameters. 
            The shape of the tensor should be (batch_size, num_params, height). 
            The parameters are interpreted differently based on the filter type:
            - For "lowpass" and "highpass" filters, param[:, 0, :] represents 
            the cutoff frequency and param[:, 1, :] represents the gain.
            - For "bandpass" filters, param[:, 0, :] represents the lower 
            cutoff frequency, param[:, 1, :] represents the upper cutoff 
            frequency, and param[:, 2, :] represents the gain.
        **Returns**:       
            - H (torch.Tensor): The frequency response of the filter.
            - B (torch.Tensor): The Fourier transformed numerator polynomial coefficients.
            - A (torch.Tensor): The Fourier transformed denominator polynomial coefficients.


        The method uses the filter type specified in `self.filter_type` to determine 
        which filter to apply. It applies an aliasing envelope decay to the filter coefficients.
        Zero values in the denominator polynomial coefficients are replaced with 
        a small constant to avoid division by zero.
        """
        match self.filter_type:
            case "lowpass":
                b, a = lowpass_filter(fc=rad2hertz(param[:, 0, :]*torch.pi, fs=self.fs), gain=param[:, 1, :], fs=self.fs, device=self.device)
            case "highpass":
                b, a = highpass_filter(fc=rad2hertz(param[:, 0, :]*torch.pi, fs=self.fs), gain=param[:, 1, :], fs=self.fs, device=self.device)
            case "bandpass":
                b, a = bandpass_filter(
                    fc1=rad2hertz(param[:, 0, :]*torch.pi, fs=self.fs), fc2=rad2hertz(param[:, 1, :]*torch.pi, fs=self.fs), gain=param[:, 2, :], fs=self.fs, device=self.device
                )
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        A[A == 0+1j*0] = torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1) )
        if torch.isnan(H).any():
            print("Warning: NaN values in the frequency response. This is a common issue with high order, we are working on it. But please rise an issue on github if you encounter it. One thing that can help is to reduce the learning rate.")
        return H, B, A
    
    def get_freq_convolve(self):
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]


class SVF(Filter):
    r"""
    IIR filter as a serially cascaded state variable filters (SVFs). 
    Inherits from the :class:`Filter` class.
    Can be used to design a variety of filters such as lowpass, highpass, bandpass, lowshelf, highshelf, peaking, and notch filters.
    The filter coefficients are parameterized by the cut-off frequency (:math:`f`) and resonance (:math:`R`) parameters.
    The mixing coefficients (:math:`m_{LP}`, :math:`m_{BP}`, :math:`m_{HP}`  for lowpass, bandpass, and highpass filters) determine the contribution of each filter type in the cascaded structure.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - param: :math:`(P, K,  N_{out}, N_{in})`
        - freq_response: :math:`(M,  N_{out}, N_{in})`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, :math:`N_{out}` is the number of output channels,
    The :attr:'param' attribute represent the biquad filter coefficents. The first dimension of the :attr:'param' tensor corresponds to the number of SVF parameters (5), the second dimension corresponds to the number of filters :math:`K`.
    Ellipsis :math:`(...)` represents additional dimensions (not tested).    

    SVF parameters are used to express biquad filters as follows:

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}

        b_0 = f^2 m_{LP} + f m_{BP} + m_{HP} \\
        b_1 = 2 f^2 m_{LP} - 2 m_{HP} \\
        b_2 = f^2 m_{LP} - f m_{BP} + m_{HP} \\
        a_0 = f^2 + 2 R f + 1 \\
        a_1 = 2 f^2 - 2 \\
        a_2 = f^2 - 2 R f + 1 \\

    For serieally cascaded SVFs, the frequency response is 

    .. math::
        H(z) = \prod_{k=1}^{K} H_k(z)

    where :math:`K` is the number of cascaded filters :attr:`n_sections`.

        **Args**:
            - size (tuple, optional): The size of the raw filter parameters. Default: (1, 1).
            - n_sections (int, optional): The number of cascaded filters. Default: 1.
            - filter_type (str, optional): The type of filter to use. Options are {"lowpass","highpass","bandpass","lowshelf","highshelf","peaking","notch",} Default: None.
            - nfft (int, optional): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - map (function, optional): The mapping function to apply to the raw parameters. Default: lambda x: x.
            - requires_grad (bool, optional): Whether the filter parameters require gradients. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str, optional): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - fs (int): The sampling frequency.
            - n_sections (int): The number of cascaded filters.
            - filter_type (str): The type of filter.
            - freq_response (torch.Tensor): The frequency response of the filter.
            - param (nn.Parameter): The parameters of the SVF filter.

        **Methods**:
            - check_param_shape(): Check the shape of the filter parameters.
            - check_input_shape(x): Check if the dimensions of the input tensor x are compatible with the module.
            - get_freq_response(): Compute the frequency response of the filter.
            - get_freq_convolve(): Compute the frequency convolution function.

    For more details, refer to the paper `Differentiable artificial reverberation <https://arxiv.org/abs/2105.13940>` by Lee, S. et al.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        n_sections: int = 1,
        filter_type: str = None,
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad: bool = True,
        alias_decay_db: float = 0.0,
        device=None
    ):
        self.fs = fs
        self.n_sections = n_sections
        assert filter_type in [
            "lowpass",
            "highpass",
            "bandpass",
            "lowshelf",
            "highshelf",
            "peaking",
            "notch",
            None,
        ], "Invalid filter type"
        self.filter_type = filter_type
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20)
        self.alias_envelope_dcy = (gamma ** torch.arange(0, 3, 1, device=device))
        super().__init__(
            size=(5, self.n_sections, *size),
            nfft=nfft,
            map=self.map_param2svf,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        assert (
            len(self.size) == 4
        ), "Filter parameter space must be 4D, for 3D (parallel) filters use parallelSVF module."

    def check_input_shape(self, x):
        r"""
        Checks if the input dimensions are compatible with the filter parameters.

            **Args**:
                - x (torch.Tensor): The input signal.
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SVF filter
        """
        f, R, mLP, mBP, mHP = param
        b = torch.zeros((3, *f.shape), device=self.device)
        a = torch.zeros((3, *f.shape), device=self.device)

        b[0] = (f**2) * mLP + f * mBP + mHP
        b[1] = 2 * (f**2) * mLP - 2 * mHP
        b[2] = (f**2) * mLP - f * mBP + mHP

        a[0] = (f**2) + 2 * R * f + 1
        a[1] = 2 * (f**2) - 2
        a[2] = (f**2) - 2 * R * f + 1

        # apply anti-aliasing
        b_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        A[A == 0+1j*0] = torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / torch.prod(A, dim=1)
        return H, B, A

    def param2freq(self, param):
        r"""
        Applies a sigmoid function to the parameter value and maps it to the range [0, fs/2],
        where :math:`f_s` is the sampling frequency according to the formula:

        .. math::
            f = \text{tan}\left(\pi \cdot \text{sigmoid}(x) \cdot 0.5\right).
        """
        sigmoid = torch.div(1, 1 + torch.exp(-param))
        return torch.tan(torch.pi * sigmoid * 0.5)

    def param2R(self, param):
        r"""
        Applies a softplus function to the parameter value and maps it to the range [0, 1],
        according to the formula:

        .. math::
            R = \text{softplus}(x) / log(2).

        """
        return torch.div(torch.log(torch.ones(1, device=self.device) + torch.exp(param)), torch.log(torch.tensor(2, device=self.device)))

    def param2mix(self, param, R=None):
        r"""
        Mapping function for the mixing coefficients relative to the filter type.
        
        - lowpass: :math:`m_{LP} = G, m_{BP} = 0, m_{HP} = 0`.
        - highpass: :math:`m_{LP} = 0, m_{BP} = 0, m_{HP} = G.`
        - bandpass: :math:`m_{LP} = 0, m_{BP} = G, m_{HP} = 0.`
        - lowshelf: :math:`m_{LP} = 1, m_{BP} = 2 \cdot R \cdot \sqrt{G}, m_{HP} = G.`
        - highshelf: :math:`m_{LP} = G, m_{BP} = 2 \cdot R \cdot \sqrt{G}, m_{HP} = 1.`
        - peaking | notch: :math:`m_{LP} = 1, m_{BP} = 2 \cdot R \cdot \sqrt{G}, m_{HP} = 1.`

        where :math:`G` is the gain parameter mapped from the raw parameters as 
        :math:`G = 10^{-\text{softplus}(x)}`.  

            **Args**:
                - param (torch.Tensor): The raw parameters.
                - R (torch.Tensor, optional): The resonance parameter. Default: None.

        """
        # activation = lambda x: 10**(-torch.log(1+torch.exp(x)) / torch.log(torch.tensor(2,  device=get_device())))
        G = 10 ** (-F.softplus(param[0]))
        match self.filter_type:
            case "lowpass":
                return torch.cat(
                    (
                        (torch.ones_like(G)).unsqueeze(0),
                        (torch.zeros_like(G)).unsqueeze(0),
                        torch.zeros_like(G).unsqueeze(0),
                    ),
                    dim=0,
                )
            case "highpass":
                return torch.cat(
                    (
                        (torch.zeros_like(G)).unsqueeze(0),
                        (torch.zeros_like(G)).unsqueeze(0),
                        torch.ones_like(G).unsqueeze(0),
                    ),
                    dim=0,
                )
            case "bandpass":
                return torch.cat(
                    (
                        (torch.zeros_like(G)).unsqueeze(0),
                        (torch.ones_like(G)).unsqueeze(0),
                        torch.zeros_like(G).unsqueeze(0),
                    ),
                    dim=0,
                )
            case "lowshelf":
                return torch.cat(
                    (
                        (torch.ones_like(G)).unsqueeze(0),
                        (2 * R * torch.sqrt(G)).unsqueeze(0),
                        (G * torch.ones_like(G)).unsqueeze(0),
                    ),
                    dim=0,
                )
            case "highshelf":
                return torch.cat(
                    (
                        (G * torch.ones_like(G)).unsqueeze(0),
                        (2 * R * torch.sqrt(G)).unsqueeze(0),
                        (torch.ones_like(G)).unsqueeze(0),
                    ),
                    dim=0,
                )
            case "peaking" | "notch":
                return torch.cat(
                    (
                        (torch.ones_like(G)).unsqueeze(0),
                        (2 * R * torch.sqrt(G)).unsqueeze(0),
                        (torch.ones_like(G)).unsqueeze(0),
                    ),
                    dim=0,
                )
            case None:
                # general SVF filter
                bias = torch.ones((param.shape), device=self.device)
                bias[1] = 2 * torch.ones((param.shape[1:]), device=self.device)
                return param + bias

    def map_param2svf(self, param):
        r"""
        Mapping function for the raw parameters to the SVF filter coefficients.
        """
        f = self.param2freq(param[0])
        r = self.param2R(param[1])
        if self.filter_type == "lowshelf" or self.filter_type == "highshelf":
            # R = r + torch.sqrt(torch.tensor(2))
            R = torch.tensor(1, device=self.device)
        if self.filter_type == "peaking":
            R = 1 / r  # temporary fix for peaking filter
            m = self.param2mix(param[2:], r)
        else:
            R = r
            m = self.param2mix(param[2:], R)
        return f, R, m[0], m[1], m[2]
    
    def get_io(self): # NOTE: This method does not need to be reimplemented here, it is inherited from Filter.
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]


class parallelSVF(SVF):
    r"""
    Parallel counterpart of the :class:`SVF` class.
    For information about **attributes** and **methods** see :class:`flamo.processor.dsp.SVF`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(P, K, N)`
        - freq_response: :math:`(M, N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    and :math:`N` is the number of input/output channels.
    The :attr:'param' attribute represent the biquad filter coefficents. The first dimension of the :attr:'param' tensor corresponds to the number of SVF parameters (5), the second dimension corresponds to the number of filters :math:`K`.
    Ellipsis :math:`(...)` represents additional dimensions (not tested).   
    """

    def __init__(
        self,
        size: tuple = (1, ),
        n_sections: int = 1,
        filter_type: str = None,
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device=None,
    ):
        super().__init__(
            size=size,
            n_sections=n_sections,
            filter_type=filter_type,
            nfft=nfft,
            fs=fs,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        assert (
            len(self.size) == 3
        ), "Filter parameter space must be 3D, for 4D filters use SVF module."

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SVF filter
        """
        f, R, mLP, mBP, mHP = param
        b = torch.zeros((3, *f.shape), device=self.device)
        a = torch.zeros((3, *f.shape), device=self.device)

        b[0] = (f**2) * mLP + f * mBP + mHP
        b[1] = 2 * (f**2) * mLP - 2 * mHP
        b[2] = (f**2) * mLP - f * mBP + mHP

        a[0] = (f**2) + 2 * R * f + 1
        a[1] = 2 * (f**2) - 2
        a[2] = (f**2) - 2 * R * f + 1

        # apply anti-aliasing
        b_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('p, pon -> pon', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        return H, B, A
    
    def get_freq_convolve(self):
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )
    
    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]


class GEQ(Filter):
    r"""
    Graphic Equilizer filter. Inherits from the :class:`Filter` class.
    It supports 1 and 1/3 octave filter bands. 
    The raw parameters are the linear gain values for each filter band.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - param: :math:`(K, N_{out}, N_{in})`
        - freq_response: :math:`(M,  N_{out}, N_{in})`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins, 
    :math:`N_{in}` is the number of input channels, :math:`N_{out}` is the number of output channels,
    The :attr:'param' attribute represent the command gains of each band + shelving filters. The first dimension of the :attr:'param' tensor corresponds to the number of command gains/filters :math:`K`.
    Ellipsis :math:`(...)` represents additional dimensions (not tested).   

        **Args**:
            - size (tuple, optional): The size of the raw filter parameters. Default: (1, 1).
            - octave_interval (int, optional): The octave interval for the center frequencies. Default: 1.
            - nfft (int, optional): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            - fs (int, optional): The sampling frequency. Default: 48000.
            - map (function, optional): The mapping function to apply to the raw parameters. Default: lambda x: 20*torch.log10(x).
            - requires_grad (bool, optional): Whether the filter parameters require gradients. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.
            - device (str, optional): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - fs (int): The sampling frequency.
            - center_freq (torch.Tensor): The center frequencies of the filter bands.
            - shelving_crossover (torch.Tensor): The shelving crossover frequencies.
            - freq_response (torch.Tensor): The frequency response of the filter.
            - param (nn.Parameter): The parameters of the GEQ filter.

        **Methods**:
            - check_param_shape(): Check the shape of the filter parameters.
            - get_freq_response(): Compute the frequency response of the filter.
            - get_freq_convolve(): Compute the frequency convolution function.

    References:
        - Schlecht, S., Habets, E. (2017). Accurate reverberation time control in
        feedback delay networks Proc. Int. Conf. Digital Audio Effects (DAFx)
        adapted to python by: Dal Santo G. 
        - Välimäki V., Reiss J. All About Audio Equal-
        ization: Solutions and Frontiers, Applied Sciences, vol. 6,
        no. 5, pp. 129, May 2016
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        octave_interval: int = 1,
        nfft: int = 2**11,
        fs: int = 48000,
        map=lambda x: 20*torch.log10(x),
        requires_grad: bool = True,
        alias_decay_db: float = 0.0,
        device=None
    ):
        self.octave_interval = octave_interval
        self.fs = fs
        self.center_freq, self.shelving_crossover = eq_freqs(
            interval=self.octave_interval)
        self.n_gains = len(self.center_freq) + 3
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db, device=device)) / (nfft) / 20)
        self.alias_envelope_dcy = (gamma ** torch.arange(0, 3, 1, device=device))
        super().__init__(
            size=(self.n_gains, *size),
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def init_param(self):
        torch.nn.init.uniform_(self.param, a=10**(-6/20), b=10**(6/20))  

    def check_param_shape(self):
        assert (
            len(self.size) == 3
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelGEQ module."

    def get_freq_response(self):
        r"""
        Compute the frequency response of the filter.
        """
        self.freq_response = lambda param: self.get_poly_coeff(self.map(param))[0]

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, *self.size), device=self.device)
        b = torch.zeros((3, *self.size), device=self.device)
        R = torch.tensor(2.7, device=self.device)
        for m_i in range(self.size[-2]):
            for n_i in range(self.size[-1]):
                a[:, :, m_i, n_i], b[:, :, m_i, n_i] = geq(
                    center_freq=self.center_freq,
                    shelving_freq=self.shelving_crossover,
                    R=R,
                    gain_db=param[:, m_i, n_i],
                    fs=self.fs,
                    device=self.device
                )
         
        b_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, a)
        a_aa = torch.einsum('p, pomn -> pomn', self.alias_envelope_dcy, b)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        A[A == 0+1j*0] = torch.tensor(1e-12)
        H = torch.prod(B, dim=1) / (torch.prod(A, dim=1))
        return H, B, A 

    def initialize_class(self):
        self.check_param_shape()
        self.get_io()
        self.freq_response = to_complex(
            torch.empty((self.nfft // 2 + 1, *self.size[1:]))
        )
        self.get_freq_response()
        self.get_freq_convolve()

    def get_io(self): # NOTE: This method does not need to be reimplemented here, it is inherited from Filter.
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]


class parallelGEQ(GEQ):
    r"""
    Parallel counterpart of the :class:`GEQ` class
    For information about **attributes** and **methods** see :class:`flamo.processor.dsp.GEQ`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(P, N)`
        - freq_response: :math:`(M, N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins, and :math:`N` is the number of input/output channels.
    The :attr:'param' attribute represent the command gains of each band + shelving filters. The first dimension of the :attr:'param' tensor corresponds to the number of command gains/filters :math:`K`.
    Ellipsis :math:`(...)` represents additional dimensions (not tested).   
    """

    def __init__(
        self,
        size: tuple = (1, ),
        octave_interval: int = 1,
        nfft: int = 2**11,
        fs: int = 48000,
        map=lambda x: 20*torch.log10(x),
        requires_grad: bool = True,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            octave_interval=octave_interval,
            nfft=nfft,
            fs=fs,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        assert (
            len(self.size) == 2
        ), "Filter must be 2D, for 3D filters use GEQ module."

    def get_poly_coeff(self, param):
        r"""
        Computes the polynomial coefficients for the SOS section.
        """
        a = torch.zeros((3, *self.size), device=self.device)
        b = torch.zeros((3, *self.size), device=self.device)
        R = torch.tensor(2.7, device=self.device)
        for n_i in range(self.size[-1]):
                a[:, :, n_i], b[:, :, n_i] = geq(
                    center_freq=self.center_freq,
                    shelving_freq=self.shelving_crossover,
                    R=R,
                    gain_db=param[:, n_i],
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

    def get_freq_convolve(self):
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]

# ============================= DELAYS ================================


class Delay(DSP):
    r"""
    Delay module that applies in frequency domain a time delay to the input signal. Inherits from :class:`DSP`.
    To improve update effectiveness, the unit of time can be adjusted via the :attr:`unit` attribute to use subdivisions or multiples of time.
    For integer Delays, the :attr:`isint` attribute can be set to True to round the delay to the nearest integer before computing the frequency response. 
    
    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - param: :math:`(M, N_{out}, N_{in})`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, and :math:`N_{out}` is the number of output channels.
    Ellipsis :math:`(...)` represents additional dimensions.

    For a delay of :math:`d` seconds, the frequency response of the delay without anti-aliasing is computed as:

    .. math::

        e^{-j \omega d}\; \text{for}\; \omega = 2\pi \frac{m}{\texttt{nfft}}


    where :math:`\texttt{nfft}` is the number of FFT points, and :math:`m` is the frequency index :math:`m=0, 1, \dots, \lfloor\texttt{nfft}/2 +1\rfloor` .

        **Args**:
            - size (tuple, optional): Size of the delay module. Default: (1, 1).
            - max_len (int, optional): Maximum length of the delay in samples. Default: 2000.
            - isint (bool, optional): Flag indicating whether the delay length should be rounded to the nearest integer. Default: False.
            - unit (int, optional): Unit value used for second-to-sample conversion. Default: 100.
            - nfft (int, optional): Number of FFT points. Default: 2 ** 11.
            - fs (int, optional): Sampling frequency. Default: 48000.
            - requires_grad (bool, optional): Flag indicating whether the module parameters require gradients. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Defaults to 0.
            - device (str, optional): The device of the constructed tensors. Default: None.
            
        **Attributes**:
            - fs (int): Sampling frequency.
            - max_len (int): Maximum length of the delay in samples.
            - unit (int): Unit value used for second-to-sample conversion.
            - isint (bool): Flag indicating whether the delay length should be rounded to the nearest integer.
            - omega (torch.Tensor): The frequency values used for the FFT.
            - freq_response (torch.Tensor): The frequency response of the delay module.
            - order (int): The order of the delay.
            - freq_convolve (function): The frequency convolution function.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
        
        **Methods**:
            - forward(x): Applies the Delay module to the input tensor x.
            - init_param(): Initializes the delay parameters.
            - s2sample(delay): Converts a delay value from seconds to samples.
            - sample2s(delay): Converts a delay value from samples to seconds.
            - get_freq_response(): Computes the frequency response of the delay module.
            - check_input_shape(x): Checks if the input dimensions are compatible with the delay parameters.
            - check_param_shape(): Checks if the shape of the delay parameters is valid.
            - get_freq_convolve(): Computes the frequency convolution function.
            - initialize_class(): Initializes the Delay module.
    """

    def __init__(
        self,
        size: tuple = (1, 1),
        max_len: int = 2000,
        isint: bool = False,
        unit: int = 100,
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad=False,
        alias_decay_db: float = 0.0,
        device=None,
    ):
        self.fs = fs  
        self.max_len = max_len  
        self.unit = unit  
        self.isint = isint  
        super().__init__(
            size=size,
            nfft=nfft,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )
        self.initialize_class()

    def forward(self, x, ext_param=None):
        r"""
        Applies the Delay module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape (B, M, N_in, ...).
                ext_paran (torch.Tensor, optional): Parameter values from outer modules. Default: None.

            **Returns**:
                torch.Tensor: Output tensor of shape (B, M, N_out, ...).
        """
        self.check_input_shape(x)
        if ext_param is None:
            return self.freq_convolve(x, self.param)
        else: 
            # log the parameters that are being passed 
            with torch.no_grad():
                self.assign_value(ext_param)
            return self.freq_convolve(x, ext_param)

    def init_param(self):
        r"""
        Initializes the delay parameters.
        """
        if self.isint:
            delay_len = torch.randint(1, self.max_len, self.size, device=self.device)
        else:
            delay_len = torch.rand(self.size, device=self.device) * self.max_len
        self.assign_value(self.sample2s(delay_len))
        self.order = (delay_len).max() + 1

    def s2sample(self, delay):
        r"""
        Converts a delay value from seconds to samples.

            **Args**:
                delay (float): The delay value in seconds.
        """
        return delay * self.fs / self.unit

    def sample2s(self, delay):
        r"""
        Converts a delay value from samples to seconds.

            **Args**:
                delay (torch.Tensor): The delay value in samples.
        """
        return delay / self.fs * self.unit

    def get_freq_response(self):
        r"""
        Computes the frequency response of the delay module.
        """
        m = self.get_delays()
        self.freq_response = lambda param: (self.gamma**m(param)) * torch.exp(
            -1j
            * torch.einsum(
                "fo, omn -> fmn",
                self.omega,
                m(param).unsqueeze(0),
            )
        )

    def get_delays(self):
        r"""
        Computes the delay values from the raw parameters.
        """
        return lambda param: self.s2sample(self.map(param))
                                      
    def check_input_shape(self, x):
        r"""
        Checks if the input dimensions are compatible with the delay parameters.

            **Args**:
                x (torch.Tensor): The input signal.
        """
        if (int(self.nfft / 2 + 1), self.input_channels) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.param.shape} not compatible with input signal of shape = ({x.shape})."
            )

    def check_param_shape(self):
        r"""
        Checks if the shape of the delay parameters is valid.
        """
        assert (
            len(self.size) == 2
        ), "delay must be 2D, for 1D (parallel) delay use parallelDelay module."

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response(param), x
        )

    def initialize_class(self):
        r"""
        Initializes the Delay module.

        This method checks the shape of the delay parameters, computes the frequency response, and initializes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_io()
        if self.requires_grad:
            if self.isint:
                self.map = lambda x: nn.functional.softplus(x).round()
            else:
                self.map = lambda x: nn.functional.softplus(x)
        self.omega = (
            2 * torch.pi * torch.arange(0, self.nfft // 2 + 1, device=self.device) / self.nfft
        ).unsqueeze(1)
        self.get_freq_response()
        self.get_freq_convolve()

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-2]

class parallelDelay(Delay):
    """
    Parallel counterpart of the :class:`Delay` class.
    For information about **attributes** and **methods** see :class:`Delay`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(M, N)`
        - output: :math:`(B, M, N, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    and :math:`N` is the number of input channels.
    Ellipsis :math:`(...)` represents additional dimensions.
    """

    def __init__(
        self,
        size: tuple = (1,),
        max_len=2000,
        unit: int = 100,
        isint: bool = False,
        nfft=2**11,
        fs: int = 48000,
        requires_grad=False,
        alias_decay_db: float = 0.0,
        device=None
    ):
        super().__init__(
            size=size,
            max_len=max_len,
            isint=isint,
            unit=unit,
            nfft=nfft,
            fs=fs,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
            device=device
        )

    def check_param_shape(self):
        """
        Checks if the shape of the delay parameters is valid.
        """
        assert len(self.size) == 1, "delays must be 1D, for 2D delays use Delay module."

    def get_freq_convolve(self):
        """
        Computes the frequency convolution function.
        """
        self.freq_convolve = lambda x, param: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response(param), x
        )

    def get_freq_response(self):
        """
        Computes the frequency response of the delay module.
        """
        m = self.get_delays()
        self.freq_response = lambda param: (self.gamma**m(param)) * torch.exp(
            -1j
            * torch.einsum(
                "fo, on -> fn",
                self.omega,
                m(param).unsqueeze(0),
            )
        )

    def get_io(self):
        r"""
        Computes the number of input and output channels based on the size parameter.
        """
        self.input_channels = self.size[-1]
        self.output_channels = self.size[-1]