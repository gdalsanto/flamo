import torch
import torch.nn as nn
from flamo.utils import to_complex
from flamo.functional import (
    skew_matrix, 
    lowpass_filter, 
    highpass_filter, 
    bandpass_filter,
    hertz2rad,
    rad2hertz)

# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    r"""
    Base class for all transformations. 

    The transformation is a callable, e.g., :class:`lambda` expression, function, :class:`nn.Module`. 

        **Args**:
            - transform (callable): The transformation function to be applied to the input. Default: lambda x: x  
        **Attributes**:
            - transform (callable): The transformation function to be applied to the input.  
        **Methods**:
            - forward(x): Applies the transformation function to the input.  

        Examples::

            >>> pow2 = Transform(lambda x: x**2)
            >>> input = torch.tensor([1, 2, 3])
            >>> pow2(input)
            tensor([1, 4, 9])
    """
    def __init__(self, transform: callable = lambda x: x):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        """
        Applies the transformation to the input tensor.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
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

if __name__ == "__main__":
    # Create an instance of the Transform class
    transform = Transform(lambda x: x ** 2)

    # Create an input tensor
    input_tensor = torch.tensor([1, 2, 3])

    # Apply the transformation to the input tensor
    output_tensor = transform(input_tensor)

    # Print the transformed tensor
    print(output_tensor)


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
    ):
        super().__init__()
        assert isinstance(size, tuple), "Size must be a tuple."
        self.size = size 
        self.nfft = nfft
        self.map = (
            map
        )
        self.new_value = 0  # flag indicating if new values have been assigned
        self.requires_grad = requires_grad 
        self.param = nn.Parameter(torch.empty(self.size), requires_grad=self.requires_grad) 
        self.fft = lambda x: torch.fft.rfft(x, n=self.nfft, dim=0)  
        self.ifft = lambda x: torch.fft.irfft(x, n=self.nfft, dim=0)  
        # initialize time anti-aliasing envelope function
        self.alias_decay_db = torch.tensor(alias_decay_db)
        self.init_param()
        self.get_gamma()

    def forward(self, x):
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

        self.gamma = torch.tensor(
            10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)
        )

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
        r"""
        Applies the Gain module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        return self.freq_convolve(x)

    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (self.size[-1]) != (x.shape[2]):
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
        self.freq_convolve = lambda x: torch.einsum(
            "mn,bfn...->bfm...", to_complex(self.map(self.param)), x
        )

    def initialize_class(self):
        r"""
        Initializes the Gain module.

        This method checks the shape of the gain parameters and computes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_freq_convolve()


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
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
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
        self.freq_convolve = lambda x: torch.einsum(
            "n,bfn...->bfn...", to_complex(self.map(self.param)), x
        )


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
    ):
        self.matrix_type = matrix_type
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
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
        self.matrix_type = self.matrix_type
        self.matrix_gallery()
        self.get_freq_convolve()


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
        r"""
        Applies the Filter module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        self.check_input_shape(x)
        if self.requires_grad or self.new_value:
            self.get_freq_response()
            self.new_value = 0
        return self.freq_convolve(x)

    def check_input_shape(self, x):
        r"""
        Checks if the dimensions of the input tensor x are compatible with the module.

            **Args**:
                x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.
        """
        if (int(self.nfft / 2 + 1), self.size[-1]) != (x.shape[1], x.shape[2]):
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
        self.ir = self.map(self.param)
        self.decaying_envelope = (self.gamma ** torch.arange(0, self.ir.shape[0])).view(
            -1, *tuple([1 for i in self.ir.shape[1:]])
        )
        self.freq_response = self.fft(self.ir * self.decaying_envelope)

    def get_freq_convolve(self):
        r"""
        Computes the frequency convolution function.

        The frequency convolution is computed using the einsum function.

            **Args**:
                x (torch.Tensor): Input tensor.

            **Returns**:
                torch.Tensor: Output tensor after frequency convolution.
        """
        self.freq_convolve = lambda x: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response, x
        )

    def initialize_class(self):
        r"""
        Initializes the Gain module.

        This method checks the shape of the gain parameters, computes the frequency response of the filter, 
        and computes the frequency convolution function.
        """
        self.check_param_shape()
        self.get_freq_response()
        self.get_freq_convolve()


class parallelFilter(Filter):
    """
    Parallel counterpart of the :class:`Filter` class.
    For information about **attributes** and **methods** see :class:`Filter`.

    Shape:
        - input: :math:`(B, M, N, ...)`
        - param: :math:`(N_{taps}, N, N)`
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
    ):
        super().__init__(
            size=size,
            nfft=nfft,
            map=map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
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
        self.freq_convolve = lambda x: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response, x
        )


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

        **Args**:
            - size (tuple, optional): Size of the filter. Default: (1, 1).
            - filter_type (str, optional): Type of the filter. Must be one of "lowpass", "highpass", or "bandpass". Default: "lowpass".
            - nfft (int, optional): Number of points for FFT. Default: 2048.
            - fs (int, optional): Sampling frequency. Default: 48000.
            - requires_grad (bool, optional): Whether the filter parameters require gradient computation. Default: True.
            - alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.

        **Attributes**:
            - filter_type (str): Type of the filter.
            - fs (int): Sampling frequency.
            - freq_response (torch.Tensor): Frequency response of the filter.

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
        filter_type: str = "lowpass",
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad: bool=True,
        alias_decay_db: float = 0.0,
    ):
        assert filter_type in ["lowpass", "highpass", "bandpass"], "Invalid filter type"
        self.filter_type = filter_type
        self.fs = fs
        self.get_map()
        gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / (nfft) / 20)
        self.alias_envelope_dcy = (gamma ** torch.arange(0, 3, 1))
        super().__init__(
            size=(*self.get_size(), *size),
            nfft=nfft,
            map=self.map,
            requires_grad=requires_grad,
            alias_decay_db=alias_decay_db,
        )
        if self.alias_decay_db != 0:
            print(
                "Warning: Anti time-aliasiang might not work properly at this stage. Work in progress."
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
        It calls the :func:`flamo.functional.lowpass_filter`, :func:`flamo.functional.highpass_filter`, or :func:`flamo.functional.bandpass_filter` functions based on the filter type.
        """
        param = self.map(self.param)
        match self.filter_type:
            case "lowpass":
                b, a = lowpass_filter(fc=rad2hertz(param[0, :]*torch.pi, fs=self.fs), gain=param[1, :], fs=self.fs)
            case "highpass":
                b, a = highpass_filter(fc=rad2hertz(param[0, :]*torch.pi, fs=self.fs), gain=param[1, :], fs=self.fs)
            case "bandpass":
                b, a = bandpass_filter(
                    fc1=rad2hertz(param[0, :]*torch.pi, fs=self.fs), fc2=rad2hertz(param[1, :]*torch.pi, fs=self.fs), gain=param[2, :], fs=self.fs
                )
        b_aa = torch.einsum('o, omn -> omn', self.alias_envelope_dcy, b)
        a_aa = torch.einsum('o, omn -> omn', self.alias_envelope_dcy, a)
        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)
        self.freq_response = B / A

    def get_map(self):
        r"""
        Get the mapping function :attr:`map` to parameter values that ensure stability.
        The type of mapping is based on the filter type.
        """
        match self.filter_type:
            case "lowpass" | "highpass":
                self.map = lambda x: torch.clamp(
                    x,
                    min=torch.tensor([0, -60]).view(-1, 1, 1).expand_as(x),
                    max=torch.tensor([torch.pi, 60]).view(-1, 1, 1).expand_as(x),
                )
            case "bandpass":
                self.map = lambda x: torch.clamp(
                    x,
                    min=torch.tensor([0, 0, -60]).view(-1, 1, 1).expand_as(x),
                    max=torch.tensor([torch.pi, torch.pi, 60])
                    .view(-1, 1, 1)
                    .expand_as(x),
                )

    def init_param(self):
        r"""
        Initialize the filter parameters.
        """
        torch.nn.init.uniform_(self.param[0, :], a=0, b=1)
        if self.filter_type == "bandpass":
            torch.nn.init.uniform_(self.param[1, :], a=0, b=1)
        torch.nn.init.uniform_(self.param[-1, :], a=-1, b=1)
    
    def check_param_shape(self):
        r"""
        Check the shape of the filter parameters.
        """
        assert (
            len(self.size) == 3
        ), "Filter must be 3D, for 2D (parallel) filters use ParallelGEQ module."

    def initialize_class(self):
        r"""
        Initialize the :class:`Biquad` class.
        """
        self.check_param_shape()
        self.freq_response = to_complex(
            torch.empty((self.nfft // 2 + 1, *self.size[1:]))
        )
        self.get_freq_response()
        self.get_freq_convolve()


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
            - nfft (int, optional): Number of FFT points. Default: 2 ** 11.
            - fs (int, optional): Sampling frequency. Default: 48000.
            - requires_grad (bool, optional): Flag indicating whether the module parameters require gradients. Default: False.
            - alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Defaults to 0.

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
        nfft: int = 2**11,
        fs: int = 48000,
        requires_grad=False,
        alias_decay_db: float = 0.0,
    ):
        self.fs = fs  
        self.max_len = max_len  
        self.unit = 100  
        self.isint = isint  
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
        r"""
        Applies the Delay module to the input tensor x.

            **Args**:
                x (torch.Tensor): Input tensor of shape (B, M, N_in, ...).

            **Returns**:
                torch.Tensor: Output tensor of shape (B, M, N_out, ...).
        """
        self.check_input_shape(x)
        if self.requires_grad or self.new_value:
            self.get_freq_response()
        return self.freq_convolve(x)

    def init_param(self):
        r"""
        Initializes the delay parameters.
        """
        if self.isint:
            delay_len = torch.randint(1, self.max_len, self.size)
        else:
            delay_len = torch.rand(self.size) * self.max_len
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
        r"""
        Checks if the input dimensions are compatible with the delay parameters.

            **Args**:
                x (torch.Tensor): The input signal.
        """
        if (int(self.nfft / 2 + 1), self.size[-1]) != (x.shape[1], x.shape[2]):
            raise ValueError(
                f"parameter shape = {self.freq_response.shape} not compatible with input signal of shape = ({x.shape})."
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
        self.freq_convolve = lambda x: torch.einsum(
            "fmn,bfn...->bfm...", self.freq_response, x
        )

    def initialize_class(self):
        r"""
        Initializes the Delay module.

        This method checks the shape of the delay parameters, computes the frequency response, and initializes the frequency convolution function.
        """
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
        """
        Checks if the shape of the delay parameters is valid.
        """
        assert len(self.size) == 1, "delays must be 1D, for 2D delays use Delay module."

    def get_freq_convolve(self):
        """
        Computes the frequency convolution function.
        """
        self.freq_convolve = lambda x: torch.einsum(
            "fn,bfn...->bfn...", self.freq_response, x
        )

    def get_freq_response(self):
        """
        Computes the frequency response of the delay module.
        """
        m = self.s2sample(self.map(self.param))
        self.freq_response = (self.gamma**m) * torch.exp(
            -1j
            * torch.einsum(
                "fo, on -> fn",
                self.omega,
                m.unsqueeze(0),
            )
        )