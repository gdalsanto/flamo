import torch
import torch.nn as nn
from flamo.utils import to_complex

# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    r"""
    Base class for all transformations. 

    The transformation is a callable, e.g., :class:`lambda` expression, function, :class:`nn.Module`. 

        **Args**:
            transform (callable): The transformation function to be applied to the input. Default: lambda x: x
        **Attributes**:
            transform (callable): The transformation function to be applied to the input.
        **Methods**:
            forward(x): Applies the transformation function to the input.

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
            nfft (int): The number of points to compute the FFT. Default: 2**11.
            norm (str): The normalization mode for the FFT. Default: "backward".
        **Attributes**:
            nfft (int): The number of points to compute the FFT.
            norm (str): The normalization mode for the FFT.
        **Methods**:
            foward(x): Apply the FFT to the input tensor x and return the one sided FFT.

    For details on the FFT function, see `torch.fft.rfft documentation <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_.
    """

    def __init__(self, nfft=2**11, norm="backward"):
        self.nfft = nfft
        self.norm = norm
        transform = lambda x: torch.fft.rfft(x, n=self.nfft, dim=1, norm=self.norm)
        super().__init__(transform=transform)


class iFFT(Transform):
    """
    Inverse Fast Fourier Transform (iFFT) class.

    The :class:`iFFT` class is an instance of the :class:`Transform` class. The transformation function is the :func:`torch.fft.irfft` function.
    Computes the inverse of the Fourier transform of a real-valued tensor. The input is interpreted as a one-sided Hermitian signal in the Fourier domain. The output is a real-valued signal in the time domain.
    
        **Args**:
            nfft (int): The size of the FFT. Default: 2**11.
            norm (str): The normalization mode. Default: "backward".
        **Attributes**:
            nfft (int): The size of the FFT.
            norm (str): The normalization mode.
        **Methods**:
            foward(x): Apply the inverse FFT to the input tensor x and returns its corresponding real valued tensor.

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
            size (tuple): The shape of the parameters before mapping.
            nfft (int, optional): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            map (function, optional): The mapping function applied to the raw parameters. Default: lambda x: x.
            requires_grad (bool, optional): Whether the parameters require gradients. Default: False.
            alias_decay_db (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.

        **Attributes**:
            size (tuple): The shape of the parameters.
            nfft (int): The number of FFT points required to compute the frequency response.
            map (function): The mapping function applied to the raw parameters.
            requires_grad (bool): Whether the parameters require gradients.
            alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            param (nn.Parameter): The parameters of the DSP module.
            fft (function): The FFT function. Calls the :func:`torch.fft.rfft` function.
            ifft (function): The Inverse FFT function. Calls the :func:`torch.fft.irfft`.
            gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.

        **Methods**:
            forward(x): Applies the processor core module to the input tensor x.
            init_param(): Initializes the parameters of the DSP module.
            get_gamma(): Computes the gamma value used for time anti-aliasing envelope.
            assign_value(new_value, indx): Assigns new values to the parameters.
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
        """
        Calculate the gamma value based on the alias decay in dB and the number of FFT points.
        The gamma value is computed as follows and saved in the attribute :attr:`gamma`:

        .. math::

            \\gamma = 10^{\\frac{-|\\alpha_{\\text{dB}}|}{20 \\cdot \\text{nfft}}}\\; \\text{and}\\; \\gamma(n) = \\gamma^{n}

        where :math:`\\alpha_{\\text{dB}}` is the alias decay in dB, :math:`\\text{nfft}` is the number of FFT points, 
        and :math:`n` is the descrete time index :math:`0\\leq n < N`, where N is the length of the signal.
        """

        self.gamma = torch.tensor(
            10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)
        )

    def assign_value(self, new_value, indx: tuple = tuple([slice(None)])):
        """
        Assigns new values to the parameters.

        **Args**:
            new_value (torch.Tensor): New values to be assigned.
            indx (tuple, optional): Index to specify the subset of values to be assigned. Default: tuple([slice(None)]).

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
            size (tuple): The size of the gain parameters. Default: (1, 1).
            nfft (int): The number of FFT points required to compute the frequency response. Default: 2 ** 11.
            map (function): A mapping function applied to the raw parameters. Default: lambda x: x.
            requires_grad (bool): Whether the parameters requires gradients. Default: False.
            alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Default: 0.

        **Attributes**:
            size (tuple): The size of the gain parameters.
            nfft (int): The number of FFT points required to compute the frequency response.
            map (function): A mapping function applied to the raw parameters.
            requires_grad (bool): Whether the parameters requires gradients.
            alias_decay_db (float): The decaying factor in dB for the time anti-aliasing envelope.
            param (nn.Parameter): The parameters of the Gain module.
            fft (function): The FFT function. Calls the torch.fft.rfft function.
            ifft (function): The Inverse FFT function. Calls the torch.fft.irfft.
            gamma (torch.Tensor): The gamma value used for time anti-aliasing envelope.

        **Methods**:
            forward(x): Applies the Gain module to the input tensor x.
            check_input_shape(x): Checks if the dimensions of the input tensor x are compatible with the module.
            check_param_shape(): Checks if the shape of the gain parameters is valid.
            get_freq_convolve(): Computes the frequency convolution function.
            initialize_class(): Initializes the Gain module.
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
        assert len(self.size) == 2, "gains must be 2D."

    def get_freq_convolve(self):
        """
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


# ============================= FILTERS ================================

