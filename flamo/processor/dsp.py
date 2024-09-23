import torch
import torch.nn as nn

# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    r"""
    Base class for all transformations. 

    The transformation is a callable, e.g., :class:`lambda` expression, function, :class:`nn.Module`. 

        Args:
            transform (callable): The transformation function to be applied to the input. Default: lambda x: x
        Attributes:
            transform (callable): The transformation function to be applied to the input.
        Methods:
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
    
        Args:
            nfft (int): The number of points to compute the FFT. Default: 2**11.
            norm (str): The normalization mode for the FFT. Default: "backward".
        Attributes:
            nfft (int): The number of points to compute the FFT.
            norm (str): The normalization mode for the FFT.
        Methods:
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
    
        Args:
            nfft (int): The size of the FFT. Default: 2**11.
            norm (str): The normalization mode. Default: "backward".
        Attributes:
            nfft (int): The size of the FFT.
            norm (str): The normalization mode.
        Methods:
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