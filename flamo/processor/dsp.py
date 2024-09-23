import torch
import torch.nn as nn

# ============================= TRANSFORMS ================================


class Transform(nn.Module):
    r"""
    Base class for all transforms. 

    The transformation is a callable of the type, e.g., Lambda expression, a function or a nn.Module. 

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
    def __init__(self, transform=lambda x: x):
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

if __name__ == "__main__":
    # Create an instance of the Transform class
    transform = Transform(lambda x: x ** 2)

    # Create an input tensor
    input_tensor = torch.tensor([1, 2, 3])

    # Apply the transformation to the input tensor
    output_tensor = transform(input_tensor)

    # Print the transformed tensor
    print(output_tensor)