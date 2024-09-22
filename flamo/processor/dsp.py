import torch
import torch.nn as nn

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
