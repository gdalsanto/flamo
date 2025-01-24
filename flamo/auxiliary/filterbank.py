import scipy
import torch
import torchaudio

from torch import nn
import numpy as np

from scipy.signal import sosfreqz
from flamo.utils import get_device


class FilterBank(nn.Module):
    r"""
    Generates a filterbank and filters tensors.

    This is gpu compatible if using torch backend, but it might be slow.
    The octave filterbanks is created using cascade Buttwerworth filters, which then are processed using
    the biquad function native to PyTorch.

    This class is useful, e.g., to get the decay curves of RIRs.

    **Arguments**:
        - **fraction** (int): Fraction of octave bands (1 or 3).
        - **order** (int): Order of the Butterworth filter.
        - **fmin** (int): Minimum frequency.
        - **fmax** (int): Maximum frequency.
        - **sample_rate** (int): Sampling rate.
        - **nfft** (int, optional): Number of FFT points.
        - **backend** (str): Backend to use ('scipy' or 'torch').

    Adapted from DecayFitNet original implementation by @georg-goetz https://github.com/georg-goetz/DecayFitNet
    """

    def __init__(
        self,
        fraction: int = 3,
        order: int = 5,
        fmin: float = 20.0,
        fmax: float = 18000.0,
        sample_rate: int = 48000,
        nfft: int = None,
        backend: str = "scipy",
    ):
        super(FilterBank, self).__init__()

        assert (fraction == 1) | (
            fraction == 3
        ), "At the moment only fractions 1 and 3 are supported"

        # nominal frequencies
        nom_freq_f1 = [
            16,
            31.5,
            63,
            125,
            250,
            500,
            1000,
            2000,
            4000,
            8000,
            16000,
            32000,
        ]
        nom_freq_f3 = [
            16,
            20,
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
            16000,
            20000,
            25000,
            32000,
        ]

        if fraction == 1:
            index = [0, len(nom_freq_f1)]
            for i, f in enumerate(nom_freq_f1):
                if fmin > f:
                    index[0] = i
                    break
            for i, f in enumerate(nom_freq_f1):
                if f > fmax:
                    index[1] = i
                    break
            center_frequencies = nom_freq_f1[index[0] : index[1]]

        if fraction == 3:
            index = [0, len(nom_freq_f3)]
            for i, f in enumerate(nom_freq_f3):
                if fmin > f:
                    index[0] = i + 1
                    break
            for i, f in enumerate(nom_freq_f3):
                if f > fmax:
                    index[1] = i
                    break
            center_frequencies = nom_freq_f3[index[0] : index[1]]

        self._center_frequencies = center_frequencies
        self._order = order
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(
            center_frequencies, self._sample_rate, self._order
        )
        self._backend = backend
        self._nfft = nfft

    def _forward_scipy(self, x: torch.Tensor):
        r"""
        Forward pass using scipy backend.

        **Arguments**:
            **x** (torch.Tensor): Input tensor.

        **Returns**:
            torch.Tensor: Filtered tensor.
        """
        out = []
        for this_sos in self._sos:
            tmp = torch.clone(x).cpu().numpy()
            tmp = scipy.signal.sosfilt(this_sos, tmp, axis=-1)
            out.append(torch.from_numpy(tmp.copy()))
        out = torch.stack(out, dim=-2)  # Stack over frequency bands

        return out

    def _forward_torch(self, x, filt_type="conv"):
        r"""
        Forward pass using torch backend.

        **Arguments**:
            - **x** (torch.Tensor): Input tensor.
            - **filt_type** (str): Filter type ('conv' or 'biquad').

        **Returns**:
            torch.Tensor: Filtered tensor.
        """
        out = []
        if filt_type == "conv":
            X = torch.fft.rfft(x, axis=1)
            for i, this_freq in enumerate(self.freqz):
                this_freq = torch.tensor(this_freq, device=x.device)
                tmp = torch.fft.irfft(
                    torch.einsum("btn, t -> btn", X, this_freq), axis=1
                )
                out.append(tmp)
        else:
            for this_sos in self._sos:
                tmp = x
                for i, sos in enumerate(this_sos):
                    tmp = torchaudio.functional.biquad(
                        tmp, sos[0], sos[1], sos[2], sos[3], sos[4], sos[5]
                    )
                out.append(tmp)
        out = torch.stack(out, dim=-1)  # Stack over frequency bands
        return out

    def set_sample_rate(self, sample_rate):
        r"""
        Sets the sample rate and updates the filters.

        **Arguments**:
            **sample_rate** (int): New sample rate.
        """
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(
            self._center_frequencies, self._sample_rate, self._order
        )

    def set_order(self, order):
        r"""
        Sets the filter order and updates the filters.

        **Arguments**:
            **order** (int): New filter order.
        """
        self._order = order
        self._sos = self._get_octave_filters(
            self._center_frequencies, self._sample_rate, self._order
        )

    def set_center_frequencies(self, center_freqs):
        r"""
        Sets the center frequencies and updates the filters.

        **Arguments**:
            **center_freqs** (list): List of new center frequencies.
        """
        center_freqs_np = np.asarray(center_freqs)
        assert not np.any(center_freqs_np < 0) and not np.any(
            center_freqs_np > self._sample_rate / 2
        ), (
            "Center Frequencies must be greater than 0 and smaller than fs/2. Exceptions: exactly 0 or fs/2 "
            "will give lowpass or highpass bands"
        )
        self._center_frequencies = np.sort(center_freqs_np).tolist()
        self._sos = self._get_octave_filters(
            center_freqs, self._sample_rate, self._order
        )

    def get_center_frequencies(self):
        r"""
        Gets the center frequencies.

        **Returns**:
            list: List of center frequencies.
        """
        return self._center_frequencies

    def forward(self, x):
        r"""
        Forward pass of the filter bank.

        **Returns**:
            **x** (torch.Tensor): Input tensor.

        **Returns**:
            torch.Tensor: Filtered tensor.
        """
        if self._nfft == None:
            self._nfft = x.size(1) // 2 + 1
        self.freqz = np.zeros((len(self._sos), self._nfft), dtype=np.complex128)
        for i, sos in enumerate(self._sos):
            _, self.freqz[i, :] = sosfreqz(sos, self._nfft, fs=self._sample_rate)

        if self._backend == "scipy":
            out = self._forward_scipy(x)
        if self._backend == "torch":
            out = self._forward_torch(x)
        else:
            raise NotImplementedError(
                "No good implementation relying solely on the pytorch backend has been found yet"
            )
        return out

    def get_filterbank_impulse_response(self):
        """
        Returns the impulse response of the filterbank.

        **Returns**:
            torch.Tensor: Impulse response of the filterbank.
        """
        impulse = torch.zeros(1, self._sample_rate * 20)
        impulse[0, self._sample_rate] = 1
        response = self.forward(impulse)
        return response

    @staticmethod
    def _get_octave_filters(center_freqs, fs, order):
        """
        Design octave band filters (butterworth filter).

        **Returns**:
            - **center_freqs** (list): List of center frequencies.
            - **fs** (int): Sampling rate.
            - **order** (int): Order of the Butterworth filter.

        **Returns**:
            list: List of SOS (second order sections) representation of the filters.
        """
        sos = []
        for band_idx in range(len(center_freqs)):
            center_freq = center_freqs[band_idx]
            if abs(center_freq) < 1e-6:
                # Lowpass band below lowest octave band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                this_sos = scipy.signal.butter(
                    N=order,
                    Wn=f_cutoff,
                    fs=fs,
                    btype="lowpass",
                    analog=False,
                    output="sos",
                )
            elif abs(center_freq - fs / 2) < 1e-6:
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                this_sos = scipy.signal.butter(
                    N=order,
                    Wn=f_cutoff,
                    fs=fs,
                    btype="highpass",
                    analog=False,
                    output="sos",
                )
            else:
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                this_sos = scipy.signal.butter(
                    N=order,
                    Wn=f_cutoff,
                    fs=fs,
                    btype="bandpass",
                    analog=False,
                    output="sos",
                )

            sos.append(torch.from_numpy(this_sos))

        return sos
