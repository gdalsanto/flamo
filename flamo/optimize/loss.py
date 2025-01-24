import torch
import torch.nn as nn
import numpy as np
from flamo.auxiliary.filterbank import FilterBank
from flamo.optimize.utils import generate_partitions


# wrapper for the sparsity loss
class sparsity_loss(nn.Module):
    r"""
    Calculates the sparsity loss for a given model.

    The sparsity loss is calculated based on the feedback loop of the FDN model's core.
    It measures the sparsity of the feedback matrix A of size (N, N).
    Note: for the loss to be compatible with the class :class:`flamo.optimize.trainer.Trainer`, it requires :attr:`y_pred` and :attr:`y_target` as arguments even if these are not being considered.
    If the feedback matrix has a third dimension C, A.size = (C, N, N), the loss is calculated as the mean of the contribution of each (N,N) matrix.

    .. math::

        \mathcal{L} = \frac{\sum_{i,j} |A_{i,j}| - N\sqrt{N}}{N(1 - \sqrt{N})}

    For more details, refer to the paper `Optimizing Tiny Colorless Feedback Delay Networks <https://arxiv.org/abs/2402.11216>`_ by Dal Santo, G. et al.

    **Arguments**:
        - **y_pred** (torch.Tensor): The predicted output.
        - **y_target** (torch.Tensor): The target output.
        - **model** (nn.Module): The model containing the core with the feedback loop.

    Returns:
        torch.Tensor: The calculated sparsity loss.
    """

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: nn.Module):
        core = model.get_core()
        try:
            A = core.feedback_loop.feedback.map(core.feedback_loop.feedback.param)
        except:
            A = core.feedback_loop.feedback.mixing_matrix.map(
                core.feedback_loop.feedback.mixing_matrix.param
            )
        N = A.shape[-1]
        if len(A.shape) == 3:
            return torch.mean(
                (torch.sum(torch.abs(A), dim=(-2, -1)) - N * np.sqrt(N))
                / (N * (1 - np.sqrt(N)))
            )
        # A = torch.matrix_exp(skew_matrix(A))
        return -(torch.sum(torch.abs(A)) - N * np.sqrt(N)) / (N * (np.sqrt(N) - 1))


class mse_loss(nn.Module):
    r"""
    Wrapper for the mean squared error loss.

    .. math::

        \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( y_{\text{pred},i} -  y_{\text{true},i} \right)^2

    where :math:`N` is the number of nfft points and :math:`M` is the number of channels.

    **Arguments / Attributes**:
        - **nfft** (int): Number of FFT points.
        - **device** (str): Device to run the calculations on.

    """

    def __init__(self, nfft: int = None, device: str = "cpu"):

        super().__init__()
        self.nfft = nfft
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        """
        Calculates the mean squared error loss.
        If :attr:`is_masked` is set to True, the loss is calculated using a masked version of the predicted output. This option is useful to introduce stochasticity, as the mask is generated randomly.

        **Arguments**:
            - **y_pred** (torch.Tensor): The predicted output.
            - **y_true** (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The calculated MSE loss.
        """
        y_pred_sum = torch.sum(y_pred, dim=-1)
        return self.mse_loss(y_pred_sum, y_true.squeeze(-1))


class masked_mse_loss(nn.Module):
    r"""
    Wrapper for the mean squared error loss with random masking.

    Calculates the mean squared error loss between the predicted and target outputs.
    The loss is calculated using a masked version of the predicted output. This option is useful to introduce stochasticity, as the mask is generated randomly.

    .. math::

        \mathcal{L} = \frac{1}{\left| \mathbb{S} \right|} \sum_{i \in \mathbb{S}} \left( y_{\text{pred}, i} - y_{\text{true},i} \right)^2

    where :math:`\mathbb{S}` is the set of indices of the mask being analyzed during the training step.

    **Arguments / Attributes**:
        - **nfft** (int): Number of FFT points.
        - **n_samples** (int): Number of samples for masking.
        - **n_sets** (int): Number of sets for masking. Default is 1.
        - **regenerate_mask** (bool): After all sets are used, if True, the mask is regenerated. Default is True.
        - **device** (str): Device to run the calculations on. Default is 'cpu'.
    """

    def __init__(
        self,
        nfft: int,
        n_samples: int,
        n_sets: int = 1,
        regenerate_mask: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_samples = n_samples
        self.n_sets = n_sets
        self.nfft = nfft
        self.regenerate_mask = regenerate_mask
        self.mask_indices = generate_partitions(
            torch.arange(self.nfft // 2 + 1), n_samples, n_sets
        )
        self.i = -1

    def forward(self, y_pred, y_true):
        """
        Calculates the masked mean squared error loss.

        **Arguments**:
            - **y_pred** (torch.Tensor): The predicted output.
            - **y_true** (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The calculated masked MSE loss.
        """
        self.i += 1
        # generate random mask for sparse sampling
        if self.i >= self.mask_indices.shape[0]:
            self.i = 0
            if self.regenerate_mask:
                # generate a new mask
                self.mask_indices = generate_partitions(
                    torch.arange(self.nfft // 2 + 1), self.n_samples, self.n_sets
                )
        mask = self.mask_indices[self.i]
        return torch.mean(torch.pow(y_pred[:, mask] - y_true[:, mask], 2))
