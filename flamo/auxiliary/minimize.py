import torch
from torch import nn
from torch.optim import LBFGS


class MLS(nn.Module):
    """
    Mean Least Squares module.
    Computes the mean of the squares of the residuals, computed as

    .. math::

        \\frac{1}{n} \sum_{i=1}^{n} (Gx_i - y_i)^2

    where :math:`G` is the matrix to be multiplied with the input, :math:`x_i` is the input, and :math:`y_i` is the target tensor.

    **Arguments**:
        - **G** (torch.Tensor): The matrix to be multiplied with the input.
        - **target_interp** (torch.Tensor): The target interpolation tensor.
    """

    def __init__(self, G: torch.Tensor, target_interp: torch.Tensor):
        super().__init__()
        self.G = G
        self.target_interp = target_interp

    def forward(self, x):
        r"""
        Computes the mean least squares loss.
        """
        return torch.mean(torch.pow(torch.matmul(self.G, x) - self.target_interp, 2))


def minimize_LBFGS(
    G: torch.Tensor,
    target_interp: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    num_freq: int,
    max_iter: int = 100,
):
    """
    Minimize the mean least square (MLS) loss using the LBFGS optimizer.

    **Arguments**:
        - **G** (torch.Tensor): The matrix to be multiplied with the input.
        - **target_interp** (torch.Tensor): The target interpolation tensor.
        - **lower_bound** (torch.Tensor): Lower bound for the optimization variables.
        - **upper_bound** (torch.Tensor): Upper bound for the optimization variables.
        - **num_freq** (int): Number of frequencies.
        - **max_iter** (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        torch.nn.Parameter: The optimized result.
    """
    initial_guess = nn.Parameter(torch.ones(num_freq + 1, device=G.device))
    assert (
        len(lower_bound) == len(upper_bound) == len(initial_guess)
    ), "The number of bounds must be equal to the number of gains."

    # Create an instance of LBFGS optimizer
    optimizer = LBFGS([initial_guess])
    criterion = MLS(G, target_interp)

    # Define a closure for the LBFGS optimizer
    def closure():
        optimizer.zero_grad()
        loss = criterion(initial_guess)
        loss.backward()
        initial_guess.data.clamp_(lower_bound, upper_bound)
        return loss

    # Perform optimization
    for i in range(max_iter):
        optimizer.step(closure)

    # Get the optimized result
    return initial_guess
