import torch
from torch import nn
import numpy as np
from typing import Optional
from flamo.utils import to_complex

torch.random.manual_seed(0)
np.random.seed(0)


class ScatteringMapping(nn.Module):
    r"""
    Class mapping an orthogonal matrix to a paraunitary matrix using sparse scattering.

    It is parameterized as a set of :math:`K` orthogonal mixing matrices (input) :math:`\mathbf{U}_k` each followed by a set of parallel delays.

    .. math::

        \mathbf{U}(z) = \mathbf{D}_{\mathbf{m}_{K+1}}(z)\mathbf{U}_K\cdots\mathbf{U}_2\mathbf{D}_{\mathbf{m}_2}(z)\mathbf{U}_1\mathbf{D}_{\mathbf{m}_1}(z)\mathbf{U}_0\mathbf{D}_{\mathbf{m}_0}(z),


    where :math:`\mathbf{U}_0, \dots, \mathbf{U}_K` are :math:`N \times N` orthogonal matrices and :math:`\mathbf{m}_0, \dots, \mathbf{m}_{K+1}` are vectors of :math:`N` integer delays.
    This parameterization ensures that the scattering matrix is paraunitary and lossless.

    For more details, refer to the paper `Scattering in Feedback Delay Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9113451>`_ by Schlecht, S. J. et al.
    Adapted to python by: Dal Santo G.

    **Arguemnts**
        - **N** (int): The size of the orthogonal matrix.
        - **n_stages** (int): The number of stages in the paraunitary matrix. Defaults to 3.
        - **sparsity** (int): The sparsity of the generated FIR filters. Defaults to 3.
        - **gain_per_sample** (float): The gain per sample for homogenous attenuation. Defaults to 0.9999.
        - **pulse_size** (int): The size of the pulse. Defaults to 1.
        - **m_L** (Tensor): The left shift :math:`\mathbf{m}_{K+1}`. Defaults to None.
        - **m_R** (Tensor): The right shift :math:`\mathbf{m}_{0}`. Defaults to None.
        - **device** (str): The device to run the calculations on. Defaults to 'cpu'.

    """

    def __init__(
        self,
        N: int,
        n_stages: int = 3,
        sparsity: int = 3,
        gain_per_sample: float = 0.9999,
        pulse_size: int = 1,
        m_L: Optional[torch.tensor] = None,
        m_R: Optional[torch.tensor] = None,
        device: str = "cpu",
    ):
        super(ScatteringMapping, self).__init__()

        self.n_stages = n_stages
        self.sparsity = sparsity
        self.gain_per_sample = gain_per_sample
        if m_L is None:
            self.m_L = torch.zeros(N, device=device)
        else:
            self.m_L = m_L
        if m_R is None:
            self.m_R = torch.zeros(N, device=device)
        else:
            self.m_R = m_R
        self.sparsity_vect = torch.ones((n_stages), device=device)
        self.sparsity_vect[0] = sparsity
        self.shifts = get_random_shifts(N, self.sparsity_vect, pulse_size)

    def forward(self, U):
        r"""
        Forward pass of the scattering mapping.
        """
        K = self.n_stages + 1
        # check that the input matrix is of correct shape
        assert U.shape[0] == K, "The input matrix must have n_stages+1 stages"
        assert U.shape[1] == U.shape[2], "The input matrix must be square"

        U = U.permute(1, 2, 0)
        V = U[:, :, 0]

        for k in range(1, K):

            G = (
                torch.diag(self.gain_per_sample ** self.shifts[k - 1, :])
                .to(torch.float32)
                .to(U.device)
            )
            R = torch.matmul(U[:, :, k], G)

            V = shift_matrix(V, self.shifts[k - 1, :], direction="left")
            V = poly_matrix_conv(R, V)

        V = shift_matrix(V, self.m_L, direction="left")
        V = shift_matrix(V, self.m_R, direction="right")

        return V.permute(2, 0, 1)


def cascaded_paraunit_matrix(
    U: torch.tensor,
    n_stages: int = 3,
    gain_per_sample: float = 0.9999,
    sparsity: int = 3,
    pulse_size: int = 1,
    m_L: Optional[torch.tensor] = None,
    m_R: Optional[torch.tensor] = None,
):
    r"""
    Creates paraunitary matrix from input orthogonal matrix.
    For details refer to :class:`flamo.auxiliary.scattering.ScatteringMapping`

    **Arguments**:
        - **U** (Tensor): The input orthogonal matrix of size (n_stages+1, N, N).
        - **n_stages** (int): The number of stages in the paraunitary matrix.
        - **gain_per_sample** (float): The gain per sample for homogenous attenuation.
        - **sparsity** (int): The sparsity.
        - **m_L** (Tensor): The left shift.
        - **m_R** (Tensor): The right shift.

    **Returns**:
        Tensor: The paraunitary scattering matrix.

    """

    K = n_stages + 1
    sparsity_vect = torch.ones((n_stages), device=U.device)
    sparsity_vect[0] = sparsity
    # check that the input matrix is of correct shape
    assert U.shape[0] == K, "The input matrix must have n_stages+1 stages"
    assert U.shape[1] == U.shape[2], "The input matrix must be square"

    U = U.permute(2, 0, 1)
    V = U[:, :, 0]
    N = V.shape[0]

    if m_L is None:
        m_L = torch.zeros(N, device=U.device)
    if m_R is None:
        m_R = torch.zeros(N, device=U.device)

    shift_L = get_random_shifts(N, sparsity_vect, pulse_size)
    for k in range(1, K):

        G = torch.diag(gain_per_sample ** shift_L[k - 1, :]).to(torch.float32)
        R = torch.matmul(U[:, :, k], G)

        V = shift_matrix(V, shift_L[k - 1, :], direction="left")
        V = poly_matrix_conv(R, V)

    V = shift_matrix(V, m_L, direction="left")
    V = shift_matrix(V, m_R, direction="right")
    V = to_complex(V)

    return V.permute(2, 0, 1)


def poly_matrix_conv(A: torch.tensor, B: torch.tensor):
    r"""Multiply two matrix polynomials A and B by convolution"""

    if len(A.shape) == 2:
        A = A.view(A.shape[0], A.shape[1], 1)
    if len(B.shape) == 2:
        B = B.view(B.shape[0], B.shape[1], 1)

    # Get the dimensions of A and B
    szA = A.shape
    szB = B.shape

    if szA[1] != szB[0]:
        raise ValueError("Invalid matrix dimension.")

    C = torch.zeros((szA[0], szB[1], szA[2] + szB[2] - 1), device=A.device)

    A = A.permute(2, 0, 1)
    B = B.permute(2, 0, 1)
    C = C.permute(2, 0, 1)

    for row in range(szA[0]):
        for col in range(szB[1]):
            for it in range(szA[1]):
                C[:, row, col] = C[:, row, col] + torch.conv1d(
                    B[:, it, col].unsqueeze(0).unsqueeze(0),
                    A[:, row, it].unsqueeze(0).unsqueeze(0),
                )

    C = C.permute(1, 2, 0)

    return C


def shift_matrix(X: torch.tensor, shift: torch.tensor, direction: str = "left"):
    r"""
    Shift in polynomial matrix in time-domain by shift samples
    direction is either Left or Right
    """

    N = X.shape[0]
    # Find the last nonzero element indices along last dim
    if len(X.shape) == 2:
        X = X.view(X.shape[0], X.shape[1], 1)
    order = torch.max(torch.nonzero(X, as_tuple=True)[-1])
    if direction.lower() == "left":
        required_space = order + shift.reshape(-1, 1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat(
            (X, torch.zeros((N, N, additional_space), device=shift.device)), dim=-1
        )
        for i in range(N):
            X[i, :, :] = torch.roll(X[i, :, :], int(shift[i].item()), dims=-1)
    elif direction.lower() == "right":
        required_space = order + shift.reshape(1, -1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat(
            (X, torch.zeros((N, N, additional_space), device=shift.device)), dim=-1
        )
        for i in range(N):
            X[:, i, :] = torch.roll(X[:, i, :], int(shift[i].item()), dims=-1)

    return X


def shift_mat_distribute(X: torch.tensor, sparsity: int, pulse_size: int):
    """shift in polynomial matrix in time-domain such that they don't overlap"""
    N = X.shape[0]
    rand_shift = torch.floor(
        sparsity * (torch.arange(0, N) + torch.rand((N), device=sparsity.device) * 0.99)
    )

    return (rand_shift * pulse_size).int()


def get_random_shifts(N, sparsity_vect, pulse_size):
    rand_shift = torch.zeros(sparsity_vect.shape[0], N, device=sparsity_vect.device)
    for k in range(sparsity_vect.shape[0]):
        temp = torch.floor(
            sparsity_vect[k]
            * (
                torch.arange(0, N, device=sparsity_vect.device)
                + torch.rand((N), device=sparsity_vect.device) * 0.99
            )
        )
        rand_shift[k, :] = (temp * pulse_size).int()
        pulse_size = pulse_size * N * sparsity_vect[k]
    return rand_shift


def hadamard_matrix(N):
    """Generate a hadamard matrix of size N"""
    X = np.array([[1]]) 
    # Create a Hadamard matrix of the specified order
    while X.shape[0] < N:
        # Kronecker product to generate a larger Hadamard matrix
        X = np.kron(X, np.array([[1, 1], [1, -1]])) / np.sqrt(2)
    return X


if __name__ == "__main__":
    matrix = (
        torch.tensor(hadamard_matrix(4), dtype=torch.float32)
        .unsqueeze(0)
        .expand(4, 4, 4)
    )
    V = cascaded_paraunit_matrix(matrix)
    # plot the matrix
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    V = V.reshape(16, -1)
    for i in range(16):
        ax = axs[i // 4, i % 4]
        ax.plot(V[i, :].squeeze().numpy())
        ax.set_title(f"Subplot {i+1}")
    plt.show()

    plt.figure()
    for i in range(16):
        plt.plot(V[i, :].squeeze().numpy(), alpha=0.5)
    plt.show()
