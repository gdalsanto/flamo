import torch
import numpy as np
torch.random.manual_seed(0)
np.random.seed(0)

def cascaded_paraunit_matrix(U: torch.tensor, 
                             n_stages: int = 3, 
                             gain_per_sample: float = 0.9999, 
                             sparsity: int = 3):
    r"""
    Creates paraunitary matrix from input othogonal matrix 
        **Args**:
            U (Tensor): The input orthogonal matrix of size (n_stages+1, N, N).
            n_stages (int): The number of stages in the paraunitary matrix.
            gain_per_sample (float): The gain per sample for homogenous attenuation.
            sparsity (int): The sparsity of the paraunitary matrix.
        **Returns**:
            Tensor: The paraunitary scattering matrix.   
    
    References:
    - Schlecht, S. J., & Habets, E. A. (2020). Scattering in feedback delay
        networks. IEEE/ACM Transactions on Audio, Speech, and Language P
        rocessing, 28, 1915-1924.
    adapted to python by: Dal Santo G. 
    """

    K = n_stages+1
    sparsity_vect = torch.ones((n_stages), device=U.device)
    sparsity_vect[0] = sparsity
    pulse_size = 1
    # check that the input matrix is of correct shape
    assert U.shape[0] == K, 'The input matrix must have n_stages+1 stages'
    assert U.shape[1] == U.shape[2], 'The input matrix must be square'

    U = U.permute(2, 0, 1)
    V = U[:,:,0]
    N = V.shape[0]

    for k in range(1,K):

        # np.random.seed(130798)
        shift_L = shift_mat_distribute(V, sparsity_vect[k-1], pulse_size)

        G = torch.diag(gain_per_sample**shift_L).to(torch.float32)
        R = torch.matmul(U[:,:,k],G)

        V = shift_matrix(V, shift_L, direction='left')
        V = poly_matrix_conv(R, V)

        pulse_size = pulse_size * N*sparsity_vect[k-1]
    
    return V.permute(2, 0, 1)

def poly_matrix_conv(A: torch.tensor, B: torch.tensor):
    ''' Multiply two matrix polynomials A and B by convolution '''
    
    if len(A.shape) == 2:
        A = A.view(A.shape[0], A.shape[1], 1)
    if len(B.shape) == 2:
        B = B.view(B.shape[0], B.shape[1], 1)

    # Get the dimensions of A and B
    szA = A.shape
    szB = B.shape

    if szA[1] != szB[0]:
        raise ValueError('Invalid matrix dimension.')

    C = torch.zeros((szA[0], szB[1], szA[2] + szB[2] - 1))

    A = A.permute(2, 0, 1)
    B = B.permute(2, 0, 1)
    C = C.permute(2, 0, 1)

    for row in range(szA[0]):
        for col in range(szB[1]):
            for it in range(szA[1]):
                C[:, row, col] = C[:, row, col] + torch.conv1d(B[:, it, col].unsqueeze(0).unsqueeze(0), A[:, row, it].unsqueeze(0).unsqueeze(0))

    C = C.permute(1, 2, 0)

    return C

def shift_matrix(X: torch.tensor, shift: torch.tensor, direction: str ='left'):
    r""" 
    Shift in polynomial matrix in time-domain by shift samples
    direction is either Left or Right
    """
    
    N = X.shape[0]
    # Find the last nonzero element indices along last dim
    if len(X.shape) == 2:
        X = X.view(X.shape[0], X.shape[1], 1)
    order = torch.max(torch.nonzero(X, as_tuple=True)[-1])
    if direction.lower() == 'left':
        required_space = order + shift.reshape(-1,1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat((X, torch.zeros((N,N,additional_space))), dim=-1)
        for i in range(N):
            X[i, :, :] = torch.roll(X[i, :, :], shift[i].item(), dims=-1)
    elif direction.lower() == 'right':
        required_space = order + shift.reshape(1,-1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat((X, torch.zeros((N,N,additional_space))), dim=-1)
        for i in range(N):
            X[:, i, :] = torch.roll(X[:, i, :], shift[i].item(), dims=-1)

    return X 

def shift_mat_distribute(X: torch.tensor, sparsity: int, pulse_size: int):
    '''shift in polynomial matrix in time-domain such that they don't overlap'''
    N = X.shape[0]
    rand_shift = torch.floor(sparsity * (torch.arange(0,N) + torch.rand((N))*0.99))
    return (rand_shift * pulse_size).int()

