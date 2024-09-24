import torch 

def get_device():
    r""" Output 'cuda' if gpu is available, 'cpu' otherwise """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_complex(x):
    r"""
    Converts a real tensor to a complex tensor.

        **Args**:
            x (torch.Tensor): The input tensor.

        **Returns**:
            torch.Tensor: The complex tensor with the same shape as the input tensor.
    """
    return torch.complex(x, torch.zeros_like(x))