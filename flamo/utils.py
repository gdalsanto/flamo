import torch 

def get_device():
    r""" Output 'cuda' if gpu is available, 'cpu' otherwise """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
