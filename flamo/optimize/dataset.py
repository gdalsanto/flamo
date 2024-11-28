import torch
import torch.utils.data as data
from flamo.utils import get_device

# ============================= DATASETS ================================

class Dataset(torch.utils.data.Dataset):                   
    r"""
    Base Dataset class. Inherits from :class:`torch.utils.data.Dataset`.
    In simple gradient descent optimization of differentiable DSP, both the input 
    and target are represented by individual tensors. Therefore, the dataset 
    consists of a single input tensor and a single target tensor. To accommodate 
    larger datasets, the input and target tensors can be expanded to the desired
    dataset length using the :attr:`expand` attribute. This enables performing 
    multiple optimization steps within a single epoch and allows for batch sizes greater than 1.
        
        **Args**:
            input (torch.Tensor, optional): The input data tensor. Default: torch.randn(100, 100).
            target (torch.Tensor, optional): The target data tensor. Default: torch.randn(100, 100).
            expand (int): The first shape dimention of the input and target tensor after expansion. Default: 1. This coincides with the length on the dataset.
            device (str, optional): The device to store the tensors on. Default: 'cpu'.
        
        **Attributes**:
            input (torch.Tensor): The input data tensor.
            target (torch.Tensor): The target data tensor.
    """
    def __init__(self, input=torch.randn(100, 100), target=torch.randn(100, 100), expand=1, device='cpu'):
        self.input = input.to(device)
        self.target = target.to(device)
        self.input = self.input.expand(tuple([expand]+[d for d in input.shape[1:]]))
        self.target = self.target.expand(tuple([expand]+[d for d in target.shape[1:]]))

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        return self.input[index], self.target[index]

    
class DatasetColorless(Dataset):
    r"""
    A dataset class for colorless optimization. Inherits from :class:`Dataset`.
    The input tensor is an impulse of length :attr:`input_shape`.
    The target is an instance of torch.ones with the shape :attr:`target_shape`, 
    corresponding to a flat magnitude spectrum.

        **Args**:
            input_shape (tuple): The shape of the input data.
            target_shape (tuple): The shape of the target data.
            expand (int, optional): The number of times to expand the dataset. Defaults to 1000.
            device (str, optional): The device to use for computation. Defaults to 'cpu'.

        **Attributes**:
            input (torch.Tensor): The input impulse tensor.
            target (torch.Tensor): The spectrally flat target tensor.
    
    For details on the colorless optimization, see `<https://arxiv.org/abs/2402.11216v2>`_.
    """

    def __init__(self, input_shape, target_shape, expand=1000, device='cpu'):
        input = torch.zeros(input_shape)
        input[:,0,:] = 1
        target = torch.ones(target_shape)
        super().__init__(input=input, target=target, expand=expand, device=device)

    def __getitem__(self, index):
        return self.input[index], self.target[index]    

# ============================= UTILS ================================

def get_dataloader(dataset, batch_size=2000, shuffle=True):
    r"""
    Create a torch dataloader (:class:`torch.utils.data.DataLoader`) from the given dataset.

        **Args**:
            dataset (torch.utils.data.Dataset): The dataset to create the dataloader from.
            batch_size (int, optional): The number of samples per batch to load. Default: 2000.
            shuffle (bool, optional): Whether to shuffle the dataset before each epoch. Default: True.
            :return: The torch dataloader.

        **Returns**:
            torch.utils.data.DataLoader: The dataloader.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last = True
    )
    return dataloader

def split_dataset(dataset, split):
    r"""
    Splits the given dataset into a training set and a validation set based on the specified split ratio.

        **Args**:
            dataset (torch.utils.data.Dataset): The dataset to be split.
            split (float): The ratio of the training set size to the total dataset size.

        **Returns**:
            tuple: A tuple containing the training set and the validation set.
    """

    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    return train_set, valid_set


def load_dataset(dataset, batch_size=2000, split=0.8, shuffle=True):
    r"""
    Load and split a dataset into training and validation sets.

        **Args**:
            dataset (torch.utils.data.Dataset): The dataset to be loaded.
            batch_size (int, optional): The batch size for the data loaders. Defaults to 2000.
            split (float, optional): The ratio to split the dataset into training and validation sets. Defaults to 0.8.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        **Returns**:
            tuple: A tuple containing the training and validation data loaders.
    """
    
    train_set, valid_set = split_dataset(
        dataset, split)

    train_loader = get_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle = shuffle,
    )
    
    valid_loader = get_dataloader(
        valid_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader, valid_loader 
    