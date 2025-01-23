import torch
from typing import Optional


def generate_partitions(
    tensor: torch.Tensor, n_samples: int, n_sets: int, seed: Optional[int] = None
):
    r"""
    Create :attr:`n_sets` sets of ``length(tensor) // n_samples`` partitions of a tensor,
    and the items are shuffled randomly for each set.

        **Arguments**:
            - **tensor** (torch.Tensor): The input tensor to partition.
            - **n_samples** (int): The number of samples in each partition.
            - **n_sets** (int): The number of different sets of partitions.
            - **seed** (int, optional): A seed for reproducibility. Default is None.

        **Returns**:
            list of lists of torch.Tensor: list of :attr:`n_sets` sets, where each set contains :attr:`length(tensor) // n_samples` partitions.
    """
    if seed is not None:
        torch.manual_seed(seed)

    length = len(tensor)
    n_partitions = length // n_samples

    # Ensure the tensor length is divisible by N
    if length % n_samples != 0:
        print(
            "Warning: Tensor length is divisible by n_samples so there will be some samples left out."
        )

    partitions_sets = []
    for _ in range(n_sets):
        # Shuffle the tensor
        shuffled_tensor = tensor[torch.randperm(length)]
        # Partition the tensor into N equal parts
        partitions = [
            shuffled_tensor[i * n_samples : (i + 1) * n_samples]
            for i in range(n_partitions)
        ]
        partitions_sets.append(torch.stack(partitions))

    partitions_sets = torch.vstack(partitions_sets)
    return partitions_sets
