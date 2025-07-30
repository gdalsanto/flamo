import torch
from flamo.processor.dsp import conv_sparse
import matplotlib.pyplot as plt

def test_conv_sparse_plot():
    x = torch.tensor([1., -0.5, 0.3])
    k = torch.tensor([0, 5, 10])
    g = torch.tensor([1., -1., 0.5])
    vL = 4
    y = conv_sparse(x, k, g, vL)

    plt.figure(figsize=(10, 3))
    plt.stem(y.cpu().numpy())  
    plt.title("Output of conv_sparse")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()