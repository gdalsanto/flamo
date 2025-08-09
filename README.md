# flamo
[Docs](https://gdalsanto.github.io/flamo/) | [PyPI](https://pypi.org/project/flamo/) | [ICASSP25-arXiv](https://arxiv.org/abs/2409.08723) 

Open-source library for frequency-domain differentiable audio processing.

It contains differentiable implementation of common LTI audio systems modules with learnable parameters.

---

### ⚙️ Optimization of audio LTI systems

Available differentiable audio signal processors - in `flamo.processor.dsp`: 
- **Gains** : Gains, Matrices, Householder Matrices
- **Filters** : Biquads, State Variable Filters (SVF), Graphic Equalizers (GEQ), Parametric Equiliers (PEQ - not released yet)
- **Delays** : Integer Delays, Fractional Delays 

Transforms  - in `flamo.processor.dsp`: 
- **Transform** : FFT, iFFT, time anti-aliasing enabled FFT and iFFT 


Utilities, system designers, and optimization - in `flamo.processor.system`:
- **Series** : Serial chaining of differentiable systems 
- **Recursion** : Closed loop with assignable feedforward and feedback paths
- **Shell**: Container class for safe interaction between system, dataset, and loss functions

Optimization - in `flamo.optimize`:
- **Trainer** : Handling of the training and validation steps 
- **Dataset** : Customizable dataset class and helper methods 

--- 

### 🛠️ Installation
To install it via pip, on a new python virtual environment `flamo-env` 
```shell
python3.10 -m venv .flamo-env
source .flamo-env/bin/activate
pip install flamo
```
If you are using conda, you might need to install `libsndfile` manually
```shell
conda create -n flamo-env python=3.10
conda activate flamo-env
pip install flamo
conda install -c conda-forge libsndfile
```

For local installation: clone and install dependencies on a new pyton virtual environment `flamo-env` 
```shell
git clone https://github.com/gdalsanto/flamo
cd flamo
python3.10 -m venv .flamo-env
source .flamo-env/bin/activate
pip install -e .
```
Note that it requires python>=3.10

---

### 💻 How to use the library

We included a few examples in [`./examples`](https://github.com/gdalsanto/flamo/tree/main/examples) that take you through the library's API. 

The following example demonstrates how to optimize the parameters of Biquad filters to match a target magnitude response. This is just a toy example; you can create and optimize much more complex systems by cascading modules either serially or recursively. 

Import modules 
```python
import torch
import torch.nn as nn
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.functional import signal_gallery, highpass_filter
```
Define parameters and target response with randomized cutoff frequency and gains

```python
in_ch, out_ch = 1, 2    # input and output channels
n_sections = 2  # number of cascaded biquad sections
fs = 48000      # sampling frequency
nfft = fs*2     # number of fft points

b, a = highpass_filter(
    fc=torch.tensor(fs/2)*torch.rand(size=(n_sections, out_ch, in_ch)), 
    gain=torch.tensor(-1) + (torch.tensor(2))*torch.rand(size=(n_sections, out_ch, in_ch)), 
    fs=fs)
B = torch.fft.rfft(b, nfft, dim=0)
A = torch.fft.rfft(a, nfft, dim=0)
target_filter = torch.prod(B, dim=1) / torch.prod(A, dim=1)
```

Define an instance of learnable Biquads

```python
filt = dsp.Biquad(
    size=(out_ch, in_ch), 
    n_sections=n_sections,
    filter_type='highpass',
    nfft=nfft,
    fs=fs,
    requires_grad=True,
    alias_decay_db=0,
)   
```

Use the `Shell` class to add input and output layers and to get the magnitude response at initialization 
Optimization is done in the frequency domain. The input will be an impulse in the time domain, thus the input layer should perform the Fourier transform.
The target is the magnitude response, so the output layer takes the absolute value of the filter's output.  

```python
input_layer = dsp.FFT(nfft)
output_layer = dsp.Transform(transform=lambda x : torch.abs(x))
model = system.Shell(core=filt, input_layer=input_layer, output_layer=output_layer)    
estimation_init = model.get_freq_response()
````

Set up optimization framework and launch it. The `Trainer` class is used to contain the model, training parameters, and training/valid steps in one class. 

```python
input = signal_gallery(1, n_samples=nfft, n=in_ch, signal_type='impulse', fs=fs)
target = torch.einsum('...ji,...i->...j', target_filter, input_layer(input))

dataset = Dataset(
    input=input,
    target=torch.abs(target),
    expand=100,
)
train_loader, valid_loader = load_dataset(dataset, batch_size=1)

trainer = Trainer(model, max_epochs=10, lr=1e-2, train_dir="./output")
trainer.register_criterion(nn.MSELoss(), 1)

trainer.train(train_loader, valid_loader)
```
end get the resulting response after optimization! 

```python
estimation = model.get_freq_response()
```

### 📖 Documentation 

A first version of the documentation is available on the repo's [Github Page](https://gdalsanto.github.io/flamo/). Note that we are currently working on improving the documentation to include examples, images, and a more pleasant template. 

---
### 📖 Reference

This work has been submitted to ICASSP 2025. Pre-print is available on [arxiv](https://arxiv.org/abs/2409.08723). 

```Dal Santo, G., De Bortoli, G. M., Prawda, K., Schlecht, S. J., & Välimäki, V. (2024). FLAMO: An Open-Source Library for Frequency-Domain Differentiable Audio Processing. arXiv preprint arXiv:2409.08723.```
