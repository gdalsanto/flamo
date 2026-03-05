import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from flamo.functional import skew_matrix
from flamo.processor import dsp, system
from flamo.utils import get_frequency_samples


np.random.seed(42)
Fs = 48000
nfft = 2**15

## ---------------- SAMPLE PARAMETERS ---------------- ##

N = 4
m = torch.tensor([101, 157, 211, 263], dtype=torch.float32)
g = 0.999
A = torch.matrix_exp(skew_matrix(torch.randn(N, N))) @ torch.diag(g**m)
b = torch.randn(N, 1)
c = torch.randn(1, N)
d = torch.tensor([[0.1]])


alias_decay_db = 0  # alias decay in dB


## ---------------- CONSTRUCT FDN ---------------- ##

# Input and output gains
input_gain = dsp.Gain(
    size=(N, 1),
    nfft=nfft,
    requires_grad=True,
    alias_decay_db=alias_decay_db,
)
input_gain.assign_value(b)
output_gain = dsp.Gain(
    size=(1, N),
    nfft=nfft,
    requires_grad=True,
    alias_decay_db=alias_decay_db,
)
output_gain.assign_value(c)
# Feedforward with delays only
delays = dsp.parallelDelay(
    size=(N,),
    max_len=int(m.max()),
    nfft=nfft,
    isint=True,
    requires_grad=False,
    alias_decay_db=alias_decay_db,
)
delays.assign_value(delays.sample2s(m))
# Feedback path with orthogonal matrix
mixing_matrix = dsp.Matrix(
    size=(N, N),
    nfft=nfft,
    matrix_type="orthogonal",
    requires_grad=True,
    alias_decay_db=alias_decay_db,
)
mixing_matrix.map = lambda x: x
mixing_matrix.assign_value(A)
# Recursion
feedback_loop = system.Recursion(fF=delays, fB=mixing_matrix)

# Full FDN
FDN = system.Series(
    OrderedDict(
        {
            "input_gain": input_gain,
            "feedback_loop": feedback_loop,
            "output_gain": output_gain,
        }
    )
)

# Create the model with Shell
input_layer = dsp.FFT(nfft)
output_layer = dsp.Transform(transform=lambda x: torch.abs(x))
model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)


# Method 1: FLAMO probe on unit circle
# Call `model.probe(z)` at each z = exp(2πjk/nfft) to get H(z) directly.

n_freq = nfft // 2 + 1
h_probe = np.zeros(n_freq, dtype=np.complex64)
all_z = []
for k in range(n_freq):
    z = np.exp(2j * np.pi * k / nfft)
    zt = torch.tensor(z, dtype=torch.complex64)
    with torch.no_grad():
        H_k = model.probe(zt)
    h_probe[k] = H_k.squeeze().cpu().numpy()
    all_z.append(z)

w = np.linspace(0, np.pi, n_freq)
print(f"H_probe shape: {h_probe.shape}")

# Method 2: FLAMO core forward
# Extract the core (bypassing the Shell), then pass an array of ones `(1, nfft//2+1, 1)` through the core.
# The core operates in frequency domain, so ones = unit amplitude at each bin; the output is H(ω) directly.

core = model.get_core()
x_ones = torch.ones(1, n_freq, 1, dtype=torch.complex64)
with torch.no_grad():
    h_core_forward = core(x_ones)
h_flamo_forward = h_core_forward.squeeze().cpu().numpy()

print(f"h_flamo_forward shape: {h_flamo_forward.shape}")


# Method 3: Probe all the frequency at ones 
zs = get_frequency_samples(n_freq, device="cpu", dtype=torch.complex64)
zs = zs.unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    h_flamo_batched = model.probe(zs).squeeze().cpu().numpy()
# Compare results

f_hz = w / np.pi * (Fs / 2)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(f_hz, 20 * np.log10(np.abs(h_probe) + 1e-12), label="FLAMO probe (unit circle)")
axes[0].plot(f_hz, 20 * np.log10(np.abs(h_flamo_forward) + 1e-12), ls="--", alpha=0.8, label="FLAMO core forward")
axes[0].plot(f_hz, 20 * np.log10(np.abs(h_flamo_batched) + 1e-12), ls=":", alpha=0.8, label="FLAMO probe (all frequencies)")
axes[0].set_ylabel("Magnitude [dB]")
axes[0].set_title("Frequency response")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

diff_mag = np.abs(h_probe - h_flamo_forward)
diff_mag_batched = np.abs(h_probe - h_flamo_batched)
axes[1].semilogy(f_hz, diff_mag)
axes[1].semilogy(f_hz, diff_mag_batched, ls=":", alpha=0.8)
axes[1].set_ylabel("|H_probe - H_forward|")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_title("Difference between methods")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Assert match

max_diff = np.max(np.abs(h_probe - h_flamo_forward))
rel_diff = np.max(np.abs(h_probe - h_flamo_forward) / (np.abs(h_probe) + 1e-12))

print(f"Max absolute difference: {max_diff:.2e}")
print(f"Max relative difference: {rel_diff:.2e}")

assert max_diff < 5e-3, "FLAMO probe and FLAMO core forward should match"
print("\n✓ Both methods give the same result.")
