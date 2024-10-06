import torch
import torch.nn as nn
import argparse
import os
import time
import auraloss 
import soundfile as sf
from collections import OrderedDict
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import sparsity_loss
from flamo.utils import save_audio
from flamo.functional import signal_gallery, find_onset

torch.manual_seed(130799)

class MultiResoSTFT(nn.Module):
    '''compute the mean absolute error between the auraloss of two RIRs'''
    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir1, rir2):
        return self.MRstft(rir1.permute(0, 2, 1), rir2.permute(0, 2, 1))

def example_fdn(args):
    """
    Example function that demonstrates the construction and training of a Feedback Delay Network (FDN) model.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """

    # FDN parameters
    N = 6  # number of delays
    alias_decay_db = 30  # alias decay in dB
    delay_lengths = torch.tensor([593, 743, 929, 1153, 1399, 1699])

    ## ---------------- CONSTRUCT FDN ---------------- ##

    # Input and output gains
    input_gain = dsp.Gain(
        size=(N, 1), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db, device=args.device
    )
    output_gain = dsp.Gain(
        size=(1, N), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db, device=args.device
    )
    # Feedback loop with delays
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=delay_lengths.max(),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device
    )
    delays.assign_value(delays.sample2s(delay_lengths))
    # Feedback path with orthogonal matrix
    mixing_matrix = dsp.Matrix(
        size=(N, N),
        nfft=args.nfft,
        matrix_type="orthogonal",
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device
    )
    attenuation = dsp.parallelGEQ(
        size=(N,),
        octave_interval=1,
        nfft=args.nfft,
        fs=args.samplerate,
        requires_grad=True,
        alias_decay_db=alias_decay_db,
        device=args.device
    )
    attenuation.map = lambda x : 20*torch.log10(torch.sigmoid(x))
    feedback = system.Series(OrderedDict({
        'mixing_matrix': mixing_matrix,
        'attenuation': attenuation
    }))

    # Recursion
    feedback_loop = system.Recursion(fF=delays, fB=feedback)

    # Full FDN
    FDN = system.Series(OrderedDict({
        'input_gain': input_gain,
        'feedback_loop': feedback_loop,
        'output_gain': output_gain
    }))

    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    # Since time aliasing mitigation is enabled, we use the iFFTAntiAlias layer
    # to undo the effect of the anti aliasing modulation introduced by the system's layers
    output_layer = dsp.iFFTAntiAlias(nfft=args.nfft, alias_decay_db=alias_decay_db, device=args.device)
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    # Get initial impulse response
    with torch.no_grad():
        ir_init =  model.get_time_response(identity=False, fs=args.samplerate).squeeze() 
        save_audio(os.path.join(args.train_dir, "ir_init.wav"), ir_init/torch.max(torch.abs(ir_init)), fs=args.samplerate)

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    # read target RIR
    input = signal_gallery(1, n_samples=args.nfft, n=1, signal_type='impulse', fs=args.samplerate, device=args.device)
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=torch.float32)
    target_rir = target_rir/torch.max(torch.abs(target_rir))
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset:(rir_onset+args.nfft)].view(1, -1, 1)

    dataset = Dataset(
        input=input,
        target=target_rir,
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(model, max_epochs=args.max_epochs, lr=args.lr,train_dir=args.train_dir, device=args.device)
    trainer.register_criterion(MultiResoSTFT(), 1)
    trainer.register_criterion(sparsity_loss(), 1, requires_model=True)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get optimized impulse response
    with torch.no_grad():
        ir_optim =  model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(os.path.join(args.train_dir, "ir_optim.wav"), ir_optim/torch.max(torch.abs(ir_optim)), fs=args.samplerate)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument('--num', type=int, default=100,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--masked_loss', type=bool, default=False, help='use masked loss')
    parser.add_argument('--target_rir', type=str, default='rirs/arni_35_3541_4_2.wav', help='filepath to target RIR')

    args = parser.parse_args()

    # check for compatible device 
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        
    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    example_fdn(args)

    