import torch
import argparse
import os
import time
from collections import OrderedDict
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import mse_loss, sparsity_loss

torch.manual_seed(130709)

def example_fdn(args):

    # FDN parameters
    N = 6  # number of delays
    alias_decay_db = 30  # alias decay in dB
    delay_lengths = torch.tensor([887, 911, 941, 1699, 1951, 2053])

    ## ---------------- CONSTRUCT FDN ---------------- ##

    # input and output gains
    input_gain = dsp.Gain(
        size=(N, 1), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db
    )
    output_gain = dsp.Gain(
        size=(1, N), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db
    )
    # feedback loop with delays
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=delay_lengths.max(),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
    )
    delays.assign_value(delays.sample2s(delay_lengths))
    # feedback path with orthogonal matrix
    feedback = dsp.Matrix(
        size=(N, N),
        nfft=args.nfft,
        matrix_type="orthogonal",
        requires_grad=True,
        alias_decay_db=alias_decay_db,
    )
    # recursion
    feedback_loop = system.Recursion(fF=delays, fB=feedback)

    # full FDN
    FDN = system.Series(OrderedDict({
        'input_gain': input_gain,
        'feedback_loop': feedback_loop,
        'output_gain': output_gain
    }))

    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x : torch.abs(x))
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    dataset = DatasetColorless(
        input_shape=(args.nfft // 2 + 1, 1),
        target_shape=(args.nfft // 2 + 1, 1),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(model, max_epochs=args.max_epochs, lr=args.lr, device=args.device, train_dir=args.train_dir)
    trainer.register_criterion(mse_loss(is_masked=args.masked_loss, n_sections=args.num, nfft=args.nfft), 1)
    trainer.register_criterion(sparsity_loss(), 0.2, requires_model=True)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    with torch.no_grad():
        ir_optim =  model.get_time_response(interior=False, fs=args.samplerate).squeeze().sum(-1) # TODO implement exp_decay unwrapper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--masked_loss', type=bool, default=False, help='use masked loss')

    args = parser.parse_args()

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

    