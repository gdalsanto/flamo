import torch
import argparse
from collections import OrderedDict
from flamo.optimize.dataset import DatasetColorless
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=96000)
    args = parser.parse_args()
    example_fdn(args)

    