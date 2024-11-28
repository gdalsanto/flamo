import os 
import argparse
import time

import torch 
import torch.nn as nn

import flamo.processor.dsp as dsp
from flamo.auxiliary.reverb import HomogeneousFDN
from flamo.auxiliary.config.config import HomogeneousFDNConfig
from flamo.optimize.trainer import Trainer
from flamo.optimize.dataset import load_dataset

class Dataset(torch.utils.data.Dataset):                   
    def __init__(self, expand=1, device='cpu'):
        # colorless optimization
        input_fdn = torch.zeros((1, 96000, 1))
        input_fdn[:,0,:] = 1
        input_fdn = input_fdn.expand(tuple([expand]+[d for d in input_fdn.shape[1:]]))
        self.input_fdn = input_fdn
        target = torch.ones((1, 96000 // 2 + 1, 1))
        # NN input 
        input_nn = torch.randn((expand, 10))
        self.input = []
        for i in range(expand):
            self.input.append((input_nn[i], input_fdn[i]))
        self.target = target.expand(tuple([expand]+[d for d in target.shape[1:]]))

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        return self.input[index], self.target[index]
    
class nnFDN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.fdn_config = HomogeneousFDNConfig()
        self.fdn_config.alias_decay_db = 30
        # self.fdn_config.input_gain_grad = False
        # self.fdn_config.output_gain_grad = False
        # self.fdn_config.mixing_matrix_grad = False
        # self.fdn_config.attenuation_grad = False
        FDN = HomogeneousFDN(config_dict=self.fdn_config) # we need to define which layers in the FDN require the extra parameters
        FDN.set_model(output_layer=dsp.Transform(transform=lambda x : torch.abs(x)))
        FDN.normalize_energy(target_energy = 1)

        self.fdn = FDN.model
        out_features = FDN.N
        in_features = 10
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.sigmoid = nn.Sigmoid()  # Activation function


    def forward(self, data):
        x = data[0] 
        z = data[1]
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.sigmoid(x) 
        # TODO: map the later to a dictionary that can be interpreted by the FDN model
        param_dict = {'input_gain': x.transpose(1, 0)}
        y = self.fdn(z, param_dict)
        return y

def example_nn(args):

    # create a model
    model = nnFDN()

    # create a dataset
    dataset = Dataset(expand=args.num)
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    trainer = Trainer(model, 
                      max_epochs=args.max_epochs, 
                      train_dir=args.train_dir, 
                      device=args.device)
    
    trainer.register_criterion(nn.MSELoss(), 1)
    
    trainer.train(train_loader, valid_loader)
    


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

    example_nn(args)

