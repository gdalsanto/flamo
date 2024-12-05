import torch 
import torch.nn as nn 
import numpy as np
from flamo.auxiliary.filterbank import FilterBank
from flamo.optimize.utils import generate_partitions

# wrapper for the sparsity loss
class sparsity_loss(nn.Module):
    """Calculates the sparsity loss for a given model."""
    def forward(self, y_pred, y_target, model):
        core = model.get_core()
        try: 
            A = core.feedback_loop.feedback.map(core.feedback_loop.feedback.param)
        except:
            A = core.feedback_loop.feedback.mixing_matrix.map(core.feedback_loop.feedback.mixing_matrix.param)
        N = A.shape[-1]
        # A = torch.matrix_exp(skew_matrix(A))
        return -(torch.sum(torch.abs(A)) - N*np.sqrt(N))/(N*(np.sqrt(N)-1))
    
class mse_loss(nn.Module):
    '''Means squared error between abs(x1) and x2'''
    def __init__(self, is_masked=False, nfft=None, n_sections=10, device='cpu'):
        super().__init__()
        self.is_masked = is_masked
        self.n_sections = n_sections    
        self.nfft = nfft
        self.device = device
        self.mask_indices = torch.chunk(torch.arange(0, self.nfft//2+1, device=device), self.n_sections)
        self.i = -1

        # create 
    def forward(self, y_pred, y_true):
        self.i += 1
        N = y_pred.size(dim=-1)
        y_pred_sum = torch.sum(y_pred, dim=-1)
        # generate random mask for sparse sampling 
        if self.is_masked:
            self.i = self.i % self.n_sections
            return torch.mean(torch.pow(torch.abs(y_pred_sum[:,self.mask_indices[self.i]])-torch.abs(y_true.squeeze(-1)[:,self.mask_indices[self.i]]), 2*torch.ones(y_pred[:,self.mask_indices[self.i]].size(1), device=self.device))) 
        else:
            return torch.mean(torch.pow(torch.abs(y_pred_sum)-torch.abs(y_true.squeeze(-1)), 2*torch.ones(y_pred.size(1), device=self.device))) 

class masked_mse_loss(nn.Module):
    '''Means squared error between abs(x1) and x2'''
    def __init__(self, nfft, n_samples, n_sets=1, device='cpu'):
        super().__init__()
        self.device = device
        self.n_samples = n_samples
        self.n_sets = n_sets
        self.nfft = nfft
        self.mask_indices = generate_partitions(torch.arange(self.nfft//2+1), n_samples, n_sets)
        self.i = -1

        # create 
    def forward(self, y_pred, y_true):
        self.i += 1
        # generate random mask for sparse sampling 
        if self.i >= self.mask_indices.shape[0]:
            # generate a new set of mask inddices 
            self.mask_indices = generate_partitions(torch.arange(self.nfft//2+1), self.n_samples, self.n_sets)
            self.i = -1
        return torch.mean((torch.abs(y_pred[:,self.mask_indices[self.i], :])-torch.abs(y_true[:,self.mask_indices[self.i], :]))**2)

class amse_loss(nn.Module):
    '''Asymmetric Means squared error between abs(x1) and x2'''
    def forward(self, y_pred, y_true):

        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = self.p_loss(y_pred_sum, y_true) # *torch.sqrt(torch.tensor(y_pred.size(0)))

        return loss
    
    def p_loss(self, y_pred, y_true):
        gT = 2*torch.ones((y_pred.size(0),y_pred.size(1)))
        gT = gT + 2*torch.gt((torch.abs(y_pred) - torch.abs(y_true.squeeze(-1))),1).type(torch.uint8)
        loss = torch.mean(torch.pow(torch.abs(y_pred)-torch.abs(y_true.squeeze(-1)),gT))   

        return loss  
    

class edc_loss(nn.Module):
    '''compute the loss on energy decay curves of two RIRs'''
    def __init__(self, sample_rate=48000, nfft=None, is_broadband=False, normalize=False, convergence=False):
        super().__init__()   
        self.sample_rate = sample_rate 
        self.is_broadband = is_broadband
        self.normalize = normalize
        self.convergence = convergence
        self.FilterBank = FilterBank(fraction=1,
                                     order = 5,
                                     fmin = 60,
                                     fmax = 15000,
                                     sample_rate= self.sample_rate, 
                                     backend='torch',
                                     nfft=nfft)
        self.mse = nn.MSELoss(reduction='mean')

    def filterbank(self, x):
        if self.is_broadband:
            return x
        else: 
            return self.FilterBank(x)

    def discard_last_n_percent(self, x, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * x.shape[1]))
        out = x[:, 0:last_id, :]

        return out
    
    def schroeder_backward_int(self, x, normalize=False):

        # Backwards integral
        out = torch.flip(x, dims=[1])
        out = torch.cumsum(out ** 2, dim=1)
        out = torch.flip(out, dims=[1])

        # Normalize to 1
        if normalize:
            norm_vals = torch.max(out, dim=1, keepdim=True)[0]  # per channel
        else: 
            norm_vals = torch.ones((x.shape[0], 1, *x.shape[2:]), device=x.device)

        out = out / norm_vals

        return out, norm_vals

    def get_edc(self, x):
        # Remove filtering artefacts (last 5 permille)
        out = self.discard_last_n_percent(x, 0.5)
        # compute EDCs
        out = self.schroeder_backward_int(self.filterbank(out))[0]
        # get energy in dB
        out = 10*torch.log10(out + 1e-32)

        return out 

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        assert (y_pred.shape == y_true.shape) & (len( y_true.shape) == 3), 'y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)'
        # compute the edcs
        y_pred_edc = self.get_edc(y_pred)
        y_true_edc = self.get_edc(y_true)

        if self.normalize: 
            norm_vals_pred = torch.max(y_pred_edc, dim=1, keepdim=True)[0]
            norm_vals_true = torch.max(y_true_edc, dim=1, keepdim=True)[0]
            y_pred_edc = y_pred_edc - norm_vals_pred
            y_true_edc = y_true_edc - norm_vals_true
            norm_vals = {
                'pred': norm_vals_pred,
                'true': norm_vals_true
            }
        else: 
            norm_vals = None
        
        
        # compute normalized mean squared error on the EDCs 
        num = self.mse(y_pred_edc, y_true_edc)
        den = torch.mean(torch.pow(y_true_edc, 2))
        if self.convergence:
            return  num / den
        else:
            return num 
    