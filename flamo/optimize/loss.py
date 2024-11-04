import torch 
import torch.nn as nn 
import numpy as np

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
        return -(torch.sum(torch.abs(A)) - N)/(N*(np.sqrt(N)-1))
    
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

class amse_loss(nn.Module):
    '''Asymmetric Means squared error between abs(x1) and x2'''
    def forward(self, y_pred, y_true):

        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = self.p_loss(y_pred_sum, y_true)*torch.sqrt(torch.tensor(y_pred.size(0)))

        return loss
    
    def p_loss(self, y_pred, y_true):
        gT = 2*torch.ones((y_pred.size(0),y_pred.size(1)))
        gT = gT + 2*torch.gt((torch.abs(y_pred) - torch.abs(y_true.squeeze(-1))),0).type(torch.uint8)
        loss = torch.mean(torch.pow(torch.abs(y_pred)-torch.abs(y_true.squeeze(-1)),gT))   

        return loss  