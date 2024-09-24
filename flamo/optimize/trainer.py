import torch
import os
import time
from tqdm import trange

class Trainer:                      
    def __init__(self, net, max_epochs=10, lr=1e-3, patience=5, patience_delta=0.01, early_stop=10, train_dir=None, device='cpu'):
        r"""
        Trainer class for training a neural network model.
        It uses :meth:`torch.optim.Adam` as the optimizer.

        **Args**:
            net (nn.Module): The neural network model to be trained.
            max_epochs (int): Maximum number of training epochs. Default: 10.
            lr (float): Learning rate for the optimizer. Default: 1e-3.
            patience (int): Number of epochs to wait for improvement in validation loss before early stopping. Default: 5.
            patience_delta (float): Minimum improvement in validation loss to be considered as an improvement. Default: 0.01.
            early_stop (int): Number of epochs to wait for improvement in validation loss before terminating training (default: 10).
            device (str): Device to use for training (default: 'cpu').
        **Attributes**:
            device (str): Device to use for training.
            net (nn.Module): The neural network model.
            max_epochs (int): Maximum number of training epochs.
            lr (float): Learning rate for the optimizer.
            patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
            early_stop (int): Number of epochs to wait for improvement in validation loss before terminating training.
            optimizer (torch.optim.Optimizer): The optimizer.
            train_dir (str): The directory for saving training outputs.
            criterion (list): List of loss functions.
            alpha (list): List of weights for the loss functions.
            requires_model (list): List of flags indicating whether the loss functions require the model as an input.
            scheduler (torch.optim.lr_scheduler.StepLR): The learning rate scheduler.    
        **Methods**:
            register_criterion(criterion, alpha, requires_model=False): Register a loss function and its weight.
            train(train_dataset, valid_dataset): Train the neural network model.
            train_step(data): Perform a single training step.
            valid_step(data): Perform a single validation step.
            print_results(epoch, time): Print the training results for an epoch.
            get_train_dir(): Get the directory path for saving training outputs.
            save_model(epoch): Save the model parameters to a file.
        """
        self.device = device
        self.net = net.to(device)
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.patience_delta = patience_delta
        self.min_val_loss = float('inf')
        self.early_stop = early_stop
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr) 

        assert os.path.isdir(train_dir), "The directory specified in train_dir does not exist."
        self.train_dir = train_dir
        
        self.criterion, self.alpha, self.requires_model = [], [], [] # list of loss functions, weights, and parameter flags
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1 )

    def register_criterion(self, criterion, alpha, requires_model=False):
        r"""
        Register a loss function and its weight in the loss function.

        Args:
            criterion (nn.Module): The loss function.
            alpha (float): The weight of the loss function.
            requires_model (bool): Whether the loss function requires the model as an input (default: False).
        """
        self.criterion.append(criterion.to(self.device))
        self.alpha.append(alpha)
        self.requires_model.append(requires_model)

    def train(self, train_dataset, valid_dataset):
        r"""
        Train the neural network model.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            valid_dataset (torch.utils.data.Dataset): The validation dataset.
        """
        self.train_loss, self.valid_loss = [], []
        
        st = time.time()    # start time
        for epoch in trange(self.max_epochs, desc='Training'):
            st_epoch = time.time()

            # training
            epoch_loss = 0
            for data in train_dataset:
                epoch_loss += self.train_step(data)
            self.scheduler.step()
            self.train_loss.append(epoch_loss/len(train_dataset))
            # validation
            epoch_loss = 0
            for data in valid_dataset:
                epoch_loss += self.valid_step(data)
            self.valid_loss.append(epoch_loss/len(valid_dataset))
            et_epoch = time.time()

            # print results
            et_epoch = time.time()
            self.print_results(epoch, et_epoch-st_epoch)

            # save checkpoints 
            self.save_model(epoch)
            if self.early_stop():
                print('Early stopping at epoch: {}'.format(epoch))
                break

        et = time.time()    # end time 
        print('Training time: {:.3f}s'.format(et-st))

    def train_step(self, data):
        """
        Perform a single training step.

        Args:
            data (tuple): A tuple containing the input data and the target data.

        Returns:
            float: The loss value for the training step.
        """
        inputs, targets = data
        # batch processing
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = 0
        for alpha, criterion, requires_model in zip(self.alpha, self.criterion, self.requires_model):
            if requires_model:
                loss += alpha*criterion(H, targets, self.net)
            else:
                loss += alpha*criterion(H, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def valid_step(self, data):
        """
        Perform a single validation step.

        Args:
            data (tuple): A tuple containing the input data and the target data.

        Returns:
            float: The loss value for the validation step.
        """
        # batch processing
        inputs, targets = data
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = 0
        for alpha, criterion, requires_model in zip(self.alpha, self.criterion, self.requires_model):
            if requires_model:
                loss += alpha*criterion(H, targets, self.net)
            else:
                loss += alpha*criterion(H, targets)
        return loss.item()

    def print_results(self, e, e_time):
        """ Print the training results for an epoch."""
        print(get_str_results(epoch=e,
                              train_loss=self.train_loss,
                              valid_loss=self.valid_loss,
                              time=e_time))

    def get_train_dir(self):
        """Get the directory path for saving training outputs."""
        if self.train_dir is not None:
            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)
        else:
            self.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.train_dir)

    def save_model(self, e):
        """
        Save the model parameters to a file.

                  e (int): The epoch number.
        """
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save model 
        torch.save(
            self.net.state_dict(),
            os.path.join(dir_path, 'model_e' + str(e) + '.pt'))

    def early_stop(self):
        r"""
        Early stopping criterion.
        """
        if self.valid_loss[-1] < (self.min_val_loss - self.patience_delta):
            # update min validation loss
            self.valid_loss = self.valid_loss[-1]
            self.counter = 0
        elif ((self.min_val_loss - self.min_delta) < self.valid_loss[-1]) and (self.valid_loss[-1] < (self.min_val_loss + self.min_delta)):
            # no improvement, so update counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_str_results(epoch=None, train_loss=None, valid_loss=None, time=None, lossF = None, lossT = None):
    """
    Construct the string that has to be printed at the end of the epoch.

    Args:
        epoch (int): The epoch number.
        train_loss (list): List of training loss values.
        valid_loss (list): List of validation loss values.
        time (float): The time taken for the epoch.
        lossF (float): The lossF value.
        lossT (float): The lossT value.

    Returns:
        str: The formatted string to be printed.
    """
    to_print=''

    if epoch is not None:
        to_print += 'epoch: {:3d} '.format(epoch)
    
    if train_loss is not None:
        to_print += '- train_loss: {:6.4f} '.format(train_loss[-1])
                        
    if valid_loss is not None:
        to_print += '- test_loss: {:6.4f} '.format(valid_loss[-1])

    if time is not None:
        to_print += '- time: {:6.4f} s'.format(time)

    if lossF is not None:
        to_print += '- lossF: {:6.4f}'.format(lossF) 

    if lossT is not None:
        to_print += '- lossT: {:6.4f}'.format(lossT) 

    return to_print