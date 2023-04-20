import numpy as np
import torch
import torchvision
# import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


def train_val_split(X, train_size, random_state=0):
    g_cuda = torch.Generator()
    seed = g_cuda.manual_seed(random_state)
    train_len = int(train_size*len(X))
    val_len = len(X) - train_len
    train_set, val_set = torch.utils.data.random_split(X, [train_len, val_len], generator=seed)

    return train_set, val_set


class LrScheduler:
    
    def __init__(self, optimizer, patience=5, min_lr=1e-8, factor=0.5):
        """
        new_lr = old_lr * factor
        
        Parameters:
        ------------
        optimizer : the optimzer the function is using
        patience  : how many epochs to wait before updating learning rate
        min_lr    : the least value the learning rate can take
        factor    : factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, 
                                                                       factor=self.factor, min_lr=self.min_lr, verbose=True)
        
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
        
        
        
        
class EarlyStopping:
    
    def __init__(self, patience=5, min_delta=0, factor=0.7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter: {self.counter} of patience: {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
                
                
class SaveBestModel:
    """
    Class to save the best model while training.
        - If the current epoch's validation loss is less than the
        previous least validation loss then save the model.
    """
    def __init__(self, model_name, best_val_loss=float('inf')):
        self.model_name = model_name
        self.best_val_loss = best_val_loss
    
    def __call__(self, curr_val_loss, epoch, model, optimizer, criterion):
        if curr_val_loss < self.best_val_loss:
            self.best_val_loss = curr_val_loss
            print(f"\nBest validation loss: {self.best_val_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({'epoch':epoch+1, 
                        'model_state_dict'     : model.state_dict(), 
                        'optimizer_state_dict' : optimizer.state_dict(), 
                        'loss'                 : criterion,
                        'last_epoch_loss'      : self.best_val_loss,
                       }, f'{self.model_name}.bin')
            
            
###############################################################################

class PositionalEncoding(torch.nn.Module):
    """
    Creates an embedding of sine and cosine curves.
        Math : Sine and Cosine are periodic functions with a period of 360 degrees or 2 * pi radians...
        The positional encodings are filled up by the mixture of these two periodic functions
        
    Paramters:
    ----------
    embedding_dim : int
        The size of the embedding.
    max_ken : int
    """
    def __init__(self, embedding_dim:int, max_len:int=1000):
        super().__init__()
        self.positional_encodings = torch.nn.Parameter(torch.zeros(max_len, embedding_dim, device=device), requires_grad=False)
        positions = torch.arange(0, max_len).reshape(max_len, 1)
        den = torch.exp(-torch.arange(0, embedding_dim, 2) * math.log(10000) / embedding_dim)
        self.positional_encodings[:, 0::2] = torch.sin(positions * den)
        self.positional_encodings[:, 1::2] = torch.cos(positions * den)

        
    def forward(self, sequence_len):
        return self.positional_encodings[:sequence_len, :].clone()
    
    
    def plot_positional_encodings(self):
        plt.plot(self.positional_encodings)
        plt.show()

    
