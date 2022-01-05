import numpy as np
import torch
import torch.nn as nn
from models.recurrent_models import LSTMGenerator
from models.convolutional_models import CausalConvGenerator

def tensor_to_string_list(tensor):
    """Convert a tensor to a list of strings representing its value"""
    scalar_list = tensor.squeeze().numpy().tolist()
    return ["%.5f" % scalar for scalar in scalar_list]

class DatasetGenerator:
    def __init__(self, generator, seq_len=20, noise_dim=100, dataset=None, mType = 'lstm'):
        """Class for fake dataset generation
        Args:
            generator (pytorch module): trained generator to use
            seq_len (int): length of the sequences to be generated
            noise_dim (int): input noise dimension for gan generator
            dataset (Dataset): dataset providing normalize and denormalize functions for deltas and series (by default, don't normalize)
        """       
        
        if mType == 'lstm':            
            model = LSTMGenerator(in_dim=noise_dim, n_layers = 1, out_dim=1, hidden_dim=256).cuda()
            model.load_state_dict(generator)
            model.eval()
            self.generator = model
            
        elif mType == 'cnn':            
            model = CausalConvGenerator(noise_size=noise_dim, output_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0.).cuda()
            model.load_state_dict(generator)
            model.eval()
            self.generator = model
            
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.dataset = dataset

    def generate_dataset(self, outfile=None, batch_size=1, size=1000):
        """Method for generating a dataset
        Args:
            outfile (string): name of the npy file to save the dataset. If None, it is simply returned as pytorch tensor
            batch_size (int): batch size for generation
            seq_len (int): sequence length of the sequences to be generated
            size (int): number of time series to generate if delta_list is present, this parameter is ignored
        """
        
        noise = torch.randn(size, self.seq_len, self.noise_dim)         
        out_list = []
        
        for batch in noise.split(batch_size):
            out_list.append(self.generator(batch.cuda()))
        out_tensor = torch.cat(out_list, dim=0)
         
        #Puts generated sequences in original range
        if self.dataset:
            out_tensor = self.dataset.denormalize(out_tensor)
            out_tensor = out_tensor.cpu()

        if outfile:
            np.save(outfile, out_tensor.detach().numpy()) 
        
        return out_tensor 


if __name__ == "__main__":
    model = torch.load('checkpoints/cnn_conditioned_alternate1_netG_epoch_85.pth') 
    gen = DatasetGenerator(model)
    print("Shape of example dataset:", gen.generate_dataset(delta_list=[i for i in range(100)]).size())
