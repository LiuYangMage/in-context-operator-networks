import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuralop.models as models


class FNO(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
        in_dim: the input dimension of features, should be  # data points * data dimension
        hidden_dim: the hidden dimension of the FFN
        emb_dim: the output dimension of the FFN, which is the embedding dimension
        hidden_layers: the number of hidden layers of the FFN
        *** only support scalar output ***
        '''
        self.config = config
        self.fno = models.FNO(n_modes=(config['n_modes'],), 
                            hidden_channels=config['hidden_channels'],
                            n_layers=config['n_layers'],
                            in_channels=config['in_channels'], 
                            out_channels=config['out_channels'])
        
    def forward(self, cond_k, cond_v, qoi_k):
        """
        cond_k:  (bs, cond_len, input_dim)
        cond_v:  (bs, cond_len, input_dim)
        qoi_k: (bs, query_len, query_dim)
        """
        grid = cond_k[...,self.config['cond_grid_idx_start']:self.config['cond_grid_idx_end']]
        grid = grid.permute(0, 2, 1)
        x = cond_v.permute(0, 2, 1)
        x = torch.cat([x, grid], dim=1) # (bs, channels, n_points)
        out = self.fno(x) # (bs, out_channels, n_points)
        out = out.permute(0, 2, 1) # (bs, n_points, out_channels)
        return out


if __name__ == "__main__":
    batchsize = 128
    x = torch.rand(batchsize, 1, 101) # (bs, channels, n_points)
    grid = torch.linspace(0, 1, 101)
    x = torch.cat([x, grid.repeat(batchsize, 1, 1)], dim=1)
    model = models.FNO(n_modes=(16,), hidden_channels=1024,
                in_channels=2, out_channels=1)
    out = model(x)
    print("x.shape", x.shape)
    print("out.shape", out.shape) # (bs, channels, n_points)


    