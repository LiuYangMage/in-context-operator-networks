import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers):
        super().__init__()
        # self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.gelu(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = F.gelu(x)
        x = self.lin2(x)
        return x


class DeepONet(nn.Module):
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

        self.branch = FFN(
            config['in_dim'],
            config['hidden_dim'],
            config['emb_dim'],
            config['hidden_layers'],
        )
        self.trunk = FFN(
            1,
            config['hidden_dim'],
            config['emb_dim'],
            config['hidden_layers'],
        )

        # trainable bias of size (1,)
        self.b = torch.nn.parameter.Parameter(torch.zeros(1,))

    def standard_forward(self, cond, query):
        """
        cond:  (bs, cond_len, input_dim)
        query: (bs, query_len, query_dim)
        """
        bs = cond.size(0)
        flat_cond = cond.reshape(bs, -1) # (bs, cond_len, input_dim) -> (bs, cond_len*input_dim)

        cond_emb = self.branch(flat_cond) # (bs, cond_len*input_dim) -> (bs, emb_dim)
        query_emd = self.trunk(query) # (bs, query_len, query_dim) -> (bs, query_len, emb_dim)
        out = torch.einsum("be,bqe->bq", cond_emb, query_emd) # (bs, emb_dim) * (bs, query_len, emb_dim) -> (bs, query_len)
        out += self.b # (bs, query_len) + (1,) -> (bs, query_len)
        out = out[...,None] # (bs, query_len, 1)
        return out

    def forward(self, cond_k, cond_v, qoi_k):
        '''
        adapt to ICON's input format
        cond_k: (bs, cond_len, dim)
        cond_v: (bs, cond_len, dim)
        qoi_k: (bs, query_len, dim)
        '''
        # slice the correct query
        query = qoi_k[...,self.config['query_idx_start']:self.config['query_idx_end']] # (bs, query_len, dim)
        out = self.standard_forward(cond_v, query)
        return out
        

if __name__ == "__main__":
    config = {
        'in_dim': 100,
        'hidden_dim': 200,
        'emb_dim': 300,
        'hidden_layers': 4,
    }
    deeponet = DeepONet(config)
    cond = torch.randn(10, 100, 1)
    query = torch.randn(10, 11, 1)
    out = deeponet.standard_forward(cond, query)
    print(out.shape)
