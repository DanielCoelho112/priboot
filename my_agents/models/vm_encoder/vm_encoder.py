import torch.nn as nn


class VMEncoder(nn.Module):
    def __init__(self, num_inputs, fc_dims, out_dims):
        super(VMEncoder, self).__init__()

        self.out_dim = out_dims

        self.encoder = nn.Sequential(nn.Linear(num_inputs, fc_dims),
                                     nn.ReLU(inplace=True), 
                                     nn.Linear(fc_dims, out_dims),
                                     nn.ReLU(inplace=True))
                
                                
    def forward(self, x):
        x = self.encoder(x)

        return x

