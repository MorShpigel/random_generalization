import torch
import torch.nn as nn
import numpy as np

class DiagonalNetworks(nn.Module):
    def __init__(self, d_input, depth=2):
        super(DiagonalNetworks, self).__init__()
        self.depth = depth
        self.d_input = d_input
        self.layers = []
        self.layers = nn.Parameter(torch.randn(d_input, depth))
        # for _ in range(depth):
        #     self.layers.append(torch.nn.Linear(d_input, depth, bias=False))
            
    def forward(self, input):
        w = torch.prod(self.layers, axis=1)
        output = torch.matmul(input.float(),w)
        return output

    @torch.no_grad()
    def forward_normalize(self, input):
        w = torch.prod(self.layers, axis=1)
        w /= 0.1*torch.norm(w, p=2, dim=0) # normalize
        output = torch.matmul(input.float(),w)
        return output
    

class DiagonalNetworks_sampled():
    def __init__(self, dist, d_input, depth=2):
        self.depth = depth
        self.d_input = d_input
        self.dist = dist
        self.w = nn.Parameter(torch.randn(d_input, 1))

    def forward(self, input):
        wx = input*self.w
        output = torch.sum(wx, axis=1)
        return output
    
    def forward_normalize(self, input):
        self.w /= 0.1*torch.norm(self.w, p=2, dim=0) # normalize
        wx = input*self.w
        output = torch.sum(wx, axis=1)
        return output
    
    @torch.no_grad()
    def sample_params(self):
        if self.dist=="Normal":
            self.w = torch.randn(self.d_input)
        elif self.dist=="Normal_uv":
            self.w = torch.ones(self.d_input)
            for _ in range(self.depth):
                self.w = torch.mul(self.w, torch.normal(0, 1, (self.d_input,)))
        elif self.dist=="Laplace":
            self.w = torch.distributions.laplace.Laplace(0, 1).sample((self.d_input,))
