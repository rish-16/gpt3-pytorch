import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, features, attn_dim):
        super(Attention, self).__init__()
        self.to_q = nn.Linear(features, attn_dim)
        self.to_k = nn.Linear(features, attn_dim)
        self.to_v = nn.Linear(features, attn_dim)
        self.project = nn.Linear(attn_dim, features)
        
    def forward(self, x):
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)
        
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = F.softmax(dots, 0)
        
        out = torch.bmm(attn, V)
        out = self.project(out)
        
        return out

class GPT3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass