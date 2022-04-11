import torch, math
import torch.nn as nn
import torch.nn.functional as F

"""
Positional Encodings from the PyTorch Text tutorial

https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, features, max_len=12288):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2) * (-math.log(10000.0) / features))
        pe = torch.zeros(max_len, 1, features)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        addi = self.pe[:x.size(0)]
        x = x + addi
        return x

class Attention(nn.Module):
    def __init__(self, features, attn_dim=128):
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
    def __init__(self, vocabsize, features, nblocks=96, k=100):
        super().__init__()
        self.attn_body = nn.Module()
        self.attn_body = Attention(features)
        
        self.pos_encoding = PositionalEncoding(features)
        self.word_embedding = nn.Linear(vocabsize, features)

        self.proj1 = nn.Linear(features, features)
        self.out_proj = nn.Linear(features, vocabsize)

        self.nblocks = nblocks
        self.k = k

    def forward(self, x):
        """
        params
            x : input sequence of shape (batchsize, ntokens, vocabsize)
        """
        x_emb = self.word_embedding(x)
        x_out = self.pos_encoding(x_emb)

        for _ in range(self.nblocks):
            x_attn = self.attn_body(x_out)

            x_res1 = x_out + x_attn

            x_norm1 = F.normalize(x_res1)

            x_proj1 = self.proj1(x_norm1)

            x_res2 = x_norm1 + x_proj1

            x_out = F.normalize(self.proj1(x_res2))
        
        x_final = self.out_proj(x_out)
        x_topk_val, x_topk_idx = torch.topk(x_final, k=self.k, dim=1)

        return x_topk_val