import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Positional Encodings from the PyTorch Text tutorial

https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, hidden, drop_prob=0.1, max_len=12288):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
    def __init__(self, vocabsize, features, nblocks=96):
        super().__init__()
        self.attn_body = nn.Module()
        self.attn_body = Attention(features)
        
        self.pos_encoding = PositionalEncoding(features)
        self.word_embedding = nn.Linear(vocabsize, features)

        self.proj1 = nn.Linear(features, features)
        self.out_proj = nn.Linear(features, vocabsize)

        self.nblocks = nblocks

    def forward(self, x):
        # x : (B, 2048, 50257)
        x_pos = self.pos_encoding(x) # (B, 2048, 12288)
        x_emb = self.word_embedding(x) # (B, 2048, 12288)

        x_out = x_pos + x_emb # (B, 2048, 12288)

        for _ in range(self.nblocks):
            x_attn = self.attn_body(x) # (B, 2048, 12288)
            b, n, d = x_attn.shape

            x_res1 = x_add + x_attn # (B, 2048, 12288)

            x_norm1 = torch.norm(x_res1) # (B, 2048, 12288)

            x_proj1 = self.proj1(x_norm1) # (B, 2048, 12288)

            x_res2 = x_norm1 + x_proj1 # (B, 2048, 12288)

            x_out = torch.normalize(self.proj1(x_res2)) # (B, 2048, 12288)
        
        x_final = self.out_proj(x_out) # (B, 2048, 50257)
        x_topk_val, x_topk_idx = torch.topk(x_final, k=self.k, dim=1) # (B, k, 50257)

        return x_topk_val