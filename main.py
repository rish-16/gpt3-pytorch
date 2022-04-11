import torch
from gpt3_pytorch import GPT3

gpt3 = GPT3(
            vocabsize=1000,
            features=256,
            nblocks=96,
            k=200 # top-k tokens
        )

vocabsize=1000
ntokens = 200
seq = torch.randint(0, vocabsize, size=(ntokens,1))
x = torch.nn.functional.one_hot(seq.squeeze(1), num_classes=vocabsize) # sequence to one-hot encodings
x = x.unsqueeze(0).float() # (batchsize, ntokens, vocabsize)

y = gpt3(x) # (batchsize, k, vocabsize)