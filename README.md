# gpt3-pytorch
Unofficial PyTorch Implementation of OpenAI's GPT-3 (discount edition)

## Foreword
I'm definitely aware I'm late to the party but here's my discount implementation of GPT-3 by OpenAI in PyTorch. I've simply used regular Attention instead of Multihead Attention and have stuck to using very few blocks because my laptop cannot handle anything more than that.

## Usage

```python
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
```

## Contributions
Any suggestions, feedback, and PRs welcome! If I've made a mistake anywhere, please do sound it out on the [`Issues`](https://github.com/rish-16/gpt3-pytorch/issues) page.

## License
[MIT](https://github.com/rish-16/gpt3-pytorch/blob/main/LICENSE)
