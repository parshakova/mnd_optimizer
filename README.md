# mndopt

PyTorch implementation of the Matrix Normal Diagonal (MND) optimizer from
[*A Statistical Viewpoint on Adam, Shampoo and Muon*](https://arxiv.org/abs/2610.24839)
by Tetiana Parshakova, Robert M. Gower, and Paulo Orenstein (2026).

Requires Python 3.10+ and PyTorch 2.2+.
Supports dense float32 and float64 parameters.

```bash
pip install mndopt
```

```python
import torch
from mndopt import MND

model = torch.nn.Linear(8, 4)
optimizer = MND(model.parameters(), lr=0.003)

x, y = torch.randn(32, 8), torch.randn(32, 4)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
optimizer.step()
```

Matrices use MND; other parameters use AdamW. Both use decoupled weight decay.

Train a small GPT on Tiny Shakespeare for 2,000 MND steps:
[notebook](examples/tiny_shakespeare.ipynb).

Licensed under [Apache 2.0](LICENSE).
