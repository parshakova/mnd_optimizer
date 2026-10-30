# mndopt

PyTorch implementation of the Matrix Normal Diagonal (MND) optimizer from
[*A Statistical Viewpoint on Adam, Shampoo and Muon*](https://arxiv.org/abs/2610.24839)
by Tetiana Parshakova, Robert M. Gower, and Paulo Orenstein (2026).

Requires Python 3.10+ and PyTorch 2.2+.

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
Set `matrix=False` to force AdamW in a parameter group:

```python
optimizer = MND([
    {"params": model.weight, "matrix": True},
    {"params": model.bias, "matrix": False, "lr": 0.001},
], weight_decay=0.01)
```

Defaults: `lr=0.003`, `beta=0.925`, `eps=1e-9`, `weight_decay=0`.
AdamW uses `adam_betas=(0.95, 0.95)` and `adam_eps=1e-8`.

From a clone, run the tests and CPU example:

```bash
pip install -e '.[test]'
pytest -q
python examples/regression.py --steps 300
```

## Citation

```bibtex
@misc{parshakova2026statistical,
  title  = {A Statistical Viewpoint on Adam, Shampoo and Muon},
  author = {Parshakova, Tetiana and Gower, Robert M. and Orenstein, Paulo},
  year   = {2026},
  eprint = {2610.24839},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2610.24839}
}
```

Licensed under [Apache 2.0](LICENSE).
