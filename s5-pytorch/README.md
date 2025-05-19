# S5: Simplified State Space Layers for Sequence Modeling

Original implementation [s5-pytorch](https://github.com/i404788/s5-pytorch).

We fix bugs on step forward mode

## Example

```py3
from s5 import S5, S5Block

# Raw S5 operator
x = torch.rand([2, 256, 32])
model = S5(32, 32)
model(x) # [2, 256, 32]

# S5-former block (S5+FFN-GLU w/ layernorm, dropout & residual)
model = S5Block(32, 32, False)
model(x) # [2, 256, 32]
```
