import torch
import time

n = 128
d_in = 1024
d_out = 3072
rank = 16

x = torch.randn(35, 10, 67, n, d_in // n)  # (..., n, i)
u = torch.randn(n, d_in // n, rank)  # (n, i, j)
v = torch.randn(n, rank, d_out // n)  # (n, j, k)

for _ in range(100):
    y_einsum = torch.einsum("...ni,nij,njk->...nk", x, u, v)

*dims, _, _ = x.shape

y = x[..., None, :] @ u @ v  # (..., n, 1, k)
y = y.reshape(*dims, n, -1)  # (..., n, 1, k) -> (..., n, k)

print(y_einsum[0, 0, 0, :5, :5])
print(y[0, 0, 0, :5, :5])


print(y_einsum.shape, y.shape)
assert torch.allclose(y_einsum, y)

torch.nn.Linear
