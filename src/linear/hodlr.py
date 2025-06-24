import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import math

split_half = lambda x: (x - x // 2, x // 2)


class HODLRLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        min_block_size: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.last_layer = False

        # base case
        if min(in_features, out_features) < 2 * min_block_size:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, **factory_kwargs)
            )
            self.last_layer = True
            return

        self.rank = min(rank, in_features // 2, out_features // 2)

        i1, i2 = split_half(in_features)
        o1, o2 = split_half(out_features)

        self.a11 = HODLRLinear(
            i1, o1, min_block_size, rank, bias=False, **factory_kwargs
        )
        self.a22 = HODLRLinear(
            i2, o2, min_block_size, rank, bias=False, **factory_kwargs
        )
        self.weight_u12 = nn.Parameter(torch.empty(i1, self.rank, **factory_kwargs))
        self.weight_v12 = nn.Parameter(torch.empty(self.rank, o2, **factory_kwargs))
        self.weight_u21 = nn.Parameter(torch.empty(i2, self.rank, **factory_kwargs))
        self.weight_v21 = nn.Parameter(torch.empty(self.rank, o1, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02):
        if self.last_layer:
            nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)
            return

        self.a11.reset_parameters(std)
        self.a22.reset_parameters(std)

        std_r = std / math.sqrt(self.rank)
        nn.init.trunc_normal_(self.weight_u12, std=std_r, a=-2 * std_r, b=2 * std_r)
        nn.init.trunc_normal_(self.weight_v12, std=std_r, a=-2 * std_r, b=2 * std_r)
        nn.init.trunc_normal_(self.weight_u21, std=std_r, a=-2 * std_r, b=2 * std_r)
        nn.init.trunc_normal_(self.weight_v21, std=std_r, a=-2 * std_r, b=2 * std_r)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def project_from(self, module: nn.Module | Tensor, *args, **kwargs):
        if isinstance(module, nn.Linear):
            weight = module.weight.T
        elif isinstance(module, Conv1D):
            weight = module.weight
        elif isinstance(module, Tensor):
            weight = module
        else:
            raise ValueError("module must be a Linear or Conv1D layer")

        try:
            self.bias.data.copy_(module.bias)  # type: ignore
        except (AttributeError, TypeError):
            pass

        if self.last_layer:
            self.weight.data.copy_(weight.T)
            return

        in_features, out_features = weight.shape
        o1, _ = split_half(out_features)
        i1, _ = split_half(in_features)

        # diagonal
        self.a11.project_from(weight[:i1, :o1])
        self.a22.project_from(weight[i1:, o1:])

        # top right
        u, s, v = torch.linalg.svd(weight[:i1, o1:], full_matrices=False)
        s = torch.diag(s[: self.rank] ** 0.5)
        self.weight_u12.data.copy_(u[:, : self.rank] @ s)
        self.weight_v12.data.copy_(s @ v[: self.rank, :])

        # bottom left
        u, s, v = torch.linalg.svd(weight[i1:, :o1], full_matrices=False)
        s = torch.diag(s[: self.rank] ** 0.5)
        self.weight_u21.data.copy_(u[:, : self.rank] @ s)
        self.weight_v21.data.copy_(s @ v[: self.rank, :])

    def forward(self, x: Tensor) -> Tensor:
        if self.last_layer:
            return F.linear(x, self.weight, self.bias)

        x1, x2 = x.chunk(2, dim=-1)

        y1 = self.a11(x1) + x2 @ self.weight_u21 @ self.weight_v21
        y2 = x1 @ self.weight_u12 @ self.weight_v12 + self.a22(x2)

        y = torch.cat((y1, y2), dim=-1)
        if self.bias is not None:
            y += self.bias
        return y


if __name__ == "__main__":
    from .hodlr_recursive import HODLR as HODLRRecursive

    linear = nn.Linear(1024, 3072, bias=False)
    hodlr = HODLRLinear(1024, 3072, min_block_size=64, rank=16)
    hodlr_recursive = HODLRRecursive(
        linear.weight,
        min_block_size=64,
        rank=16,
        device=linear.weight.device,
        transpose=True,
    )
    hodlr.project_from(linear)

    # print number of parameters
    print(f"Linear parameters: {sum(p.numel() for p in linear.parameters())}")
    print(f"HODLR parameters: {sum(p.numel() for p in hodlr.parameters())}")
    print(
        f"HODLR Recursive parameters: {sum(p.numel() for p in hodlr_recursive.parameters())}"
    )

    x = torch.randn(35, 10, 67, 1024)
    y1 = linear(x)
    y2 = hodlr(x)
    print("=" * 50)
    y3 = hodlr_recursive(x)
    print("=" * 50)

    print(y1.shape, y2.shape, y3.shape)
    assert torch.allclose(
        y2, y3, atol=1e-5
    ), "HODLR and HODLR Recursive outputs do not match"

    print(y1[0, 0, 0, :5])
    print(y2[0, 0, 0, :5])
    print(y3[0, 0, 0, :5])
