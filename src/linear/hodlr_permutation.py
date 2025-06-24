import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import math
from absl import logging


class HODLRLinear(nn.Module):
    num_layers: int
    permutations: list[Tensor]

    weight_us: nn.ParameterList
    weight_vs: nn.ParameterList
    weight_ds: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        min_block_size: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.real_in_features = in_features
        self.real_out_features = out_features
        self.min_block_size = min_block_size
        self.rank = rank
        self.num_layers = max(
            int(math.log2(min(in_features, out_features) / min_block_size)), 0
        )
        self.permutations = []
        self.weight_us = nn.ParameterList()
        self.weight_vs = nn.ParameterList()

        # 1. align the input and output size to multiple of 2**num_layers
        in_features = int(
            2**self.num_layers * math.ceil(in_features / (2**self.num_layers))
        )
        out_features = int(
            2**self.num_layers * math.ceil(out_features / (2**self.num_layers))
        )

        if in_features != self.real_in_features:
            logging.warning(
                f"Input size {self.real_in_features} is not a multiple of 2**{self.num_layers}, "
                f"aligning to {in_features}"
            )
        if out_features != self.real_out_features:
            logging.warning(
                f"Output size {self.real_out_features} is not a multiple of 2**{self.num_layers}, "
                f"aligning to {out_features}"
            )

        # TODO(gary): for speed purpose, only support in_features == real_in_features
        assert (
            in_features == self.real_in_features
            and out_features == self.real_out_features
        ), "Input and output size must be equal to the real input and output size."

        # 2. create the weight matrices
        for i in range(1, self.num_layers + 1):
            current_rank = min(rank, min(in_features, out_features) // (2**i))

            # For each layer, it tries to construct a sequence like
            # [s1, s0, s3, s2, s5, s4, ...]
            # and each s_i is a sequence of contiguous indices like [64, 65, 66, ..., 128]
            self.permutations.append(
                torch.arange(out_features, **factory_kwargs)
                .reshape(2 ** (i - 1), 2, -1)[:, (1, 0), :]
                .flatten()
            )
            n = 2**i
            self.weight_us.append(
                nn.Parameter(
                    torch.empty(n, in_features // n, current_rank, **factory_kwargs)
                )
            )
            self.weight_vs.append(
                nn.Parameter(
                    torch.empty(n, current_rank, out_features // n, **factory_kwargs)
                )
            )

        self.weight_ds = nn.Parameter(
            torch.empty(
                2**self.num_layers,
                in_features // (2**self.num_layers),
                out_features // (2**self.num_layers),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the weights of the HODLR layer.
        """
        for i in range(self.num_layers):
            nn.init.kaiming_uniform_(self.weight_us[i], a=math.sqrt(5))
            self.weight_vs[i].data.zero_()
        nn.init.kaiming_uniform_(self.weight_ds, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def init_weights_with_dense(self, dense: nn.Module, **kwargs) -> None:
        assert isinstance(dense, (nn.Linear, Conv1D))

        if isinstance(dense, Conv1D):
            in_features = dense.nx
            out_features = dense.nf
            weight = dense.weight
        elif isinstance(dense, nn.Linear):
            in_features = dense.in_features
            out_features = dense.out_features
            weight = dense.weight.transpose(0, 1)
        else:
            raise ValueError("dense must be a Linear or Conv1D layer")

        weight = F.pad(
            weight,
            (0, self.in_features - in_features, 0, self.out_features - out_features),
            mode="constant",
            value=0,
        )

        stack = [weight]
        for i in range(self.num_layers):
            us, vs = [], []
            for j in range(2**i):
                w = stack.pop(0)

                # Push on-diagonal blocks
                stack.append(w[: w.shape[0] // 2, : w.shape[1] // 2])
                stack.append(w[w.shape[0] // 2 :, w.shape[1] // 2 :])

                # SVD off-diagonal blocks
                # A12, the upper right block
                u, s, v = torch.linalg.svd(
                    w[: w.shape[0] // 2, w.shape[1] // 2 :], full_matrices=False
                )
                r = min(self.rank, len(s))
                us.append(u[:, :r] @ torch.diag(s[:r]))
                vs.append(v[:r, :])
                # A21, the lower left block
                u, s, v = torch.linalg.svd(
                    w[w.shape[0] // 2 :, : w.shape[1] // 2], full_matrices=False
                )
                r = min(self.rank, len(s))
                us.append(u[:, :r] @ torch.diag(s[:r]))
                vs.append(v[:r, :])
            self.weight_us[i].data.copy_(torch.stack(us, dim=0))
            self.weight_vs[i].data.copy_(torch.stack(vs, dim=0))

        del stack

        # Block-diagonal part
        _m = self.in_features // (2**self.num_layers)
        _n = self.out_features // (2**self.num_layers)
        ds = [
            weight[i * _m : (i + 1) * _m, i * _n : (i + 1) * _n]
            for i in range(2**self.num_layers)
        ]
        self.weight_ds.data.copy_(torch.stack(ds, dim=0))
        if self.bias is not None and dense.bias is not None:
            self.bias.data.copy_(dense.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (b, ..., d_in)

        return: (b, ..., d_out)
        """
        *dims, d_in = x.shape
        # if d_in != self.in_features:
        #     x = F.pad(
        #         x,
        #         (0, self.in_features - d_in),
        #         mode="constant",
        #         value=0,
        #     )

        # 1. pre-allocate the output tensor
        output = torch.zeros(*dims, self.out_features, device=x.device, dtype=x.dtype)

        # 2. compute the recursive HODLR matrix-matrix multiplication
        for i in range(self.num_layers):
            n = 2 ** (i + 1)
            # ! Gary: Note that the following sliding and updating are in-place computation
            # ! that only one 1 copy is created as the output of einsum
            output[..., self.permutations[i]] += torch.einsum(
                "...ni,nij,njk->...nk",
                x.reshape(*dims, n, -1),
                self.weight_us[i],
                self.weight_vs[i],
            ).reshape(*dims, -1)

            # output += (
            #     x.reshape(*dims, n, 1, -1) @ self.weight_us[i] @ self.weight_vs[i]
            # ).reshape(*dims, -1)[..., self.permutations[i]]

        # 3. compute the block-diagonal matrix-matrix multiplication
        output += torch.einsum(
            "...ni,nij->...nj",
            x.reshape(*dims, 2**self.num_layers, -1),
            self.weight_ds,
        ).reshape(*dims, self.out_features)
        # output += (
        #     x.reshape(*dims, 2**self.num_layers, 1, -1) @ self.weight_ds
        # ).reshape(*dims, -1)

        # 4. add the bias
        if self.bias is not None:
            output += self.bias

        # if d_in != self.in_features:
        #     # ! This returns a view of the output tensor, not a copy. Be aware of memory leaks.
        #     # https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905
        #     return output[..., : self.real_out_features]

        return output
