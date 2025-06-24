import torch
from torch import nn


class HODLR(nn.Module):

    def __init__(
        self,
        A: torch.Tensor,
        min_block_size,
        rank,
        device=None,
        dtype=None,
        transpose=False,
        first_call=True,
    ):
        super().__init__()
        if transpose:
            A = A.transpose(0, 1)

        self.in_dim, self.out_dim = A.shape
        self.min_block_size = min_block_size
        self.rank = rank
        # print("CUDA available:", torch.cuda.is_available())
        # print("Device count:", torch.cuda.device_count())
        # print("Current device:", torch.cuda.current_device())
        if first_call:
            self.bias = nn.Parameter(
                torch.zeros(self.out_dim, device=device, dtype=dtype)
            )

        if self.in_dim <= self.min_block_size or self.out_dim <= self.min_block_size:
            self.A = nn.Parameter(A.to(device=device, dtype=dtype).contiguous())
        else:
            in_dim_half = self.in_dim // 2 + self.in_dim % 2
            out_dim_half = self.out_dim // 2 + self.out_dim % 2
            # in_dim_half = math.ceil(self.in_dim / 2)
            # out_dim_half = math.ceil(self.out_dim / 2)

            # Partition A into blocks
            A11 = A[:in_dim_half, :out_dim_half]
            A12 = A[:in_dim_half, out_dim_half:]

            # bottom_left_r, bottom_left_c = bottom_left_corner
            # top_right_r, top_right_c = top_right_corner

            # A12_bottom_left = (bottom_left_r, bottom_left_c)
            # A12_top_right = (bottom_left_r - in_dim_half, bottom_left_c + out_dim_half)

            A21 = A[in_dim_half:, :out_dim_half]
            # A21_bottom_left = (bottom_left_r - in_dim_half, bottom_left_c + out_dim_half)
            # A21_top_right = (top_right_r, top_right_c)

            A22 = A[in_dim_half:, out_dim_half:]

            # Low-rank approximations of off-diagonal blocks
            U1, S1, V1_t = torch.linalg.svd(A12, full_matrices=False)
            U2, S2, V2_t = torch.linalg.svd(A21, full_matrices=False)
            r1 = min(self.rank, len(S1))
            r2 = min(self.rank, len(S2))

            self.U12 = U1[:, :r1] @ torch.diag(S1[:r1])
            self.V12_H = V1_t[:r1, :]

            self.U21 = U2[:, :r2] @ torch.diag(S2[:r2])
            self.V21_H = V2_t[:r2, :]

            # Recursive construction
            self.A11 = HODLR(
                A11,
                self.min_block_size,
                rank,
                device=device,
                dtype=dtype,
                first_call=False,
            )
            self.A22 = HODLR(
                A22,
                self.min_block_size,
                rank,
                device=device,
                dtype=dtype,
                first_call=False,
            )

            self.in_dim_half = in_dim_half
            self.out_dim_half = out_dim_half

            self.U12 = nn.Parameter(self.U12.to(device=device, dtype=dtype).contiguous())
            self.V12_H = nn.Parameter(self.V12_H.to(device=device, dtype=dtype).contiguous())
            self.U21 = nn.Parameter(self.U21.to(device=device, dtype=dtype).contiguous())
            self.V21_H = nn.Parameter(self.V21_H.to(device=device, dtype=dtype).contiguous())

    def init_weights_with_dense(self, *args, **kwargs):
        pass

    def forward(self, x):
        """
        x: (b, ..., d_in)

        return: (b, ..., d_out)
        """
        if self.in_dim <= self.min_block_size or self.out_dim <= self.min_block_size:
            A = self.A.to(x.device)
            if hasattr(self, "bias"):
                return (x @ A) + self.bias.to(x.device)
            else:
                return x @ A

        X1 = x[..., : self.in_dim_half]  # (b, ..., d_in / 2)
        X2 = x[..., self.in_dim_half :]  # (b, ..., d_in / 2)

        """
         [X1, X2]   @   [[A11, U12@V12_H],   =   [[X1@A11 + X2@U21@V21_H],
                         [U21@V21_H, A22]]        [X1@U12@V12_H + X2@A22]]
        """

        # Recursively apply multiplication
        # self.U12: (d_in / 2, r1), self.V12: (r1, d_out / 2)
        Y1 = self.A11.forward(X1) + X2 @ self.U21.to(x.device) @ self.V21_H.to(x.device)
        Y2 = X1 @ self.U12.to(x.device) @ self.V12_H.to(x.device) + self.A22.forward(X2)
        print(f"{X1[0,0,0,:3]} {X2[0,0,0,:3]} {Y1[0,0,0,:3]} {Y2[0,0,0,:3]}")
        if hasattr(self, "bias"):
            return torch.cat([Y1, Y2], dim=-1) + self.bias.to(x.device)
        return torch.cat([Y1, Y2], dim=-1)

