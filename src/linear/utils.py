from lightning.pytorch.utilities import rank_zero_info

import torch.nn as nn

from transformers.pytorch_utils import Conv1D

from typing import Callable

from src.linear.hodlr import HODLRLinear
from src.linear.hodlr_recursive import HODLR as HODLRRecursive


def replace_module(
    module: nn.Module,
    structured_type: str,
    init_strategy: str,
    **kwargs,
):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            in_dim, out_dim = child_module.in_features, child_module.out_features
        elif isinstance(child_module, Conv1D):
            in_dim, out_dim = child_module.nx, child_module.nf
        else:
            replace_module(
                child_module, structured_type, init_strategy, **kwargs
            )
            continue

        if structured_type == "dense":
            return
        elif structured_type == "hodlr":
            new_layer = HODLRLinear(
                in_features=in_dim,
                out_features=out_dim,
                min_block_size=kwargs["min_block_size"],
                rank=kwargs["max_rank"],
            )
            rank_zero_info(
                f"Replacing {child_name} with HODLRLinear: in_dim={in_dim}, out_dim={out_dim}, "
                f"min_block_size={kwargs['min_block_size']}, rank={kwargs['max_rank']}"
            )
        else:
            raise ValueError(f"Unsupported structured type: {structured_type}")
        if init_strategy == "projection":
            new_layer.project_from(child_module, **kwargs)
        elif init_strategy == "random":
            ...
        else:
            raise ValueError(f"Unsupported init strategy: {init_strategy}")

        setattr(module, child_name, new_layer)
