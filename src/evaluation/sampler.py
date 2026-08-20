import numpy as np
import torch
import torch.nn as nn


class LocalNegativesSampler(nn.Module):
    def __init__(
            self,
            num_items: int,
    ) -> None:
        super().__init__()

        self.num_items: int = num_items

    def forward(
            self,
            positive_ids: torch.Tensor,
            num_to_sample: int,
    ) -> torch.Tensor:
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_ids = torch.randint(
            low=0,
            high=self.num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        return sampled_ids
