from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..policy import STACK_TYPE

IMAGE_TYPE = Literal["rgb", "grayscale"]


class Encoder(nn.Module):
    def __init__(
        self,
        in_shape: int | np.ndarray,
        out_dim: int,
        stacks=1,
        stack_type: STACK_TYPE = None,
        last_layer: Literal["lstm", "linear"] = "linear",
        image_type: IMAGE_TYPE = "rgb",
    ):
        super().__init__()
        self.in_shape = in_shape
        self.stacks = stacks
        self.last_layer = last_layer

        layers = []
        in_shape = list(in_shape)

        conv_stacks = stacks if last_layer == "linear" else 1

        if len(in_shape) == 1:
            layers.extend(
                [
                    nn.Linear(in_shape[0], 128),
                    nn.ReLU(True),
                    nn.Linear(128, 128),
                    nn.ReLU(True),
                ]
            )
        elif len(in_shape) == 3:
            image_dim = image_type == "rgb" and 3 or 1

            layers.extend(
                [
                    nn.AdaptiveAvgPool2d((84, 84)),
                    nn.Conv2d(image_dim * conv_stacks, 32, 8, 4),
                    nn.ReLU(True),
                    nn.Conv2d(32, 64, 4, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Flatten(),
                ]
            )
        else:
            raise ValueError(f"Invalid shape {in_shape}")

        self.encoder = nn.Sequential(*layers)

        if self.last_layer == "lstm":
            self.lstm = nn.LSTM(3136, 512, 1, batch_first=True)
            self.l1 = nn.Linear(512, out_dim)
        else:
            self.l1 = nn.Linear(3136, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Encode x
        if x.ndim == 5:
            S = x.shape[1]
            x = rearrange(x, "b s c h w -> (b s) c h w")
            x = self.encoder(x)  # (B*S, 3136)
            x = rearrange(x, "(b s) n -> b s n", s=S)
        else:
            x = self.encoder(x)

        # Optionally use an LSTM
        if self.last_layer == "lstm":
            if mask is not None:
                # We need to mask some (B, S) to not calculate their hidden_states
                mask_lengths = mask.sum(dim=1)  # (B)
                packed = pack_padded_sequence(
                    x, mask_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.lstm(packed)
                x, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                x, _ = self.lstm(x)
            x = x.squeeze(dim=0)

        # Return logits in shape of the output dimension
        return self.l1(x)
