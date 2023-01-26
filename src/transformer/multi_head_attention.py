from typing import TYPE_CHECKING

from torchtyping import TensorType

import torch

from .attention import Attention
from .linear import Linear
from .positional_encodings import RotaryEncoding

if TYPE_CHECKING:
    from .types import batch, d_model, sequence


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_head: int,
        rotary: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.in_proj = Linear(d_model, 3 * d_model, gain=2 ** 0.5, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)

        self.use_rotary = rotary
        self.rotary = RotaryEncoding(self.d_head)

        self.attention = Attention(d_model, n_head)

    def forward(
        self,
        *,
        sequence: TensorType["sequence", "batch", "d_model"],
        mask: TensorType["batch", "sequence", "sequence"],
    ):
        batch_size = sequence.size(1)
        batch_head_size = batch_size * self.n_head

        qkv = self.in_proj(sequence)

        # combine batch and head dim
        qkv = qkv.view(sequence.size(0), batch_head_size, 3, self.d_head)

        queries = qkv[:, :, 0]
        keys = qkv[:, :, 1]
        values = qkv[:, :, 2]

        if self.use_rotary:
            queries, keys = self.rotary(queries, keys)

        output, probabilities = self.attention(queries, keys, values, mask)

        attn_sequence = self.out_proj(output)

        return attn_sequence
