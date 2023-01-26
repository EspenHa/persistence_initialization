from typing import TYPE_CHECKING, Type, Union

from torchtyping import TensorType

import torch

from src.conf import LayerEnum, PersistenceInitEnum, PositionalEncodingEnum

from .feedforward import FeedForwardLayer
from .layer_norm import LayerNorm
from .linear import Linear
from .multi_head_attention import MultiHeadAttention
from .positional_encodings import SinusoidalEncoding

if TYPE_CHECKING:
    from .types import batch, channels, d_model, sequence


class ReZero(torch.nn.Module):

    """
    ReZero Transformer layer.

    ReZero is All You Need: Fast Convergence at Large Depth
    https://arxiv.org/abs/2003.04887
    """

    def __init__(self, feed_forward: FeedForwardLayer, mha: MultiHeadAttention):
        super().__init__()

        self.mha = mha
        self.feed_forward = feed_forward

        self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        sequence: TensorType["sequence", "batch", "d_model"],
        mask: TensorType["batch", "sequence", "sequence"],
    ) -> TensorType["sequence", "batch", "d_model"]:

        attn_sequence = self.mha(sequence=sequence, mask=mask)
        attn_sequence = sequence + self.alpha * attn_sequence

        ff_sequence = self.feed_forward(attn_sequence)
        ff_sequence = attn_sequence + self.alpha * ff_sequence

        return ff_sequence


class PreNorm(torch.nn.Module):
    def __init__(self, feed_forward: FeedForwardLayer, mha: MultiHeadAttention):
        super().__init__()

        self.mha = mha
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(self.mha.d_model)
        self.norm2 = LayerNorm(self.feed_forward.d_model)

    def forward(
        self,
        *,
        sequence: TensorType["sequence", "batch", "d_model"],
        mask: TensorType["batch", "sequence", "sequence"],
    ) -> TensorType["sequence", "batch", "d_model"]:

        attn_sequence = self.norm1(sequence)
        attn_sequence = self.mha(sequence=attn_sequence, mask=mask)
        attn_sequence = sequence + attn_sequence

        ff_sequence = self.norm2(attn_sequence)
        ff_sequence = self.feed_forward(ff_sequence)
        ff_sequence = ff_sequence + attn_sequence

        return ff_sequence


class PostNorm(torch.nn.Module):
    def __init__(self, feed_forward: FeedForwardLayer, mha: MultiHeadAttention):
        super().__init__()

        self.mha = mha
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(self.mha.d_model)
        self.norm2 = LayerNorm(self.feed_forward.d_model)

    def forward(
        self,
        *,
        sequence: TensorType["sequence", "batch", "d_model"],
        mask: TensorType["batch", "sequence", "sequence"],
    ) -> TensorType["sequence", "batch", "d_model"]:

        attn_sequence = self.mha(sequence=sequence, mask=mask)
        attn_sequence = self.norm1(sequence + attn_sequence)

        ff_sequence = self.feed_forward(attn_sequence)
        ff_sequence = self.norm2(attn_sequence + ff_sequence)

        return ff_sequence


class DecoderTransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 4,
        n_head: int = 4,
        d_model: int = 32,
        persistence_init: PersistenceInitEnum = PersistenceInitEnum.SKIP_AND_GATING,
        pos_enc: PositionalEncodingEnum = PositionalEncodingEnum.ROTARY,
        layer: LayerEnum = LayerEnum.REZERO,
    ):
        super().__init__()

        self.up_proj = Linear(1, d_model, bias=False)
        self.down_proj = Linear(d_model, 1, bias=False)

        if pos_enc is PositionalEncodingEnum.SINUSOIDAL:
            self.initial_pos_encoder: torch.nn.Module = SinusoidalEncoding(d_model)
            rotary = False
        else:
            assert pos_enc is PositionalEncodingEnum.ROTARY
            self.initial_pos_encoder = torch.nn.Identity()
            rotary = True

        cls: Union[Type[ReZero], Type[PreNorm], Type[PostNorm]]
        if layer is LayerEnum.REZERO:
            cls = ReZero
        elif layer is LayerEnum.PRE_NORM:
            cls = PreNorm
        elif layer is LayerEnum.POST_NORM:
            cls = PostNorm
        else:
            raise ValueError

        self.layers = torch.nn.ModuleList(
            [
                cls(
                    FeedForwardLayer(d_model=d_model),
                    MultiHeadAttention(d_model=d_model, n_head=n_head, rotary=rotary),
                )
                for _ in range(num_layers)
            ]
        )

        if persistence_init is PersistenceInitEnum.SKIP_AND_GATING:
            self.gamma = torch.nn.Parameter(torch.zeros(1))
            self.skip = True
        elif persistence_init is PersistenceInitEnum.SKIP:
            self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.skip = True
        elif persistence_init is PersistenceInitEnum.NONE:
            self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.skip = False
        else:
            raise ValueError

    def forward(
        self,
        sequence: TensorType["sequence", "batch", "channels"],
        *,
        mask: TensorType["batch", "sequence", "sequence"] = None,
    ) -> TensorType["sequence", "batch", "channels"]:

        sequence.nan_to_num_()

        x = self.up_proj(sequence)
        x = self.initial_pos_encoder(x)

        for layer in self.layers:
            x = layer(sequence=x, mask=mask)

        x = self.down_proj(x)

        if self.skip:
            x = sequence + self.gamma * x

        return x
