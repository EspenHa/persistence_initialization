import torch

try:
    import fused_layer_norm_cuda  # noqa: F401
    from apex.normalization.fused_layer_norm import (
        fused_layer_norm,
        fused_layer_norm_affine,
    )

    apex_available = True
except Exception:
    apex_available = False


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, x):
        if not x.is_cuda or not apex_available:
            return super().forward(x)

        if self.elementwise_affine:
            return fused_layer_norm_affine(
                x, self.weight, self.bias, self.normalized_shape, self.eps
            )
        else:
            return fused_layer_norm(x, self.normalized_shape, self.eps)
