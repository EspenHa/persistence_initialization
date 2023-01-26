import torch


class SinusoidalEncoding(torch.nn.Module):
    def __init__(self, dim, *, max_length=1024, base=10000):
        super().__init__()

        self.pe: torch.Tensor
        self.register_buffer("pe", self._compute_buffers(dim, base, max_length))

    @staticmethod
    def _compute_buffers(dim, base, length):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = freqs[:, None, :]  # add batch dim

        encoding = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        return encoding

    def forward(self, x):
        seq_len = x.size(0)

        pe = self.pe[:seq_len]

        x = x + pe

        return x


class RotaryEncoding(torch.nn.Module):
    def __init__(self, dim, *, max_length=1024, base=10000):
        super().__init__()

        sin, cos = self._compute_buffers(dim, base, max_length)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

        self.sin: torch.Tensor
        self.cos: torch.Tensor

    @staticmethod
    def _compute_buffers(dim, base, length):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = freqs.repeat(1, 2)  # repeat each frequency twice
        freqs = freqs[:, None, :]  # add batch dim

        return freqs.sin(), freqs.cos()

    def forward(self, queries, keys):
        return apply_rotary(queries, keys, self.sin, self.cos)


def rotate_half(x):
    middle = x.size(-1) // 2

    x1 = x[..., :middle]
    x2 = x[..., middle:]

    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x, sin, cos):
    return x * cos[: x.size(0)] + rotate_half(x) * sin[: x.size(0)]


@torch.jit.script
def apply_rotary(queries, keys, sin, cos):
    return _apply_rotary(queries, sin, cos), _apply_rotary(keys, sin, cos)
