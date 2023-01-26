import torch
import torch.nn.functional as F


class Attention(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.d_head = d_model // n_head

        self.scale = self.d_head ** -0.5

    def forward(self, queries, keys, values, mask):
        return apply_attention(
            queries,
            keys,
            values,
            mask,
            self.n_head,
            self.scale,
        )


@torch.jit.script
def apply_attention(queries, keys, values, mask, n_head: int, scale: float):
    # need batch dim first for bmm
    batch_head_size = queries.size(1)
    batch_size = batch_head_size // n_head
    q_size = queries.size(0)
    k_size = keys.size(0)

    # need batch dim first for bmm
    queries = queries.transpose(0, 1)
    keys = keys.transpose(0, 1)
    values = values.transpose(0, 1)

    keys = keys.transpose(1, 2)  # transpose so dims align for matrix multiply

    score = torch.bmm(queries, keys)
    score.mul_(scale)

    score.view(
        batch_size,
        n_head,
        q_size,
        k_size,
    ).masked_fill_(mask[:, None], float("-inf"))

    probabilities = F.softmax(score, dim=-1)

    output = torch.bmm(probabilities, values)

    output = output.transpose(0, 1).contiguous()
    output = output.view(q_size, batch_size, -1)

    return output, probabilities
