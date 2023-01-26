import torch

from .linear import Linear


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, *, d_model: int):
        super().__init__()

        self.d_model = d_model
        self.d_feedforward = 4 * d_model

        self.layers = torch.nn.Sequential(
            Linear(d_model, self.d_feedforward, gain="relu"),
            torch.nn.ReLU(),
            Linear(self.d_feedforward, d_model),
        )

    def forward(self, x):
        return self.layers(x)
