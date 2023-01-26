import torch


class Linear(torch.nn.Linear):
    def __init__(self, *args, gain="linear", **kwargs):
        if isinstance(gain, str):
            self.gain = torch.nn.init.calculate_gain(gain)
        elif isinstance(gain, float):
            self.gain = gain
        else:
            raise ValueError

        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=self.gain)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
