"""MLP projector mapping backbone features h to the SSL embedding z.

Barlow Twins / VICReg style: Linear -> BatchNorm -> ReLU in every hidden layer,
plain Linear on the output layer (no BatchNorm on z).
"""

from torch import nn


class MLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: tuple[int, ...] = (1024, 1024),
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = (input_dim, *hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out, bias=False), nn.BatchNorm1d(d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
