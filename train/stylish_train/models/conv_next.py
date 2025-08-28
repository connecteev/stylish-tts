import torch


class GRN(torch.nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class BasicConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int = 7,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim
        )  # depthwise conv

        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
