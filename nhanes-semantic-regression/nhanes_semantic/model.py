from __future__ import annotations

import math

import torch
from torch import nn


class LowRankBilinearRegressor(nn.Module):
    """A = L R^T with a frozen task-target embedding bank."""

    def __init__(self, target_bank: torch.Tensor, rank: int) -> None:
        super().__init__()
        if target_bank.ndim != 2:
            raise ValueError("target_bank must be [n_tasks, embedding_dimension]")
        dimension = int(target_bank.shape[1])
        rank = min(int(rank), dimension)
        self.dimension = dimension
        self.rank = rank
        self.register_buffer("target_bank", target_bank.float())
        self.left = nn.Parameter(torch.empty(dimension, rank))
        self.right = nn.Parameter(torch.empty(dimension, rank))
        self.bias = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.left)
        nn.init.orthogonal_(self.right)
        with torch.no_grad():
            self.left.mul_(0.5)
            self.right.mul_(0.5)
            self.bias.zero_()

    def forward(self, z: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        target = self.target_bank[task_index]
        z_factors = z @ self.left
        target_factors = target @ self.right
        return (z_factors * target_factors).sum(dim=-1) / math.sqrt(self.rank) + self.bias

    def dense_operator(self) -> torch.Tensor:
        return self.left @ self.right.T


def identity_prediction(
    z: torch.Tensor, task_index: torch.Tensor, target_bank: torch.Tensor
) -> torch.Tensor:
    return (z * target_bank[task_index]).sum(dim=-1)

