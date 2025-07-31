from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeVar

import torch

if TYPE_CHECKING:
    import numpy
    import polars
    T = TypeVar('T')
    U = TypeVar('U')

class Cast():
    def __init__(self, column, transformation: Callable[[T], U]):
        self.column = column
        self.transformation = transformation
    
    def __len__(self):
        return len(self.column)
    
    def __getitem__(self, idx):
        return self.transformation(self.column[idx])

class TorchDataset(torch.utils.data.Dataset):
    """Base class for PyTorch datasets that."""
    
    def __init__(self, input: Cast, target : Cast):
        assert len(input) == len(target), "Input and target must have the same length."
        self.input = input
        self.target = target
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]