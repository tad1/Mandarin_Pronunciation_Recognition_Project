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
    
    def __init__(self, *columns : Cast):
        assert all(len(column) == len(columns[0]) for column in columns), "All columns must have the same length."
        self.columns = columns
    def __len__(self):
        return len(self.columns[0])

    def __getitem__(self, idx):
        return (column[idx] for column in self.columns)