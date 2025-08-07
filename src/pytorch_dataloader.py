from typing import Literal, Union
from torch.utils.data import Dataset
from torch import Tensor
import torch

class DefaultCollate():
    def __call__(self, batch):
        return torch.utils.data.default_collate(batch)

class ReshapeCollate(DefaultCollate):
    def __init__(self, shape: tuple):
        self.shape = shape

    def __call__(self, batch):
        return torch.utils.data.default_collate(batch).reshape(self.shape)

class PaddingCollate(DefaultCollate):
    def __init__(self, mode: Literal["SET_MAX_LEN"], max_len: int, pad_dim: int = 2, pad_value: Union[int, float] = 0):
        self.pad_value = pad_value
        self.max_len = max_len
        self.pad_dim = pad_dim
    
    def __call__(self, batch):
        res = []
        for tensor in batch:
            size = tensor.shape[self.pad_dim]
            if size < self.max_len:
                pad_amount = self.max_len - size
                tensor = torch.nn.functional.pad(tensor, (0, pad_amount))
            else:
                tensor: torch.Tensor = tensor
                tensor = tensor.narrow(self.pad_dim, 0, self.max_len)
            res.append(tensor)
        return torch.stack(res)

def build_collate_fn(*columns: DefaultCollate):
    def collate_fn(batches):
        return tuple(column(batch) for batch, column in zip(zip(*batches), columns))
    return collate_fn

class MemoryLoadedDataLoader:
    def __init__(self, data_loader, device="cpu"):
        self.data_loader = data_loader
        self.mem = [
            tuple(item.to(device) for item in t)
            for t in data_loader
        ]

    def __iter__(self):
        return iter(self.mem)


class MemoryLoadedDataset(Dataset):
    def __init__(self, dataset, device="cpu"):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.mem = [
            tuple(item.to(self.device) for item in self.dataset[idx])
            for idx in range(len(self.dataset))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.mem[index]
