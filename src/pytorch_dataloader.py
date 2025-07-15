from torch.utils.data import Dataset


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
