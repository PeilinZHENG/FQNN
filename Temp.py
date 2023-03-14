import torch
from torch.utils.data import DataLoader, Dataset
from utils import CtrlRandomSampler

torch.manual_seed(0)

class LoadData(Dataset):
    def __init__(self, dataload, labels):
        self.dataload = dataload
        self.labels = labels

    def __getitem__(self, index):
        return self.dataload[index], self.labels.long()

    def __len__(self):
        return self.dataload.shape[0]


if __name__ == "__main__":
    bz = 16
    traindata = 0.1 * torch.arange(100).unsqueeze(-1).tile(1, 2)
    trainlabels = ((1 - (-1) ** torch.arange(100)) / 2)
    valdata = -0.1 * torch.arange(50).unsqueeze(-1).tile(1, 2)
    vallabels = ((1 - (-1) ** torch.arange(50)) / 2)
    train_dataset = LoadData(traindata, trainlabels)
    val_dataset = LoadData(valdata, vallabels)
    sampler = CtrlRandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=bz, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bz, shuffle=False, num_workers=2, pin_memory=True)
    for epoch in range(3):
        for i, (x, target) in enumerate(train_loader):
            if i == 1:
                print(epoch, 'train\n', x)

        for i, (x, target) in enumerate(val_loader):
            if i == 1:
                print(epoch, 'val\n', x)

