import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.utils.data import _utils
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


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        super(MyDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, pin_memory=pin_memory)

    def _get_iterator(self):
        return MySingleProcessDataLoaderIter(self)


class MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data, index

if __name__ == "__main__":
    bz = 16
    traindata = 0.1 * torch.arange(100).unsqueeze(-1).tile(1, 2)
    trainlabels = ((1 - (-1) ** torch.arange(100)) / 2)
    valdata = -0.1 * torch.arange(50).unsqueeze(-1).tile(1, 2)
    vallabels = ((1 - (-1) ** torch.arange(50)) / 2)
    train_dataset = LoadData(traindata, trainlabels)
    val_dataset = LoadData(valdata, vallabels)
    # generator = torch.Generator()
    # generator.manual_seed(0)
    # sampler = RandomSampler(train_dataset, replacement=True, generator=generator)
    # sampler = CtrlRandomSampler(train_dataset)
    train_loader = MyDataLoader(train_dataset, batch_size=bz, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bz, shuffle=False, num_workers=2, pin_memory=True)
    iterable = iter(train_loader)
    for epoch in range(3):
        for i, ((x, target), index) in enumerate(train_loader):
            print(epoch, i, 'train\n', x, '\n', index)
            # print(next(iter(train_loader._index_sampler)))

        # train_loader.dataset.dataload = 0.111 * torch.arange(100).unsqueeze(-1).tile(1, 2)

