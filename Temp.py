import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.utils.data import _utils
from utils import CtrlRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from FK_Data import Ham
from utils import mymkdir
import time
np.set_printoptions(precision=3, linewidth=95, suppress=True)

torch.manual_seed(0)


class TestNN(nn.Module):
    def __init__(self, scale):
        super(TestNN, self).__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.scale = torch.tensor(5.)
        self.lin = nn.Linear(5, 5)


class LoadData(Dataset):
    def __init__(self, dataload, labels, SEinit):
        self.dataload = dataload
        self.labels = labels
        self.SEinit = SEinit

    def __getitem__(self, index):
        return self.dataload[index], self.SEinit[index], index, self.labels.long()

    def __len__(self):
        return self.dataload.shape[0]


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, pin_memory):
        super(MyDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=0, pin_memory=pin_memory)

    def _get_iterator(self):
        return MySingleProcessDataLoaderIter(self)


class MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(MySingleProcessDataLoaderIter, self).__init__(loader)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data, index


if __name__ == "__main__":
    OP = np.load('results/FK_12_QPT_/SE+OP/kOP_0.005.npy')
    data = np.load('results/FK_12_QPT_/Naive_3/m=0.7/PD_0.005.npy')
    x, P = data[0], data[1]
    np.save('results/FK_12_QPT_/QPT_T=0.005.npy', np.concatenate((np.stack((x, P), axis=0), OP), axis=0))
    labels = ['cb', 'stripe']
    fig, ax1 = plt.subplots()
    plt.axis([x[0], x[-1], 0., 1.])
    plt.title('Checkerboard VS Stripe / T=0.005, L=12')
    ax1.set_xlabel("t'")
    ax1.set_xlim([x[0], x[-1]])
    ax1.set_ylabel('P', c='r')
    ax1.set_ylim([0., 1.])
    ax1.set_yticks(0.1 * np.arange(11))
    ax1.scatter(x, P, s=10, c='r', marker='o')
    ax1.plot([x[0], x[-1]], [0.5, 0.5], 'ko--', linewidth=0.5, markersize=0.1)
    ax1.plot([0.575, 0.575], [0., 1.], 'go--', linewidth=0.5, markersize=0.1)
    ax1.plot([1 / np.sqrt(2), 1 / np.sqrt(2)], [0., 1.], 'yo--', linewidth=0.5, markersize=0.1)
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('OP', c='b')
    ax2.set_ylim([0., 1.])
    ax2.set_yticks(0.1 * np.arange(11))
    for i, op in enumerate(OP):
        ax2.scatter(x, op, s=10, marker='^', label=labels[i])
    ax2.legend(loc='lower right')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.show()
    plt.close()



