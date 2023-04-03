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
    data = torch.load('datasets/FK_12_QPT/train/dataset.pt')
    labels = torch.load('datasets/FK_12_QPT/train/labels.pt')
    torch.save(data.cpu(), 'datasets/FK_12_QPT/train/dataset.pt')
    torch.save(labels.cpu(), 'datasets/FK_12_QPT/train/dataset.pt')

    exit(0)
    a = torch.rand(50, device='cuda') * 2 - 1
    good_idx = torch.nonzero(a > 0, as_tuple=True)[0]
    t = time.time()
    bad_idx = torch.tensor([i for i in range(len(a)) if i not in good_idx], dtype=good_idx.dtype,
                           device=good_idx.device)
    t0 = time.time()
    print(t0 - t)
    bad_idx = torch.nonzero(a < 0, as_tuple=True)[0]
    t1 = time.time()
    print(t1 - t0)



    exit(0)


    data = np.load('results/result.npz')
    L = 12
    SE = torch.from_numpy(data['SE'])
    OP = torch.from_numpy(data['op'])
    torch.save(SE, 'results/FK_12_/SE+OP/SE_0.020.pt')
    torch.save(OP, 'results/FK_12_/SE+OP/OP_0.020.pt')
    exit(0)
    U = torch.from_numpy(data['U'])
    T = torch.from_numpy(data['T'])
    mu = U / 2
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1)
    mymkdir(f'datasets/FK_{L}_')
    mymkdir(f'datasets/FK_{L}_/train')
    mymkdir(f'datasets/FK_{L}_/test')
    torch.save(H0, f'datasets/FK_{L}_/train/dataset.pt')
    torch.save(H0, f'datasets/FK_{L}_/test/dataset.pt')
    torch.save(SE, f'datasets/FK_{L}_/train/SE.pt')
    torch.save(SE, f'datasets/FK_{L}_/test/SE.pt')
    labels = torch.stack((U, T, torch.cat((torch.zeros(23), torch.ones(47)))), dim=1)
    print(labels)
    torch.save(labels, f'datasets/FK_{L}_/train/labels.pt')
    torch.save(labels, f'datasets/FK_{L}_/test/labels.pt')


    exit(0)

    L = 10
    T = 0.1
    Net = 'Naive_2d_sf_0'
    data = np.load(f'results/FK_{L}/{Net}/PD_{T:.3f}.npy')
    U = data[0]
    P = data[1]
    OP = data[2]
    PTPs = {'0.100': (1.5, 1.76), '0.110': (1.7, 2.01), '0.120': (1.8, 2.28), '0.130': (2.0, 2.6), '0.140': (2.2, 3.03),
            '0.150': (2.4, 3.99)}
    PTP, QMCPTP = PTPs[f'{T:.3f}']

    '''plot phase diagram'''
    fig, ax1 = plt.subplots()
    plt.title(f'Metal VS Insulator / T={T:.3f}, L={L}')
    # plt.axis([U[0], U[-1], 0., 1.])
    ax1.set_xlim([U[0], U[-1]])
    ax1.set_xlabel('U')
    ax1.set_ylabel('P', c='r')
    ax1.set_ylim([0.4, 0.6])
    ax1.set_yticks(0.4 + 0.02 * np.arange(11))
    ax1.scatter(U, P, s=10, c='r', marker='o')
    ax1.plot([U[0], U[-1]], [0.5, 0.5], 'ko--', linewidth=0.5, markersize=0.1)
    if PTP is not None: ax1.plot([PTP, PTP], [0., 1.], 'go--', linewidth=0.5, markersize=0.1)
    if QMCPTP is not None: ax1.plot([QMCPTP, QMCPTP], [0., 1.], 'yo--', linewidth=0.5, markersize=0.1)
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('OP', c='b')
    ax2.set_ylim([0., 1.])
    ax2.set_yticks(0.1 * np.arange(11))
    ax2.scatter(U, OP, s=10, c='b', marker='^')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.show()
    plt.close()


    exit(0)
    bz = 16
    traindata = 0.1 * torch.arange(100).unsqueeze(-1).tile(1, 2)
    trainlabels = ((1 - (-1) ** torch.arange(100)) / 2)
    trainSEinit = torch.ones(100, 2)
    train_dataset = LoadData(traindata, trainlabels, trainSEinit)
    # generator = torch.Generator()
    # generator.manual_seed(0)
    # sampler = RandomSampler(train_dataset, replacement=True, generator=generator)
    # sampler = CtrlRandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True, num_workers=0, pin_memory=True)
    for epoch in range(2):
        for i, (x, se, index, target) in enumerate(train_loader):
            print(epoch, i, 'train\n', x, '\n', index)
            print(se)

            train_loader.dataset.SEinit[index] = index.float().unsqueeze(-1).tile(1, 2)


    exit(0)

    model = TestNN(True).to('cuda')
    print(model.scale.device)
    torch.save(model.state_dict(), 'test.pth.tar')

    check = torch.load('test.pth.tar', map_location="cpu")
    model.load_state_dict(check)
    print(check)
