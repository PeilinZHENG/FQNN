import os
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from functools import partial
import mkl, warnings
from utils import mymkdir
from Data_Ins import Ham

warnings.filterwarnings('ignore')
mkl.set_num_threads(1)

L = 12

TYPE = 'train'
k_amount, E_amount = 90, 1     # amount = k_amount * (E_amount * 8)
processors = 50                # total_amount = amount * processors

# torch.manual_seed(0)


def Energy(kx, ky, k):
    return 2 * np.sqrt(np.cos(kx) ** 2 + np.cos(ky) ** 2 + (np.sin(ky) ** 2) * (1 - k + k * np.sin(kx)) ** 2)


def Bound(k):
    index = np.array([[0, 0], [np.pi, 0], [-np.pi / 2, np.pi / 2], [np.pi / 2, 0], [np.arcsin(k / (1 + k)), np.pi / 2],
                      [-np.pi / 2, 0]])
    Es = Energy(index[:, 0], index[:, 1], k)
    return min(Es), max(Es)


def genData(k_amount, E_amount, L):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, Es, labels = [], [], []
    for i in range(k_amount):
        if i < int(k_amount / 3):
            k = 0.1 + 0.4 * torch.rand(1)
            H = Ham(k, L)
            low, up = Bound(k.item())
            for _ in range(E_amount):
                E = torch.cat((up + torch.rand(2), low + (up - low) * torch.rand(4), low * (2 * torch.rand(2) - 1)))
                E *= torch.tensor([1, -1, 1, 1, -1, -1, 1, 1])
                Es.append(E)
                labels.extend([0, 0, 2, 2, 2, 2, 0, 0])     # 0: Normal, 1: Topological, 2: Metal
        else:
            k = 0.5 + 0.5 * torch.rand(1)
            H = Ham(k, L)
            low, up = Bound(k.item())
            for _ in range(E_amount):
                E = torch.cat((up + torch.rand(2), low + (up - low) * torch.rand(2), low * (2 * torch.rand(4) - 1)))
                E *= torch.tensor([1, -1, 1, -1, 1, 1, 1, 1])
                Es.append(E)
                labels.extend([0, 0, 2, 2, 1, 1, 1, 1])     # 0: Normal, 1: Topological, 2: Metal
        Hs.extend([H] * 8 * E_amount)
    Hs = torch.stack(Hs, dim=0)        # (k_amount * (E_amount * 8), L ** 2, L ** 2)
    Es = torch.cat(Es, dim=0)          # (k_amount * (E_amount * 8),)
    labels = torch.tensor(labels)      # (k_amount * (E_amount * 8),)
    return Hs, Es, labels


if __name__ == "__main__":
    path = 'datasets/Ins_{}_nfz'.format(L)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t= time.time()

    Hs, Es, labels = [], [], []
    mp.set_start_method('fork', force=True)
    pool = mp.Pool(processes=processors)
    res = pool.imap(partial(genData, E_amount=E_amount, L=L), k_amount * torch.ones(processors, dtype=torch.int32))
    for (h, E, l) in res:
        Hs.append(h)
        Es.append(E)
        labels.append(l)
    pool.close()
    pool.join()
    Hs = torch.cat(Hs, dim=0).unsqueeze(1)     # (amount * processors, 1, L ** 2, L ** 2)
    Es = torch.cat(Es, dim=0)                  # (amount * processors,)
    labels = torch.cat(labels, dim=0)          # (amount * processors,)

    delta_t = time.time() - t
    print(delta_t, '\n', L, Hs.shape, Es.shape, labels.shape)
    f = open('{}/info.txt'.format(path), 'w')
    f.write('time={}\nL={}\ndataset.shape={}\nEs.shape={}\nlabels.shape={}'.format(
        delta_t, L, Hs.shape, Es.shape, labels.shape))
    f.close()

    # save
    torch.save(Hs, '{}/dataset.pt'.format(path))                  # (amount * processors, 1, L ** 2, L ** 2)
    torch.save(Es, '{}/Es.pt'.format(path))                       # (amount * processors,)
    torch.save(labels.long(), '{}/labels.pt'.format(path))        # (amount * processors,)

    # H = Ham(0.1, L)
    # print(H)