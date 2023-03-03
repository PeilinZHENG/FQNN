import os
import math
import time
import torch
import torch.multiprocessing as mp
from functools import partial
from utils import mymkdir
from Data_Ins import l2c
import mkl, warnings

warnings.filterwarnings('ignore')
mkl.set_num_threads(1)


def Ham(L, mu):
    H = torch.diag_embed(-mu * torch.ones(L ** 2)).type(torch.complex128)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            # nearest neighbor
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            H[nx, n] = H[nx, n] - 1.
            H[n, nx] = H[n, nx] - 1.
            H[ny, n] = H[ny, n] - 1.
            H[n, ny] = H[n, ny] - 1.
            # # next nearest neighbor
            # n1 = l2c(x + 1, y + 1, L)
            # n2 = l2c(x + 1, y - 1, L)
            # H[n1, n] = H[n1, n] - 1.
            # H[n, n1] = H[n, n1] - 1.
            # H[n2, n] = H[n2, n] - 1.
            # H[n, n2] = H[n, n2] - 1.
    return H


def genData(amount, L):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, labels = [], []
    E_mu = -0.06
    for i in range(int(amount / 2)):
        U = torch.rand(1) + 2.5
        H = Ham(L, U / 2)
        Hs.append(H)
        labels.append([E_mu, U.item(), 0.])
        U = torch.rand(1) + 0.5
        H = Ham(L, U / 2)
        Hs.append(H)
        labels.append([E_mu, U.item(), 1.])
    Hs = torch.stack(Hs, dim=0)        # (amount, L ** 2, L ** 2)
    labels = torch.tensor(labels)      # (amount, 3)
    return Hs, labels


if __name__ == "__main__":
    L = 20

    TYPE = 'test'
    amount, processors = 4, 50  # total_amount = amount * processors

    # torch.manual_seed(0)

    path = 'datasets/FK_{}'.format(L)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t= time.time()

    Hs, labels = [], []
    mp.set_start_method('fork', force=True)
    pool = mp.Pool(processes=processors)
    res = pool.imap(partial(genData, L=L), amount * torch.ones(processors, dtype=torch.int32))
    for (h, l) in res:
        Hs.append(h)
        labels.append(l)
    pool.close()
    pool.join()
    Hs = torch.cat(Hs, dim=0).unsqueeze(1)     # (amount * processors, 1, L ** 2, L ** 2)
    labels = torch.cat(labels, dim=0)          # (amount * processors, 3)

    delta_t = time.time() - t
    print(delta_t, '\n', L, Hs.shape, labels.shape)
    f = open('{}/info.txt'.format(path), 'w')
    f.write('time={}\nL={}\ndataset.shape={}\nlabels.shape={}'.format(delta_t, L, Hs.shape, labels.shape))
    f.close()

    # save
    torch.save(Hs, '{}/dataset.pt'.format(path))           # (amount * processors, 1, L ** 2, L ** 2)
    torch.save(labels, '{}/labels.pt'.format(path))        # (amount * processors, 3)