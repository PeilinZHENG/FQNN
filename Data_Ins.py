import time
import torch
import torch.multiprocessing as mp
from functools import partial
from utils import mymkdir


def pbc(x, L):
    return x % L


def l2c(x, y, L):
    x, y = pbc(x, L), pbc(y, L)
    return x + y * L


def c2l(n, L):
    assert n < L ** 2
    return n % L, n // L


def Ham(k, L):
    H = torch.zeros((L ** 2, L ** 2), dtype=torch.complex64)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            H[nx, n] = H[nx, n] + (-1) ** y
            H[n, nx] = H[n, nx] + (-1) ** y
            H[ny, n] = H[ny, n] + 1. + (-1) ** y * (1 - k)
            H[n, ny] = H[n, ny] + 1. + (-1) ** y * (1 - k)
            H[n1, n] = H[n1, n] + 1j * (-1) ** y * k / 2
            H[n, n1] = H[n, n1] - 1j * (-1) ** y * k / 2
            H[n2, n] = H[n2, n] - 1j * (-1) ** y * k / 2
            H[n, n2] = H[n, n2] + 1j * (-1) ** y * k / 2
    return H


def nnn_mask(L):
    m = torch.eye(L ** 2, dtype=torch.complex64)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            m[nx, n] = m[nx, n] + 1.
            m[n, nx] = m[n, nx] + 1.
            m[ny, n] = m[ny, n] + 1.
            m[n, ny] = m[n, ny] + 1.
            m[n1, n] = m[n1, n] + 1.
            m[n, n1] = m[n, n1] + 1.
            m[n2, n] = m[n2, n] + 1.
            m[n, n2] = m[n, n2] + 1.
    return m


def genData(amount, L):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, labels = [], []
    for i in range(int(amount / 2)):
        H = Ham(0.1 + 0.4 * torch.rand(1), L)
        Hs.append(H)
        labels.append(0)
        H = Ham(0.5 + 0.5 * torch.rand(1), L)
        Hs.append(H)
        labels.append(1)
    Hs = torch.stack(Hs, dim=0)        # (amount, L ** 2, L ** 2)
    labels = torch.tensor(labels)      # (amount,)
    return Hs, labels


if __name__ == "__main__":
    import mkl, warnings

    warnings.filterwarnings('ignore')
    mkl.set_num_threads(1)

    L = 16

    TYPE = 'train'
    amount, processors = 400, 50  # total_amount = amount * processors

    # torch.manual_seed(0)

    path = 'datasets/Ins_{}'.format(L)
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
    labels = torch.cat(labels, dim=0)          # (amount * processors,)

    delta_t = time.time() - t
    print(delta_t, '\n', L, Hs.shape, labels.shape)
    f = open('{}/info.txt'.format(path), 'w')
    f.write('time={}\nL={}\ndataset.shape={}\nlabels.shape={}'.format(delta_t, L, Hs.shape, labels.shape))
    f.close()

    # save
    torch.save(Hs, '{}/dataset.pt'.format(path))                  # (amount * processors, 1, L ** 2, L ** 2)
    torch.save(labels.long(), '{}/labels.pt'.format(path))        # (amount * processors,)

    # H = Ham(0.1, L)
    # print(H)