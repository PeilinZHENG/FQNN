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


def Ham2(L, mu):
    device = mu.device
    H = torch.zeros((L ** 2, L ** 2)).type(torch.complex128).to(device)  # (2**L,2**L)
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
    mu = mu.unsqueeze(-1)  # (amount/2,1)
    H = H.unsqueeze(0) + torch.diag_embed(-mu * torch.ones(L ** 2,device=device)).type(torch.complex128)  # (amount/2,2**L,2**L)
    return H


def genData(amount, L):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, labels = [], []
    for i in range(int(amount / 2)):
        U = torch.rand(1) + 2.5
        H = Ham(L, U / 2)
        Hs.append(H)
        labels.append([(U/2).item(),U.item(), 0.])
        U = torch.rand(1) + 0.5
        H = Ham(L, U / 2)
        Hs.append(H)
        labels.append([(U/2).item(),U.item(), 1.])
    Hs = torch.stack(Hs, dim=0)        # (amount, L ** 2, L ** 2)
    labels = torch.tensor(labels)      # (amount, 3)
    return Hs, labels


def genData2(amount, L):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    # random phase
    T_points = [0.05, 0.2, 0.2]  # [T1,T2,T3]
    U_points = [0.5, 0.5, 2.5]  # [U1,U2,U3]
    rd1 = torch.rand(size=[int(amount / 2)])
    rd2 = torch.rand(size=[int(amount / 2)])
    rd2 = torch.sqrt(rd2)
    U_random = rd2 * rd1 * U_points[0] + rd2 * (1 - rd1) * U_points[1] + (1 - rd2) * rd1 * U_points[2]
    T_random = rd2 * rd1 * T_points[0] + rd2 * (1 - rd1) * T_points[1] + (1 - rd2) * rd1 * T_points[2]
    H_random = Ham2(L, U_random / 2)  # (amount/2,2**L,2**L)

    # ordered phase
    T_points = [0.05, 0.05, 0.15]
    U_points = [1., 5.0, 4.0]
    rd1 = torch.rand(size=[int(amount / 2)])
    rd2 = torch.rand(size=[int(amount / 2)])
    rd2 = torch.sqrt(rd2)
    U_order = rd2 * rd1 * U_points[0] + rd2 * (1 - rd1) * U_points[1] + (1 - rd2) * rd1 * U_points[2]  # (amount/2)
    T_order = rd2 * rd1 * T_points[0] + rd2 * (1 - rd1) * T_points[1] + (1 - rd2) * rd1 * T_points[2]
    H_order = Ham2(L, U_order / 2)  # (amount/2,2**L,2**L)
    Hs = torch.cat((H_random, H_order), dim=0)
    Us = torch.cat((U_random, U_order), dim=0)  # (amount)
    Ts = torch.cat((T_random, T_order), dim=0)
    phase_labels = torch.repeat_interleave(torch.tensor([0, 1]), (int(amount / 2)))  # (amount)
    labels = torch.stack((Us, Ts, phase_labels), dim=-1)  # (amount,3)
    return Hs, labels


if __name__ == "__main__":
    L = 12

    TYPE = 'train'
    amount, processors = 200, 50  # total_amount = amount * processors

    # torch.manual_seed(0)
    # genData2(amount, L)
    # exit(0)

    path = 'datasets/FK_{}'.format(L)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t = time.time()

    Hs, labels = [], []
    mp.set_start_method('fork', force=True)
    pool = mp.Pool(processes=processors)
    res = pool.imap(partial(genData2, L=L), amount * torch.ones(processors, dtype=torch.int32))
    for (h, l) in res:
        Hs.append(h)
        labels.append(l)
    pool.close()
    pool.join()
    Hs = torch.cat(Hs, dim=0).unsqueeze(1)  # (amount * processors, 1, L ** 2, L ** 2)
    labels = torch.cat(labels, dim=0)  # (amount * processors, 2)

    delta_t = time.time() - t
    print(delta_t, '\n', L, Hs.shape, labels.shape)
    f = open('{}/info.txt'.format(path), 'w')
    f.write('time={}\nL={}\ndataset.shape={}\nlabels.shape={}'.format(delta_t, L, Hs.shape, labels.shape))
    f.close()

    # save
    torch.save(Hs, '{}/dataset.pt'.format(path))  # (amount * processors, 1, L ** 2, L ** 2)
    torch.save(labels, '{}/labels.pt'.format(path))  # (amount * processors, 2)
