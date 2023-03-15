import os
import time
import torch
import torch.multiprocessing as mp
from functools import partial
from utils import mymkdir
from Data_Ins_d import Ham, ChernNumber, countriangle, Stri, Stri_
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)

L = 12

TYPE, gap = 'test', 0.2
amount, processors = 80, 50  # total_amount = amount * processors

# torch.manual_seed(0)


def genData(strength, amount, L, qlt, S, gap=0.):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, labels = [], []
    for i in range(amount):
        if i < amount / 3:
            H = Ham(0.1, L, strength)
            Hs.append(H)
            labels.append(0)
        else:
            while True:
                H = Ham(1., L, strength)
                cn = abs(ChernNumber(H, L, qlt, S))
                if cn < 0.5 - gap:
                    Hs.append(H)
                    labels.append(0)
                    break
                elif cn > 0.5 + gap:
                    Hs.append(H)
                    labels.append(1)
                    break
    Hs = torch.stack(Hs, dim=0)        # (amount, L ** 2, L ** 2)
    labels = torch.tensor(labels)      # (amount,)
    return Hs, labels


if __name__ == "__main__":
    path = 'datasets/Ins_{}_db{}'.format(L, gap)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t= time.time()

    strength = torch.stack(
        (torch.linspace(1., 3, amount), torch.linspace(0., 1., amount), torch.linspace(0., 0.5, amount)), dim=1)
    qlt = countriangle(2)
    S = Stri(qlt)
    Hs, labels = [], []
    mp.set_start_method('fork', force=True)
    pool = mp.Pool(processes=processors)
    res = pool.imap(partial(genData, amount=processors, L=L, qlt=qlt, S=S, gap=gap), strength)
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

    # kappa = 1.
    # H = Ham(kappa, L, (3, 1., 0.5))
    # print(torch.dist(H, H.mH))
    # t = time.time()
    # qlt = countriangle(10)
    # S = Stri(qlt)
    # print(ChernNumber(H, L, qlt, S))
    # print(time.time() - t)

    # E, V = torch.linalg.eigh(H.type(torch.complex128))
    # E = E.numpy()
    #
    # L = 10000
    # kx = np.arange(L) / L * 2 * np.pi - np.pi
    # ky = np.arange(int(L / 2)) / (L / 2) * np.pi - np.pi / 2
    #
    # E1 = []
    # for x in kx:
    #     for y in ky:
    #         e = 2 * np.sqrt(np.cos(y) ** 2 + np.cos(x) ** 2 + np.sin(y) ** 2 * (1 - kappa + kappa * np.sin(x)) ** 2)
    #         E1.append(e)
    #         E1.append(-e)
    # E1 = np.array(sorted(E1))
    #
    # # plt.figure()
    # # plt.scatter(np.zeros(len(E)), E, c='b', s=10)
    # # plt.scatter(np.ones(len(E1)), E1, c='r', s=10)
    # # plt.xlabel('Count')
    # # plt.ylabel('E')
    # # plt.show()
    # # plt.close()
    #
    # for e in E:
    #     temp = np.abs(e - E1)
    #     m = np.min(temp)
    #     if m > 1e-6:
    #         print(e, m)
    #     else:
    #         index = np.argmin(temp)
    #         E1 = np.delete(E1, index)
