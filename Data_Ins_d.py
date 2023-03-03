import os
import math
import time
import torch
import torch.multiprocessing as mp
from functools import partial
from ent import correlation
from utils import mymkdir
from Data_Ins import l2c


def Ham(k, L, strength=(1., 0., 0.)):
    H = torch.diag_embed((2 * torch.rand(L ** 2) - 1) * strength[0]).type(torch.complex64)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            disorder = (2 * torch.rand(2) - 1) * strength[1]
            H[nx, n] = H[nx, n] + (-1) ** y + disorder[0]
            H[n, nx] = H[n, nx] + (-1) ** y + disorder[0]
            H[ny, n] = H[ny, n] + 1. + (-1) ** y * (1 - k) + disorder[1]
            H[n, ny] = H[n, ny] + 1. + (-1) ** y * (1 - k) + disorder[1]
            disorder = (2 * torch.rand(2) - 1) * strength[2]
            H[n1, n] = H[n1, n] + 1j * (-1) ** y * k / 2 + disorder[0]
            H[n, n1] = H[n, n1] - 1j * (-1) ** y * k / 2 + disorder[0]
            H[n2, n] = H[n2, n] - 1j * (-1) ** y * k / 2 + disorder[1]
            H[n, n2] = H[n, n2] + 1j * (-1) ** y * k / 2 + disorder[1]
    return H


def ChernNumber(H, L, qlt, S):
    C = correlation(H).transpose(-2, -1)
    cn = 0.
    for x in range(L):
        for y in range(L):
            i = l2c(x, y, L)
            for iqlt in range(len(qlt)):
                j = l2c(x + qlt[iqlt, 0, 0], y + qlt[iqlt, 0, 1], L)
                k = l2c(x + qlt[iqlt, 1, 0], y + qlt[iqlt, 1, 1], L)
                cn += ((C[i, j] * C[j, k] * C[k, i]).imag * S[iqlt]).item()
    return -cn * 24 * math.pi / L ** 2


def Stri(x):
    # x = torch.cat((x, torch.zeros((x.shape[0], 2, 1))), dim=-1)
    # return torch.linalg.cross(x[:, 0], x[:, 1])[:, -1] / 2
    return torch.linalg.det(x.double()) / 2


def Stri_(x): # Heron's formula, no sign
    a = torch.linalg.norm(x[:, 0], dim=1)
    b = torch.linalg.norm(x[:, 1], dim=1)
    c = torch.linalg.norm(x[:, 0] - x[:, 1], dim=1)
    s = (a + b + c) / 2
    return torch.sqrt(torch.abs(s * (s - a) * (s - b) * (s - c)))


def countriangle(dmax):
    """The list of triangles within a cut-off scale dmax."""
    qlt = []
    for dx in range(-dmax, dmax + 1):
        for dy in range(dmax + 1):
            for dx2 in range(-dmax, dmax + 1):
                for dy2 in range(-dy, dmax + 1):
                    if (abs(dx + dx2) <= dmax and abs(dy + dy2) <= dmax and (dy2 * dx - dy * dx2) > 0 and (
                            dy > 0 or dx > 0) and (dx + dx2 > 0 or dy + dy2 > 0)):
                        qlt.append([[dx, dy], [dx + dx2, dy + dy2]])
    return torch.tensor(qlt)


def disp_mask(L):
    x_mask, y_mask = torch.zeros((L ** 2, L ** 2)), torch.zeros((L ** 2, L ** 2))
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            x_mask[nx, n] = x_mask[nx, n] - 1.
            x_mask[n, nx] = x_mask[n, nx] + 1.
            x_mask[n1, n] = x_mask[n1, n] - 1.
            x_mask[n, n1] = x_mask[n, n1] + 1.
            x_mask[n2, n] = x_mask[n2, n] - 1.
            x_mask[n, n2] = x_mask[n, n2] + 1.
            y_mask[ny, n] = y_mask[ny, n] - 1.
            y_mask[n, ny] = y_mask[n, ny] + 1.
            y_mask[n1, n] = y_mask[n1, n] - 1.
            y_mask[n, n1] = y_mask[n, n1] + 1.
            y_mask[n2, n] = y_mask[n2, n] + 1.
            y_mask[n, n2] = y_mask[n, n2] - 1.
    return x_mask, y_mask


def velocity(H):
    L = int(math.sqrt(H.shape[-1]))
    if os.path.exists('datasets/disp_mask_{}.pt'.format(L)):
        x_mask, y_mask = torch.load('datasets/disp_mask_{}.pt'.format(L))
    else:
        x_mask, y_mask = disp_mask(L)
        torch.save((x_mask, y_mask), 'datasets/disp_mask_{}.pt'.format(L))
    vx, vy = 1j * x_mask * H, 1j * y_mask * H
    # print(vx.norm(1) + vy.norm(1) - (x_mask * y_mask * H).norm(1) + torch.diag(H).norm(1) - H.norm(1))
    return vx.type(torch.complex128), vy.type(torch.complex128)


def Kubo(H):
    vx, vy = velocity(H)
    values, states = torch.linalg.eigh(H.type(torch.complex128))
    M, N = None, None
    for i, v in enumerate(values):
        if v > 0:
            M, N = states[:, :i].T, states[:, i:].T
            EM, EN = values[:i], values[i:]
            break
    if M is None or N is None: return 0
    cn = 0.
    for m, em in zip(M, EM):
        for n, en in zip(N, EN):
            temp = (m.conj().unsqueeze(0) @ vy @ n.unsqueeze(1)) * (n.conj().unsqueeze(0) @ vx @ m.unsqueeze(1))
            temp = temp - (m.conj().unsqueeze(0) @ vx @ n.unsqueeze(1)) * (n.conj().unsqueeze(0) @ vy @ m.unsqueeze(1))
            temp = temp / (en - em) ** 2
            cn += temp.item()
    return (cn * 1j * 2 * math.pi / H.shape[-1]).real


def genData(strength, amount, L, qlt, S, gap=0.):
    torch.manual_seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    Hs, labels = [], []
    for i in range(amount):
        if i < amount / 3:
            while True:
                H = Ham(0.1 + 0.4 * torch.rand(1), L, strength)
                values = torch.linalg.eigvalsh(H.type(torch.complex128))
                minval = torch.min(torch.abs(values))
                if minval < 1e-2:
                    continue
                else:
                    Hs.append(H)
                    labels.append(0)
                    break
        else:
            while True:
                H = Ham(0.5 + 0.5 * torch.rand(1), L, strength)
                values = torch.linalg.eigvalsh(H.type(torch.complex128))
                minval = torch.min(torch.abs(values))
                if minval < 1e-2: continue
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
    import mkl, warnings

    warnings.filterwarnings('ignore')
    mkl.set_num_threads(1)

    L = 12

    TYPE, gap = 'test', 0.2
    amount, processors = 120, 50  # total_amount = amount * processors

    # torch.manual_seed(0)

    path = 'datasets/Ins_{}_d{}'.format(L, gap)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t= time.time()

    strength = torch.stack(
        (torch.linspace(1., 3, amount), torch.linspace(0., 1., amount), torch.linspace(0., 0.5, amount)), dim=1)
    qlt = countriangle(3)
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
