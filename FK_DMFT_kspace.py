import numpy as np
from functools import partial
from FK_Data import Ham
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, linewidth=110, suppress=True)
np.random.seed(4396)


L = 12
size = 4
T = 0.005
U = 1.
tp = np.linspace(0., 1.3, 131)
epochs = 2000
count = 20
gap = 1
tol_bi = 1e-7
iota = 0
momentum = 0.5
# adjMu = 0.25 * np.ones(len(tp))
adjMu = np.concatenate((np.linspace(0.5, 0.2, 71), np.linspace(0.2, 0.5, len(tp) - 71)))
iomega = 1j * (2 * np.arange(-count, count)[:, None] + 1) * np.pi * T  # (count * 2, 1)


def fourier():
    n = L // 2
    temp = np.arange(-np.pi / 2, np.pi / 2, np.pi / n)
    kx, ky = np.repeat(temp, n), np.tile(temp, n)
    idx = np.arange(n * n)
    res = np.zeros((L * L, L * L), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            x, y, kidx = 2 * i, 2 * j, 4 * idx
            res[x + y * L, kidx] = np.exp(-1j * (kx * x + ky * y))
            x, y, kidx = 2 * i + 1, 2 * j, 4 * idx + 1
            res[x + y * L, kidx] = np.exp(-1j * (kx * x + ky * y))
            x, y, kidx = 2 * i, 2 * j + 1, 4 * idx + 2
            res[x + y * L, kidx] = np.exp(-1j * (kx * x + ky * y))
            x, y, kidx = 2 * i + 1, 2 * j + 1, 4 * idx + 3
            res[x + y * L, kidx] = np.exp(-1j * (kx * x + ky * y))
    return res / n


def Hk(tp):
    n = L // 2
    temp = np.arange(-np.pi / 2, np.pi / 2, np.pi / n)
    # res, i = np.zeros((L * L, L * L)), 0
    # for kx in temp:
    #     for ky in temp:
    #         x = 2 * np.cos(kx)
    #         y = 2 * np.cos(ky)
    #         xy = 2 * tp * np.cos(kx + ky)
    #         x_y = 2 * tp * np.cos(kx - ky)
    #         res[i:i + 4, i:i + 4] = np.array([[0, x, y, xy], [x, 0, x_y, y], [y, x_y, 0, x], [xy, y, x, 0]])
    #         i += 4
    kx, ky = np.repeat(temp, n), np.tile(temp, n)
    onsite = np.zeros(n * n)
    x = 2 * np.cos(kx)
    y = 2 * np.cos(ky)
    xy = 2 * tp * (np.cos(kx + ky) + np.cos(kx - ky))
    A = np.stack((onsite, x, y, xy), axis=1)
    B = np.stack((x, onsite, xy, y), axis=1)
    C = np.stack((y, xy, onsite, x), axis=1)
    D = np.stack((xy, y, x, onsite), axis=1)
    return np.stack((A, B, C, D), axis=1)  # (L * L / size, size, size)


def diag_embed(A):
    n = A.shape[-1]
    B = np.zeros(list(A.shape) + [n], dtype=A.dtype)
    r = np.arange(n)
    B[..., r, r] = A[..., r]
    return B


def calc_Gloc(H0, SE):
    temp = diag_embed(iomega - SE)[:, :, None]
    Gloc = np.diagonal(np.linalg.inv(temp - H0), axis1=-2, axis2=-1)  # (bz, count * 2, L * L / size, size)
    return np.mean(Gloc, axis=2)  # (bz, count * 2, size)


def calc_nf(E_mu, UoverWI):
    z = E_mu / T - np.sum(np.log(1 - UoverWI) * np.exp(iomega * iota), axis=1)  # (bz, size)
    return np.clip(np.nan_to_num((1 / (1 + np.exp(z))).real, nan=0.), a_min=0., a_max=1.)  # (bz, size)


def calc_nd0_avg(mu, H0):
    FDD = np.nan_to_num(1 / (1 + np.exp((np.linalg.eigvalsh(H0) - mu) / T)), nan=0.)
    return np.mean(FDD, axis=(1, 2, 3))  # (bz,)


def fix_filling(a, args, f_ele=True):
    if f_ele:
        return np.mean(calc_nf(a, args), axis=1) - 0.5  # (bz,)
    else:
        return calc_nd0_avg(a, args) - 0.5  # (bz,)


def bisearch(fun, a, args):
    sbz = a.shape[0] ** 0.5
    fa = fun(a, args)
    b = a + gap
    fb = fun(b, args)
    for i in np.nonzero(np.sign(fa) * np.sign(fb) > 0)[0]:
        if fa[i] > 0:
            if fa[i] < fb[i]:
                while fa[i] > 0:
                    b[i], fb[i] = a[i], fa[i]
                    a[i] = a[i] - gap
                    fa[i] = fun(a[i:i + 1], args[i:i + 1])
            else:
                while fb[i] > 0:
                    a[i], fa[i] = b[i], fb[i]
                    b[i] = b[i] + gap
                    fb[i] = fun(b[i:i + 1], args[i:i + 1])
        else:
            if fa[i] < fb[i]:
                while fb[i] < 0:
                    a[i], fa[i] = b[i], fb[i]
                    b[i] = b[i] + gap
                    fb[i] = fun(b[i:i + 1], args[i:i + 1])
            else:
                while fa[i] < 0:
                    b[i], fb[i] = a[i], fa[i]
                    a[i] = a[i] - gap
                    fa[i] = fun(a[i:i + 1], args[i:i + 1])
    index = np.nonzero(np.abs(fa) < tol_bi)
    if len(index[0]) > 0: b[index] = a[index]
    index = np.nonzero(np.abs(fb) < tol_bi)
    if len(index[0]) > 0: a[index] = b[index]
    while True:
        c = (a + b) / 2
        fc = fun(c, args)
        index = np.nonzero(np.abs(fc) < tol_bi)
        if len(index[0]) > 0: a[index], b[index] = c[index], c[index]
        if np.linalg.norm((b - a) / 2) < tol_bi * sbz: return c
        index = np.nonzero(np.sign(fc) * np.sign(fun(a, args)) < 0)
        b[index] = c[index]
        c[index] = a[index]
        a = c


def op_cb(nf):
    L = round(nf.shape[-1] ** 0.5)
    line = (-1) ** np.arange(L)
    mask = np.concatenate([line * (-1) ** i for i in range(L)])
    return 2 * np.abs(np.mean(nf * mask, axis=-1))  # (bz,)


def op_str(nf):
    L = round(nf.shape[-1] ** 0.5)
    line = np.ones(L)
    mask1 = np.concatenate([line * (-1) ** i for i in range(L)])
    mask2 = np.tile((-1) ** np.arange(L), L)
    return 2 * np.maximum(np.abs(np.mean(nf * mask1, axis=-1)), np.abs(np.mean(nf * mask2, axis=-1)))  # (bz,)


if __name__ == "__main__":
    if size == 4:
        H0 = np.stack([Hk(i) for i in tp], axis=0)[:, None]  # (bz, 1, L * L / size, size, size)
    else:
        H0 = np.stack([Ham(L, 0., j.item()).numpy() for j in tp], axis=0)[:, None, None]  # (bz, 1, 1, L * L, L * L)
    # print(np.linalg.eigvalsh(H0)[-1].flatten().tolist())
    mu = bisearch(partial(fix_filling, f_ele=False), np.zeros((len(tp), 1, 1, 1)), H0)  # (bz, 1, 1, 1)
    print('<nd>: {:.3f}'.format(np.mean(calc_nd0_avg(mu, H0))))
    H0 = H0 - diag_embed(np.tile(mu + adjMu[:, None, None, None], (1, 1, 1, size)))
    SE = 0.1 * (2 * np.random.rand(len(tp), count * 2, size) - 1).astype(np.complex128)  # (bz, count * 2, size)
    E_mu = np.zeros((len(tp), 1))  # (bz, 1)
    best_error = 1e10
    for l in range(epochs):
        Gloc = calc_Gloc(H0, SE)  # (bz, count * 2, size)
        WeissInv = 1 / Gloc + SE
        UoverWI = U / WeissInv  # (bz, count * 2, size)
        E_mu = bisearch(fix_filling, E_mu, UoverWI)  # (bz, 1)
        nf = calc_nf(E_mu, UoverWI)[:, None]  # (bz, 1, size)
        print('{} loop <nf>: {:.3f}'.format(l, np.mean(nf)))
        Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv
        error = np.linalg.norm(Gimp - Gloc)
        print("{} loop error: {:.3e}".format(l, error))
        if error < best_error:
            best_error = error
            best_nf = nf.squeeze(1)
        SE = momentum * SE + (1. - momentum) * (WeissInv - 1 / Gimp)  # (bz, count * 2, size)
    for i, op in enumerate(best_nf):
        print(i, '\t', op)
    cb, st = op_cb(best_nf), op_str(best_nf)
    np.save(f'results/FK_{L}_QPT_/SE+OP/kOP_0.005.npy', np.stack((cb, st), axis=0))
    print('checkerboard:\n', cb)
    print('stripe:\n', st)
