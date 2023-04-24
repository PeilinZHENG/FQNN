import numpy as np
from functools import partial
np.set_printoptions(precision=3, linewidth=80, suppress=True)

L = 12
T = 0.005
tp = np.linspace(0., 1.3, 66)
U = np.ones(len(tp))
epochs = 100
count = 100
gap = 1
tol_bi = 1e-7
iota = 0
momentum = 0.5
iomega = 1j * (2 * np.arange(-count, count)[:, None] + 1) * np.pi * T  # (count * 2, 1)
adjMu = np.concatenate((np.linspace(0.5, -0.1, 36), np.linspace(-0.1, 0.05, len(tp) - 36)))
size = L * L


def Hk(tp):
    n = L // 2
    temp = np.arange(-np.pi, np.pi, 2 * np.pi / n)
    kx, ky = np.tile(temp, n), np.repeat(temp, n)
    AA = np.zeros(n * n)
    AB = 2 * np.cos(kx / 2)
    AC = 2 * np.cos(ky / 2)
    AD = 2 * tp * np.cos((kx + ky) / 2)
    BC = 2 * tp * np.cos((kx - ky) / 2)
    A = np.stack((AA, AB, AC, AD), axis=1)
    B = np.stack((AB, AA, BC, AC), axis=1)
    C = np.stack((AC, BC, AA, AB), axis=1)
    D = np.stack((AD, AC, AB, AA), axis=1)
    return np.stack((A, B, C, D), axis=1)  # (size / 4, 4, 4)

def diag_embed(A):
    n = A.shape[-1]
    B = np.zeros_like(list(A.shape) + [n])
    r = np.arange(n)
    B[..., r, r] = A[..., r]
    return B


def calc_Gloc(H0, SE):
    temp = diag_embed(iomega - SE)[:, :, None]
    Gloc = np.diagonal(np.linalg.inv(temp - H0), axis1=-2, axis2=-1) # (bz, count * 2, size / 4, 4)
    return np.sum(Gloc, axis=2)  # (bz, count * 2, 4)


def calc_nf(E_mu, UoverWI):
    z = E_mu / T - np.sum(np.log(1 - UoverWI) * np.exp(iomega * iota), axis=1)  # (bz, 4)
    return np.nan_to_num((1 / (1 + np.exp(z))).real, nan=0.)  # (bz, 4)


def calc_nd0_avg(mu, H0):
    FDD = np.nan_to_num(1 / 1 + np.exp((np.linalg.eigvalsh(H0) - mu) / T), nan=0.)
    return np.mean(FDD, axis=(1, 2, 3))  # (bz,)


def fix_filling(a, args, f_ele=True):
    if f_ele:
        return np.mean(calc_nf(a, args), axis=1) - 0.5  # (bz,)
    else:
        return calc_nd0_avg(a, args) - 0.5  # (bz,)


def bisearch(fun, a, args):
    sbz = a.size(0) ** 0.5
    fa = fun(a, args)
    b = a + gap
    fb = fun(b, args)
    for i in np.nonzero(fa.sign() * fb.sign() > 0)[0]:
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
    index = np.nonzero(fa.abs() < tol_bi)
    if len(index[0]) > 0: b[index] = a[index]
    index = np.nonzero(fb.abs() < tol_bi)
    if len(index[0]) > 0: a[index] = b[index]
    while True:
        c = (a + b) / 2
        fc = fun(c, args)
        index = np.nonzero(fc.abs() < tol_bi)
        if len(index[0]) > 0: a[index], b[index] = c[index], c[index]
        if np.linalg.norm((b - a) / 2) < tol_bi * sbz: return c
        index = np.nonzero(fc.sign() * fun(a, args).sign() < 0)
        b[index] = c[index]
        c[index] = a[index]
        a = c


def calc_sigma(Gloc, nf):
    return U / 2 - (1 - np.sqrt(1 + (U * Gloc) ** 2 + (U * Gloc) * (4 * nf - 2))) / 2 / Gloc  # (bz, count * 2, 4)


if __name__ == "__main__":
    H0 = np.stack([Hk(i) for i in tp], axis=0)[:, None]   # (bz, 1, size / 4, 4, 4)
    mu = bisearch(partial(fix_filling, f_ele=False), np.zeros((len(tp), 1, 1, 1)), H0)  # (bz, 1, 1, 1)
    print('<nd>: {:.3f}'.format(np.mean(calc_nd0_avg(mu, H0))))
    H0 = H0 - diag_embed(np.tile(mu + adjMu[:, None, None, None], (1, 1, 1, 4)))
    sigma = np.random.rand((len(mu), count * 2, 4)) * 0.1 + np.zeros((len(mu), count * 2, 4)) * 1j  # (bz, count * 2, 4)
    E_mu = np.zeros((len(mu), 1))  # (bz, 1)
    for l in range(epochs):
        Gloc = calc_Gloc(H0, sigma)  # (bz, count * 2, 4)
        UoverWI = U / (1 / Gloc + sigma)  # (bz, count * 2, 4)
        E_mu = bisearch(fix_filling, E_mu, UoverWI)  # (bz, 1)
        nf = calc_nf(E_mu, UoverWI)[:, None]  # (bz, 1, 4)
        print('{} loop <nf>: {:.3f}'.format(l, np.mean(nf)))
        sigma = momentum * sigma + (1. - momentum) * calc_sigma(Gloc, nf)   # (bz, count * 2, 4)
    for i, op in enumerate(nf):
        print(i, '\n', op)
