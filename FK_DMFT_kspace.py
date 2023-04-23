import numpy as np
import cmath, math

T = 0.005
U = 10.0
mu = np.zeros(61)
tp = np.arange(61)
count = 100
gap = 1
tol_bi = 1e-7
iota = 0
iomega = 1j * (2 * np.arange(-count, count)[:, None] + 1) * np.pi * T       # (count * 2, 1)


def calc_Gloc(sigma, kstep=100):
    Gloc = np.zeros_like(sigma, dtype=sigma.dtype)
    for ikx in range(kstep):
        for iky in range(kstep):
            mepsilonk = 2 * math.cos(ikx * np.pi * 2 / kstep) + 2 * math.cos(iky * np.pi * 2 / kstep)
            for n in range(count * 2):
                iomegan = 1j * (n * 2 - count * 2 + 1) * np.pi * T
                ga[n] += (iomegan + mu - sigmab[n]) / (
                            (iomegan + mu - sigmab[n]) * (iomegan + mu - sigmaa[n]) - mepsilonk ** 2) / kstep ** 2
                gb[n] += (iomegan + mu - sigmaa[n]) / (
                            (iomegan + mu - sigmab[n]) * (iomegan + mu - sigmaa[n]) - mepsilonk ** 2) / kstep ** 2
    return Gloc  # (bz, count * 2, 4)


def calc_nf(E_mu, UoverWI):
    z = E_mu / T - np.sum(np.log(1 - UoverWI) * np.exp(iomega * iota), axis=1)  # (bz, 4)
    return np.nan_to_num((1 / (1 + np.exp(z))).real, nan=0.)                    # (bz, 4)


def fix_filling(E, UoverWI):
    return calc_nf(E, UoverWI).mean(axis=-1) - 0.5   # (bz,)


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
        if np.linalg.norm((b - a) / 2) < tol_bi * sbz: return c  # (bz, 1)
        index = np.nonzero(fc.sign() * fun(a, args).sign() < 0)
        b[index] = c[index]
        c[index] = a[index]
        a = c


def calc_sigma(Gloc, nf):
    return U / 2 - (1 - np.sqrt(1 + (U * Gloc) ** 2 + (U * Gloc) * (4 * nf - 2))) / 2 / Gloc   # (bz, count * 2, 4)


if __name__ == "__main__":
    sigma = np.random.rand((len(mu), count * 2, 4)) * 0.1 + np.zeros((len(mu), count * 2, 4)) * 1j # (bz, count * 2, 4)
    E_mu = np.zeros((len(mu), 1))   # (bz, 1)
    for step in range(20):
        Gloc = calc_Gloc(sigma)                       # (bz, count * 2, 4)
        UoverWI = U / (1 / Gloc + sigma)              # (bz, count * 2, 4)
        E_mu = bisearch(fix_filling, E_mu, UoverWI)   # (bz, 1)
        nf = calc_nf(E_mu, UoverWI)[:, None, :]       # (bz, 1, 4)
        sigma = calc_sigma(Gloc, nf)                  # (bz, count * 2, 4)
