import torch
from utils import mymkdir


class DMFT:
    def __init__(self, count=100, iota=0., momentum=0., maxEpoch=100, filling=None, tol_sc=1e-8, tol_bi=1e-6,
                 device=torch.device('cpu'), double=True):
        self.count = count * 2
        self.iota = iota
        self.momentum = momentum
        self.MAXEPOCH = maxEpoch
        self.filling = filling
        self.tol_sc = tol_sc
        self.tol_bi = tol_bi
        self.iomega0 = 1j * (2 * torch.arange(-count, count, device=device).unsqueeze(0) + 1) * torch.pi  # (1, self.count)
        if double: self.iomega0 = self.iomega0.type(torch.complex128)

    def calc_nf(self, WeissInv, T, iomega, U, E_mu):
        z = torch.sum(torch.log(1 - U / WeissInv) * torch.exp(iomega * self.iota), dim=1) - E_mu / T # (bz, size)
        return torch.nan_to_num(torch.sigmoid(z).real, nan=0.).unsqueeze(1)  # (bz, 1, size)

    def calc_nf_(self, WeissInv, T, iomega, U, E_mu):
        z = torch.prod((1 - U / WeissInv).pow(torch.exp(iomega * self.iota)), dim=1).pow(-1) * torch.exp(E_mu / T) # (bz, size)
        return torch.nan_to_num((1 + z).pow(-1).real, nan=0.).unsqueeze(1)  # (bz, 1, size)

    def fix_filling(self, WeissInv, T, iomega, U, E_mu):
        nf = self.calc_nf(WeissInv, T, iomega, U, E_mu)
        return torch.mean(nf, dim=-1) - self.filling  # (bz, 1)

    def bisection(self, fun, WeissInv, T, iomega, U, a, mingap=5.):
        fa = fun(WeissInv, T, iomega, U, a)
        b = a + mingap
        fb = fun(WeissInv, T, iomega, U, b)
        for i in torch.nonzero(fa * fb > 0, as_tuple=True)[0]:
            if fa[i] > 0:
                if fa[i] < fb[i]:
                    while fa[i] > 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - mingap
                        fa[i] = fun(WeissInv[i:i+1], T[i:i+1], iomega[i:i+1], U[i:i+1], a[i:i+1])
                else:
                    while fb[i] > 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + mingap
                        fb[i] = fun(WeissInv[i:i+1], T[i:i+1], iomega[i:i+1], U[i:i+1], b[i:i+1])
            else:
                if fa[i] < fb[i]:
                    while fb[i] < 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + mingap
                        fb[i] = fun(WeissInv[i:i+1], T[i:i+1], iomega[i:i+1], U[i:i+1], b[i:i+1])
                else:
                    while fa[i] < 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - mingap
                        fa[i] = fun(WeissInv[i:i+1], T[i:i+1], iomega[i:i+1], U[i:i+1], a[i:i+1])
        index = torch.nonzero(torch.abs(fa) < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: b[index] = a[index]
        index = torch.nonzero(torch.abs(fb) < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: a[index] = b[index]
        while True:
            c = (a + b) / 2
            fc = fun(WeissInv, T, iomega, U, c)
            index = torch.nonzero(torch.abs(fc) < self.tol_bi, as_tuple=True)
            if len(index[0]) > 0:
                a[index] = c[index]
                b[index] = c[index]
            if torch.linalg.norm((b - a) / 2) < self.tol_bi:
                return (b + a) / 2   # (bz, 1)
            index = torch.nonzero(fc * fun(WeissInv, T, iomega, U, a) < 0, as_tuple=True)
            b[index] = c[index]
            c[index] = a[index]
            a = c

    def calc_OP(self, nf, prinfo=False):
        op = (nf[:, 0, 0] - nf[:, 0, 1]).abs()
        if prinfo: print('order parameter:\n', torch.round(op, decimals=3))
        return op

    @torch.no_grad()
    def __call__(self, T, H0, U, E_mu=None, model=None, SEinit=None, reOP=False, prinfo=False):  # E_mu, U: (bz,)
        device, dtype = self.iomega0.device, self.iomega0.dtype
        bz, _, _, size = H0.shape
        if E_mu is None:
            assert self.filling is not None
            E_mu = torch.zeros((bz, 1), device=device, dtype=dtype)  # (bz, 1)
        else:
            E_mu = E_mu.unsqueeze(-1).to(device=device, dtype=dtype)  # (bz, 1)
        T = T.unsqueeze(-1).to(device=device, dtype=dtype) # (bz, 1)
        iomega = torch.matmul(T, self.iomega0).unsqueeze(-1) #(bz, self.count, 1)
        H0 = H0.tile(1, self.count, 1, 1).to(device=device, dtype=dtype) # (bz, self.count, size, size)
        U = U[:, None, None].to(device=device, dtype=dtype) # (bz, 1, 1)
        if model is None:
            H_omega = torch.diag_embed(iomega.expand(bz, self.count, size)) - H0  # (bz, self.count, size, size)
        else:
            model.z = iomega
        '''0. initialize self-energy'''
        if SEinit is None:
            # SE = 0.01 * torch.randn((bz, self.count, size), device=device).type(dtype) # (bz, self.count, size)
            SE = 0.01 * (2. * torch.rand((bz, self.count, size), device=device).type(dtype) - 1.) # (bz, self.count, size)
        else:
            SE = SEinit.to(device=device, dtype=dtype)
        min_error = 1e10
        best_SE = None
        if prinfo: best_nf = None
        for l in range(self.MAXEPOCH):
            '''1. compute G_{loc}'''
            if model is None:
                Gloc = torch.diagonal((H_omega - torch.diag_embed(SE)).inverse(), dim1=-2, dim2=-1)  # (bz, self.count, size)
            else:
                Gloc = model(H0 + torch.diag_embed(SE), selfcons=True)  # (bz, self.count, size)
            '''2. compute Weiss field \mathcal{G}_0'''
            WeissInv = Gloc.pow(-1) + SE  # (bz, self.count, size)
            '''3. compute G_{imp}'''
            if self.filling is not None:
                E_mu = self.bisection(self.fix_filling, WeissInv, T, iomega, U, E_mu)
            nf = self.calc_nf(WeissInv, T, iomega, U, E_mu) # (bz, 1, size)
            Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv  # (bz, self.count, size)
            if prinfo: print('<nf>={}'.format(torch.mean(nf).item()))
            '''4. compute new self-energy'''
            error = torch.linalg.norm(Gimp - Gloc).item()
            if error < self.tol_sc:
                if prinfo: print("final error: {}".format(error))
                return (SE, self.calc_OP(nf, prinfo)) if reOP else SE   # (bz, self.count, size)
            else:
                if error < min_error:
                    min_error = error
                    best_SE = SE
                    if prinfo: best_nf = nf
                SE = self.momentum * SE + (1. - self.momentum) * (WeissInv - Gimp.pow(-1))
                if prinfo: print("{} loop error: {}".format(l, error))
        return (best_SE, self.calc_OP(best_nf, prinfo)) if reOP else best_SE  # (bz, self.count, size)


if __name__ == "__main__":
    from FK_Data import Ham
    import os, time, warnings
    warnings.filterwarnings('ignore')

    threads = 8
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    L = 14  # size = L ** 2
    save = True
    show = True

    '''construct DMFT'''
    count = 20
    iota = 0.
    momentum = 0.5
    maxEpoch = 5000
    filling = 0.5
    tol_sc = 1e-6
    tol_bi = 1e-6
    scf = DMFT(count, iota, momentum, maxEpoch, filling, tol_sc, tol_bi, device)

    '''2D test'''
    # T = torch.tensor([0.15, 0.25], device=device)
    # U = torch.tensor([4., 4.], device=device)
    # mu = U / 2.
    # H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1).to(device)
    # t = time.time()
    # SE = scf(T, H0, U, prinfo=True)  # (bz, 1, size)
    # print(time.time() - t)
    # exit(0)

    from FK_rgfnn import Network
    from utils import myceil
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.functional import softmax

    '''construct FQNN'''
    data = 'FK_{}'.format(L)
    Net = 'Naive_1'
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network('Naive', L ** 2, 2, 100, 64, None, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    '''construct Hamiltonians'''
    PTP, QMCPTP = 2.4, 3.99
    U = torch.linspace(1., 4., 150)
    T = 0.15 * torch.ones(len(U))
    mu = U / 2.
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1)

    '''compute self-energy by DMFT'''
    bz = 75
    P, OP = [], []
    for i in range(myceil(len(U) / bz)):
        H0_batch = H0[i * bz:(i + 1) * bz].to(device)
        T_batch = T[i * bz:(i + 1) * bz].to(device)
        U_batch = U[i * bz:(i + 1) * bz].to(device)
        SE, op = scf(T_batch, H0_batch, U_batch, model=model, reOP=True, prinfo=True if i == 0 else False)  # (bz, scf.count, size)
        '''compute phase diagram'''
        H = H0_batch + torch.diag_embed(SE)
        LDOS = model(H)
        P.append(softmax(LDOS, dim=1)[:, 1].data.cpu())
        OP.append(op.cpu())
    P = torch.cat(P, dim=0).numpy()
    OP = torch.cat(OP, dim=0).numpy()
    U = U.numpy()

    '''plot phase diagram'''
    fig, ax1 = plt.subplots()
    plt.title(f'Metal VS Insulator / T={T[0].item():.3f}, L={L}')
    plt.axis([U[0], U[-1], 0., 1.])
    ax1.set_xlim([U[0], U[-1]])
    ax1.set_xlabel('U')
    ax1.set_ylabel('P', c='r')
    ax1.set_ylim([0., 1.])
    ax1.set_yticks(0.1 * np.arange(11))
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
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/PD_{:.3f}.jpg'.format(path, T[0].item()))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/PD_{:.3f}.npy'.format(path, T[0].item()), np.stack((U, P, OP), axis=0))
