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

    def saveOP(self, nf, T, U, error, L):
        op = torch.round((nf[:, 0, 0] - nf[:, 0, 1]).abs().cpu(), decimals=3).numpy()
        print('order parameter:\n', op)
        mymkdir(f'results/FK_{L}')
        torch.save({'OP': op, 'T': T.cpu(), 'U': U.cpu(), 'error': error}, f'results/FK_{L}/OP.pt')

    @torch.no_grad()
    def __call__(self, T, H0, U, E_mu=None, model=None, SEinit=None, prinfo=False):  # E_mu, U: (bz,)
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
            if prinfo: print('<nf>={}'.format(torch.mean(nf).item()))
            Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv  # (bz, self.count, size)
            '''4. compute new self-energy'''
            error = torch.linalg.norm(Gimp - Gloc).item()
            if error < self.tol_sc:
                if prinfo:
                    print("final error: {}".format(error))
                    self.saveOP(nf, T[:, 0].real, U[:, 0, 0].real, error, int(size ** 0.5))
                return SE  # (bz, self.count, size)
            else:
                if error < min_error:
                    min_error = error
                    best_SE = SE
                    if prinfo: best_nf = nf
                SE = self.momentum * SE + (1. - self.momentum) * (WeissInv - Gimp.pow(-1))
                if prinfo:
                    print("{} loop error: {}".format(l, error))
        if prinfo:
            self.saveOP(best_nf, T[:, 0].real, U[:, 0, 0].real, min_error, int(size ** 0.5))
        return best_SE # (bz, self.count, size)


if __name__ == "__main__":
    from FK_Data import Ham
    import os, time, warnings
    warnings.filterwarnings('ignore')

    threads = 8
    os.environ['CUDA_VISIBLE_DEVICES'] = str(threads)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    L = 10  # size = L ** 2
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
    Net = 'Naive_0'
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network('Naive', L ** 2, 2, 100, 64, None, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    '''construct Hamiltonians'''
    U = torch.linspace(1., 4., 150)
    T = 0.15 * torch.ones(len(U))
    mu = U / 2.
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1)

    '''compute self-energy by DMFT'''
    bz = 150
    P = []
    for i in range(myceil(len(U) / bz)):
        H0_batch = H0[i * bz:(i + 1) * bz].to(device)
        T_batch = T[i * bz:(i + 1) * bz].to(device)
        U_batch = U[i * bz:(i + 1) * bz].to(device)
        SE = scf(T_batch, H0_batch, U_batch, model=model, prinfo=True if i == 0 else False)  # (bz, scf.count, size)
        '''compute phase diagram'''
        H = H0_batch + torch.diag_embed(SE)
        LDOS = model(H)
        P.append(softmax(LDOS, dim=1)[:, 1].data.cpu())
    P = torch.cat(P, dim=0).numpy()
    U = U.numpy()

    '''plot phase diagram'''
    plt.figure()
    plt.axis([U[0], U[-1], 0., 1.])
    # plt.plot([2., 2.], [0., 1.], 'bo--', linewidth=0.5, markersize=0.1)
    # plt.plot([2.14, 2.14], [0., 1.], 'ko--', linewidth=0.5, markersize=0.1)
    plt.plot([U[0], U[-1]], [0.5, 0.5], 'ko--', linewidth=0.5, markersize=0.1)
    plt.scatter(U, P, s=20, c='r', marker='o')
    plt.xlabel('U')
    plt.ylabel('P')
    plt.title('Metal VS Insulator / T={:.3f}, L={}'.format(T[0].item(), L))
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/PD_{:.3f}.jpg'.format(path, T[0].item()))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/PD_{:.3f}.npy'.format(path, T[0].item()), np.stack((U, P), axis=0))
