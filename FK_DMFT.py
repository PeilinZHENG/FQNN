import torch


class DMFT:
    def __init__(self, T=1e-2, count=100, iota=0., momentum=0., maxEpoch=100, filling=None, tol_sc=1e-8, tol_bi=1e-6,
                 device=torch.device('cpu'), double=True):
        self.T = T
        self.count = count * 2
        self.iota = iota
        self.momentum = momentum
        self.MAXEPOCH = maxEpoch
        self.filling = filling
        self.tol_sc = tol_sc
        self.tol_bi = tol_bi
        self.iomega = 1j * (2 * torch.arange(-count, count, device=device).unsqueeze(1) + 1) * torch.pi * self.T  # (count, 1)
        if double: self.iomega = self.iomega.type(torch.complex128)

    def nansum(self, x, dim, keepdim=False):
        return torch.nansum(x.real, dim=dim, keepdim=keepdim) + 1j * torch.nansum(x.imag, dim=dim, keepdim=keepdim)

    def nan_to_num(self, x, nan=0.):
        return torch.nan_to_num(x.real, nan=nan) + 1j * torch.nan_to_num(x.imag, nan=nan)

    def calc_nf(self, WeissInv, iomega, E_mu, U):
        z = torch.sum(torch.log(1 - U / WeissInv) * torch.exp(iomega * self.iota), dim=1) - E_mu / self.T # (bz, size)
        return self.nan_to_num(torch.sigmoid(z)).unsqueeze(1).real  # (bz, 1, size)

    def fix_filling(self, E_mu, WeissInv, U):
        nf = self.calc_nf(WeissInv, self.iomega, E_mu, U)
        return torch.mean(nf, dim=-1) - self.filling  # (bz, 1)

    def bisection(self, fun, a, WeissInv, U, mingap=5.):
        fa = fun(a, WeissInv, U)
        b = a + mingap
        fb = fun(b, WeissInv, U)
        for i in torch.nonzero(fa * fb > 0, as_tuple=True)[0]:
            if fa[i] > 0:
                if fa[i] < fb[i]:
                    while fa[i] > 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - mingap
                        fa[i] = fun(a[i:i+1], WeissInv[i:i+1], U[i:i+1])
                else:
                    while fb[i] > 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + mingap
                        fb[i] = fun(b[i:i+1], WeissInv[i:i+1], U[i:i+1])
            else:
                if fa[i] < fb[i]:
                    while fb[i] < 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + mingap
                        fb[i] = fun(b[i:i+1], WeissInv[i:i+1], U[i:i+1])
                else:
                    while fa[i] < 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - mingap
                        fa[i] = fun(a[i:i+1], WeissInv[i:i+1], U[i:i+1])
        index = torch.nonzero(torch.abs(fa) < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: b[index] = a[index]
        index = torch.nonzero(torch.abs(fb) < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: a[index] = b[index]
        while True:
            c = (a + b) / 2
            fc = fun(c, WeissInv, U)
            index = torch.nonzero(torch.abs(fc) < self.tol_bi, as_tuple=True)
            if len(index[0]) > 0:
                a[index] = c[index]
                b[index] = c[index]
            if torch.linalg.norm((b - a) / 2) < self.tol_bi:
                return (b + a) / 2
            index = torch.nonzero(fc * fun(a, WeissInv, U) < 0, as_tuple=True)
            b[index] = c[index]
            c[index] = a[index]
            a = c

    @torch.no_grad()
    def __call__(self, H0, E_mu, U, model=None, SEinit=None, prinfo=False):  # E_mu, U: (bz,)
        bz, _, _, size = H0.shape
        H0 = H0.tile(1, self.count, 1, 1) # (bz, count, size, size)
        device, dtype = H0.device, H0.dtype
        E_mu = E_mu.unsqueeze(1).type(dtype)  # (bz, 1)
        U = U[:, None, None].type(dtype) # (bz, 1, 1)
        if model is None:
            H_omega = torch.diag_embed(self.iomega.expand(-1, size)) - H0  # (bz, count, size, size)
        '''0. initialize self-energy'''
        if SEinit is None:
            # SE = torch.zeros((bz, self.count, size), device=device, dtype=dtype) # (bz, count, size)
            # SE = 0.01 * torch.randn((bz, self.count, size), device=device).type(dtype) # (bz, count, size)
            SE = 0.01 * (2. * torch.rand((bz, self.count, size), device=device).type(dtype) - 1.) # (bz, count, size)
        else:
            SE = SEinit
        min_error = 1e10
        best_SE = None
        for l in range(self.MAXEPOCH):
            '''1. compute G_{loc}'''
            if model is None:
                Gloc = torch.diagonal((H_omega - torch.diag_embed(SE)).inverse(), dim1=-2, dim2=-1)  # (bz, count, size)
            else:
                Gloc = model(H0 + torch.diag_embed(SE), selfcons=True)  # (bz, count, size)
            '''2. compute Weiss field \mathcal{G}_0'''
            WeissInv = Gloc.pow(-1) + SE  # (bz, count, size)
            '''3. compute G_{imp}'''
            if self.filling is not None:
                E_mu = self.bisection(self.fix_filling, E_mu, WeissInv, U)
            nf = self.calc_nf(WeissInv, self.iomega, E_mu, U) # (bz, 1, size)
            if prinfo: print('<nf>={}'.format(torch.mean(nf).item()))
            Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv  # (bz, count, size)
            '''4. compute new self-energy'''
            error = torch.linalg.norm(Gimp - Gloc).item()
            if error < self.tol_sc:
                if prinfo:
                    print("final error: {}".format(error))
                    print(torch.round(nf.cpu(), decimals=3).numpy())
                return SE  # (bz, count, size)
            else:
                if error < min_error:
                    min_error = error
                    best_SE = SE
                SE = self.momentum * SE + (1. - self.momentum) * (WeissInv - Gimp.pow(-1))
                if prinfo:
                    print("{} loop error: {}".format(l, error))
        return best_SE


if __name__ == "__main__":
    from FK_Data import Ham
    import os, mkl
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mkl.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    L = 12  # size = L ** 2
    save = True
    show = True

    '''construct DMFT'''
    T = 0.15
    count = 20
    momentum = 0.5
    maxEpoch = 2000
    scf = DMFT(T, count, momentum=momentum, maxEpoch=maxEpoch, filling=0.5, device=device)

    '''2D test'''
    U = torch.tensor([4.], device=device)
    mu = U / 2.
    E_mu = torch.zeros(len(U), device=device) - mu  # E - mu  (-0.066)
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1).to(device)
    SE = scf(H0, E_mu, U, prinfo=True)  # (bz, 1, size)
    exit(0)

    from FK_rgfnn import Network
    from utils import mymkdir, myceil
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.functional import softmax

    '''construct FQNN'''
    data = 'FK_{}'.format(L)
    Net = 'Naive_1'
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network('Naive', L ** 2, 2, 100, 64, scf.iomega, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    '''construct Hamiltonians'''
    U = torch.linspace(1.5, 2.5, 50)
    mu = U / 2.
    E_mu = torch.zeros(len(U)) - mu  # E - mu  (-0.066)
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1)

    '''compute self-energy by DMFT'''
    bz = 20
    P = []
    for i in range(myceil(len(U) / bz)):
        H0_batch = H0[i * bz:(i + 1) * bz].to(device)
        E_mu_batch = E_mu[i * bz:(i + 1) * bz].to(device)
        U_batch = U[i * bz:(i + 1) * bz].to(device)
        SE = scf(H0_batch, E_mu_batch, U_batch, model, prinfo=True if i == 0 else False)  # (bz, 1, size)
        '''compute phase diagram'''
        H = H0_batch + torch.diag_embed(SE)
        LDOS = model(H)
        P.append(softmax(LDOS, dim=1)[:, 1].data.cpu())
    P = torch.cat(P, dim=0).numpy()
    U = U.numpy()

    '''plot phase diagram'''
    plt.figure()
    plt.axis([U[0], U[-1], 0., 1.])
    plt.plot([2., 2.], [0., 1.], 'bo--', linewidth=0.5, markersize=0.1)
    plt.plot([2.14, 2.14], [0., 1.], 'ko--', linewidth=0.5, markersize=0.1)
    plt.scatter(U, P, s=20, c='r', marker='o')
    plt.xlabel('U')
    plt.ylabel('P')
    plt.title('Metal VS Insulator')
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/PD.jpg'.format(path))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/{}.npy'.format(path, 'PD'), np.stack((U, P), axis=0))
