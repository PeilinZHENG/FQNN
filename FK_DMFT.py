import torch
from typing import Union
from functools import partial


class DMFT:
    def __init__(self, count: int = 100, iota: float = 0., momentum: float = 0., maxEpoch: int = 100,
                 milestone: Union[int, None] = None, f_filling: Union[float, None] = None,
                 d_filling: Union[float, None] = None, tol_sc: float = 1e-8, tol_bi: float = 1e-6, mingap : float = 5.,
                 device: torch.device = torch.device('cpu'), double: bool = True):
        self.count = count * 2
        self.iota = iota
        self.momentum = momentum
        self.MAXEPOCH = maxEpoch
        self.MILESTONE = maxEpoch if milestone is None or milestone > maxEpoch else milestone
        self.f_filling = f_filling
        self.d_filling = d_filling
        self.tol_sc = tol_sc
        self.tol_bi = tol_bi
        self.mingap = mingap
        self.iomega0 = 1j * (2 * torch.arange(-count, count, device=device).unsqueeze(0) + 1) * torch.pi  # (1, self.count)
        if double: self.iomega0 = self.iomega0.type(torch.complex128)

    def calc_Gloc(self, mu, iomega, H, model=None):
        if model is None:
            return torch.linalg.diagonal((torch.diag_embed((iomega + mu).tile(1, 1, H.shape[-1])) - H).inverse())  # (bz, self.count, size)
        else:
            return model(H - torch.diag_embed(mu.tile(1, 1, H.shape[-1])), selfcons=True)  # (bz, self.count, size)

    def calc_nd(self, Gloc, T, iomega):
        return (T * torch.sum(Gloc * torch.exp(iomega * self.iota), dim=1)).real  # (bz, size)

    def calc_nf(self, E_mu, T, iomega, UoverWI):
        z = torch.sum(torch.log(1 - UoverWI) * torch.exp(iomega * self.iota), dim=1) - E_mu / T # (bz, size)
        return torch.nan_to_num(torch.sigmoid(z).real, nan=0.)  # (bz, size)

    def calc_nf_(self, E_mu, T, iomega, UoverWI):
        z = torch.prod((1 - UoverWI).pow(torch.exp(iomega * self.iota)), dim=1).pow(-1) * torch.exp(E_mu / T) # (bz, size)
        return torch.nan_to_num((1 + z).pow(-1).real, nan=0.)  # (bz, size)

    def fix_filling(self, a, T, iomega, args, model=None, f_ele=True):
        if f_ele:
            n = torch.mean(self.calc_nf(a, T, iomega, args), dim=-1)
        else:
            n = torch.mean(self.calc_nd(self.calc_Gloc(a, iomega, args, model), T, iomega), dim=-1)
        return (n - self.f_filling) if f_ele else (n - self.d_filling)  # (bz,)

    def detr_intvl(self, fun, a, T, iomega, args):
        fa = fun(a, T, iomega, args)
        b = a + self.mingap
        fb = fun(b, T, iomega, args)
        for i in torch.nonzero(fa.sign() * fb.sign() > 0, as_tuple=True)[0]:
            if fa[i] > 0:
                if fa[i] < fb[i]:
                    while fa[i] > 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - self.mingap
                        fa[i] = fun(a[i:i + 1], T[i:i + 1], iomega[i:i + 1], args[i:i + 1])
                else:
                    while fb[i] > 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + self.mingap
                        fb[i] = fun(b[i:i + 1], T[i:i + 1], iomega[i:i + 1], args[i:i + 1])
            else:
                if fa[i] < fb[i]:
                    while fb[i] < 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + self.mingap
                        fb[i] = fun(b[i:i + 1], T[i:i + 1], iomega[i:i + 1], args[i:i + 1])
                else:
                    while fa[i] < 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - self.mingap
                        fa[i] = fun(a[i:i + 1], T[i:i + 1], iomega[i:i + 1], args[i:i + 1])
        return a, b, fa, fb

    def bisection(self, fun, a, T, iomega, args):
        a, b, fa, fb = self.detr_intvl(fun, a, T, iomega, args)
        best = torch.zeros_like(a, dtype=a.dtype, device=a.device)
        idx = torch.nonzero(fa.abs() < self.tol_bi, as_tuple=True)[0]
        if len(idx) > 0: best[idx] = a[idx]
        idx = torch.nonzero(fb.abs() < self.tol_bi, as_tuple=True)[0]
        if len(idx) > 0: best[idx] = b[idx]
        idx = torch.nonzero((fa.abs() >= self.tol_bi) & (fb.abs() >= self.tol_bi), as_tuple=True)[0]
        a, b, T, iomega, args = a[idx], b[idx], T[idx], iomega[idx], args[idx]
        while True:
            c = (a + b) / 2
            fc = fun(c, T, iomega, args)
            good_idx = torch.nonzero((fc.abs() < self.tol_bi) | ((b - a).abs().view(-1) / 2 < self.tol_bi),
                                     as_tuple=True)[0]
            if len(good_idx) > 0: best[idx[good_idx]] = c[good_idx]
            if len(good_idx) < len(c):
                bad_idx = torch.nonzero((fc.abs() >= self.tol_bi) & ((b - a).abs().view(-1) / 2 >= self.tol_bi),
                                        as_tuple=True)[0]
                a, b, c, fc, idx = a[bad_idx], b[bad_idx], c[bad_idx], fc[bad_idx], idx[bad_idx]
                T, iomega, args = T[bad_idx], iomega[bad_idx], args[bad_idx]
                index = torch.nonzero(fc.sign() * fun(a, T, iomega, args).sign() < 0, as_tuple=True)
                b[index] = c[index]
                c[index] = a[index]
                a = c
            else:
                break
        return best

    def bisection_(self, fun, a, T, iomega, args):
        sbz = a.size(0) ** 0.5
        a, b, fa, fb = self.detr_intvl(fun, a, T, iomega, args)
        index = torch.nonzero(fa.abs() < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: b[index] = a[index]
        index = torch.nonzero(fb.abs() < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: a[index] = b[index]
        while True:
            c = (a + b) / 2
            fc = fun(c, T, iomega, args)
            index = torch.nonzero(fc.abs() < self.tol_bi, as_tuple=True)
            if len(index[0]) > 0:
                a[index] = c[index]
                b[index] = c[index]
            if torch.linalg.norm((b - a) / 2) < self.tol_bi * sbz:
                return c  # (bz, 1)
            index = torch.nonzero(fc * fun(a, T, iomega, args) < 0, as_tuple=True)
            b[index] = c[index]
            c[index] = a[index]
            a = c

    def calc_OP(self, fun, nf, prinfo=False):
        op = fun(nf.squeeze(1))
        if prinfo: print('order parameter:\n', torch.round(op, decimals=3))
        return op

    @torch.no_grad()
    def __call__(self, T, H0, U, E_mu=None, model=None, SEinit=None, reOP=False, reBad=False,
                 OPfuns=(lambda n: (n[:, 0] - n[:, 1]).abs(),), prinfo=False):  # E_mu, U: (bz,)
        device, dtype = self.iomega0.device, self.iomega0.dtype
        bz, _, _, size = H0.shape
        if E_mu is None:
            assert self.f_filling is not None
            E_mu = torch.zeros((bz, 1), device=device, dtype=dtype)  # (bz, 1)
        else:
            E_mu = E_mu.unsqueeze(-1).to(device=device, dtype=dtype)  # (bz, 1)
        mu = torch.zeros((bz, 1, 1), device=device, dtype=dtype)  # (bz, 1, 1)
        T = T.unsqueeze(-1).to(device=device, dtype=dtype) # (bz, 1)
        iomega = torch.matmul(T, self.iomega0).unsqueeze(-1) # (bz, self.count, 1)
        H0 = H0.tile(1, self.count, 1, 1).to(device=device, dtype=dtype) # (bz, self.count, size, size)
        U = U[:, None, None].to(device=device, dtype=dtype) # (bz, 1, 1)
        if model is not None: model.z = iomega
        '''0. initialize self-energy'''
        if SEinit is None:
            # SE = torch.zeros((bz, self.count, size), device=device).type(dtype)
            # SE = 0.01 * torch.randn((bz, self.count, size), device=device).type(dtype)
            SE = 0.01 * (2. * torch.rand((bz, self.count, size), device=device).type(dtype) - 1.)
        else:
            SE = SEinit.to(device=device, dtype=dtype)
        min_error, min_errors = 1e10, None
        best_SE, best_nf, best_mu = None, None, mu
        cur_tol_sc = self.tol_sc
        avg_tol_sc = self.tol_sc / bz ** 0.5
        for l in range(self.MAXEPOCH):
            '''1. compute G_{loc}'''
            H = H0 + torch.diag_embed(SE)
            if self.d_filling is not None:
                mu = self.bisection(partial(self.fix_filling, model=model, f_ele=False), mu, T, iomega, H)
            Gloc = self.calc_Gloc(mu, iomega, H, model)
            if prinfo: print('{} loop <nd>: {:.8f}'.format(l, torch.mean(self.calc_nd(Gloc, T, iomega)).item()))
            exit(0)
            '''2. compute Weiss field \mathcal{G}_0'''
            WeissInv = Gloc.pow(-1) + SE  # (bz, self.count, size)
            '''3. compute G_{imp}'''
            UoverWI = U / WeissInv
            if self.f_filling is not None:
                E_mu = self.bisection_(self.fix_filling, E_mu, T, iomega, UoverWI)
            nf = self.calc_nf(E_mu, T, iomega, UoverWI).unsqueeze(1) # (bz, 1, size)
            Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv  # (bz, self.count, size)
            if prinfo: print('{} loop <nf>: {:.3f}'.format(l, torch.mean(nf).item()))
            '''4. compute new self-energy'''
            new_errors = torch.linalg.norm(Gimp - Gloc, dim=(1, 2))
            tot_error = torch.linalg.norm(new_errors).item()
            if prinfo: print("{} loop tol: {:.3e}".format(l, cur_tol_sc))
            if tot_error < cur_tol_sc:
                if prinfo: print("final error: {:.5e}".format(tot_error))
                if l <= self.MILESTONE:
                    best_SE, best_nf = SE, nf
                    if self.d_filling is not None: best_mu = mu
                else:
                    best_SE[idx], best_nf[idx] = SE, nf
                    if self.d_filling is not None: best_mu[idx] = mu
                if reBad:
                    idx = torch.tensor([], dtype=torch.long, device=new_errors.device)
                    min_errors = torch.tensor([], dtype=new_errors.dtype, device=new_errors.device)
                break
            else:
                if prinfo: print("{} loop error: {:.5e}".format(l, tot_error))
                if l <= self.MILESTONE and tot_error < min_error:
                    min_error, min_errors = tot_error, new_errors
                    best_SE, best_nf = SE, nf
                    if self.d_filling is not None: best_mu = mu
                if l >= self.MILESTONE:
                    if l == self.MILESTONE:
                        bad_idx = torch.nonzero(min_errors >= avg_tol_sc, as_tuple=True)[0]
                        idx = bad_idx
                    else:
                        better_idx = torch.nonzero((new_errors - min_errors) < 0, as_tuple=True)[0]
                        if len(better_idx) > 0:
                            min_errors[better_idx] = new_errors[better_idx]
                            best_SE[idx[better_idx]] = SE[better_idx]
                            best_nf[idx[better_idx]] = nf[better_idx]
                            if self.d_filling is not None: best_mu[idx[better_idx]] = mu[better_idx]
                        bad_idx = torch.nonzero(min_errors >= avg_tol_sc, as_tuple=True)[0]
                        idx = idx[bad_idx]
                    H0, SE, iomega = H0[bad_idx], SE[bad_idx], iomega[bad_idx]
                    T, U, mu, E_mu = T[bad_idx], U[bad_idx], mu[bad_idx], E_mu[bad_idx]
                    min_errors, WeissInv, Gimp = min_errors[bad_idx], WeissInv[bad_idx], Gimp[bad_idx]
                    if model is not None: model.z = iomega
                    cur_tol_sc = self.tol_sc * (len(idx) / bz) ** 0.5
                    if prinfo: print("{} loop remain: {}".format(l, len(idx)))
                SE = self.momentum * SE + (1. - self.momentum) * (WeissInv - Gimp.pow(-1))
        OP = torch.stack([self.calc_OP(fun, best_nf, prinfo) for fun in OPfuns], dim=0)
        if reOP:
            if reBad:
                return best_SE - best_mu, OP, [idx, min_errors]
            else:
                return best_SE - best_mu, OP
        elif reBad:
            return best_SE - best_mu, [idx, min_errors]
        else:
            return best_SE - best_mu  # (bz, self.count, size)


def opf(nf):
    L = int(nf.shape[-1] ** 0.5)
    nf = nf.view(-1, L, L)
    op = torch.stack(((nf[:, 0, 0] - nf[:, 0, 1]).abs(), (nf[:, 0, 0] - nf[:, 1, 0]).abs()), dim=0)
    return torch.min(op, dim=0)[0]


if __name__ == "__main__":
    from FK_Data import Ham
    import os, time, warnings
    warnings.filterwarnings('ignore')

    threads = 8
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    L = 12  # size = L ** 2
    data = 'FK_{}'.format(L)
    Net = 'Naive_2d_0'
    T = 0.005
    save = True
    show = True

    '''construct DMFT'''
    count = 20
    iota = 0.1
    momentum = 0.5
    maxEpoch = 5000
    milestone = 30
    f_filling = 0.5
    d_filling = 0.5
    tol_sc = 1e-6
    tol_bi = 1e-6
    mingap = 5.
    scf = DMFT(count, iota, momentum, maxEpoch, milestone, f_filling, d_filling, tol_sc, tol_bi, mingap, device)

    '''2D test'''
    tp = torch.tensor([0.1, 1.4])
    U = torch.ones(len(tp))
    T = T * torch.ones(len(U))
    mu = torch.zeros(len(U))
    H0 = torch.stack([Ham(L, i.item(), j.item()) for i, j in zip(mu, tp)], dim=0).unsqueeze(1)
    print((1 / (torch.exp(torch.linalg.eigvalsh(H0).squeeze(1) / T.unsqueeze(-1)) + 1)).mean(dim=1))
    t = time.time()
    SE, OP = scf(T, H0, U, reOP=True, OPfuns=(opf,), prinfo=True)  # (bz, 1, size)
    print(time.time() - t)
    exit(0)

    from FK_rgfnn import Network
    from utils import myceil, mymkdir
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.functional import softmax, tanh, relu

    '''construct FQNN'''
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network(Net[:Net.index('_')], L ** 2, 1 if Net.startswith('C') else 2, 100, 64, None, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    '''construct Hamiltonians'''
    U = torch.linspace(1., 4., 150)
    mu = U / 2.
    H0 = torch.stack([Ham(L, i.item()) for i in mu], dim=0).unsqueeze(1)
    PTPs = {'0.020':(0.33, 0.392), '0.100': (1.5, 1.76), '0.110': (1.7, 2.01), '0.120': (1.8, 2.28),
            '0.130': (2.0, 2.6), '0.140': (2.2, 3.03), '0.150': (2.4, 3.99)}
    try:
        PTP, QMCPTP = PTPs[f'{T:.3f}']
    except KeyError:
        PTP, QMCPTP = None, None
    T = T * torch.ones(len(U))

    '''compute self-energy by DMFT'''
    bz = 75
    P = []
    if '2d' in Net:
        mymkdir(f'results/{data}/SE+OP')
        try:
            SEs = torch.load(f'results/{data}/SE+OP/SE_{T[0].item():.3f}.pt')
            OP = torch.load(f'results/{data}/SE+OP/OP_{T[0].item():.3f}.pt')
        except FileNotFoundError:
            SEs, OP = [], []
    else:
        OP = []
    for i in range(myceil(len(U) / bz)):
        H0_batch = H0[i * bz:(i + 1) * bz].to(device)
        T_batch = T[i * bz:(i + 1) * bz].to(device)
        U_batch = U[i * bz:(i + 1) * bz].to(device)
        if '2d' in Net:
            if type(SEs) is torch.Tensor:
                SE = SEs[i * bz:(i + 1) * bz].to(device)
            else:
                SE, op = scf(T_batch, H0_batch, U_batch, model=None, reOP=True, prinfo=True if i == 0 else False)  # (bz, scf.count, size)
                SEs.append(SE.cpu())
                OP.append(op.cpu().squeeze(0))
        else:
            SE, op = scf(T_batch, H0_batch, U_batch, model=model, reOP=True, prinfo=True if i == 0 else False)  # (bz, scf.count, size)
            OP.append(op.cpu().squeeze(0))
        if Net.startswith('C') or 'sf' in Net:
            SE = SE[:, count:count + 1]   # (bz, 1, size)
            if '2d' in Net or model.z.size(0) != T_batch.shape[0]:
                model.z = T_batch.to(device=device, dtype=scf.iomega0.dtype) * scf.iomega0[0, count]  # (bz,)
            else:
                model.z = model.z[:, count, 0]  # (bz,)
        elif '2d' in Net or model.z.size(0) != T_batch.shape[0]:
            model.z = (T_batch[:, None].to(device=device, dtype=scf.iomega0.dtype) @ scf.iomega0).unsqueeze(-1)  # (bz, scf.count, 1)

        '''compute phase diagram'''
        H = H0_batch + torch.diag_embed(SE)
        LDOS = model(H)
        if model.out.size == 1:
            P.append(tanh(relu(LDOS / 2)).data.cpu())
        else:
            P.append(softmax(LDOS, dim=1)[:, 1].data.cpu())
    P = torch.cat(P, dim=0).numpy()
    if type(OP) is list: OP = torch.cat(OP, dim=0)
    if '2d' in Net and type(SEs) is list:
        torch.save(torch.cat(SEs, dim=0), f'results/{data}/SE+OP/SE_{T[0].item():.3f}.pt')
        torch.save(OP, f'results/{data}/SE+OP/OP_{T[0].item():.3f}.pt')
    OP = OP.numpy()
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
