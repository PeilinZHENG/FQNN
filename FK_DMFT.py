import torch
from typing import Union
from functools import partial
from Data_Ins import l2c


class DMFT:
    def __init__(self, count: int = 100, iota: float = 0., momentum: float = 0., momDisor: float = 0.,
                 maxEpoch: int = 100, milestone: Union[int, None] = None, f_filling: Union[float, None] = None,
                 d_filling: Union[float, None] = None, tol_sc: float = 1e-8, tol_bi: float = 1e-6, gap : float = 5.,
                 device: torch.device = torch.device('cpu'), double: bool = True):
        self.count = count * 2
        self.iota = iota
        self.momentum = momentum
        self.momDisor = momDisor
        self.MAXEPOCH = maxEpoch
        self.MILESTONE = maxEpoch - 1 if milestone is None or milestone >= maxEpoch else milestone
        self.f_filling = f_filling
        self.d_filling = d_filling
        self.tol_sc = tol_sc
        self.tol_bi = tol_bi
        self.gap = gap
        self.iomega0 = 1j * (2 * torch.arange(-count, count, device=device).unsqueeze(0) + 1) * torch.pi  # (1, self.count)
        if double: self.iomega0 = self.iomega0.type(torch.complex128)

    def init_E_mu(self, E_mu, device, dtype, bz=None):
        if E_mu is None:
            assert self.f_filling is not None
            return torch.zeros((bz, 1), device=device, dtype=dtype)  # (bz, 1)
        else:
            return E_mu.unsqueeze(-1).to(device=device, dtype=dtype)  # (bz, 1)

    def init_H0(self, H0, device, dtype, adjMu=None, T=None, bz=None, size=None, prinfo=False):
        H0 = H0.to(device=device, dtype=dtype)
        if adjMu is not None and self.d_filling is not None:
            mu = self.bisearch_dec(partial(self.fix_filling, f_ele=False),
                                   torch.zeros((bz, 1, 1), device=device, dtype=torch.float32
                                   if dtype == torch.complex64 else torch.float64), H0, T)
            if prinfo: print('<nd>: {:.3f}'.format(torch.mean(self.calc_nd0_avg(mu, H0, T)).item()))
            return H0 - torch.diag_embed((mu + adjMu.to(device)[:, None, None]).tile(1, 1, size))
        else:
            return H0   # (bz, 1, size, size)

    def init_SE(self, SEinit, device, dtype, bz=None, size=None):
        if SEinit is None:
            # return torch.zeros((bz, self.count, size), device=device).type(dtype)
            # return 0.01 * torch.randn((bz, self.count, size), device=device).type(dtype)
            return 0.01 * (2. * torch.rand((bz, self.count, size), device=device).type(dtype) - 1.)
        else:
            return SEinit.to(device=device, dtype=dtype)

    def calc_Gloc(self, H, iomega, model=None):
        if model is None:
            return torch.linalg.diagonal((torch.diag_embed(iomega.tile(1, 1, H.shape[-1])) - H).inverse())  # (bz, self.count, size)
        else:
            return model(H, selfcons=True)  # (bz, self.count, size)

    def calc_nd(self, mu, H, T, iomega, model=None):
        Gloc = self.calc_Gloc(H - torch.diag_embed(mu.tile(1, 1, H.shape[-1])), iomega, model)
        return (T * torch.sum(Gloc * (iomega * self.iota).exp(), dim=1)).real  # (bz, size)

    def calc_FDD(self, mu, E, T):
        return torch.nan_to_num(torch.sigmoid((mu - E).squeeze(1) / T), nan=0.)  # (bz, size)

    def calc_nd0(self, mu, H0, T):
        E, V = torch.linalg.eigh(H0)   # (bz, 1, size) (bz, 1, size, size)
        fdd = self.calc_FDD(mu, E, T).unsqueeze(-1)  # (bz, size, 1)
        return torch.matmul(V.squeeze(1).abs().pow(2), fdd).squeeze(-1)  # (bz, size)

    def calc_nd0_avg(self, mu, H0, T):
        return torch.mean(self.calc_FDD(mu, torch.linalg.eigvalsh(H0), T), dim=-1)  # (bz,)

    def calc_nf(self, E_mu, UoverWI, T, iomega):
        z = torch.sum((1 - UoverWI).log() * (iomega * self.iota).exp(), dim=1) - E_mu / T # (bz, size)
        return torch.nan_to_num(torch.sigmoid(z).real, nan=0.)  # (bz, size)

    def fix_filling(self, a, args, T, iomega=None, f_ele=True):
        if f_ele:
            return self.calc_nf(a, args, T, iomega).mean(dim=-1) - self.f_filling   # (bz,)
        else:
            return self.calc_nd0_avg(a, args, T) - self.d_filling # (bz,)

    def detr_intvl(self, fun, a, args, T, iomega):
        fa = fun(a, args, T, iomega)
        b = a + self.gap
        fb = fun(b, args, T, iomega)
        for i in torch.nonzero(fa.sign() * fb.sign() > 0, as_tuple=True)[0]:
            if fa[i] > 0:
                if fa[i] < fb[i]:
                    while fa[i] > 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - self.gap
                        fa[i] = fun(a[i:i + 1], args[i:i + 1], T[i:i + 1], None if iomega is None else iomega[i:i + 1])
                else:
                    while fb[i] > 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + self.gap
                        fb[i] = fun(b[i:i + 1], args[i:i + 1], T[i:i + 1], None if iomega is None else iomega[i:i + 1])
            else:
                if fa[i] < fb[i]:
                    while fb[i] < 0:
                        a[i], fa[i] = b[i], fb[i]
                        b[i] = b[i] + self.gap
                        fb[i] = fun(b[i:i + 1], args[i:i + 1], T[i:i + 1], None if iomega is None else iomega[i:i + 1])
                else:
                    while fa[i] < 0:
                        b[i], fb[i] = a[i], fa[i]
                        a[i] = a[i] - self.gap
                        fa[i] = fun(a[i:i + 1], args[i:i + 1], T[i:i + 1], None if iomega is None else iomega[i:i + 1])
        return a, b, fa, fb

    def bisearch(self, fun, a, args, T, iomega=None):
        sbz = a.size(0) ** 0.5
        a, b, fa, fb = self.detr_intvl(fun, a, args, T, iomega)
        index = torch.nonzero(fa.abs() < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: b[index] = a[index]
        index = torch.nonzero(fb.abs() < self.tol_bi, as_tuple=True)
        if len(index[0]) > 0: a[index] = b[index]
        while True:
            c = (a + b) / 2
            fc = fun(c, args, T, iomega)
            index = torch.nonzero(fc.abs() < self.tol_bi, as_tuple=True)
            if len(index[0]) > 0: a[index], b[index] = c[index], c[index]
            if torch.linalg.norm((b - a) / 2) < self.tol_bi * sbz: return c  # (bz, 1)
            index = torch.nonzero(fc.sign() * fun(a, args, T, iomega).sign() < 0, as_tuple=True)
            b[index] = c[index]
            c[index] = a[index]
            a = c

    def bisearch_dec(self, fun, a, args, T, iomega=None):
        a, b, fa, fb = self.detr_intvl(fun, a, args, T, iomega)
        best = torch.zeros_like(a, dtype=a.dtype, device=a.device)
        idx = torch.nonzero(fa.abs() < self.tol_bi, as_tuple=True)[0]
        if len(idx) > 0: best[idx] = a[idx]
        idx = torch.nonzero(fb.abs() < self.tol_bi, as_tuple=True)[0]
        if len(idx) > 0: best[idx] = b[idx]
        idx = torch.nonzero((fa.abs() >= self.tol_bi) & (fb.abs() >= self.tol_bi), as_tuple=True)[0]
        if len(idx) > 0:
            a, b, T, args = a[idx], b[idx], T[idx], args[idx]
            if iomega is not None: iomega = iomega[idx]
        while len(idx) > 0:
            c = (a + b) / 2
            fc = fun(c, args, T, iomega)
            good_idx = torch.nonzero((fc.abs() < self.tol_bi) | ((b - a).abs().view(-1) / 2 < self.tol_bi), as_tuple=True)[0]
            if len(good_idx) > 0: best[idx[good_idx]] = c[good_idx]
            if len(good_idx) < len(c):
                bad_idx = torch.nonzero((fc.abs() >= self.tol_bi) & ((b - a).abs().view(-1) / 2 >= self.tol_bi), as_tuple=True)[0]
                a, b, c, fc, idx = a[bad_idx], b[bad_idx], c[bad_idx], fc[bad_idx], idx[bad_idx]
                T, args = T[bad_idx], args[bad_idx]
                if iomega is not None: iomega = iomega[bad_idx]
                index = torch.nonzero(fc.sign() * fun(a, args, T, iomega).sign() < 0, as_tuple=True)
                b[index] = c[index]
                c[index] = a[index]
                a = c
            else:
                break
        return best

    def calc_OP(self, fun, nf):
        return fun(nf.squeeze(1))

    @torch.no_grad()
    def __call__(self, T, H0, U, E_mu=None, model=None, SEinit=None, adjMu=None, reOP=False, reNf=False, reBad=False,
                 OPfuns=(lambda n: (n[:, 0] - n[:, 1]).abs(),), prinfo=False):  # T, U, E_mu: (bz,)
        '''-3. get parameters'''
        device, dtype = self.iomega0.device, torch.float32 if self.iomega0.dtype == torch.complex64 else torch.float64
        bz, _, _, size = H0.shape
        '''-2. initialize variables'''
        T = T.unsqueeze(-1).to(device=device, dtype=dtype)                             # (bz, 1)
        iomega = torch.matmul(T + 1j * 0., self.iomega0).unsqueeze(-1)                 # (bz, self.count, 1)
        U = U[:, None, None].to(device=device, dtype=dtype)                            # (bz, 1, 1)
        E_mu = self.init_E_mu(E_mu, device, dtype, bz)                                 # (bz, 1)
        H0 = self.init_H0(H0, device, self.iomega0.dtype, adjMu, T, bz, size, prinfo)  # (bz, 1, size, size)
        if model is not None: model.z = iomega
        '''-1. initialize records and parameters'''
        min_error, min_errors = 1e10, None
        best_SE, best_nf = None, None
        cur_tol_sc, avg_tol_sc = self.tol_sc, self.tol_sc / bz ** 0.5
        '''0. initialize self-energy'''
        SE = self.init_SE(SEinit, device, self.iomega0.dtype, bz, size)                # (bz, self.count, size)
        for l in range(self.MAXEPOCH):
            '''1. compute G_{loc}'''
            Gloc = self.calc_Gloc(H0 + torch.diag_embed(SE), iomega, model)
            '''2. compute Weiss field \mathcal{G}_0'''
            WeissInv = Gloc.pow(-1) + SE  # (bz, self.count, size)
            '''3. compute G_{imp}'''
            UoverWI = U / WeissInv
            if self.f_filling is not None: E_mu = self.bisearch(self.fix_filling, E_mu, UoverWI, T, iomega)
            nf = self.calc_nf(E_mu, UoverWI, T, iomega).unsqueeze(1) # (bz, 1, size)
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
                else:
                    best_SE[idx], best_nf[idx] = SE, nf
                if reBad:
                    idx = torch.tensor([], dtype=torch.long, device=new_errors.device)
                    min_errors = torch.tensor([], dtype=dtype, device=new_errors.device)
                break
            else:
                m = self.momentum
                if prinfo: print("{} loop error: {:.5e}".format(l, tot_error))
                if l <= self.MILESTONE and tot_error < min_error:
                    min_error, min_errors = tot_error, new_errors
                    best_SE, best_nf = SE, nf
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
                        bad_idx = torch.nonzero(min_errors >= avg_tol_sc, as_tuple=True)[0]
                        idx = idx[bad_idx]
                    H0, T, U, E_mu, iomega = H0[bad_idx], T[bad_idx], U[bad_idx], E_mu[bad_idx], iomega[bad_idx]
                    SE, min_errors, WeissInv, Gimp = SE[bad_idx], min_errors[bad_idx], WeissInv[bad_idx], Gimp[bad_idx]
                    if model is not None: model.z = iomega
                    cur_tol_sc = self.tol_sc * (len(idx) / bz) ** 0.5
                    if prinfo: print("{} loop remain: {}".format(l, len(idx)))
                    m = torch.normal(self.momentum, self.momDisor, (1,)).clamp(min=0., max=1.).item()
                SE = m * SE + (1. - m) * (WeissInv - Gimp.pow(-1))
        res = [best_SE, torch.stack([self.calc_OP(fun, best_nf) for fun in OPfuns], dim=0) if reOP else None,
               best_nf if reNf else None, [idx, min_errors] if reBad else None]
        res = [x for x in res if x is not None]
        return res if len(res) > 1 else res[0]


def op_loc(nf):
    nf = nf * 2 - 1
    bz, size = nf.shape
    L = int(size ** (1 / 2))
    dtype, device = nf.dtype, nf.device
    correlation = torch.zeros(bz, dtype=dtype, device=device)
    for rx in range(L):
        for ry in range(L):
            n = l2c(rx, ry, L)
            for delta_x in [-1, 0, 1]:
                for delta_y in [-1, 0, 1]:
                    if delta_x == 0 and delta_y == 0: continue
                    n_delta = l2c(rx + delta_x, ry + delta_y, L)
                    correlation += nf[:, n] * nf[:, n_delta] * (-1) ** (delta_x + delta_y)
    return correlation / (size * 8)


def op_cb(nf):
    L = int(nf.shape[-1] ** 0.5)
    line = (-1) ** torch.arange(L, device=nf.device)
    mask = torch.cat([line * (-1) ** i for i in range(L)])
    return 2 * torch.mean(nf * mask, dim=-1).abs()


def op_str(nf):
    L = int(nf.shape[-1] ** 0.5)
    line = torch.ones(L, device=nf.device)
    mask1 = torch.cat([line * (-1) ** i for i in range(L)])
    mask2 = ((-1) ** torch.arange(L, device=nf.device)).tile(L)
    return 2 * torch.max(torch.mean(nf * mask1, dim=-1).abs(), torch.mean(nf * mask2, dim=-1).abs())  # (bz,)


if __name__ == "__main__":
    from FK_Data import Ham, Ham2
    import numpy as np
    import os, time, warnings
    warnings.filterwarnings('ignore')
    np.set_printoptions(precision=3, linewidth=80, suppress=True)

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
    torch.manual_seed(10)

    L = 12  # size = L ** 2
    data = f'FK_{L}'
    Net = 'Naive_1'
    T = 0.13
    save = True
    show = True

    '''construct DMFT'''
    count = 20
    iota = 0.
    momentum = 0.5
    momDisor = 0.
    maxEpoch = 2000
    milestone = 30
    f_filling = 0.5
    d_filling = None
    tol_sc = 1e-6
    tol_bi = 1e-7
    gap = 1.
    scf = DMFT(count, iota, momentum, momDisor, maxEpoch, milestone, f_filling, d_filling, tol_sc, tol_bi, gap, device)

    '''2D test'''
    # tp = torch.linspace(0., 1.2, 61)
    # mu = torch.zeros(len(tp))#torch.linspace(-0.5, 0.1, 31)
    # U = torch.ones(len(tp))
    # T = T * torch.ones(len(U))
    # adjMu = torch.cat((torch.linspace(0.5, -0.1, 36), torch.linspace(-0.1, 0.05, 25)))
    # H0 = torch.stack([Ham(L, i.item(), j.item()) for i, j in zip(mu, tp)], dim=0).unsqueeze(1)
    # t = time.time()
    # SE, OP, nf, Bad = scf(T, H0, U, adjMu=adjMu, reOP=True, reNf=True, reBad=True, OPfuns=(op_cb, op_str), prinfo=True)
    # print(time.time() - t)
    # for i, op in enumerate(nf.cpu().numpy()):
    #     print(i, '\n', op)
    # print('order parameter:\n', OP.cpu().numpy())
    # print('order:\n', torch.max(OP, dim=0)[1].cpu().numpy())
    # print('bad index:', Bad[0].cpu().numpy())
    # print('bad error:', Bad[1].cpu().numpy())
    # exit(0)

    from FK_rgfnn import Network
    from utils import myceil, mymkdir
    import matplotlib.pyplot as plt
    from torch.nn.functional import softmax, tanh, relu
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)

    '''construct FQNN'''
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network(Net[:Net.index('_')], L ** 2, 1 if Net.startswith('C') else 2, 100, 64, None, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    '''construct Hamiltonians'''
    # tp = torch.linspace(0., 1.2, 61)
    # U = torch.ones(len(tp))
    # mu = torch.zeros(len(tp))
    # adjMu = torch.cat((torch.linspace(0.5, -0.1, 36), torch.linspace(-0.1, 0.05, 25)))
    # H0 = torch.stack([Ham(L, i.item(), j.item()) for i, j in zip(mu, tp)], dim=0).unsqueeze(1)
    U = torch.linspace(1., 4., 150)
    mu = U / 2.
    adjMu = None
    H0 = Ham2(L, mu).unsqueeze(1)
    PTPs = {'0.005':(0.575, 1 / np.sqrt(2)), '0.020':(0.33, 0.392), '0.100': (1.47, 1.76), '0.110': (1.63, 2.01),
            '0.120': (1.8, 2.28), '0.130': (1.96, 2.6), '0.140': (2.14, 3.03), '0.150': (2.34, 3.99),
            '0.160': (2.54, 4), '0.170': (2.74, 4), '0.180': (2.98, 4), '0.190': (3.27, 4), '0.200': (3.57, 4)}
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
                SE, op = scf(T_batch, H0_batch, U_batch, model=None, adjMu=adjMu, reOP=True, OPfuns=(op_cb, op_str),
                             prinfo=True if i == 0 else False)  # (bz, scf.count, size)
                SEs.append(SE.cpu())
                OP.append(op.cpu())
        else:
            SE, op = scf(T_batch, H0_batch, U_batch, model=model, adjMu=adjMu, reOP=True, OPfuns=(op_cb, op_str),
                         prinfo=True if i == 0 else False)  # (bz, scf.count, size)
            OP.append(op.cpu())
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
    if type(OP) is list: OP = torch.cat(OP, dim=1)
    if '2d' in Net and type(SEs) is list:
        torch.save(torch.cat(SEs, dim=0), f'results/{data}/SE+OP/SE_{T[0].item():.3f}.pt')
        torch.save(OP, f'results/{data}/SE+OP/OP_{T[0].item():.3f}.pt')
    OP = OP.numpy()
    x = U.numpy() # tp.numpy()

    '''plot phase diagram'''
    labels = ['cb', 'stripe']
    fig, ax1 = plt.subplots()
    plt.axis([x[0], x[-1], 0., 1.])
    # plt.title(f'Checkerboard VS Stripe / T={T[0].item():.3f}, L={L}')
    # ax1.set_xlabel("t'")
    plt.title(f'Metal VS Insulator / T={T[0].item():.3f}, L={L}')
    ax1.set_xlabel("U")
    ax1.set_xlim([x[0], x[-1]])
    ax1.set_ylabel('P', c='r')
    ax1.set_ylim([0., 1.])
    ax1.set_yticks(0.1 * np.arange(11))
    ax1.scatter(x, P, s=10, c='r', marker='o')
    ax1.plot([x[0], x[-1]], [0.5, 0.5], 'ko--', linewidth=0.5, markersize=0.1)
    if PTP is not None: ax1.plot([PTP, PTP], [0., 1.], 'go--', linewidth=0.5, markersize=0.1)
    if QMCPTP is not None: ax1.plot([QMCPTP, QMCPTP], [0., 1.], 'yo--', linewidth=0.5, markersize=0.1)
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('OP', c='b')
    ax2.set_ylim([0., 1.])
    ax2.set_yticks(0.1 * np.arange(11))
    for i, op in enumerate(OP):
        ax2.scatter(x, op, s=10, marker='^', label=labels[i])
    ax2.legend(loc='lower right')
    ax2.tick_params(axis='y', labelcolor='b')
    if save:
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/PD_{:.3f}.jpg'.format(path, T[0].item()))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/PD_{:.3f}.npy'.format(path, T[0].item()), np.concatenate((np.stack((x, P), axis=0), OP), axis=0))
