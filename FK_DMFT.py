import torch


class DMFT:
    def __init__(self, T=1e-2, count=100, iota=0., momentum=0., device=torch.device('cpu'), double=True):
        self.T = T
        self.count = count * 2
        self.iota = iota
        self.momentum = momentum
        self.iomega = 1j * (2 * torch.arange(-int(count), int(count), device=device).unsqueeze(1) + 1) * torch.pi * self.T  # (count, 1)
        if double: self.iomega = self.iomega.type(torch.complex128)

    def checkerboard(self, L, device, dtype):
        line = torch.tensor([0., 1.], device=device, dtype=dtype).tile(int(L / 2))
        return torch.cat((line, 1. - line)).tile(int(L / 2))

    def nansum(self, x, dim, keepdim=False):
        return torch.nansum(x.real, dim=dim, keepdim=keepdim) + 1j * torch.nansum(x.imag, dim=dim, keepdim=keepdim)

    def nan_to_num(self, x, nan=0.):
        return torch.nan_to_num(x.real, nan=nan) + 1j * torch.nan_to_num(x.imag, nan=nan)

    def calc_nf(self, WeissInv, iomega, E_mu, U):
        z = torch.sum(torch.log(1 - U / WeissInv) * torch.exp(iomega * self.iota), dim=1) - E_mu / self.T # (bz, size)
        return self.nan_to_num(torch.sigmoid(z)).unsqueeze(1)  # (bz, 1, size)

    def fourier(self, SE, iomega, tau):
        return -torch.sum(SE * torch.exp(iomega * tau), dim=1, keepdim=True).real / self.count / self.T / torch.pi  # (bz, 1, size)

    @torch.no_grad()
    def __call__(self, H0, E_mu, U, model=None, prinfo=False):  # E_mu, U: (bz,)
        bz, _, _, size = H0.shape
        H0 = H0.tile(1, self.count, 1, 1) # (bz, count, size, size)
        device, dtype = H0.device, H0.dtype
        E_mu = E_mu.unsqueeze(1).type(dtype)  # (bz, 1)
        U = U[:, None, None].type(dtype) # (bz, 1, 1)
        if model is None:
            H_omega = torch.diag_embed(self.iomega.expand(-1, size)) - H0  # (bz, count, size, size)
        else:
            z = model.z.clone()
            model.z = self.iomega
        '''0. initialize self-energy'''
        SE = torch.zeros((bz, self.count, size), device=device, dtype=dtype) # (bz, count, size)
        # SE = 0. + 1. * torch.randn((bz, self.count, size), device=device, dtype=dtype) # (bz, count, size)
        if prinfo: l = 1
        while True:
            '''1. compute G_{loc}'''
            if model is None:
                Gloc = torch.diagonal((H_omega - torch.diag_embed(SE)).inverse(), dim1=-2, dim2=-1)  # (bz, count, size)
            else:
                Gloc = model(H0 + torch.diag_embed(SE), selfcons=True)  # (bz, count, size)
            '''2. compute Weiss field \mathcal{G}_0'''
            WeissInv = Gloc.pow(-1) + SE  # (bz, count, size)
            '''3. compute G_{imp}'''
            nf = self.calc_nf(WeissInv, self.iomega, E_mu, U) # (bz, 1, size)
            # nf = 0.5 * torch.ones((bz, 1, size), device=device, dtype=dtype)
            # nf = self.checkerboard(int(size ** (0.5)), device, dtype).expand((bz, 1, size))
            Gimp = nf / (WeissInv - U) + (1. - nf) / WeissInv  # (bz, count, size)
            '''4. compute new self-energy'''
            error = torch.linalg.norm(Gimp - Gloc).item()
            if error < 1e-5:
                if prinfo:
                    print("final error: {}".format(error))
                    print(torch.round(nf.real.cpu(), decimals=3).numpy())
                if model is not None:
                    model.z = z
                return self.fourier(SE, self.iomega, tau=0.5 / self.T)  # (bz, 1, size)
            else:
                SE = self.momentum * SE + (1. - self.momentum) * (WeissInv - Gimp.pow(-1))
                if prinfo:
                    print("{} loop error: {}".format(l, error))
                    l += 1


def pbc(x, L):
    return x % L


def l2c(x, y, L):  # 2D coordinate -> 1D coordinate
    x, y = pbc(x, L), pbc(y, L)
    return x + y * L


def Ham(L, mu):
    H = torch.diag_embed(-mu * torch.ones(L ** 2)).type(torch.complex128)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            # nearest neighbor
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            H[nx, n] = H[nx, n] - 1.
            H[n, nx] = H[n, nx] - 1.
            H[ny, n] = H[ny, n] - 1.
            H[n, ny] = H[n, ny] - 1.
            # # next nearest neighbor
            # n1 = l2c(x + 1, y + 1, L)
            # n2 = l2c(x + 1, y - 1, L)
            # H[n1, n] = H[n1, n] - 1.
            # H[n, n1] = H[n, n1] - 1.
            # H[n2, n] = H[n2, n] - 1.
            # H[n, n2] = H[n, n2] - 1.
    return H


if __name__ == "__main__":
    from FK_rgfnn import Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    L = 16  # size = L ** 2

    '''construct DMFT'''
    T = 0.01
    count = 100
    momentum = 0.8
    scf = DMFT(T, count, momentum=momentum, device=device)

    '''construct FQNN'''
    model = Network('Naive', L ** 2, 2, 100, 64, 1e-3j, double=True).to(device)

    '''construct Hamiltonian'''
    U = torch.tensor([3.], device=device)
    mu = (U / 2).item()
    E_mu = torch.tensor([0.], device=device) - mu # E - mu
    H0 = Ham(L, mu)[None, None, ...].to(device)

    '''compute self-energy by DMFT'''
    SE = scf(H0, E_mu, U, model, prinfo=True)  # (bz, 1, size)

    '''compute ground state energy'''
    H = H0 + torch.diag_embed(SE)
    Es = torch.linalg.eigvalsh(H).squeeze()
    E_gs = torch.nansum(Es * torch.sigmoid(-Es / T)).item() / L ** 2
    print('E_gs={}'.format(E_gs))