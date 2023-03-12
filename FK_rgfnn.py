import torch
from typing import Union, Optional
from rgfnn import GLinear, Abstract


def Network(Net, input_size, output_size=2, embedding_size: Union[int, None] = 100,
            hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
            diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
            init_bound: float = 1., scale: bool = True, drop: float = 0., sigma: float = 0., double: bool = False):
    if Net == 'Naive':
        return Naive(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                     scale, drop, sigma, double)
    elif Net == 'CNaive':
        return CNaive(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                      scale, drop, sigma, double)
    elif Net == 'Simple':
        return Simple(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                      scale, drop, sigma, double)
    elif Net == 'CSimple':
        return CSimple(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                       scale, drop, sigma, double)
    elif Net == 'Median':
        return Median(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                       scale, drop, sigma, double)
    elif Net == 'CMedian':
        return CMedian(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                       scale, drop, sigma, double)
    else:
        raise NameError('Wrong Net Type')


class AbstractSC(Abstract):
    def __init__(self, z: Optional[complex] = None, scale: Optional[bool] = None):
        super(AbstractSC, self).__init__(z, scale)

    def z_wrapper(self, z, size):
        if z.dim() == 0:
            return z * torch.eye(size, device=z.device)
        elif z.dim() == 1:  # (bz,)
            return torch.diag_embed(z[:, None, None].tile(1, 1, size))
        elif z.dim() == 2:  # (count, 1)
            return torch.diag_embed(z.tile(1, size))
        elif z.dim() == 3:  # (bz, count, 1)
            return torch.diag_embed(z.tile(1, 1, size))
        else:
            raise RuntimeError('Wrong dim of z')

    def postproc(self, g):
        count = g.shape[1]
        if count == 1:
            g = g.imag.squeeze(1)
        else:
            n = int(count / 2)
            g = torch.sum(g * (-1) ** torch.arange(-n, n, device=g.device), dim=1).real * 1j / torch.pi
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(-1)
        return g * self.scale

    def FQNN(self, g):
        raise NotImplementedError

    @torch.no_grad()
    def G_diag(self, g, all):
        raise NotImplementedError

    @torch.no_grad()
    def rG_11(self, h):
        raise NotImplementedError

    def forward(self, h, selfcons=False, rev=True, all=False):
        if selfcons:
            if rev:
                return self.rG_11(h)
            else:
                return self.G_diag((self.z_wrapper(self.z, h.shape[-1]) - h).inverse(), all)
        else:
            return self.FQNN((self.z_wrapper(self.z, h.shape[-1]) - h).inverse())


class Naive(AbstractSC):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(Naive, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 3)
        self.pre = Gloc(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = Gloc(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.out = Gloc(self.emb.size, output_size, hermi, diago[2], False, real, drop, sigma, init_bound, double)

    def FQNN(self, g):
        g = self.pre(g, self.z)
        g = self.emb(g, self.z)
        g = self.out(g, self.z)
        return self.postproc(g)

    @torch.no_grad()
    def G_diag(self, g, all):
        g = self.pre([g, g, g, torch.diagonal(g, dim1=-2, dim2=-1)], self.z, all=all)
        g = self.emb(g, self.z, all=all)
        return self.out(g, self.z, all=all)[-1]

    @torch.no_grad()
    def rG_11(self, h):
        tgt = self.out(None, self.z, rev=True)
        tgt = self.emb(tgt, self.z, rev=True)
        tgt = self.pre(tgt, self.z, rev=True)
        return torch.diagonal((self.z_wrapper(self.z, h.shape[-1]) - h - tgt).inverse(), dim1=-2, dim2=-1)


class CNaive(Naive):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(CNaive, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real,
                                     init_bound, scale, drop, sigma, double)
        self.scale = torch.tensor(scale)

    def FQNN(self, g):
        g = self.pre([g, g], self.z)
        g = self.emb(g, self.z)
        g = self.out(g, self.z)[-1]
        g = g.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 8.  # 1 / localisation length
            # return -8. / torch.log(g)  # localisation length


class Simple(AbstractSC):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(Simple, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 5)
        self.pre = Gloc(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = Gloc(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = Gloc(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = Gloc(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.out = Gloc(self.fc2.size, output_size, hermi, diago[4], False, real, drop, sigma, init_bound, double)

    def FQNN(self, g):
        g = self.pre(g, self.z)
        g = self.emb(g, self.z)
        g = self.fc1(g, self.z)
        g = self.fc2(g, self.z)
        g = self.out(g, self.z)
        return self.postproc(g)

    @torch.no_grad()
    def G_diag(self, g, all):
        g = self.pre([g, g, g, torch.diagonal(g, dim1=-2, dim2=-1)], self.z, all=all)
        g = self.emb(g, self.z, all=all)
        g = self.fc1(g, self.z, all=all)
        g = self.fc2(g, self.z, all=all)
        return self.out(g, self.z, all=all)[-1]

    @torch.no_grad()
    def rG_11(self, h):
        tgt = self.out(None, self.z, rev=True)
        tgt = self.fc2(tgt, self.z, rev=True)
        tgt = self.fc1(tgt, self.z, rev=True)
        tgt = self.emb(tgt, self.z, rev=True)
        tgt = self.pre(tgt, self.z, rev=True)
        return torch.diagonal((self.z_wrapper(self.z, h.shape[-1]) - h - tgt).inverse(), dim1=-2, dim2=-1)


class CSimple(Simple):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(CSimple, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                                      real, init_bound, scale, drop, sigma, double)
        self.scale =  torch.tensor(scale)

    def FQNN(self, g):
        g = self.pre([g, g], self.z)
        g = self.emb(g, self.z)
        g = self.fc1(g, self.z)
        g = self.fc2(g, self.z)
        g = self.out(g, self.z)[-1]
        g = g.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 12.  # 1 / localisation length


class Median(AbstractSC):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(Median, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 7)
        self.pre = Gloc(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = Gloc(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = Gloc(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = Gloc(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.fc3 = Gloc(self.fc2.size, hidden_size[3], hermi, diago[4], restr[4], real, drop, sigma, init_bound, double)
        self.fc4 = Gloc(self.fc3.size, hidden_size[4], hermi, diago[5], restr[5], real, drop, sigma, init_bound, double)
        self.out = Gloc(self.fc4.size, output_size, hermi, diago[6], False, real, drop, sigma, init_bound, double)

    def FQNN(self, g):
        g = self.pre(g, self.z)
        g = self.emb(g, self.z)
        g = self.fc1(g, self.z)
        g = self.fc2(g, self.z)
        g = self.fc3(g, self.z)
        g = self.fc4(g, self.z)
        g = self.out(g, self.z)
        return self.postproc(g)

    @torch.no_grad()
    def G_diag(self, g, all):
        g = self.pre([g, g, g, torch.diagonal(g, dim1=-2, dim2=-1)], self.z, all=all)
        g = self.emb(g, self.z, all=all)
        g = self.fc1(g, self.z, all=all)
        g = self.fc2(g, self.z, all=all)
        g = self.fc3(g, self.z, all=all)
        g = self.fc4(g, self.z, all=all)
        return self.out(g, self.z, all=all)[-1]

    @torch.no_grad()
    def rG_11(self, h):
        tgt = self.out(None, self.z, rev=True)
        tgt = self.fc4(tgt, self.z, rev=True)
        tgt = self.fc3(tgt, self.z, rev=True)
        tgt = self.fc2(tgt, self.z, rev=True)
        tgt = self.fc1(tgt, self.z, rev=True)
        tgt = self.emb(tgt, self.z, rev=True)
        tgt = self.pre(tgt, self.z, rev=True)
        return torch.diagonal((self.z_wrapper(self.z, h.shape[-1]) - h - tgt).inverse(), dim1=-2, dim2=-1)


class CMedian(Median):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(CMedian, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                                      real, init_bound, scale, drop, sigma, double)
        self.scale = torch.tensor(scale)

    def FQNN(self, g):
        g = self.pre([g, g], self.z)
        g = self.emb(g, self.z)
        g = self.fc1(g, self.z)
        g = self.fc2(g, self.z)
        g = self.fc3(g, self.z)
        g = self.fc4(g, self.z)
        g = self.out(g, self.z)[-1]
        g = g.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 16.  # 1 / localisation length





'''
Basic Layers: Gloc
'''


class Gloc(GLinear):
    def __init__(self, input_size, output_size, hermi: Union[bool, int] = True, diago: Union[bool, int] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, drop : float = 0., sigma: float = 0.,
                 init_bound: float = 1., double: bool = False):
        super(Gloc, self).__init__(input_size, output_size, hermi, diago, restr, real, drop, sigma, init_bound, double)

    def G_NN(self, g, z):
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        return torch.linalg.inv(self.z_bias(z) - hb - tw.mH @ g @ tw)

    def rG_NN(self, tgt, z):
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        if tgt is None:
            return tw @ (self.z_bias(z) - hb).inverse() @ tw.mH
        else:
            return tw @ (self.z_bias(z) - hb - tgt).inverse() @ tw.mH

    def G_1N(self, g, z):
        gd, god = g
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        gd = torch.linalg.inv(self.z_bias(z) - hb - tw.mH @ gd @ tw)
        god = god @ tw @ gd
        return [gd, god]

    def G_ii(self, g, z):
        gd, giN, gNi, gii = g
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        gd = torch.linalg.inv(self.z_bias(z) - hb - tw.mH @ gd @ tw)
        giN = giN @ tw @ gd
        temp = tw.mH @ gNi
        gii = gii + torch.diagonal(giN @ temp, dim1=-2, dim2=-1)
        gii = torch.cat((gii, torch.diagonal(gd, dim1=-2, dim2=-1)), dim=-1)
        giN = torch.cat((giN, gd), dim=-2)
        gNi = torch.cat((gd @ temp, gd), dim=-1)
        return [gd, giN, gNi, gii]

    def G_11(self, g, z):
        gd, g1N, gN1, g11 = g
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        gd = torch.linalg.inv(self.z_bias(z) - hb - tw.mH @ gd @ tw)
        g1N = g1N @ tw @ gd
        temp = tw.mH @ gN1
        g11 = g11 + torch.diagonal(g1N @ temp, dim1=-2, dim2=-1)
        return [gd, g1N, gd @ temp, g11]

    def forward(self, g, z, rev=False, all=False):
        if rev:
            return self.rG_NN(g, z)
        elif type(g) is torch.Tensor:
            return self.G_NN(g, z)
        elif len(g) == 2:
            return self.G_1N(g, z)
        elif all:
            return self.G_ii(g, z)
        else:
            return self.G_11(g, z)


