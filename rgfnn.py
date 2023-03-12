import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


def Network(Net, input_size, output_size=10, embedding_size: Union[int, None] = 100,
            hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
            diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
            init_bound: float = 1., scale: bool = True, drop: float = 0., sigma: float = 0., double: bool = False,
            trs=True):
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
    elif Net == 'NaiveL':
        return NaiveL(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                      scale, drop, sigma, double)
    elif Net == 'ClassicalNaive':
        return ClassicalNaive(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real,
                              init_bound, scale, drop, sigma, double)
    elif Net == 'NaiveB':
        return NaiveB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                      scale, drop, sigma, double, trs)
    elif Net == 'NaiveCB':
        return NaiveCB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                       scale, drop, sigma, double, trs)
    elif Net == 'ClassicalNaiveB':
        return ClassicalNaiveB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real,
                               init_bound, scale, drop, sigma, double, trs)
    elif Net == 'ClassicalNaiveCB':
        return ClassicalNaiveCB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real,
                                init_bound, scale, drop, sigma, double, trs)
    elif Net == 'SimpleB':
        return SimpleB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                       scale, drop, sigma, double, trs)
    elif Net == 'SimpleCB':
        return SimpleCB(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real, init_bound,
                        scale, drop, sigma, double, trs)
    else:
        raise NameError('Wrong Net Type')


class Abstract(nn.Module):
    def __init__(self, z: Optional[complex] = None, scale: Optional[bool] = None):
        super(Abstract, self).__init__()
        if z is None:
            self.z = None
        else:
            self.register_buffer('z', z if type(z) is torch.Tensor else torch.tensor(z))
        if scale is not None:
            self.register_buffer('scale', torch.tensor(-1. / torch.pi if scale else 1.))

    def parameters_wrapper(self, hidden_size, diago, restr, length):
        if type(hidden_size) is not tuple:
            hidden_size = [hidden_size] * (length - 2)
        if type(diago) is not tuple:
            diago = [diago] * length
        if type(restr) is not tuple or len(restr) == 3:
           restr = [restr] * (length - 1)
        return hidden_size, diago, restr

    def extra_repr(self) -> str:
        if self.z is None:
            return 'z=None, scale={:.5f}'.format(self.scale)
        elif self.z.dim() == 0:
            return 'z={:.3e}, scale={:.5f}'.format(self.z.item(), self.scale)
        else:
            return 'scale={:.5f}'.format(self.scale)


class Naive(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(Naive, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 3)
        self.pre = GLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.out = GLinear(self.emb.size, output_size, hermi, diago[2], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        g = self.pre(g, self.z).inverse()
        g = self.emb(g, self.z).inverse()
        g = self.out(g, self.z).inverse()
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class CNaive(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(CNaive, self).__init__(z)
        self.register_buffer('scale', torch.tensor(scale))
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 3)
        self.pre = GcorLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GcorLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.out = GcorLinear(self.emb.size, output_size, hermi, diago[2], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        gd, god = self.pre([g, g], self.z)
        gd, god = self.emb([gd, god], self.z)
        _, god = self.out([gd, god], self.z)
        g = god.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 8.  # 1 / localisation length
            # return -8. / torch.log(g)  # localisation length


class Simple(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(Simple, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 5)
        self.pre = GLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = GLinear(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = GLinear(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.out = GLinear(self.fc2.size, output_size, hermi, diago[4], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        g = self.pre(g, self.z).inverse()
        g = self.emb(g, self.z).inverse()
        g = self.fc1(g, self.z).inverse()
        g = self.fc2(g, self.z).inverse()
        g = self.out(g, self.z).inverse()
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class CSimple(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(CSimple, self).__init__(z, scale)
        self.register_buffer('scale', torch.tensor(scale))
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 5)
        self.pre = GcorLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GcorLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = GcorLinear(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = GcorLinear(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.out = GcorLinear(self.fc2.size, output_size, hermi, diago[4], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        gd, god = self.pre([g, g], self.z)
        gd, god = self.emb([gd, god], self.z)
        gd, god = self.fc1([gd, god], self.z)
        gd, god = self.fc2([gd, god], self.z)
        _, god = self.out([gd, god], self.z)
        g = god.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 12.  # 1 / localisation length


class Median(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(Median, self).__init__(z, scale)
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 7)
        self.pre = GLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = GLinear(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = GLinear(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.fc3 = GLinear(self.fc2.size, hidden_size[3], hermi, diago[4], restr[4], real, drop, sigma, init_bound, double)
        self.fc4 = GLinear(self.fc3.size, hidden_size[4], hermi, diago[5], restr[5], real, drop, sigma, init_bound, double)
        self.out = GLinear(self.fc4.size, output_size, hermi, diago[6], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        g = self.pre(g, self.z).inverse()
        g = self.emb(g, self.z).inverse()
        g = self.fc1(g, self.z).inverse()
        g = self.fc2(g, self.z).inverse()
        g = self.fc3(g, self.z).inverse()
        g = self.fc4(g, self.z).inverse()
        g = self.out(g, self.z).inverse()
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class CMedian(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False):
        super(CMedian, self).__init__(z, scale)
        self.register_buffer('scale', torch.tensor(scale))
        hidden_size, diago, restr = self.parameters_wrapper(hidden_size, diago, restr, 7)
        self.pre = GcorLinear(input_size, embedding_size, hermi, diago[0], restr[0], real, drop, sigma, init_bound, double)
        self.emb = GcorLinear(self.pre.size, hidden_size[0], hermi, diago[1], restr[1], real, drop, sigma, init_bound, double)
        self.fc1 = GcorLinear(self.emb.size, hidden_size[1], hermi, diago[2], restr[2], real, drop, sigma, init_bound, double)
        self.fc2 = GcorLinear(self.fc1.size, hidden_size[2], hermi, diago[3], restr[3], real, drop, sigma, init_bound, double)
        self.fc3 = GcorLinear(self.fc2.size, hidden_size[3], hermi, diago[4], restr[4], real, drop, sigma, init_bound, double)
        self.fc4 = GcorLinear(self.fc3.size, hidden_size[4], hermi, diago[5], restr[5], real, drop, sigma, init_bound, double)
        self.out = GcorLinear(self.fc4.size, output_size, hermi, diago[6], False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        gd, god = self.pre([g, g], self.z)
        gd, god = self.emb([gd, god], self.z)
        gd, god = self.fc1([gd, god], self.z)
        gd, god = self.fc2([gd, god], self.z)
        gd, god = self.fc3([gd, god], self.z)
        gd, god = self.fc4([gd, god], self.z)
        _, god = self.out([gd, god], self.z)
        g = god.squeeze(1).squeeze(-1).sum(dim=1)
        g = g.real ** 2 + g.imag ** 2
        if self.scale:
            return g
        else:
            return -torch.log(g) / 16.  # 1 / localisation length


class NaiveL(Abstract):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(NaiveL, self).__init__(z, scale)
        if type(restr) is tuple:
            if len(restr) == 2:
                pre_restr, emb_restr = restr
            else:
                pre_restr, emb_restr = restr, restr
        else:
            pre_restr, emb_restr = restr, restr
        if type(diago) is tuple:
            pre_diago, emb_diago, out_diago = diago
        else:
            pre_diago, emb_diago, out_diago = diago, diago, diago
        self.pre = GLinear(input_size, embedding_size, hermi, pre_diago, pre_restr, real, drop, sigma, init_bound, double)
        self.emb = GLinear(self.pre.size, hidden_size, hermi, emb_diago, emb_restr, real, drop, sigma, init_bound, double)
        self.out = GLinear(self.emb.size, output_size, hermi, out_diago, False, real, drop, sigma, init_bound, double)

    def forward(self, g):
        g = self.pre(g, self.z)
        g = self.emb(g, self.z)
        g = self.out(g, self.z)
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class ClassicalNaive(NaiveL):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0.,
                 double: bool = False):
        super(ClassicalNaive, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago,
                                             restr, real, init_bound, scale, drop, sigma, double)
        # self.pre = Linear2D(input_size, embedding_size, hermi, diago, restr)
        # self.emb = Linear2D(embedding_size, hidden_size, hermi, diago, restr)
        # self.out = Linear2D(hidden_size, output_size, hermi, diago, False)

    def forward(self, g):
        g = complex_relu(self.pre(g, self.z))
        g = complex_relu(self.emb(g, self.z))
        g = self.out(g, self.z)
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class NaiveB(Naive):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False,
                 trs=True):
        super(NaiveB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, real,
                                     init_bound, scale, drop, sigma, double)
        self.bnp = NaiveComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = NaiveComplexBatchNorm2d(1, track_running_stats=trs)

    def forward(self, g):
        g = self.bnp(self.pre(g, self.z)).inverse()
        g = self.bne(self.emb(g, self.z)).inverse()
        g = self.out(g, self.z).inverse()
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class NaiveCB(NaiveB):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False,
                 trs=True):
        super(NaiveCB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                                      real, init_bound, scale, drop, sigma, double)
        self.bnp = ComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = ComplexBatchNorm2d(1, track_running_stats=trs)


class ClassicalNaiveB(ClassicalNaive):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False,
                 trs=True):
        super(ClassicalNaiveB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago,
                                              restr, real, init_bound, scale, drop, sigma, double)
        self.bnp = NaiveComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = NaiveComplexBatchNorm2d(1, track_running_stats=trs)

    def forward(self, g):
        g = complex_relu(self.bnp(self.pre(g, self.z)))
        g = complex_relu(self.bne(self.emb(g, self.z)))
        g = self.out(g, self.z)
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class ClassicalNaiveCB(ClassicalNaiveB):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, None] = 64, z: Union[complex, None] = 1e-3j, hermi: Union[bool, int] = True,
                 diago: Union[bool, int, tuple] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 init_bound: float = 1., scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False,
                 trs=True):
        super(ClassicalNaiveCB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago,
                                               restr, real, init_bound, scale, drop, sigma, double)
        self.bnp = ComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = ComplexBatchNorm2d(1, track_running_stats=trs)


class SimpleB(Simple):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False, trs=True):
        super(SimpleB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                                      real, init_bound, scale, drop, sigma, double)
        self.bnp = NaiveComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = NaiveComplexBatchNorm2d(1, track_running_stats=trs)
        self.bn1 = NaiveComplexBatchNorm2d(1, track_running_stats=trs)
        self.bn2 = NaiveComplexBatchNorm2d(1, track_running_stats=trs)

    def forward(self, g):
        g = self.bnp(self.pre(g, self.z)).inverse()
        g = self.bne(self.emb(g, self.z)).inverse()
        g = self.bn1(self.fc1(g, self.z)).inverse()
        g = self.bn2(self.fc2(g, self.z)).inverse()
        g = self.out(g, self.z).inverse()
        g = torch.diagonal(g, dim1=-2, dim2=-1).squeeze(1).squeeze(-1)
        g = g.imag * self.scale
        return g


class SimpleCB(SimpleB):
    def __init__(self, input_size, output_size=10, embedding_size: Union[int, None] = 100,
                 hidden_size: Union[int, tuple, None] = 64, z: Union[complex, None] = 1e-3j,
                 hermi: Union[bool, int] = True, diago: Union[bool, int, tuple] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, init_bound: float = 1.,
                 scale: bool = False, drop: float = 0., sigma: float = 0., double: bool = False, trs=True):
        super(SimpleCB, self).__init__(input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                                       real, init_bound, scale, drop, sigma, double)
        self.bnp = ComplexBatchNorm2d(1, track_running_stats=trs)
        self.bne = ComplexBatchNorm2d(1, track_running_stats=trs)
        self.bn1 = ComplexBatchNorm2d(1, track_running_stats=trs)
        self.bn2 = ComplexBatchNorm2d(1, track_running_stats=trs)


'''
Basic Layers: GLinear, GinvLinear, Linear2D
'''


class GLinear(nn.Module):
    def __init__(self, input_size, output_size: Union[int, None]=None, hermi: Union[bool, int] = True,
                 diago: Union[bool, int] = False, restr: Union[bool, int, tuple] = False, real: bool = False,
                 drop: float = 0., sigma: float = 0., init_bound: float = 1., double: bool = False):
        '''
        :param hermi: True, False, 0: naive hermi
        :param diago: True, False, 1: 1D NN, 2: 2D NN, 3: 1D NNN, 4: 2D NNN
        :param restr: False/0: fc, 1: 1D NN, 2: 2D NN, 3: 1D NNN, 4: 2D NNN
        '''
        super(GLinear, self).__init__()
        self.register_buffer('probs', torch.clamp(torch.tensor(1 - drop, dtype=torch.float32), min=0, max=1))
        self.sigma = abs(sigma)
        if output_size is None:
            assert type(restr) is tuple
            self.restr = restr[0]
            self.kernel_size = restr[1]
            self.stride = restr[2] if type(restr[2]) is int else self.kernel_size
            assert self.stride <= self.kernel_size
            if self.restr <= 2:
                a = self.square_check(input_size)
                if a < self.kernel_size:
                    self.restr = False
                    output_size = a
                    b = None
                else:
                    b = self.calculate_output_size(a)
                    if self.restr == 1:
                        output_size = b
                    else:
                        output_size = b ** 2
            else:
                b = self.square_check(input_size)
                a = self.calculate_output_size_inv(b)
                if self.restr == 3:
                    output_size = a
                else:
                    output_size = a ** 2
        else:
            assert type(restr) is int or type(restr) is bool
            if input_size >= output_size:
                a, b = input_size, output_size
            else:
                a, b = output_size, input_size
            self.restr = restr
        self.fan_in = input_size
        self.size = output_size
        self.hermi = hermi
        self.diago = diago
        self.sanity_check()
        if double:
            self.dtype = torch.float64 if real else torch.complex128
            self.cdtype = torch.complex128
        else:
            self.dtype = torch.float32 if real else torch.complex64
            self.cdtype = torch.complex64
        self.size_check(b, checked=True if type(restr) is tuple else False)

        # Initilizatize t
        if not self.restr:
            self.t = nn.Parameter(torch.zeros((input_size, output_size), dtype=self.dtype))
        else:
            # compute index
            if type(restr) is int:
                a_index = torch.arange(a)
                if self.restr % 2 == 0:
                    a_sqr_float, b_sqr_float = math.sqrt(a), math.sqrt(b)
                    a_sqr_int, b_sqr_int = int(a_sqr_float), int(b_sqr_float)
                    if abs(a_sqr_float - a_sqr_int) > 1e-6 or abs(b_sqr_float - b_sqr_int) > 1e-6:
                        self.restr -= 1
                    else:
                        a, b = a_sqr_int, b_sqr_int
                if self.restr == 3:
                    a_index = a_index.tile(3)
                elif self.restr == 4:
                    a_index = a_index.tile(5)
                b_index = self.calculate_b_index(a, b)
                if self.restr == 3:
                    b_index = torch.cat((b_index, self.pbc(b_index - 1, b), self.pbc(b_index + 1, b)))
                elif self.restr % 2 == 0:
                    x, y = b_index.tile(a), b_index.repeat_interleave(a)
                    b_index = self.l2c(x, y, b)
                    if self.restr == 4:
                        index1 = self.l2c(self.pbc(x - 1, b), y, b)
                        index2 = self.l2c(x, self.pbc(y - 1, b), b)
                        index3 = self.l2c(self.pbc(x + 1, b), y, b)
                        index4 = self.l2c(x, self.pbc(y + 1, b), b)
                        b_index = torch.cat((b_index, index1, index2, index3, index4))
            else:
                b_index = torch.arange(b).repeat_interleave(self.kernel_size)
                a_index = [torch.arange(self.kernel_size)]
                for _ in range(1, b):
                    a_index.append(a_index[-1] + self.stride)
                a_index = torch.cat(a_index, dim=0)
                tail = a - (b - 1) * self.stride - self.kernel_size
                if tail < 0:
                    a_index, b_index = a_index[:tail], b_index[:tail]
                length = len(a_index)
                if self.restr % 2 == 0:
                    a_index = self.l2c(a_index.tile(length), a_index.repeat_interleave(length), a)
                    b_index = self.l2c(b_index.tile(length), b_index.repeat_interleave(length), b)
            if input_size >= output_size:
                self.t_index = (a_index, b_index)
            else:
                self.t_index = (b_index, a_index)
            self.t = nn.Parameter(torch.zeros(len(a_index), dtype=self.dtype))
        gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        std = gain / math.sqrt(input_size)
        bound = math.sqrt(3.0) * std * init_bound
        nn.init.uniform_(self.t, -bound, bound)

        # Initilizatize h
        bound = init_bound / math.sqrt(input_size)
        if self.diago:
            if self.hermi:
                self.h = nn.Parameter(torch.zeros(self.size, dtype=torch.float64 if double else torch.float32))
            else:
                self.h = nn.Parameter(torch.zeros(self.size, dtype=self.dtype))
            if type(self.diago) is int:
                b_index = torch.arange(self.size)
                if self.diago % 2 == 0:
                    a_float = math.sqrt(self.size)
                    a = int(a_float)
                    if abs(a_float - a) > 1e-6:
                        self.diago -= 1
                if self.diago == 1:
                    a_index = self.pbc(b_index + 1, self.size)
                    if not self.hermi:
                        a_index = torch.cat((self.pbc(b_index - 1, self.size), a_index))
                elif self.diago == 3:
                    a_index = torch.cat((self.pbc(b_index + 1, self.size), self.pbc(b_index + 2, self.size)))
                    b_index = b_index.tile(2)
                    if not self.hermi:
                        temp = torch.cat((self.pbc(b_index - 2, self.size), self.pbc(b_index - 1, self.size)))
                        a_index = torch.cat((temp, a_index))
                        b_index = b_index.tile(2)
                else:
                    x, y = torch.arange(a).tile(a), torch.arange(a).repeat_interleave(a)
                    ix1, iy1 = self.l2c(self.pbc(x + 1, a), y, a), self.l2c(x, self.pbc(y + 1, a), a)
                    if not self.hermi:
                        ix_1, iy_1 = self.l2c(self.pbc(x - 1, a), y, a), self.l2c(x, self.pbc(y - 1, a), a)
                    if self.diago == 2:
                        a_index = torch.cat((ix1, iy1))
                        b_index = b_index.tile(2)
                        if not self.hermi:
                            a_index = torch.cat((ix_1, iy_1, a_index))
                            b_index = b_index.tile(2)
                    elif self.diago == 4:
                        ix1y1 = self.l2c(self.pbc(x + 1, a), self.pbc(y + 1, a), a)
                        ix1y_1 = self.l2c(self.pbc(x + 1, a), self.pbc(y - 1, a), a)
                        a_index = torch.cat((ix1y_1, ix1, ix1y1, iy1))
                        b_index = b_index.tile(4)
                        if not self.hermi:
                            ix_1y_1 = self.l2c(self.pbc(x - 1, a), self.pbc(y - 1, a), a)
                            ix_1y1 = self.l2c(self.pbc(x - 1, a), self.pbc(y + 1, a), a)
                            a_index = torch.cat((ix_1y1, ix_1, ix_1y_1, iy_1, a_index))
                            b_index = b_index.tile(2)
                    else:
                        raise NameError('Wrong diago!')
                self.h_index = (a_index, b_index)
                self.hod = nn.Parameter(torch.zeros(len(b_index), dtype=self.dtype))
                nn.init.uniform_(self.hod, -bound, bound)
        elif self.hermi:
            self.h_index = tuple(torch.triu_indices(output_size, output_size, offset=1))
            self.h = nn.Parameter(torch.zeros(output_size, dtype=torch.float64 if double else torch.float32))
            self.hod = nn.Parameter(torch.zeros(int(output_size * (output_size - 1) / 2), dtype=self.dtype))
            nn.init.uniform_(self.hod, -bound, bound)
        else:
            self.h = nn.Parameter(torch.zeros((output_size, output_size), dtype=self.dtype))
        nn.init.uniform_(self.h, -bound, bound)
        # nn.init.zeros_(self.h)

    def sanity_check(self):
        if type(self.hermi) is not bool:
            assert self.hermi == 0
        if type(self.diago) is not bool:
            assert self.diago in (1, 2, 3, 4)
        if type(self.restr) is bool:
            assert self.restr == False
        else:
            assert self.restr in (1, 2, 3, 4)

    def size_check(self, b, checked=False):
        if type(self.restr) is int and not checked:
            if b <= 3:
                if self.restr > 2:
                    self.restr = False
            elif b <= 4:
                if self.restr == 4:
                    self.restr = False
        if type(self.diago) is int:
            if self.size <= 3:
                self.diago = False
            elif self.size <= 5:
                if self.diago > 1:
                    self.diago = False
            elif self.size <= 9:
                if self.diago == 4:
                    self.diago = False

    def square_check(self, input_size):
        if self.restr % 2 == 0:
            a_sqr_float = math.sqrt(input_size)
            a_sqr_int = int(a_sqr_float)
            if abs(a_sqr_float - a_sqr_int) > 1e-6:
                self.restr -= 1
                return input_size
            else:
                return a_sqr_int
        else:
            return input_size

    def calculate_output_size(self, input_size):
        return math.ceil((input_size - self.kernel_size) / self.stride + 1)

    def calculate_output_size_inv(self, input_size):
        return (input_size - 1) * self.stride + self.kernel_size

    def divide_b_into_a(self, a, b):
        q = a // b
        if a % b == 0:
            return b, q, 0, 0
        else:
            x = (a - b) // q
            return x, q + 1, a - (q + 1) * x, b - x

    def calculate_b_index(self, a, b):
        b_index, start = [], 0
        while b > 0:
            x, num, a, b = self.divide_b_into_a(a, b)
            end = start + x
            b_index.append(torch.arange(start, end).repeat_interleave(num))
            start = end
        return torch.cat(b_index, dim=0)

    def pbc(self, x, L):
        return x % L

    def l2c(self, x, y, L):
        return x + y * L

    def t_weight(self):
        if self.restr:
            return torch.zeros((self.fan_in, self.size), dtype=self.cdtype, device=self.t.device).index_put(
                self.t_index, self.t.type(self.cdtype))
        else:
            return self.t.type(self.cdtype)

    def z_bias(self, z):
        if z.dim() == 0:
            return z * torch.eye(self.size, device=z.device)
        elif z.dim() == 1:   # (bz,)
            return torch.diag_embed(z[:, None, None].tile(1, 1, self.size))
        elif z.dim() == 2:   # (count, 1)
            return torch.diag_embed(z.tile(1, self.size))
        elif z.dim() == 3:   # (bz, count, 1)
            return torch.diag_embed(z.tile(1, 1, self.size))
        else:
            raise RuntimeError('Wrong dim of z')

    def h_bias(self):
        if self.diago:
            b = torch.diag_embed(self.h.type(self.cdtype))
            if type(self.diago) is int:
                b = b.index_put(self.h_index, self.hod.type(self.cdtype))
                if self.hermi:
                    b = b + b.mH
        elif self.hermi:
            b = torch.diag_embed(self.h.type(self.cdtype)).index_put(self.h_index, self.hod.type(self.cdtype))
            b = b + b.mH
        else:
            b = self.h.type(self.cdtype)
        if self.hermi == 0:
            return b + b.mH
        else:
            return b

    def train_wrap(self, tw, hb):
        if self.training:
            if self.sigma > 1e-10:
                hb = hb + torch.diag_embed(torch.normal(mean=0., std=self.sigma, size=(self.size,), device=hb.device))
            if self.probs < 1 - 1e-10:
                mask = torch.bernoulli(self.probs.expand((1, self.size))) / torch.sqrt(self.probs)
                tw = tw * mask.tile((self.fan_in, 1))
                # hb = hb * torch.matmul(mask.T, mask) / torch.sqrt(self.probs)
        return tw, hb

    def forward(self, g, z):
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        return self.z_bias(z) - hb - tw.mH @ g @ tw

    def extra_repr(self) -> str:
        try:
            return 'input_size={}, output_size={}, restr=({},{},{}), diago={}, hermi={}, dropout={:.3e}, disorder=(0, {:.3e}), dtype={}'.format(
                self.fan_in, self.size, self.restr, self.kernel_size, self.stride, self.diago, self.hermi, self.dropout,
                self.disorder[1], self.dtype)
        except AttributeError:
            return 'input_size={}, output_size={}, restr={}, diago={}, hermi={}, dropout={:.3e}, disorder=(0, {:.3e}), dtype={}'.format(
                self.fan_in, self.size, self.restr, self.diago, self.hermi, self.dropout, self.disorder[1], self.dtype)

    @property
    def dropout(self):
        return 1 - self.probs.item()

    @property
    def disorder(self):
        return (0., self.sigma)


class GinvLinear(GLinear):
    def __init__(self, input_size, output_size, hermi: Union[bool, int] = True, diago=False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, drop : float = 0., sigma: float = 0.,
                 init_bound: float = 1., double: bool = False):
        super(GinvLinear, self).__init__(input_size, output_size, hermi, diago, restr, real, drop, sigma, init_bound,
                                         double)

    def forward(self, g, z):
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        return self.z_bias(z) - hb - tw.mH @ torch.linalg.solve(g, tw)


class GcorLinear(GLinear):
    def __init__(self, input_size, output_size, hermi: Union[bool, int] = True, diago: Union[bool, int] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, drop : float = 0., sigma: float = 0.,
                 init_bound: float = 1., double: bool = False):
        super(GcorLinear, self).__init__(input_size, output_size, hermi, diago, restr, real, drop, sigma, init_bound,
                                         double)

    def forward(self, g, z):
        gd, god = g
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        gd = torch.linalg.inv(self.z_bias(z) - hb - tw.mH @ gd @ tw)
        god = god @ tw @ gd
        return [gd, god]


class Linear2D(GLinear):
    def __init__(self, input_size, output_size, hermi: Union[bool, int] = True, diago: Union[bool, int] = False,
                 restr: Union[bool, int, tuple] = False, real: bool = False, drop : float = 0., sigma: float = 0.,
                 init_bound: float = 1., double: bool = False):
        super(Linear2D, self).__init__(input_size, output_size, hermi, diago, restr, real, drop, sigma, init_bound,
                                       double)

    def forward(self, g, z=None):
        tw, hb = self.train_wrap(self.t_weight(), self.h_bias())
        return hb + tw.mH @ g @ tw


def complex_relu(input):
    return F.relu(input.real).type(torch.complex64) + 1j * F.relu(input.imag).type(torch.complex64)


class NaiveComplexBatchNorm2d(nn.Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.bn_r(input.real).type(torch.complex64) + 1j * self.bn_i(input.imag).type(torch.complex64)


class _ComplexBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (
                        1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1. / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + (
                        1 - exponential_average_factor) * self.running_covar[:, 0]
                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + (
                        1 - exponential_average_factor) * self.running_covar[:, 1]
                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + (
                        1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None] * input.real + Rri[None, :, None, None] * input.imag).type(
            torch.complex64) + 1j * (
                        Rii[None, :, None, None] * input.imag + Rri[None, :, None, None] * input.real).type(
            torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None,
                                                                        None] * input.imag + self.bias[None, :, 0, None,
                                                                                             None]).type(
                torch.complex64) + 1j * (
                            self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None,
                                                                               None] * input.imag + self.bias[None,
                                                                                                    :, 1, None,
                                                                                                    None]).type(
                torch.complex64)

        return input


def complex_dropout(input, p=0.5, training=True):
    mask = torch.ones(*input.shape, dtype=torch.float32, device=input.device)
    mask = F.dropout(mask, p, training) * 1 / (1 - p)
    mask.type(input.dtype)
    return mask * input


def complex_dropout2d(input, p=0.5, training=True):
    mask = torch.ones(*input.shape, dtype=torch.float32, device=input.device)
    mask = F.dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(input.dtype)
    return mask * input


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout2d(input, self.p)
        else:
            return input

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)
