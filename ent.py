import torch
from typing import Optional
from rgfnn import GLinear, GinvLinear, GcorLinear


def orthogonalize(U, eps=1e-8): #GS-orthogonalization
    n = U.shape[1]
    V = torch.zeros((n, n), dtype=U.dtype)
    V[0] = U[0] / torch.linalg.norm(U[0]) #first vector normalization
    count = 1
    for u in U[1:]:
        prev_basis = V[:count]
        projector = torch.matmul(prev_basis.T, prev_basis.conj())
        u -= torch.matmul(projector, u)
        norm = torch.linalg.norm(u)
        if norm > eps:
            V[count] = u / norm
            count += 1
        if count == n: break
    return V


def updateH(H, t, h):
    padding = torch.zeros((H.shape[0] - t.shape[0], t.shape[1]), dtype=torch.complex64, device=t.device)
    t = torch.cat((padding, t), dim=0)
    H = torch.cat((H, t), dim=1)
    temp = torch.cat((t.mH, h), dim=1)
    return torch.cat((H, temp), dim=0)


def constrModH(input_size, model, device='cpu', double=False):
    with torch.no_grad():
        H = torch.zeros((input_size, input_size), dtype=torch.complex128 if double else torch.complex64, device=device)
        modules = set()
        t_out = None
        for k in model.state_dict().keys():
            try:
                m = k[:k.index('.')]
            except:
                continue
            if m not in modules:
                modules.add(m)
                module = model.get_submodule(m)
                if isinstance(module, GLinear) or isinstance(module, GinvLinear) or isinstance(module, GcorLinear):
                    # print(m, module)
                    t, h = module.t_weight().to(device), module.h_bias().to(device)
                    if m == 'out':
                        t_out, h_out = t, h
                    else:
                        H = updateH(H, t, h)
        if t_out is None:
            return H
        else:
            return updateH(H, t_out, h_out)


def updateG(G, g, t):
    size = t.shape[0]
    Gh, Gv = G[..., -size:, :], G[..., :, -size:]
    G = G + Gv @ t @ g @ t.mH @ Gh
    G = torch.cat((G, Gv @ t @ g), dim=-1)
    temp = torch.cat((g @ t.mH @ Gh, g), dim=-1)
    return torch.cat((G, temp), dim=-2)


def RecurGF(G0, model, device='cpu'):
    with torch.no_grad():
        G, g, modules = G0.to(device), G0.to(device), set()
        t_out = None
        for k in model.state_dict().keys():
            try:
                m = k[:k.index('.')]
            except:
                continue
            if m not in modules:
                modules.add(m)
                module = model.get_submodule(m)
                if isinstance(module, GLinear) or isinstance(module, GinvLinear) or isinstance(module, GcorLinear):
                    # print(m, module)
                    t, h, z = module.t_weight().to(device), module.h_bias().to(device), module.z_bias(model.z).to(device)
                    if m == 'out':
                        t_out, h_out, z_out = t, h, z
                    else:
                        g = (z - h - t.mH @ g @ t).inverse()
                        G = updateG(G, g, t)
        if t_out is None:
            return G
        else:
            g = (z_out - h_out - t_out.mH @ g @ t_out).inverse()
            return updateG(G, g, t_out)


def H2G(H, z):
    if type(z) is not torch.Tensor: z = torch.tensor(z)
    if z.dim() == 0:
        z = z * torch.eye(H.shape[-1], device=z.device)
    else:
        z = torch.diag_embed(z.expand(H.shape[-1], -1).transpose(0, 1))
        if H.dim() > 3: z = z.unsqueeze(1)
    return torch.linalg.inv(z - H)


def G2H(G, z):
    if type(z) is not torch.Tensor: z = torch.tensor(z)
    if z.dim() == 0:
        z = z * torch.eye(G.shape[-1], device=z.device)
    else:
        z = torch.diag_embed(z.expand(G.shape[-1], -1).transpose(0, 1))
        if G.dim() > 3: z = z.unsqueeze(1)
    return z - torch.linalg.inv(G)


def getIndex(input_size, embedding_size, hidden_size, output_size, tc: Optional[int] = None):
    size = input_size + embedding_size + sum(hidden_size) + output_size

    # A, B, C, D
    if tc is None or output_size == 1:
        A, B = torch.arange(size - output_size, size), None
    else:
        A = torch.arange(size - output_size + tc, size - output_size + tc + 1)
        B = torch.cat((torch.arange(size - output_size, A[0].item()), torch.arange(A[-1].item() + 1, size)))
    C = torch.arange(0, input_size)
    # D = torch.arange(input_size, size - output_size)
    D = torch.arange(input_size, input_size + embedding_size)
    AC, AD, CD = torch.cat((C, A)), torch.cat((D, A)), torch.cat((C, D))
    ACD = torch.cat((CD, A))
    if B is not None:
        BC, BD = torch.cat((C, B)), torch.cat((D, B))
        BCD = torch.cat((CD, B))
    else:
        BC, BD, BCD = None, None, None

    return A, B, C, D, AC, AD, BC, BD, CD, ACD, BCD


@torch.no_grad()
def entropy(c, order=1):
    expeigs = 1 / torch.linalg.eigvalsh(c) - 1
    # print(torch.linalg.eigvalsh(c))
    if order == 1:        # Von Neumann (Renyi-1)
        return torch.nansum(torch.log(1 + 1 / expeigs) + torch.log(expeigs) / (1 + expeigs), dim=-1)
    elif order > 1:       # Renyi-alpha (alpha > 1)
        return torch.nansum(torch.log(1 + expeigs ** (-order)) - order * torch.log(1 + 1 / expeigs), dim=-1) / (1 - order)
    else:
        raise NameError('Wrong order')


def mutInfo(c1, c2, c12, order=1):
    return entropy(c1, order) + entropy(c2, order) - entropy(c12, order)


def triInfo(c1, c2, c3, c12, c13, c23, c123, order=1):
    S1, S2, S3 = entropy(c1, order), entropy(c2, order), entropy(c3, order)
    S12, S13, S23 = entropy(c12, order), entropy(c13, order), entropy(c23, order)
    S123 = entropy(c123, order)
    I12 = S1 + S2 - S12
    I13 = S1 + S3 - S13
    I1_23 = S1 + S23 - S123
    return torch.stack((I12 + I13 - I1_23, I12, I13, I1_23), dim=0)


def correlation(H):
    e, v = torch.linalg.eigh(H.type(torch.complex128))
    return v.conj() @ torch.diag_embed(
        1. - torch.heaviside(e, values=torch.zeros([1] * e.dim(), device=e.device, dtype=e.dtype))).type(
        v.dtype) @ v.transpose(-2, -1)


def Cor2I(Cor, A, B, C, D, AC, AD, BC, BD, CD, ACD, BCD, order):
    Ca, Cc, Cd = Cor[:, A][:, :, A], Cor[:, C][:, :, C], Cor[:, D][:, :, D]
    Cac, Cad, Ccd = Cor[:, AC][:, :, AC], Cor[:, AD][:, :, AD], Cor[:, CD][:, :, CD]
    Cacd = Cor[:, ACD][:, :, ACD]
    Ia = triInfo(Ca, Cc, Cd, Cac, Cad, Ccd, Cacd, order)
    if B is not None:
        Cb = Cor[:, B][:, :, B]
        Cbc, Cbd = Cor[:, BC][:, :, BC], Cor[:, BD][:, :, BD]
        Cbcd = Cor[:, BCD][:, :, BCD]
        Ib = triInfo(Cb, Cc, Cd, Cbc, Cbd, Ccd, Cbcd, order)
    else:
        Ib = None
    return Ia, Ib


def Cor2Ic(Cor, A, B, C, AC, BC, order):
    Ca, Cc, Cac = Cor[:, A][:, :, A], Cor[:, C][:, :, C], Cor[:, AC][:, :, AC]
    Iac = mutInfo(Ca, Cc, Cac, order).unsqueeze(0)
    if B is not None:
        Cb, Cbc = Cor[:, B][:, :, B], Cor[:, BC][:, :, BC]
        Ibc = mutInfo(Cb, Cc, Cbc, order).unsqueeze(0)
    else:
        Ibc = None
    return Iac, Ibc


def computeI(model, input_size, H0, index, order: int = 1, delta: float = 0., device='cpu', double=False, full=True):
    H = constrModH(input_size, model, device, double)
    # Add H0
    H = H.repeat(H0.shape[0], 1, 1)
    H[:, :input_size, :input_size] = H0.squeeze(1)
    delta = abs(delta)
    A, B, C, D, AC, AD, BC, BD, CD, ACD, BCD = index
    if model.z.real.dim() == 0:
        H = H - (model.z.real.to(device) + delta) * torch.eye(H.shape[-1], device=device)
    else:
        H = H - torch.diag_embed((model.z.real.to(device) + delta).expand(H.shape[-1], -1).T)
    Cor = correlation(H)
    if full:
        Ia, Ib = Cor2I(Cor, A, B, C, D, AC, AD, BC, BD, CD, ACD, BCD, order)
    else:
        Ia, Ib = Cor2Ic(Cor, A, B, C, AC, BC, order)
    if delta < 1e-8:
        return Ia, Ib
    else:
        H = H + 2 * delta * torch.eye(H.shape[-1], device=device)
        Cor = correlation(H)
        if full:
            Ia_, Ib_ = Cor2I(Cor, A, B, C, D, AC, AD, BC, BD, CD, ACD, BCD, order)
        else:
            Ia_, Ib_ = Cor2Ic(Cor, A, B, C, AC, BC, order)
        if Ib is None:
            return (Ia - Ia_) / 2 / delta, None
        else:
            return (Ia - Ia_) / 2 / delta, (Ib - Ib_) / 2 / delta


if __name__ == "__main__":
    import os
    import numpy as np
    from rgfnn import Network
    from utils import readinfo, emb_hid_size_wrapper

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    order, delta, tc = 1, 1e-2, 1
    data = 'Ins_12_d0.2'
    Net = 'Simple_'
    file = 'ENTANGLEMENT'
    index = np.random.randint(4000, size=5).tolist()

    model_path = 'models/{}/{}/{}'.format(data, file, Net)
    input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, scale, real, double \
        = readinfo('{}/info.txt'.format(model_path))

    if data.endswith('MNIST'):
        images, labels = torch.load('datasets/{}/processed/test.pt'.format(data))
        images, labels = images[index], labels[index]
        if tc is not None:
            index = torch.nonzero(labels == tc, as_tuple=True)
            if len(index[0]) == 0: exit(0)
            images, labels = torch.flatten(images[index].unsqueeze(1), 2), labels[index].tolist()
        else:
            images, labels = torch.flatten(images.unsqueeze(1), 2), labels.tolist()
        adj = torch.load('datasets/AdjMat.pt')
        H0 = torch.diag_embed(-images * torch.pi * 1j) + adj * 1e-3
        H0 = z * torch.eye(H0.shape[-1]) - torch.linalg.inv(H0)
        # H0 = torch.diag_embed(z - 1 / (images * torch.pi * 1j + 1e-3))
    else:
        data_path = 'datasets/{}/test'.format(data)
        H0 = torch.load('{}/dataset.pt'.format(data_path))[index]
        labels = torch.load('{}/labels.pt'.format(data_path))[index]
        if tc is not None:
            index = torch.nonzero(labels == tc, as_tuple=True)
            if len(index[0]) == 0: exit(0)
            H0, labels = H0[index], labels[index].tolist()
        else:
            labels = labels.tolist()
        if type(z) is float:
            E = torch.load('{}/Es.pt'.format(data_path))[index]
            z = E + z * 1j
    if double: H0 = H0.type(torch.complex128)
    print('labels={}'.format(labels))

    model = Network(Net[:Net.index('_')], input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                    scale=scale, double=double)

    embedding_size, hidden_size = emb_hid_size_wrapper(model)

    size = getIndex(input_size, embedding_size, hidden_size, output_size, tc=tc)
    # print(size)

    file_list = sorted(os.listdir(model_path))
    for file in file_list:
        if file.startswith('checkpoint_'):
            print('\n{}'.format(file))
            state_dict = torch.load(os.path.join(model_path, file), map_location="cpu")['state_dict']
            # if 'z' in state_dict.keys():
            #     del state_dict['z']
            model.load_state_dict(state_dict, strict=False)
            Ia, Ib = computeI(model, input_size, H0, size, order=order, delta=delta, double=double)
            print('Iacd={}'.format(Ia[0].tolist()))
            print('Iac={}'.format(Ia[1].tolist()))
            print('Iad={}'.format(Ia[2].tolist()))
            print('Ia_cd={}'.format(Ia[3].tolist()))
            if Ib is not None:
                print('Ibcd={}'.format(Ib[0].tolist()))
                print('Ibc={}'.format(Ib[1].tolist()))
                print('Ibd={}'.format(Ib[2].tolist()))
                print('Ib_cd={}'.format(Ib[3].tolist()))
