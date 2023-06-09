import os
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import optimize
from functools import partial
from typing import Union
from utils import readinfo, Optimizer, dircheck
from ent import constrModH, correlation, H2G
from rgfnn import Network
from Data_Ins import Ham, pbc, l2c, c2l, nnn_mask
from Data_Ins_d import countriangle, Stri, ChernNumber, Kubo
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)

np.random.seed(0)
torch.manual_seed(2200)


L = 12
data = 'Ins_12_d0.2'
Net = 'CMedian_'
file = 'ENTANGLEMENT'
Hdiago, Hreal, index = 4, False, np.random.randint(5000, size=5).tolist()
print('Hdiago={}\tHreal={}\tindex={}'.format(Hdiago, Hreal, index))
delta = np.arange(0, 1.005, 0.005)
trunc, kappa = 3, 0.52
no, rp = 3, True
amount, n = None, None
print('trunc={}\tkappa={}'.format(trunc, kappa))
print('no={}\trandom phi={}'.format(no, rp))
print('amount={}\tn={}\n'.format(amount, n))
load, save, show = False, True, True


class H_delta_torch(nn.Module):
    def __init__(self, input_size, diago: Union[bool, int] = False, real: bool = False, init_bound: float = 1.):
        super(H_delta_torch, self).__init__()
        self.diago = diago
        dtype = torch.float64 if real else torch.complex128
        bound = init_bound / math.sqrt(input_size)
        if self.diago:
            self.h = nn.Parameter(torch.zeros(input_size, dtype=torch.float64))
            if type(self.diago) is int:
                b_index = torch.arange(input_size)
                if self.diago % 2 == 0:
                    a_float = math.sqrt(input_size)
                    a = int(a_float)
                    if abs(a_float - a) > 1e-6:
                        self.diago -= 1
                if self.diago == 1:
                    a_index = self.pbc(b_index + 1, input_size)
                elif self.diago == 3:
                    a_index = torch.cat((self.pbc(b_index + 1, input_size), self.pbc(b_index + 2, input_size)))
                    b_index = b_index.tile(2)
                else:
                    x, y = torch.arange(a).tile(a), torch.arange(a).repeat_interleave(a)
                    ix1, iy1 = self.l2c(self.pbc(x + 1, a), y, a), self.l2c(x, self.pbc(y + 1, a), a)
                    if self.diago == 2:
                        a_index = torch.cat((ix1, iy1))
                        b_index = b_index.tile(2)
                    elif self.diago == 4:
                        ix1y1 = self.l2c(self.pbc(x + 1, a), self.pbc(y + 1, a), a)
                        ix1y_1 = self.l2c(self.pbc(x + 1, a), self.pbc(y - 1, a), a)
                        a_index = torch.cat((ix1y_1, ix1, ix1y1, iy1))
                        b_index = b_index.tile(4)
                    else:
                        raise NameError('Wrong diago!')
                self.h_index = (a_index, b_index)
                self.hod = nn.Parameter(torch.zeros(len(b_index), dtype=dtype))
        else:
            self.h_index = tuple(torch.triu_indices(input_size, input_size, offset=1))
            self.h = nn.Parameter(torch.zeros(input_size, dtype=torch.float64))
            self.hod = nn.Parameter(torch.zeros(int(input_size * (input_size - 1) / 2), dtype=dtype))
        nn.init.uniform_(self.hod, -bound, bound)
        nn.init.uniform_(self.h, -bound, bound)

    def pbc(self, x, L):
        return x % L

    def l2c(self, x, y, L):
        return x + y * L

    def Hamil(self):
        if self.diago:
            b = torch.diag_embed(self.h.type(torch.complex128))
            if type(self.diago) is int:
                b = b.index_put(self.h_index, self.hod.type(torch.complex128))
                b = b + b.mH
        else:
            b = torch.diag_embed(self.h.type(torch.complex128)).index_put(self.h_index, self.hod.type(torch.complex128))
            b = b + b.mH
        return b

    def Hphi(self, phi):
        return torch.matmul(self.Hamil(), phi)

    def forward(self, phi):
        L = self.Hphi(phi)
        return torch.sum(L * L.conj()).real
        # return torch.linalg.norm(L)


def H_delta(v, size, index, phi):
    H = np.zeros((size, size), dtype=np.complex128)
    H[index] = v
    H = H + H.T.conj()
    L = np.matmul(H, phi)
    return np.sum(L * L.conj())


def H_t(Hc, HQ):
    return Hc @ HQ.inverse() @ Hc.mH


def H_delta_theory(Hi, Hc, HQ):
    return Hi - H_t(Hc, HQ)


def H_input(Hd, Ht, delta=1.):
    return (Hd * delta + Ht)[None, None, ...]


def wrap_model(model, Hi):
    Gi = H2G(Hi, model.z.to(Hi.device))
    return F.tanh(F.relu(model(Gi) / 2)).data.numpy()


class HQ_det_torch(nn.Module):
    def __init__(self, HQ, init_bound: float = 1.):
        super(HQ_det_torch, self).__init__()
        self.HQ = HQ
        self.h = nn.Parameter(torch.zeros(self.HQ.shape[1], dtype=torch.complex128))
        bound = init_bound / math.sqrt(self.HQ.shape[1])
        nn.init.uniform_(self.h, -bound, bound)

    def forward(self, eta):
        L = torch.matmul(self.HQ, self.h)
        return torch.sum(L * L.conj()).real + eta * (torch.sum(self.h * self.h.conj()).real - 1) ** 2
        # return torch.linalg.norm(L) + eta * (torch.linalg.norm(self.h) - 1) ** 2


class MyPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eigh(np.dot(X.T.conj(), X))
        Y = np.dot(X, M[:, -self.n_components:])
        return Y.real


def PCADemo(datas, labels):
    clf = MyPCA(n_components=2)  # sklearn.decomposition.PCA
    outputs = clf.fit_transform(datas)
    plotfig_pca(outputs, labels)


def TSNEDemo(datas, labels):
    # parameters
    initial_dims = 50  # 用pca进行数据预处理降到的维度，可选择不进行预处理(None or int)
    perplexity, early_exaggeration = 30, 4
    init = 'pca'  # 降维后的坐标的初始值，可用pca降维结果或随机生成(pca or random)
    n_iter, lr, weight_decay, momentum = 1000, 'auto', 0, 0.5  # lr可选择'auto'

    pca = PCA(n_components=initial_dims)  # sklearn.decomposition.PCA
    clf = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter,
               learning_rate=lr, init=init, method='exact')  # sklearn.manifold.TSNE
    if initial_dims is None:
        outputs = clf.fit_transform(datas)
    else:
        outputs = clf.fit_transform(pca.fit_transform(datas))

    plotfig_tsne(outputs, labels)


def plotfig_pca(outputs, labels):
    plt.figure()
    plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
    plt.title("PCA_{}".format(Net))
    plt.show()
    plt.close()


def plotfig_tsne(outputs, labels):
    plt.figure()
    plt.scatter(outputs[:, 0], outputs[:, 1], c=labels)
    plt.title("TSNE_{}".format(Net))
    plt.show()
    plt.close()


def plotcor(c, label, y_label):
    plt.figure()
    plt.xlim(-1, len(c) + 1)
    plt.plot(np.arange(len(c)), c, 'o-', linewidth=1, markersize=2, color='r' if y_label == 'c' else 'b')
    plt.xlabel('Site')
    plt.ylabel(y_label)
    plt.title('{} / {}'.format(y_label, label))
    plt.show()
    plt.close()


def plotdelta(delta, v, y_label):
    plt.figure()
    plt.plot(delta, v, 'o-', linewidth=2, markersize=0.01)
    plt.xlabel('delta')
    plt.ylabel(y_label)
    plt.title('{} / kappa={:.3f}'.format(y_label, kappa))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}_{}{}_{:.3f}.jpg'.format(path, y_label, 'rand' if rp else '', no, kappa))
    if show: plt.show()
    plt.close()


def spectrum(H, numpy=True):
    values, states = torch.linalg.eigh(H.type(torch.complex128))
    data = []
    values, indices = torch.min(torch.abs(values), dim=1)
    for state, index in zip(states, indices):
        data.append(state[:, index])
    data = torch.stack(data, dim=0)  # (bz, number of sites)
    return data.numpy() if numpy else data, values.numpy() if numpy else values


def corfunc(c, C_r, C_i, eta):
    return -(np.sum(c * C_r) ** 2 + np.sum(c * C_i) ** 2) + eta * (np.sum(c ** 2) - 1) ** 2


'''main functions'''
def Kubo_test(L):
    kappa = np.arange(0, 0.46, 0.01)
    cn = []
    for k in kappa:
        cn.append(Kubo(Ham(k, L)))
    plt.figure()
    plt.plot(kappa, cn)
    print(cn)
    kappa = np.arange(0.55, 1.01, 0.01)
    cn = []
    for k in kappa:
        cn.append(Kubo(Ham(k, L)))
    plt.plot(kappa, cn)
    plt.xlabel('kappa')
    plt.ylabel('Chern Number')
    plt.show()
    plt.close()
    print(cn)


def SpecAna(model, model_path, data_path):
    labels = torch.load('{}/labels.pt'.format(data_path))
    index = torch.nonzero(labels, as_tuple=True)
    H0 = torch.load('{}/dataset.pt'.format(data_path))[index]
    if double: H0 = H0.type(torch.complex128)
    # model_path_ = '{}-'.format(model_path)
    # state_dict_ = torch.load('{}/model_best.pth.tar'.format(model_path_), map_location="cpu")['state_dict']
    # index_ = torch.nonzero(1 - labels, as_tuple=True)
    # H0_ = torch.load('{}/dataset.pt'.format(data_path))[index_]
    # model_ = Network(Net[:Net.index('_')], input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
    #                  real, scale=scale, double=double)
    # model_.load_state_dict(state_dict_, strict=False)
    # model_.eval()

    H = constrModH(input_size, model, double=double)
    H = H.repeat(H0.shape[0], 1, 1)
    H[:, :input_size, :input_size] = H0.squeeze(1)

    states = spectrum(H0)[0]
    datas = states
    labels = np.ones(len(states))
    # states_ = spectrum(model_, H0_, input_size)[0]
    # datas = np.concatenate((states, states_), axis=0)
    # labels = np.concatenate((np.ones(len(states)), np.zeros(len(states_))), axis=0)

    PCADemo(datas, labels)
    # TSNEDemo(datas, labels)


def CorChannel(model, input_size, data_path, eta=1000):
    H0 = torch.load('{}/dataset.pt'.format(data_path))[index]
    if double: H0 = H0.type(torch.complex128)
    labels = torch.load('{}/labels.pt'.format(data_path))[index].numpy().tolist()
    H = constrModH(input_size, model, double=double)
    H = H.repeat(H0.shape[0], 1, 1)
    H[:, :input_size, :input_size] = H0.squeeze(1)
    H = H - model.z.cpu().real * torch.eye(H.shape[-1])
    Cor = correlation(H)[:, :input_size, -1].numpy()
    for cor, label in zip(Cor, labels):
        plotcor(cor.real, label, 'Re(Cor)')
        plotcor(cor.imag, label, 'Im(Cor)')
        while True:
            init = np.random.rand(input_size)
            init /= np.linalg.norm(init)
            result = optimize.minimize(partial(corfunc, C_r=cor.real, C_i=cor.imag, eta=eta), init, tol=1e-8,
                                       method="L-BFGS-B")
            if result.success:
                print('{}, corfunc={}'.format(label, -corfunc(result.x, cor.real, cor.imag, eta=0)))
                plotcor(result.x, label, 'c')
                break


def FindHdelta(model, data_path, input_size, index=None):
    labels = torch.load('{}/labels.pt'.format(data_path))
    t_index = torch.nonzero(labels, as_tuple=True)
    H0 = torch.load('{}/dataset.pt'.format(data_path))[t_index]
    if index is not None: H0 = H0[index]
    if double: H0 = H0.type(torch.complex128)
    print(wrap_model(model, H0))

    H = constrModH(input_size, model, double=double)
    HQ = H[input_size:, input_size:]
    Hc = H[:input_size, input_size:]
    Ht = H_t(Hc, HQ)
    Hd_theory = H_delta_theory(H0, Hc, HQ).squeeze(1).numpy()
    H = H.repeat(H0.shape[0], 1, 1)
    H[:, :input_size, :input_size] = H0.squeeze(1)

    phi = spectrum(H)[0][:, :input_size].T
    triu = np.triu_indices(input_size)
    size = len(triu[0])
    while True:
        init = 0.1 * ((2 * np.random.rand(size) - 1) + 1j * (2 * np.random.rand(size) - 1))
        result = optimize.minimize(partial(H_delta, size=input_size, index=triu, phi=phi), init, tol=1e-8,
                                   method="Nelder-Mead")
        if result.success:
            print('H_delta={}'.format(H_delta(result.x, input_size, triu, phi)))
            break
    Hd = np.zeros((size, size), dtype=np.complex128)
    Hd[index] = result.x
    Hd = Hd + Hd.T.conj()
    print(np.sum(np.abs(Hd)))
    print(np.sum(np.abs(Hd_theory - Hd), axis=(1, 2)))
    Hi = H_input(torch.from_numpy(Hd).type(torch.complex64), Ht.type(torch.complex64), delta)
    print(wrap_model(model, Hi))


def FindHdelta_torch(model, input_size, delta, diago, real, qlt, S, data_path, index=None):
    if data_path is None:
        H0 = Ham(1.0, L)[None, None, ...]
    else:
        H0 = torch.load('{}/dataset.pt'.format(data_path))
        labels = torch.load('{}/labels.pt'.format(data_path))
        if data != 'Ins_12':
            H0 = torch.cat((H0, torch.load('datasets/Ins_12/test/dataset.pt')), dim=0)
            labels = torch.cat((labels, torch.load('datasets/Ins_12/test/labels.pt')), dim=0)
        t_index = torch.nonzero(labels, as_tuple=True)
        H0 = H0[t_index]
        if index is not None: H0 = H0[index]
    if double: H0 = H0.type(torch.complex128)
    print('H0\nQNN=\n{}'.format(wrap_model(model, H0)))
    cn = []
    for h in H0.squeeze(1):
        cn.append(abs(ChernNumber(h, L, qlt, S)))
        print(Kubo(h))
    print('Chern number=\n{}\n'.format(np.array(cn)))

    H = constrModH(input_size, model, double=double)
    HQ = H[input_size:, input_size:]
    Hc = H[:input_size, input_size:]
    # Ht = H_t(Hc, HQ)
    Ht = Ham(kappa, L)
    print(Kubo(Ht))
    print('Ht\nQNN={}\tChern Number={}\n'.format(wrap_model(model, Ht[None, None, ...]), abs(ChernNumber(Ht, L, qlt, S))))
    Hd_theory = H_delta_theory(H0, Hc, HQ)
    print('H_delta_theory\nQNN=\n{}'.format(wrap_model(model, Hd_theory)))
    cn = []
    for h in Hd_theory.squeeze(1):
        cn.append(abs(ChernNumber(h, L, qlt, S)))
    print('Chern number=\n{}\n'.format(np.array(cn)))
    H = H.repeat(H0.shape[0], 1, 1)
    H[:, :input_size, :input_size] = H0.squeeze(1)
    phi, values = spectrum(H, numpy=False)
    phi = phi[:, :input_size].T
    print('H\nEigenvalues\n={}\n'.format(values.numpy()))
    exit(0)

    if load and os.path.exists('results/{}/{}/{}/H_delta.pt'.format(data, file, Net)):
        print('Load H_delta')
        Hd = torch.load('results/{}/{}/{}/H_delta.pt'.format(data, file, Net))
        # print(Hd[:10, :10])
    else:
        print('Start training')
        H = H_delta_torch(input_size, diago, real)
        optimizer = Optimizer('Adam', H.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.999))
        best, Hd = 100., None
        for i in range(2001):
            loss = H(phi)
            if i % 200 == 0:
                print('{}: {}'.format(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            cur = H(phi).item()
            if cur <= best:
                best = cur
                Hd = H.Hamil().detach().data
        print('Final best: {}'.format(best))
        print('Finish training\n')
        if save: torch.save(Hd, '{}/H_delta.pt'.format(dircheck(data, file, Net)))
    print('H_delta\nnorm={}'.format(Hd.norm(2).item()))
    print('dist={}'.format((Hd_theory.squeeze(1) - Hd).norm(2, dim=(1, 2)).numpy()))
    print('QNN={}\tChern Number={}\n'.format(
        wrap_model(model, Hd[None, None, ...].type(torch.complex128 if double else torch.complex64)),
        abs(ChernNumber(Hd, L, qlt, S))))
    Hi = H_input(Hd.type(torch.complex64), Ht, delta)
    print('Hi\nQNN={}\tChern Number={}\n'.format(wrap_model(model, Hi), abs(ChernNumber(Hi.squeeze(), L, qlt, S))))


def DetTran_torch(model, input_size, delta, n, qlt, S, diago, amount=amount, print_tl=False):
    H_model = constrModH(input_size, model, double=double).type(torch.complex128)
    # Ht = H_t(H_model[:input_size, input_size:], H_model[input_size:, input_size:])
    Ht = Ham(kappa, L)
    H = H_model.clone()
    H[:input_size, :input_size] = Ht
    print('Ht\nQNN={:.5f}\tChern number={:.5f}\tHall={:.5f}'.format(
        wrap_model(model, Ht[None, None, ...].type(torch.complex128 if double else torch.complex64))[0],
        abs(ChernNumber(Ht, L, qlt, S)), Kubo(Ht)))
    print('Eigenvalues={:.5f}\thermi={:.4e}\n'.format(
        torch.min(torch.abs(torch.linalg.eigvalsh(H))).item(), (H - H.mH).norm(2).item()))

    # load or compute phis
    if rp:
        if load and os.path.exists('results/{}/{}/{}/phis_rand{}.pt'.format(data, file, Net, no)):
            print('Load phis_rand{}'.format(no))
            phis = torch.load('results/{}/{}/{}/phis_rand{}.pt'.format(data, file, Net, no))
        else:
            if os.path.exists('datasets/nnn_mask_{}.pt'.format(L)):
                mask = torch.load('datasets/nnn_mask_{}.pt'.format(L))
            else:
                mask = nnn_mask(L)
                torch.save(mask, 'datasets/nnn_mask_{}.pt'.format(L))
            phis = torch.zeros(H.shape[-1], input_size, dtype=torch.complex128)
            phis[:input_size] = (2 * torch.rand(input_size, input_size, dtype=torch.float64) - 1 + 1j * (
                        2 * torch.rand(input_size, input_size, dtype=torch.float64) - 1)) * mask
            phis = phis / phis.norm(2, dim=0, keepdim=True)
            torch.save(phis, '{}/phis_rand{}.pt'.format(dircheck(data, file, Net), no))
    elif load and os.path.exists('results/{}/{}/{}/phis_{}.pt'.format(data, file, Net, no)):
        print('Load phis_{}'.format(no))
        phis = torch.load('results/{}/{}/{}/phis_{}.pt'.format(data, file, Net, no))
    else:
        phis = []
        if n is None: amount = None
        if amount is None: amount = input_size
        for i in range(amount):
            if n is None:
                y_index = torch.from_numpy(np.delete(np.arange(H.shape[-1]), i)).long()
                if not diago:
                    x_index = torch.arange(input_size, H.shape[-1])
                    HQ = H[input_size:, y_index]
                else:
                    assert diago > 2
                    x, y = c2l(i, L)
                    nz = [i, l2c(x, y + 1, L), l2c(x + 1, y, L), l2c(x, y - 1, L), l2c(x - 1, y, L)]
                    if diago % 2 == 0:
                        nz += [l2c(x + 1, y + 1, L), l2c(x + 1, y - 1, L), l2c(x - 1, y - 1, L), l2c(x - 1, y + 1, L)]
                    x_index = torch.from_numpy(np.delete(np.arange(H.shape[-1]), nz)).long()
                    HQ = H[x_index][:, y_index]
                HQc = H[:, y_index]
            else:
                x_index = torch.from_numpy(np.delete(np.arange(H.shape[-1]), i)).long()
                HQ = H[input_size:, input_size - n:]
                HQc = H[:, input_size - n:]
            while True:
                F = HQ_det_torch(HQ)
                optimizer = Optimizer('Adam', F.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.999))
                if print_tl: print('Start training')
                best, best_h = 100., None
                for j in range(5001):
                    loss = F(1)
                    if print_tl and j % 500 == 0:
                        print('{}: {}'.format(i, loss.item()))
                    optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    optimizer.step()
                    cur = F(0).item()
                    if cur <= best:
                        best = cur
                        best_h = F.h.detach().data
                if print_tl: print('Finish training\n')
                phi = torch.matmul(HQc, best_h)
                phi = (phi / phi.norm(2)).unsqueeze(1)
                error = phi[x_index].norm(2).item()
                if error < 1e-5:
                    phis.append(phi)
                    break
            print('{}\tFinal best: {:.4e}\tnorm={:.3f}\terror={:.4e}'.format(i, best, best_h.norm(2).item(), error))
        phis = torch.cat(phis, dim=1)
        torch.save(phis, '{}/phis_{}.pt'.format(dircheck(data, file, Net), no))
    print(phis.norm(2, dim=0).numpy())

    # add phis to H
    if n is None:
        Hs = []
        for d in delta:
            temp = H.clone()
            temp[:, :input_size] = temp[:, :input_size] + phis * d
            temp[:input_size] = temp[:input_size] + phis.mH * d.conjugate()
            Hs.append(temp)
    else:
        phi = phis.sum(dim=1, keepdim=True)
        indices = torch.tensor([random.sample(range(input_size - n), amount) for _ in range(4)])
        Hs = []
        for d in delta:
            temp = H.clone()
            temp[:, :input_size - n] = temp[:, :input_size - n] + phi * d
            temp[:input_size - n] = temp[:input_size - n] + phi.mH * d.conjugate()
            Hs.append(temp)
            for index in indices:
                temp = H.clone()
                temp[:, index] = temp[:, index] + phi * d
                temp[index] = temp[index] + phi.mH * d.conjugate()
                Hs.append(temp)
    Hs = torch.stack(Hs, dim=0)
    Hi = Hs[:, :input_size, :input_size]
    qnn = wrap_model(model, Hi.unsqueeze(1).type(torch.complex128 if double else torch.complex64))
    print('Hi\nQNN=\n{}'.format(qnn))
    plotdelta(delta, qnn, 'qnn')
    gap = torch.min(torch.abs(torch.linalg.eigvalsh(Hi.type(torch.complex128))), dim=1)[0].numpy()
    eigenvalues = torch.min(torch.abs(torch.linalg.eigvalsh(Hs.type(torch.complex128))), dim=1)[0].numpy()
    print('Gap=\n{}'.format(gap))
    print('Eigenvalues=\n{}'.format(eigenvalues))
    plotdelta(delta, gap, 'gap')
    plotdelta(delta, eigenvalues, 'eigenvalues')
    cn, hall = [], []
    for h in Hi:
        cn.append(abs(ChernNumber(h, L, qlt, S)))
        hall.append(Kubo(h))
    cn = np.array(cn)
    hall = np.array(hall)
    print('Chern number=\n{}'.format(cn))
    print('Hall=\n{}'.format(hall))
    plotdelta(delta, cn, 'cn')
    plotdelta(delta, hall, 'hall')
    if save:
        path = dircheck(data, file, Net)
        np.save('{}/qnn_{}{}_{:.3f}.npy'.format(path, 'rand' if rp else '', no, kappa), qnn)
        np.save('{}/gap_{}{}_{:.3f}.npy'.format(path, 'rand' if rp else '', no, kappa), gap)
        np.save('{}/eigenvalues_{}{}_{:.3f}.npy'.format(path, 'rand' if rp else '', no, kappa), eigenvalues)
        np.save('{}/cn_{}{}_{:.3f}.npy'.format(path, 'rand' if rp else '', no, kappa), cn)
        np.save('{}/hall_{}{}_{:.3f}.npy'.format(path, 'rand' if rp else '', no, kappa), hall)
    print('hermi=\n{}'.format((Hs - Hs.mH).norm(2, dim=(1, 2)).numpy()))
    Hs[:, :input_size, :input_size] = torch.zeros(input_size, input_size)
    print('dist=\n{}\n'.format((Hs - H_model).norm(2, dim=(1, 2)).numpy()))


if __name__ == "__main__":
    model_path = 'models/{}/{}/{}'.format(data, file, Net)
    state_dict = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")['state_dict']
    if 'z' in state_dict.keys():
        del state_dict['z']
    input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, scale, real, double \
        = readinfo('{}/info.txt'.format(model_path))
    data_path = 'datasets/{}/test'.format(data)
    model = Network(Net[:Net.index('_')], input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                    real, scale=scale, double=double)
    model.load_state_dict(state_dict, strict=False)
    model.eval()


    qlt = countriangle(trunc)
    S = Stri(qlt)

    DetTran_torch(model, input_size, delta, n, qlt, S, Hdiago)

    # FindHdelta_torch(model, input_size, delta, Hdiago, Hreal, qlt, S, data_path, index=index)

    # Kubo_test(L)

    # SpecAna(model, model_path, data_path)

    # CorChannel(model, input_size, data_path)

    # FindHdelta(model, data_path, input_size, index)