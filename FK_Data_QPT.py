from FK_DMFT import *
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from functools import partial
from utils import mymkdir, myceil
from Data_Ins import l2c
import mkl, warnings

warnings.filterwarnings('ignore')
mkl.set_num_threads(1)


def Ham(L, mu, t2=0.):
    H = torch.diag_embed(-mu * torch.ones(L ** 2)).type(torch.complex128)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            # nearest neighbor
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            H[nx, n] = H[nx, n] + 1.
            H[n, nx] = H[n, nx] + 1.
            H[ny, n] = H[ny, n] + 1.
            H[n, ny] = H[n, ny] + 1.
            # next nearest neighbor
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            H[n1, n] = H[n1, n] + t2
            H[n, n1] = H[n, n1] + t2
            H[n2, n] = H[n2, n] + t2
            H[n, n2] = H[n, n2] + t2
    return H


def genData(L, t2, mu_devi, DMFT_data=False, phase_set=1):
    t = time.time()
    torch.manual_seed(0)
    '''construct DMFT'''
    count = 20
    iota = 0.
    momentum = 0.5
    momDisor = 0.
    maxEpoch = 10
    milestone = 50
    f_filling = 0.5
    d_filling = 0.5
    tol_sc = 1e-6
    tol_bi = 1e-7
    gap = 5.
    T_float = 0.005
    scf = DMFT(count, iota, momentum, momDisor, maxEpoch, milestone, f_filling, d_filling, tol_sc, tol_bi, gap, device)
    dtype = torch.double

    '''gen Ham data'''
    # t2 = torch.arange(0, 0.1, 0.05, device=device)
    t2_amount = t2.shape[0]
    T = T_float * torch.ones(t2_amount, device=device, dtype=dtype).unsqueeze(-1)  # (bz,1)
    mu0 = torch.zeros((t2_amount, 1, 1), device=device, dtype=dtype)  # (t2_amount, 1, 1)
    H0 = torch.stack([Ham(L, mu=0., t2=i.item()) for i in t2], dim=0).unsqueeze(1).to(device)  # (t2_amount,1,size,size)
    mu = scf.bisearch_dec(partial(scf.fix_filling, f_ele=False), mu0, H0, T)  # (t2_amount,1,1)
    # mu_devi = torch.arange(-0.1, 0.5, 0.05, device=device)

    H0_samples = []
    mu_samples = []
    for i in range(t2_amount):
        mu_sample = mu[i, :, :] + mu_devi.unsqueeze(-1).unsqueeze(-1)  # (mu_devi_amount,1,1)
        H0_sample = H0[i, :, :, :] - torch.diag_embed(
            mu_sample * torch.ones(L ** 2, device=device, dtype=dtype))  # (mu_devi_amount,1,size,size)
        mu_samples.append(mu_sample)
        H0_samples.append(H0_sample)
    bz = mu_devi.shape[0] * t2_amount  # bz = mu_devi_amount * t2_amount
    mu_samples = torch.concatenate(mu_samples, dim=0).squeeze()  # (bz)
    H0_samples = torch.concatenate(H0_samples, dim=0)  # (bz,1,size,size)
    t2_samples = torch.repeat_interleave(t2, mu_devi.shape[0])  # (bz)
    T_samples = T_float * torch.ones(bz, device=device, dtype=dtype)  # (bz)
    U_samples = torch.ones(bz, device=device, dtype=dtype)  # (bz)

    if not DMFT_data:
        phase_labels = phase_set * torch.ones_like(T_samples)
        labels = torch.stack((U_samples, T_samples, phase_labels), dim=-1)  # (bz,3)
        return H0_samples, labels
    else:
        '''gen DMFT data'''
        t = time.time()
        train_batchsize = 100
        OPs = []
        nfs = []
        bad_idxs = []
        errors = []
        for i in range(myceil(bz / train_batchsize)):
            H0_batch = H0_samples[i * train_batchsize:(i + 1) * train_batchsize].to(device)
            T_batch = T_samples[i * train_batchsize:(i + 1) * train_batchsize].to(device)
            U_batch = U_samples[i * train_batchsize:(i + 1) * train_batchsize].to(device)
            SE_batch, OP_batch, nf_batch, bad_idx_batch, bad_error_batch = scf(T_batch, H0_batch, U_batch, reOP=True,
                                                                               reBad=True,
                                                                               OPfuns=(op_loc,),
                                                                               prinfo=True)  # (bz, 1, size)
            OPs.append(OP_batch)
            nfs.append(nf_batch)
            bad_idxs.append(bad_idx_batch + i * train_batchsize)
            errors.append(bad_error_batch)

        OP = torch.cat(OPs, dim=0)
        nf = torch.cat(nfs, dim=0)
        idx = torch.cat(bad_idxs, dim=0)
        errors = torch.cat(errors, dim=0)
        labels = torch.stack((U_samples, T_samples, OP), dim=-1)  # (amount,3)
        return H0_samples, labels
        # OP_np = torch.cat(OPs, dim=0).cpu().numpy()
        # nf_np = torch.cat(nfs, dim=0).cpu().numpy()
        # idx_np = torch.cat(bad_idxs, dim=0).cpu().numpy()
        # errors_np = torch.cat(errors, dim=0).cpu().numpy()
        # mu_np = mu_samples.cpu().numpy()
        # t2_np = t2_samples.cpu().numpy()
        # '去除bad data'
        # mu_np = np.delete(mu_np, idx_np)
        # t2_np = np.delete(t2_np, idx_np)
        # OP_np = np.delete(OP_np, idx_np, axis=1)
        # nf_np = np.delete(nf_np, idx_np, axis=0)
        # np.savez("data_result", mu=mu_np, t2=t2_np, OP=OP_np, nf=nf_np, idx_bad=idx_np, error=errors_np)
        # for i in np.arange(mu_np.shape[0]):
        #     print(f"mu={mu_np[i]},t2={t2_np[i]},OP={OP_np[:,i]},best_op={np.argmax(OP_np[:,i])}")
        #     print(np.round(nf_np[i,:].reshape(12,12),2))
        # print(time.time() - t)


def main(L, t2_amount, mu_amount):
    '生成不含DMFT的data'
    t2 = torch.linspace(0, 0.35, t2_amount, device=device)
    mu_devi = torch.linspace(-0.1, 0.5, mu_amount, device=device)
    H1s, label1s = genData(L, t2, mu_devi, DMFT_data=False, phase_set=1)  # label=1,checkboard
    t2 = torch.linspace(0.8, 1.2, t2_amount, device=device)
    H2s, label2s = genData(L, t2, mu_devi, DMFT_data=False, phase_set=0)  # label=0,stripe
    Hs = torch.cat((H1s, H2s))
    labels = torch.cat((label1s, label2s))
    return Hs, labels


if __name__ == '__main__':
    device = torch.device("cpu")
    L = 12
    TYPE = 'test'
    t2_amount, mu_amount = 100, 5  # total_amount = t2_amount * mu_amount * 2

    path = 'datasets/FK_{}'.format(L)
    mymkdir(path)
    path = '{}/{}'.format(path, TYPE)
    mymkdir(path)

    t = time.time()
    Hs, labels = main(L, t2_amount, mu_amount)

    delta_t = time.time() - t
    print(delta_t, '\n', L, Hs.shape, labels.shape)
    f = open('{}/info.txt'.format(path), 'w')
    f.write('time={}\nL={}\ndataset.shape={}\nlabels.shape={}'.format(delta_t, L, Hs.shape, labels.shape))
    f.close()

    # save
    torch.save(Hs, '{}/dataset.pt'.format(path))  # (amount * processors, 1, L ** 2, L ** 2)
    torch.save(labels, '{}/labels.pt'.format(path))  # (amount * processors, 2)
