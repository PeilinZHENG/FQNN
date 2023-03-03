import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import warnings, mkl
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)

N = 16
numzero = 1e-6
path = 'data/Boltz_{}+'.format(N)
np.random.seed(58)


# Generate the Hamiltonian of the configuration
def genHamil(data, N):
    # Be careful for the basis matrices are pauli matrices not spin-representation
    Ham = sparse.coo_matrix((np.append(np.repeat(data[:N], 2 ** N),
                                       np.add(np.dot(data[N:2 * N], sz), np.dot(data[2 * N:], szsz))), (srow, scol)),
                            shape=(2 ** N, 2 ** N))
    return Ham


def pbc(x, y, L):
    return x % L, y % L


def l2c(x, y, L):
    x, y = pbc(x, y, L)
    return x + y * L


def c2l(n, L):
    assert n < L ** 2
    return n % L, n // L


def Ham(k, L, mode=True):
    if mode:
        H = np.zeros((L ** 2, L ** 2), dtype=np.complex128)
    else:
        H = torch.zeros((L ** 2, L ** 2), dtype=torch.complex128)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            H[nx, n] = H[nx, n] + (-1) ** y
            H[n, nx] = H[n, nx] + (-1) ** y
            H[ny, n] = H[ny, n] + 1. + (-1) ** y * (1 - k)
            H[n, ny] = H[n, ny] + 1. + (-1) ** y * (1 - k)
            H[n1, n] = H[n1, n] + 1j * (-1) ** y * k / 2
            H[n, n1] = H[n, n1] - 1j * (-1) ** y * k / 2
            H[n2, n] = H[n2, n] - 1j * (-1) ** y * k / 2
            H[n, n2] = H[n, n2] + 1j * (-1) ** y * k / 2
    return H


def orthogonalize(U, eps=1e-6): #GS-orthogonalization
    n = U.shape[1]
    V = U.T
    zero_count = 0
    # V[0] = V[0] / torch.linalg.norm(V[0]) #first vector normalization
    for i in range(1, n):
        prev_basis = V[:i]     # orthonormal basis before V[i]
        projector = torch.matmul(prev_basis.T, prev_basis)
        V[i] -= torch.matmul(projector, V[i])
        if torch.linalg.norm(V[i]) < eps:
            V[i][:] = 0.   # set the small entries to 0
            zero_count += 1
        else:
            V[i] = V[i] / torch.linalg.norm(V[i])
    return V


def check_numpy(E, v):
    norm = np.linalg.norm(v, axis=0)
    for i, x in enumerate(norm):
        if np.abs(x - 1) > numzero:
            print('norm', i, x, np.abs(x - 1))
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            overlap = np.abs(np.dot(v[:, i].conj(), v[:, j]))
            if overlap > numzero:
                print('ortho', overlap, E[i], E[j])


def check_torch(E, v):
    norm = torch.linalg.norm(v, dim=0)
    for i, x in enumerate(norm):
        if torch.abs(x - 1) > numzero:
            print('norm', i, x.item(), torch.abs(x - 1).item())
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            overlap = torch.abs(torch.dot(v[:, i].conj(), v[:, j])).item()
            if overlap > numzero:
                print('ortho', overlap, E[i].item(), E[j].item())


if __name__ == "__main__":
    gap = np.load('results/data/Fig4/gap_rand_0.520.npy')
    print(np.min(gap), np.argmin(gap))
    exit(0)

    loss, acc, muti = [], [], []
    path = 'results/MNIST/STRUCTURE'
    for i in range(10):
        model = 'Naive_h_N_{}'.format(i)
        loss.append(np.load('{}/{}/loss.npy'.format(path, model)))
        acc.append(np.load('{}/{}/accuracy.npy'.format(path, model)))
        # muti.append(np.load('{}/{}/Iac.npy'.format(path, model)))
    loss = np.stack(loss, axis=0)
    acc = np.stack(acc, axis=0)
    # muti = np.stack(muti, axis=0)
    np.save('{}/LDOS_loss.npy'.format(path), loss)
    np.save('{}/LDOS_accuarcy.npy'.format(path), acc)
    # np.save('{}/muti_reg.npy'.format(path), muti)
    print(loss.shape, acc.shape)
    exit(0)



    labels = torch.load('datasets/Ins_12_d0.2/train/labels.pt')

    c0, c1 = 0, 0
    for label in labels:
        if label == 0:
            c0 += 1
        elif label == 1:
            c1 += 1
    print(c0, c1)
    exit(0)

    dtype_np, dtype_t = np.complex64, torch.complex64
    dtypec_np, dtypec_t = np.complex128, torch.complex128

    ham0 = Ham(0.1, N).astype(dtype_np)
    ham1 = (2 * np.random.rand(N ** 2, N ** 2) - 1) + 1j * (2 * np.random.rand(N ** 2, N ** 2) - 1)
    ham1 = (ham1 + ham1.T.conj()).astype(dtype_np)
    # Configuration
    N = int(N / 2)
    """
    Hamiltonian Skeleton
    """
    s0 = np.arange(2 ** N)
    sz = np.array([np.repeat(np.tile([1, -1], 2 ** i), 2 ** (N - 1 - i)) for i in np.arange(N)])
    # sz = -np.array([np.floor_divide(np.remainder(s0,2*2**i),2**i)*2-1 for i in np.arange(N)])
    szsz = np.array([sz[i] * sz[j] for i in np.arange(N - 1) for j in np.arange(i + 1, N)])
    sx = np.array([s0 + sz[i] * 2 ** (N - 1 - i) for i in np.arange(N)])
    # sx = np.array([np.subtract(s0,-sz[i]*2**i) for i in np.arange(N)])
    scol = np.append(sx.flatten(), s0)
    srow = np.tile(s0, N + 1)

    data = np.ones(int(2 * N + N * (N - 1) / 2))
    ham2 = genHamil(data, N).toarray().astype(dtype_np)
    data = np.zeros(int(2 * N + N * (N - 1) / 2))
    data[:2 * N] = -0.1 * np.random.rand(2 * N)
    temp = -np.ones(1)
    for i in range(1, N - 1):
        temp = np.concatenate((-np.ones(1), np.zeros(i), temp))
    data[2 * N:] = temp
    ham3 = genHamil(data, N).toarray().astype(dtype_np)

    # numpy
    print('======numpy======')
    # t = time.time()
    # for _ in range(1000):
    #     E0, v0 = np.linalg.eigh(ham0)
    # print((time.time() - t) / 1000)
    E0, v0 = np.linalg.eigh(ham0.astype(dtypec_np))
    E1, v1 = np.linalg.eigh(ham1.astype(dtypec_np))
    E2, v2 = np.linalg.eigh(ham2.astype(dtypec_np))
    E3, v3 = np.linalg.eigh(ham3.astype(dtypec_np))

    print('------ham0 check------')
    print('E0=\n', E0)
    check_numpy(E0, v0)
    print('dist=', np.linalg.norm(v0 @ np.diag(E0).astype(v0.dtype) @ v0.T.conj() - ham0))
    print(np.max(np.abs((v3 @ np.diag(E3).astype(v3.dtype) @ v3.T.conj() - ham3))))
    print('------ham1 check------')
    print('E1=\n', E1)
    check_numpy(E1, v1)
    print('dist=', np.linalg.norm(v1 @ np.diag(E1).astype(v1.dtype) @ v1.T.conj() - ham1))
    print(np.max(np.abs((v3 @ np.diag(E3).astype(v3.dtype) @ v3.T.conj() - ham3))))
    print('------ham2 check------')
    print('E2=\n', E2)
    check_numpy(E2, v2)
    print('dist=', np.linalg.norm(v2 @ np.diag(E2).astype(v2.dtype) @ v2.T.conj() - ham2))
    print(np.max(np.abs((v3 @ np.diag(E3).astype(v3.dtype) @ v3.T.conj() - ham3))))
    print('------ham3 check------')
    print('E3=\n', E3)
    check_numpy(E3, v3)
    print('dist=', np.linalg.norm(v3 @ np.diag(E3).astype(v3.dtype) @ v3.T.conj() - ham3))
    print(np.max(np.abs((v3 @ np.diag(E3).astype(v3.dtype) @ v3.T.conj() - ham3))))

    # torch
    print('======torch======')
    ham0 = Ham(0.1, N, False).type(dtype_t)
    ham1 = torch.from_numpy(ham1).type(dtype_t)
    ham2 = torch.from_numpy(ham2).type(dtype_t)
    ham3 = torch.from_numpy(ham3).type(dtype_t)
    E0, v0 = torch.linalg.eigh(ham0.type(dtypec_t))
    E1, v1 = torch.linalg.eigh(ham1.type(dtypec_t))
    E2, v2 = torch.linalg.eigh(ham2.type(dtypec_t))
    E3, v3 = torch.linalg.eigh(ham3.type(dtypec_t))

    print('------ham0 check------')
    print('E0=\n', E0)
    check_torch(E0, v0)
    print('dist=', torch.dist(v0 @ torch.diag_embed(E0).type(v0.dtype) @ v0.mH, ham0))
    print(torch.max(torch.abs(v0 @ torch.diag_embed(E0).type(v0.dtype) @ v0.mH - ham0)))
    print('------ham1 check------')
    print('E1=\n', E1)
    check_torch(E1, v1)
    print('dist=', torch.dist(v1 @ torch.diag_embed(E1).type(v1.dtype) @ v1.mH, ham1))
    print(torch.max(torch.abs(v1 @ torch.diag_embed(E1).type(v1.dtype) @ v1.mH - ham1)))
    print('------ham2 check------')
    print('E2=\n', E2)
    check_torch(E2, v2)
    print('dist=', torch.dist(v2 @ torch.diag_embed(E2).type(v2.dtype) @ v2.mH, ham2))
    print(torch.max(torch.abs(v2 @ torch.diag_embed(E2).type(v2.dtype) @ v2.mH - ham2)))
    print('------ham3 check------')
    print('E3=\n', E3)
    check_torch(E3, v3)
    print('dist=', torch.dist(v3 @ torch.diag_embed(E3).type(v3.dtype) @ v3.mH, ham3))
    print(torch.max(torch.abs(v3 @ torch.diag_embed(E3).type(v3.dtype) @ v3.mH - ham3)))



    # eigvs = torch.linalg.qr(eigvs)[0]
    # eigvs = orthogonalize(eigvs)


