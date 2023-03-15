import torch, os
from FK_DMFT import DMFT
import torch.multiprocessing as mp
from utils import myceil
import mkl, warnings
warnings.filterwarnings('ignore')

# parameters
L = 10
TYPE = 'train'
processors = 0
if processors == 0: bz = 50
# DMFT configs
count = 20
iota = 0.
momentum = 0.5
maxEpoch = 5000
filling = 0.5
tol_sc = 1e-6
tol_bi = 1e-6
double = True


if processors > 0:
    mkl.set_num_threads(1)
    torch.set_num_threads(1)
    os.environ ['OMP_NUM_THREADS'] = '1'
    os.environ ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ ['MKL_NUM_THREADS'] = '1'
    os.environ ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ ['NUMEXPR_NUM_THREADS'] = '1'


path = 'datasets/FK_{}/{}'.format(L, TYPE)
H0 = torch.load('{}/dataset.pt'.format(path))     # (amount, scf.count, L ** 2, L ** 2)
target = torch.load('{}/labels.pt'.format(path))  # (amount, 3)
scf = DMFT(count, iota, momentum, maxEpoch, filling, tol_sc, tol_bi, double=double)
if processors > 0: bz = myceil(len(H0) / processors)


def computeSE(i):
    return scf(target[i * bz:(i + 1) * bz, 1], H0[i * bz:(i + 1) * bz], target[i * bz:(i + 1) * bz, 0]) # (bz, scf.count, L ** 2)


if __name__ == "__main__":
    SE = []
    if processors > 0:
        mp.set_start_method('fork', force=True)
        pool = mp.Pool(processes=processors)
        res = pool.map(computeSE, range(processors))
        for se in res: SE.append(se)
        pool.close()
        pool.join()
    else:
        for i in range(myceil(len(H0) / bz)):
            SE.append(computeSE(i))
    SE = torch.cat(SE, dim=0)
    torch.save(SE, '{}/SE.pt'.format(path))  # (amount, scf.count, L ** 2)