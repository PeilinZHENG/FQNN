import torch, os
from FK_DMFT import DMFT
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
from utils import myceil
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# parameters
L = 10
TYPE = 'train'
processors = 0
if processors == 0: bz = 250
# DMFT configs
count = 20
iota = 0.
momentum = 0.5
momDisor = 0.
maxEpoch = 5000
milestone = 30
f_filling = 0.5
d_filling = None
tol_sc = 1e-6
tol_bi = 1e-7
gap = 1.
double = True
device = torch.device("cuda")


def computeSE(i, bz, scf, path):
    SE = scf(target[i * bz:(i + 1) * bz, 1], H0[i * bz:(i + 1) * bz], target[i * bz:(i + 1) * bz, 0])
    torch.save(SE, f'{path}/SE_{i}.pt')
    return SE   # (bz, scf.count, L ** 2)


if __name__ == "__main__":
    if processors > 0:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        torch.set_num_threads(1)

    path = 'datasets/FK_{}/{}'.format(L, TYPE)
    H0 = torch.load('{}/dataset.pt'.format(path))  # (amount, scf.count, L ** 2, L ** 2)
    target = torch.load('{}/labels.pt'.format(path))  # (amount, 3)
    scf = DMFT(count, iota, momentum, momDisor, maxEpoch, milestone, f_filling, d_filling, tol_sc, tol_bi, gap, device, double)

    if processors > 0:
        SE = []
        mp.set_start_method('fork', force=True)
        pool = mp.Pool(processes=processors)
        res = pool.map(partial(computeSE, bz=myceil(len(H0) / processors), scf=scf, path=path), range(processors))
        for se in res: SE.append(se)
        pool.close()
        pool.join()
        torch.save(torch.cat(SE, dim=0), '{}/SE.pt'.format(path))  # (amount, scf.count, L ** 2)
    else:
        SE = torch.zeros((0, scf.count, L ** 2), dtype=scf.iomega0.dtype)
        for i in tqdm(range(myceil(len(H0) / bz))):
            SE = torch.cat((SE, scf(target[i * bz:(i + 1) * bz, 1], H0[i * bz:(i + 1) * bz],
                                    target[i * bz:(i + 1) * bz, 0]).cpu()), dim=0)
            torch.save(SE, '{}/SE.pt'.format(path))  # (amount, scf.count, L ** 2)
