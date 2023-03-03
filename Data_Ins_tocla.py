import torch
from utils import mymkdir
from Data_Ins import l2c
import mkl, warnings
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)


L = 12
TYPE = 'train'
path = 'datasets/Ins_{}_d0.2'.format(L)
mymkdir(path)
path = '{}/{}'.format(path, TYPE)
mymkdir(path)


def getElements(H):
    data = []
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            data += [H[n, n].real, H[nx, n].real, H[nx, n].imag, H[ny, n].real, H[ny, n].imag, H[n1, n].real,
                     H[n1, n].imag, H[n2, n].real, H[n2, n].imag]
    return torch.stack(data)


def Q2C(quandata):
    datas = []
    for H in quandata.squeeze(1):
        datas.append(getElements(H))
    datas = torch.stack(datas, dim=0)        # (amount, 9 * L ** 2)
    return datas


def gapCheck(quandata):
    errors = []
    for i, H in enumerate(quandata.squeeze(1)):
        values = torch.linalg.eigvalsh(H.type(torch.complex128))
        minval = torch.min(torch.abs(values))
        if minval < 1e-2:
            errors.append(i)
    return errors

if __name__ == "__main__":
    quandata = torch.load('{}/dataset.pt'.format(path))
    errors = gapCheck(quandata)
    print(len(errors))
    print(errors)
    # cladata = Q2C(quandata)
    # print(cladata.shape)
    # torch.save(cladata, '{}/cla_dataset.pt'.format(path))
