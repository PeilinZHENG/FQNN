import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import readinfo, mymkdir, image2H, emb_hid_size_wrapper, dircheck
from ent import constrModH, correlation, mutInfo, orthogonalize, RecurGF, H2G
from rgfnn import Network
import mkl, warnings
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
np.random.seed(0)


data = 'Ins_12_d0.2'
Net = 'Simple_en_12'
file = 'ENTANGLEMENT'
adj = None
index = np.random.randint(10000, size=1).tolist()
print('index={}'.format(index))
save, show = True, False


def labelchange(label, data):
    if not data.endswith('MNIST'):
        if label == 0:
            return 'Normal'
        elif label == 1:
            return 'Topological'
        elif label == 2:
            return 'Metal'
        else:
            return label
    else:
        return label


def plotEntangFlow(E, label, input_size, embedding_size, hidden_size, allop, transform):
    plt.figure()
    plt.xlim(-1, len(E) + 1)
    plt.bar(np.arange(input_size), E[:input_size], width=1.0, color='blue')
    size = input_size + embedding_size
    plt.bar(np.arange(input_size, size), E[input_size:size], width=1.0, color='green')
    for hs in hidden_size:
        plt.bar(np.arange(size, size + hs), E[size:size + hs], width=1.0)
        size += hs
    plt.xlabel('Site')
    plt.ylabel('I')
    label = labelchange(label, data)
    plt.title('{}'.format(label))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/EntangFlow{}_{}{}.jpg'.format(path, 'Tran' if transform else '', label, '+' if allop else ''))
    if show: plt.show()
    plt.close()


def plotAllDOS(E, label, input_size, embedding_size, hidden_size, output_size, transform):
    plt.figure()
    plt.xlim(-1, len(E) + 1)
    plt.bar(np.arange(input_size), E[:input_size], width=1.0, color='blue')
    size = input_size + embedding_size
    plt.bar(np.arange(input_size, size), E[input_size:size], width=1.0, color='green')
    for hs in hidden_size:
        plt.bar(np.arange(size, size + hs), E[size:size + hs], width=1.0)
        size += hs
    plt.bar(np.arange(size, size + output_size), E[size:], width=1.0, color='red')
    plt.xlabel('Site')
    plt.ylabel('LDOS')
    label = labelchange(label, data)
    plt.title('{} / {}'.format(label, 'Transformation' if transform else 'GF'))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/AllLDOS{}_{}.jpg'.format(path, 'Tran' if transform else '', label))
    if show: plt.show()
    plt.close()


def Unitary(C, A, output_size, size_list):
    cur_size = 0
    U = torch.zeros_like(C)
    U[-output_size:][:, -output_size:] = torch.eye(output_size)
    for size in size_list:
        Cba = C[cur_size:cur_size + size][:, A]
        Ul = torch.cat((Cba.mH, torch.eye(size)), dim=0)
        Ul = orthogonalize(Ul)
        U[cur_size:cur_size + size][:, cur_size:cur_size + size] = Ul
        cur_size += size
    # print(torch.sum(torch.abs(U @ U.mH - torch.eye(U.shape[0]))))
    return U


# main functions
def EntangFlow(Net, model, H0, labels, input_size, embedding_size, hidden_size, output_size, order, allop=False,
               transform=False, adj=None):
    z = model.z.cpu()
    size = input_size + embedding_size + sum(hidden_size) + output_size
    size_list = [input_size, embedding_size] + list(hidden_size)

    H = constrModH(input_size, model, double=double)
    H = H.repeat(H0.shape[0], 1, 1)
    if data.endswith('MNIST'):
        H0 = image2H(H0, input_size, adj)
    H[:, :input_size, :input_size] = H0.squeeze(1)
    H = (H - z.real * torch.eye(H.shape[-1])).type(torch.complex128)
    Cor = correlation(H)

    if output_size == 1 or allop:
        A = torch.arange(size - output_size, size)
    else:
        A = []
        for l in labels:
            A.append(torch.arange(size - output_size + l, size - output_size + l + 1))

    I = []
    if type(A) is list:
        DOS = []
        for C, h, a in zip(Cor, H, A):
            Ca = C[a][:, a]
            if transform:
                U = Unitary(C, a, output_size, size_list)
                C = U @ C @ U.mH
                h = U.conj() @ h @ U.T
            DOS.append(torch.diagonal(H2G(h, z), dim1=-2, dim2=-1).imag * scale)
            temp = []
            for i in range(size - output_size):
                B = torch.arange(i, i + 1)
                AB = torch.cat((B, a))
                Cb, Cab = C[B][:, B], C[AB][:, AB]
                temp.append(mutInfo(Ca, Cb, Cab, order).item())
            I.append(temp)
        I = torch.tensor(I)
        DOS = torch.stack(DOS, dim=0)
    else:
        if transform:
            for i in range(len(Cor)):
                U = Unitary(Cor[i], A, output_size, size_list)
                Cor[i] = U @ Cor[i] @ U.mH
                H[i] = U.conj() @ H[i] @ U.T
        DOS = torch.diagonal(H2G(H, z), dim1=-2, dim2=-1).imag * scale
        Ca = Cor[:, A][:, :, A]
        for i in range(size - output_size):
            B = torch.arange(i, i + 1)
            AB = torch.cat((B, A))
            Cb, Cab = Cor[:, B][:, :, B], Cor[:, AB][:, :, AB]
            I.append(mutInfo(Ca, Cb, Cab, order))
        I = torch.stack(I, dim=0).T
    for E, D, label in zip(I, DOS, labels):
        plotEntangFlow(E, label, input_size, embedding_size, hidden_size, allop, transform)
        print(D[:input_size])
        plotAllDOS(D, label, input_size, embedding_size, hidden_size, output_size, transform)
        if save:
            np.save('{}/I{}_{}{}.npy'.format(dircheck(data, file, Net), 'Tran' if transform else '',
                                             labelchange(label, data), '+' if allop else ''), E.numpy())


if __name__ == "__main__":
    model_path = 'models/{}/{}/{}'.format(data, file, Net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")['state_dict']
    if 'z' in state_dict.keys():
        del state_dict['z']
    input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, scale, real, double \
        = readinfo('{}/info.txt'.format(model_path))
    if data.endswith('MNIST'):
        dataset = torch.load('datasets/{}/processed/test.pt'.format(data))
        H0 = dataset[0][index].unsqueeze(1) / 255
        labels = dataset[1][index].numpy().tolist()
    else:
        data_path = 'datasets/{}/test'.format(data)
        H0 = torch.load('{}/dataset.pt'.format(data_path))
        labels = torch.load('{}/labels.pt'.format(data_path))
        if data != 'Ins_12':
            H0 = torch.cat((H0, torch.load('datasets/Ins_12/test/dataset.pt')), dim=0)
            labels = torch.cat((labels, torch.load('datasets/Ins_12/test/labels.pt')), dim=0)
        H0, labels = H0[index], labels[index].numpy().tolist()
        if type(z) is float:
            E = torch.load('{}/Es.pt'.format(data_path))[index]
            z = E + z * 1j
    if double: H0 = H0.type(torch.complex128)

    print('labels={}'.format(labels))

    model = Network(Net[:Net.index('_')], input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr,
                    real, scale=scale, double=double)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(model)
    embedding_size, hidden_size = emb_hid_size_wrapper(model)
    loc = 'upper right' if scale else 'lower left'
    scale = -1. / torch.pi if scale else 1.
    order = 1
    if adj is not None: adj = adj * torch.load('datasets/AdjMat.pt')

    '''
    main functions
    '''
    # EntangFlow(Net, model, H0, labels, input_size, embedding_size, hidden_size, output_size, order, allop=False,
    #            transform=False, adj=adj)
    # EntangFlow(Net, model, H0, labels, input_size, embedding_size, hidden_size, output_size, order, allop=False,
    #            transform=True, adj=adj)
    if output_size > 1:
        EntangFlow(Net, model, H0, labels, input_size, embedding_size, hidden_size, output_size, order, allop=True,
                   transform=False, adj=adj)
        EntangFlow(Net, model, H0, labels, input_size, embedding_size, hidden_size, output_size, order, allop=True,
                   transform=True, adj=adj)

