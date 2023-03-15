import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import readinfo, mymkdir, image2G, image2H, emb_hid_size_wrapper, LossPrepro, dircheck
from ent import updateH, H2G, G2H, constrModH, correlation, mutInfo, RecurGF
from rgfnn import Network, GLinear, GinvLinear, GcorLinear
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)

np.random.seed(9)


data = 'Ins_12_d0.2'
Net = 'Simple_en_12'
file = 'ENTANGLEMENT'
adj = None
index = np.random.randint(10000, size=1).tolist()
print('index={}'.format(index))
save, show, amount = True, True, None #4001


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


def H_magnitude(H):
    temp, count = torch.abs(H), torch.count_nonzero(H)
    if save: f.write('\nmodel\nmax={}, mean={}\n'.format(torch.max(temp).item(), (torch.sum(temp) / count).item()))
    print('model\nmax={}, mean={}'.format(torch.max(temp).item(), (torch.sum(temp) / count).item()))
    print('Im(H)={}'.format(torch.sum(torch.abs(H.imag)).item()))


def plotEnergy(E, label, symbol):
    plt.figure()
    plt.xlim(-1, len(E) + 1)
    plt.plot([-1, len(E) + 1], [0, 0], 'ko--', linewidth=1., markersize=0.1)
    plt.scatter(np.arange(len(E)), E, c='r', s=10)
    plt.xlabel('Count')
    plt.ylabel('E')
    label = labelchange(label, data)
    plt.title('{} / {}'.format(label, symbol))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/spectrum_{}.jpg'.format(path, label))
    if show: plt.show()
    plt.close()


def plotOutputVSE(Es, DOS, label, symbol, loc):
    plt.figure()
    plt.axis([min(Es), max(Es), -np.max(DOS) / 50, np.max(DOS) * 1.02])
    plt.plot([0., 0.], [-np.max(DOS) / 50, np.max(DOS) * 1.02], 'ko--', linewidth=0.5, markersize=0.1)
    if len(DOS) == 1:
        plt.plot(Es, DOS[0], 'bo-', linewidth=0.5, markersize=0.01, label='Topological')
    else:
        for i, D in enumerate(DOS):
            plt.plot(Es, D, 'o-', linewidth=0.5, markersize=0.01, label=i)
    plt.legend(loc=loc)
    plt.xlabel('E')
    plt.ylabel('Output')
    label = labelchange(label, data)
    plt.title('{} / {}'.format(label, symbol))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/QNNOutput_{}.jpg'.format(path, label))
    if show: plt.show()
    plt.close()


def plotAllDOS(E, label, symbol, input_size, embedding_size, hidden_size, output_size):
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
    plt.title('{} / {}'.format(label, symbol))
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/AllLDOS_{}.jpg'.format(path, label))
    if show: plt.show()
    plt.close()


def EnergyCount(states, values, z, scale, n):
    return (-torch.sum(torch.abs(states[:, -n:]) ** 2 / torch.abs(z - values.unsqueeze(1)) ** 2,
                       dim=2) * z.imag * scale).numpy()


# main functions
def spectrum(model, H0, labels, input_size, embedding_size, hidden_size):
    # Compute DOS
    with torch.no_grad():
        H, modules = torch.zeros(input_size, input_size, dtype=torch.complex128), set()
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
                    t, h = module.t_weight().type(H.dtype), module.h_bias().type(H.dtype)
                    if m == 'out':
                        t_out, h_out = t, h
                    else:
                        # eigs = torch.linalg.eigvalsh(h)
                        # plotEnergy(eigs, 'Model', m)
                        H = updateH(H, t, h)
        # eigs = torch.linalg.eigvalsh(h_out)
        # plotEnergy(eigs, 'Model', 'out')
        H = updateH(H, t_out, h_out)
        H_magnitude(H)
        # eigs = torch.linalg.eigvalsh(H[input_size:, input_size:])
        # plotEnergy(eigs, 'Model', 'QNN')
        # print(eigs)

        # Add H0
        H = H.repeat(H0.shape[0], 1, 1)
        if data.endswith('MNIST'):
            H0 = image2H(H0, input_size)
        H[:, :input_size, :input_size] = H0.squeeze(1)
        values = torch.linalg.eigvalsh(H)
        eigs = [[] for _ in range(len(labels))]

        # temp = torch.linalg.eigvalsh(H0.squeeze(1))
        # for i in range(len(temp)):
        #     eigs[i].append(temp[i])
        # cur_size = input_size + embedding_size
        # temp = torch.linalg.eigvalsh(H[:, :cur_size, :cur_size])
        # for i in range(len(temp)):
        #     eigs[i].append(temp[i])
        # for hs in hidden_size:
        #     cur_size += hs
        #     temp = torch.linalg.eigvalsh(H[:, :cur_size, :cur_size])
        #     for i in range(len(temp)):
        #         eigs[i].append(temp[i])

        for i in range(len(values)):
            eigs[i].append(values[i])
        for e, label in zip(eigs, labels):
            for i in range(len(e)):
                plotEnergy(e[i], label, i)
                if save: np.save('{}/spectrum_{}_{}.npy'.format(dircheck(data, file, Net), label, i), e[i].numpy())


def alldos(model, H0, labels, input_size, embedding_size, hidden_size, output_size, scale, adj=None):
    z = model.z.cpu()
    # Compute DOS_RGF
    if data.endswith('MNIST'):
        G0 = image2G(H0, input_size, adj, z=z if Net.endswith('-') else None)
    else:
        G0 = H2G(H0, z)
    G = RecurGF(G0, model).squeeze(1)
    G = torch.diagonal(G, dim1=-2, dim2=-1)
    DOS_RGF = (G.imag * scale).numpy()
    for d, label in zip(DOS_RGF, labels):
        plotAllDOS(d, label, "RGF", input_size, embedding_size, hidden_size, output_size)

    # # Compute DOS
    # H = constrModH(input_size, model, double=double)
    # H = H.repeat(H0.shape[0], 1, 1)
    # H[:, :input_size, :input_size] = H0.squeeze(1)
    # values, states = torch.linalg.eigh(H.type(torch.complex128))
    #
    # # Compute DOS_EC
    # DOS_EC = EnergyCount(states, values, z, scale, H.shape[-1])
    # for d, label in zip(DOS_EC, labels):
    #     plotAllDOS(d, label, 'Energy Count', input_size, embedding_size, hidden_size, output_size)
    #     # plotAllDOS(d[:input_size + embedding_size + 1], label, 'Energy Count', input_size, embedding_size, hidden_size, output_size)
    #
    # # Compute DOS_GF
    # G = H2G(H, z)
    # G = torch.diagonal(G, dim1=-2, dim2=-1)
    # DOS_GF = (G.imag * scale).numpy()
    # for d, label in zip(DOS_GF, labels):
    #     plotAllDOS(d, label, "Green's Function", input_size, embedding_size, hidden_size, output_size)
    #
    # # Compute DOS_NN
    # model = model.to(device)
    # model.eval()
    # DOS_NN = model(H2G(H0, z).to(device))
    # if DOS_NN.dim() == 1:
    #     DOS_NN = DOS_NN.reshape(DOS_GF.shape)
    # DOS_NN = DOS_NN.data.cpu().numpy()
    # print(DOS_NN, DOS_RGF[:, -2:])


def dos(model, H0, labels, input_size, output_size, scale, loc, device, adj=None):
    z = model.z.cpu()
    # # Compute DOS
    # H = constrModH(input_size, model, double=double)
    # H = H.repeat(H0.shape[0], 1, 1)
    # if data.endswith('MNIST'):
    #     H0 = image2H(H0, input_size)
    # H[:, :input_size, :input_size] = H0.squeeze(1)
    Es = None if amount is None else torch.linspace(-1, 1, amount)

    # # Compute DOS_EC
    # values, states = torch.linalg.eigh(H)
    # DOS_EC = EnergyCount(states, values, z, scale, output_size)
    # print('DOS_EC=\n{}'.format(DOS_EC))
    # DOS = []
    # for E in Es:
    #     DOS.append(EnergyCount(states, values, E + z.imag * 1j, scale, output_size))
    # DOS = np.stack(DOS, axis=2)
    # for D, label in zip(DOS, labels):
    #     plotDOS(Es, D, label, 'Energy Count', loc)

    # # Compute DOS_GF
    # G = H2G(H, z)
    # G = torch.diagonal(G, dim1=-2, dim2=-1)
    # G = G.imag * scale
    # DOS_GF = G[:, -output_size:].numpy()
    # print('DOS_GF=\n{}'.format(DOS_GF))
    # DOS = []
    # for E in Es:
    #     G = torch.diagonal(H2G(H, E + z.imag * 1j), dim1=-2, dim2=-1).imag * scale
    #     DOS.append(G[:, -output_size:].numpy())
    # DOS = np.stack(DOS, axis=2)
    # for D, label in zip(DOS, labels):
    #     plotDOS(Es, D, label, "Green's Function", loc)

    # Compute DOS_NN
    model = model.to(device)
    model.eval()
    if data.endswith('MNIST'):
        G0 = image2G(H0, input_size, adj, z=z if Net.endswith('-') else None).to(device)
    else:
        G0 = H2G(H0, z).to(device)
    DOS_NN_0 = model(G0)
    if DOS_NN_0.dim() == 1:
        output = LossPrepro(DOS_NN_0, 'BCE', scale=True if scale < 0 else False)
        output = torch.vstack((1 - output, output)).T
        DOS_NN_0 = DOS_NN_0.unsqueeze(1)
    else:
        output = torch.softmax(DOS_NN_0, dim=1)
    values, indices = torch.max(output, dim=1)
    print('Prediction={}'.format(indices.tolist()))
    print('Confidence={}'.format(values.tolist()))
    if save:
        f.write('Prediction={}\n'.format(indices.tolist()))
        f.write('Confidence={}\n'.format(values.tolist()))
    # print('Distance={}'.format(np.sum(np.abs(DOS_GF - DOS_NN))))
    if Es is not None:
        DOS = []
        for E in Es:
            if E == z.real:
                DOS.append(DOS_NN_0.data.cpu().numpy())
                continue
            temp = E + z.imag * 1j
            model.z = temp.to(device)
            if data.endswith('MNIST'):
                G0 = image2G(H0, input_size, adj, z=temp if Net.endswith('-') else None).to(device)
            else:
                G0 = H2G(H0, temp).to(device)
            DOS_NN = model(G0)
            if DOS_NN.dim() == 1:
                DOS_NN = DOS_NN.unsqueeze(1)
            DOS.append(DOS_NN.data.cpu().numpy())
        DOS = np.stack(DOS, axis=2)
        for D, label in zip(DOS, labels):
            plotOutputVSE(Es, D, label, "FBQNN", loc)


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
    temp = torch.abs(torch.flatten(H0, 1))
    count = torch.count_nonzero(temp, dim=1)
    print('H0\nmax={}\nmean={}'.format(torch.max(temp, dim=1)[0].numpy(), (torch.sum(temp, dim=1) / count).numpy()))
    if save:
        f = open('{}/magnitude.txt'.format(dircheck(data, file, Net)), 'w')
        f.write('index={}\n'.format(index))
        f.write('labels={}\n'.format(labels))
        f.write('H0\nmax={}\nmean={}\n'.format(torch.max(temp, dim=1)[0].numpy(), (torch.sum(temp, dim=1) / count).numpy()))


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
    if not data.endswith('MNIST') or Net.endswith('-'):
        spectrum(model, H0, labels, input_size, embedding_size, hidden_size)
    else:
        H_magnitude(constrModH(input_size, model, double=double))
    alldos(model, H0, labels, input_size, embedding_size, hidden_size, output_size, scale, adj=adj)
    dos(model, H0, labels, input_size, output_size, scale, loc, device, adj=adj)

    if save: f.close()
