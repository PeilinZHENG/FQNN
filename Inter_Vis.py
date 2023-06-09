import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sklearn.metrics as metrics
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import warnings
from ent import H2G
from rgfnn import Network
from Data_Ins import Ham
from utils import readinfo, mymkdir, LoadHamData, image2G, LossPrepro, dircheck
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)


amount = 100
L = 12
data = 'MNIST'
Net = 'Naive_-'
file = 'STRUCTURE'
save, show = False, False
addpath = 'models/QLT/Sig_'


def plotfig(x, x_label, title):
    plt.figure()
    plt.plot(x, c='r')
    plt.xlim([-0.5, len(x) - 0.5])
    plt.ylabel(x_label)
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def plotbargrads(x, title):
    plt.figure()
    plt.xlim(0, len(x) + 1)
    plt.bar(np.arange(1, len(x) + 1), x, width=0.8)
    plt.ylabel('Gradient')
    plt.xlabel('Layer')
    plt.title(title)
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def plotbar2grads(x, x_label, y, y_label, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim(0, len(x) + 1)
    plt.bar(np.arange(1, len(x) + 1), x, width=0.8, color='b')
    plt.ylabel('Normalized Gradient')
    plt.title(x_label)
    plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=0.3)
    plt.xlim(0, len(y) + 1)
    plt.bar(np.arange(1, len(y) + 1), y, width=0.8, color='r')
    plt.ylabel('Normalized Gradient')
    plt.xlabel('Layer')
    plt.title(y_label)
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def plotgrads(X, title, g=None):
    plt.figure()
    if g is not None: plt.plot(g, c='k', label='grads')
    for i, x in enumerate(X, 1):
        plt.plot(x, label='grads({})'.format(i))
    plt.xlim([-0.5, X.shape[1] - 0.5])
    plt.legend(loc='upper right')
    plt.ylabel('Gradient')
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def plot3grads(x, x_label, y, y_label, z, z_label, title):
    plt.figure()
    plt.plot(x, c='r', label=x_label)
    plt.plot(y, c='b', label=y_label)
    plt.plot(z, c='g', label=z_label)
    plt.xlim([-0.5, len(x) - 0.5])
    plt.legend(loc='right')
    plt.ylabel('Gradient')
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def plot2axisfig(x, x_label, y, y_label, title):
    fig, ax1 = plt.subplots()
    plt.title(title)
    ax1.set_xlim([-0.5, len(x) - 0.5])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(x_label, c='g')
    # ax1.set_ylim([min(x), max(x)])
    # ax1.set_yticks(0.1 * max(loss) * np.arange(11))
    ax1.plot(x, c='g')
    ax1.tick_params(axis='y', labelcolor='g')
    # ax1.legend(loc='lower left')
    ax2 = ax1.twinx()
    ax2.set_ylabel(y_label, c='b')
    # ax2.set_ylim([my_floor(min(x), 1), 1.001])
    # ax2.set_yticks(0.1 * np.arange(11))
    ax2.plot(y, c='b')
    ax2.tick_params(axis='y', labelcolor='b')
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/{}.jpg'.format(path, title))
    if show: plt.show()
    plt.close()


def readdata(filename):
    with open(filename, "r") as f:
        cont = f.readlines()
    for i, line in enumerate(cont):
        if re.match(r"loss=.*", line):
            break
    cont = cont[i:]
    loss = np.fromstring(re.match(r"loss=\[(.*)\]", cont[0]).group(1), dtype=float, sep=",")
    accuracy = np.fromstring(re.match(r"accuracy=\[(.*)\]", cont[1]).group(1), dtype=float, sep=",")
    index = 2
    try:
        grads = np.fromstring(re.match(r"grads=\[(.*)\]", cont[index]).group(1), dtype=float, sep=",")
        agrads = np.fromstring(re.match(r"agrads=\[(.*)\]", cont[index + 1]).group(1), dtype=float, sep=",")
        grads1 = np.fromstring(re.match(r"grads1=\[(.*)\]", cont[index + 2]).group(1), dtype=float, sep=",")
        agrads1 = np.fromstring(re.match(r"agrads1=\[(.*)\]", cont[index + 3]).group(1), dtype=float, sep=",")
        index = index + 4
    except:
        grads, agrads, grads1, agrads1 = None, None, None, None
    try:
        grads2 = np.fromstring(re.match(r"grads2=\[(.*)\]", cont[index]).group(1), dtype=float, sep=",")
        agrads2 = np.fromstring(re.match(r"agrads2=\[(.*)\]", cont[index + 1]).group(1), dtype=float, sep=",")
        grads3 = np.fromstring(re.match(r"grads3=\[(.*)\]", cont[index + 2]).group(1), dtype=float, sep=",")
        agrads3 = np.fromstring(re.match(r"agrads3=\[(.*)\]", cont[index + 3]).group(1), dtype=float, sep=",")
        grads4 = np.fromstring(re.match(r"grads4=\[(.*)\]", cont[index + 4]).group(1), dtype=float, sep=",")
        agrads4 = np.fromstring(re.match(r"agrads4=\[(.*)\]", cont[index + 5]).group(1), dtype=float, sep=",")
        grads5 = np.fromstring(re.match(r"grads5=\[(.*)\]", cont[index + 6]).group(1), dtype=float, sep=",")
        agrads5 = np.fromstring(re.match(r"agrads5=\[(.*)\]", cont[index + 7]).group(1), dtype=float, sep=",")
        grads6 = np.fromstring(re.match(r"grads6=\[(.*)\]", cont[index + 8]).group(1), dtype=float, sep=",")
        agrads6 = np.fromstring(re.match(r"agrads6=\[(.*)\]", cont[index + 9]).group(1), dtype=float, sep=",")
        index = index + 10
    except:
        grads2, agrads2, grads3, agrads3, grads4, agrads4 = None, None, None, None, None, None
        grads5, agrads5, grads6, agrads6 = None, None, None, None
    try:
        grads_1 = np.fromstring(re.match(r"grads_1=\[(.*)\]", cont[index]).group(1), dtype=float, sep=",")
        agrads_1 = np.fromstring(re.match(r"agrads_1=\[(.*)\]", cont[index + 1]).group(1), dtype=float, sep=",")
        index = index + 2
    except:
        grads_1, agrads_1 = None, None
    try:
        Iacd = np.fromstring(re.match(r"Iacd=\[(.*)\]", cont[index]).group(1), dtype=float, sep=",")
        Iac = np.fromstring(re.match(r"Iac=\[(.*)\]", cont[index + 1]).group(1), dtype=float, sep=",")
        Iad = np.fromstring(re.match(r"Iad=\[(.*)\]", cont[index + 2]).group(1), dtype=float, sep=",")
        Ia_cd = np.fromstring(re.match(r"Ia_cd=\[(.*)\]", cont[index + 3]).group(1), dtype=float, sep=",")
        index = index + 4
    except:
        Iacd, Iac, Iad, Ia_cd = None, None, None, None
    try:
        Ibcd = np.fromstring(re.match(r"Ibcd=\[(.*)\]", cont[index]).group(1), dtype=float, sep=",")
        Ibc = np.fromstring(re.match(r"Ibc=\[(.*)\]", cont[index + 1]).group(1), dtype=float, sep=",")
        Ibd = np.fromstring(re.match(r"Ibd=\[(.*)\]", cont[index + 2]).group(1), dtype=float, sep=",")
        Ib_cd = np.fromstring(re.match(r"Ib_cd=\[(.*)\]", cont[index + 3]).group(1), dtype=float, sep=",")
    except:
        Ibcd, Ibc, Ibd, Ib_cd = None, None, None, None
    return loss, accuracy, grads, agrads, grads1, agrads1, grads2, agrads2, grads3, agrads3, grads4, agrads4, grads5, \
           agrads5, grads6, agrads6, grads_1, agrads_1, Iacd, Iac, Iad, Ia_cd, Ibcd, Ibc, Ibd, Ib_cd


def normalgrads(Grads):
    Grads = np.mean(Grads[:, :3], axis=1)
    Grads = Grads / Grads[-1]
    return Grads[:-1]


def addgrads(path):
    grads = np.load('{}/grads.npy'.format(path))
    agrads = np.load('{}/agrads.npy'.format(path))
    Grads = np.load('{}/grads_layers.npy'.format(path))
    Agrads = np.load('{}/agrads_layers.npy'.format(path))
    return grads, agrads, Grads, Agrads


# main functions
def LossAndAccuracy(path):
    loss, accuracy, grads, agrads, grads1, agrads1, grads2, agrads2, grads3, agrads3, grads4, agrads4, grads5, \
    agrads5, grads6, agrads6, grads_1, agrads_1, Iacd, Iac, Iad, Ia_cd, Ibcd, Ibc, Ibd, Ib_cd = readdata(path)

    accuracy = [x / 100 for x in accuracy]
    print('best:\tloss={}\taccuarcy={}'.format(min(loss), max(accuracy)))

    if addpath is not None:
        cgrads, cagrads, Cgrads, Cagrads = addgrads(addpath)

    if data.endswith('MNIST'):
        title = data
    else:
        title = 'Normal VS Topological'
    plot2axisfig(loss, 'loss', accuracy, 'accuarcy', title)
    if save:
        np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'loss'), loss)
        np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'accuracy'), accuracy)

    if grads2 is not None:
        if Net.startswith('Naive') or Net.startswith('CNaive'):
            Grads = np.stack((grads1, grads2, grads_1), axis=0)
            Agrads = np.stack((agrads1, agrads2, agrads_1), axis=0)
        elif Net.startswith('Simple') or Net.startswith('CSimple'):
            Grads = np.stack((grads1, grads2, grads3, grads4, grads_1), axis=0)
            Agrads = np.stack((agrads1, agrads2, agrads3, agrads4, agrads_1), axis=0)
        else:
            Grads = np.stack((grads1, grads2, grads3, grads4, grads5, grads6, grads_1), axis=0)
            Agrads = np.stack((agrads1, agrads2, agrads3, agrads4, agrads5, agrads6, agrads_1), axis=0)
        if save:
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'grads'), grads)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'agrads'), agrads)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'grads_layers'), Grads)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'agrads_layers'), Agrads)
        if addpath is None:
            plotgrads(Grads, 'Gradients', grads)
            plotgrads(Agrads, 'Average Gradients', agrads)
            plotbargrads(normalgrads(Grads), 'Normalized Gradients ({})'.format(Net[:Net.index('_')]))
            plotbargrads(normalgrads(Agrads), 'Normalized Average Gradients ({})'.format(Net[:Net.index('_')]))
        else:
            plotbar2grads(normalgrads(Grads), 'QNN (CC)', normalgrads(Cgrads), 'ANN (Sigmoid)', 'Gradient')
            plotbar2grads(normalgrads(Agrads), 'QNN (CC)', normalgrads(Cagrads), 'ANN (Sigmoid)', 'Average Gradient')
            # np.save('results/QNN_avg_grads.npy', normalgrads(Agrads))
            # np.save('results/ANN_avg_grads.npy', normalgrads(Cagrads))
    elif grads is not None:
        # plot3grads(grads, 'grads', grads1, 'grads(1)', grads_1, 'grads(-1)', 'Total VS First VS Last Gradients')
        plot3grads(agrads, 'agrads', agrads1, 'agrads(1)', agrads_1, 'agrads(-1)',
                   'Total VS First VS Last Average Gradients')

    if Iacd is not None:
        if Ibcd is None:
            plotfig(Iacd, 'Iacd', 'a ~ c ~ d')
            plotfig(Iac, 'Iac', 'a ~ c')
            plotfig(Iad, 'Iad', 'a ~ d')
            plotfig(Ia_cd, 'Ia_cd', 'a ~ c+d')
        else:
            plot2axisfig(Iacd, 'Iacd', Ibcd, 'Ibcd', 'a/b ~ c ~ d')
            plot2axisfig(Iac, 'Iac', Ibc, 'Ibc', 'a/b ~ c')
            plot2axisfig(Iad, 'Iad', Ibd, 'Ibd', 'a/b ~ d')
            plot2axisfig(Ia_cd, 'Ia_cd', Ib_cd, 'Ib_cd', 'a/b ~ c+d')
            if save:
                np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Ibcd'), Ibcd)
                np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Ibc'), Ibc)
                np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Ibd'), Ibd)
                np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Ib_cd'), Ib_cd)
        if save:
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Iacd'), Iacd)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Iac'), Iac)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Iad'), Iad)
            np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'Ia_cd'), Ia_cd)


def InsulatorTest(model, device):
    ks = np.linspace(0.1, 1, amount)
    Hs = torch.stack([Ham(k, L) for k in ks], dim=0).unsqueeze(1)  # (amount, 1, L ** 2, L ** 2)
    Gs = H2G(Hs, z).to(device)
    if double: Gs = Gs.type(torch.complex128)
    if output_size == 1:
        P = F.tanh(F.relu(model(Gs) / 2)).data.cpu().numpy()
        # P = model(Gs).data.cpu().numpy()
    else:
        P = F.softmax(model(Gs), dim=1)[:, 1].data.cpu().numpy()
    # print(P)

    plt.figure()
    plt.axis([0., 1., min(P), max(P)])
    plt.plot([0.5, 0.5], [min(P), max(P)], 'ko--', linewidth=0.5, markersize=0.1)
    plt.scatter(ks, P, s=20, c='r', marker='o')
    plt.xlabel('kappa')
    plt.ylabel('P')
    # plt.ylabel('localisation length')
    plt.title('Normal VS Topological')
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/PD.jpg'.format(path))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/{}.npy'.format(dircheck(data, file, Net), 'PD'), np.stack((ks, P), axis=0))


def roc_curve(model, z, double, scale, device):
    dataset = torch.load('datasets/{}/test/dataset.pt'.format(data))
    labels = torch.load('datasets/{}/test/labels.pt'.format(data))
    if data != 'Ins_12':
        dataset = torch.cat((dataset, torch.load('datasets/Ins_12/test/dataset.pt')), dim=0)
        labels = torch.cat((labels, torch.load('datasets/Ins_12/test/labels.pt')), dim=0)
    if double: dataset = dataset.type(torch.complex128)
    if Net.endswith('-'): labels = 1 - labels
    testdata = LoadHamData(dataset, labels, z)
    loader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=False)
    with torch.no_grad():
        y_true, y_pred = [], []
        for g, label in loader:
            y_true.append(label.numpy())
            output = model(g.to(device))
            if output.dim() == 1:
                pred = LossPrepro(output, 'BCE', scale=scale)
            else:
                pred = torch.softmax(output, dim=1)[:, 1]
            y_pred.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)
    # print(y_true.shape, y_pred.shape)
    predlabel = (y_pred > 0.5).astype(np.int16)
    acc = y_true == predlabel
    print('accuarcy={}'.format(acc.sum() / len(acc)))
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot()
    plt.title('Normal VS Topological')
    plt.plot(fpr, tpr, 'b-', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if save:
        path = dircheck(data, file, Net)
        plt.savefig('{}/ROC_curve.jpg'.format(path))
    if show: plt.show()
    plt.close()


if __name__ == "__main__":
    path = 'models/{}/{}/{}/info.txt'.format(data, file, Net)
    LossAndAccuracy(path)

    if not data.endswith('MNIST') and file != 'CLASSICAL' and file != 'GRADIENT':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = 'models/{}/{}/{}'.format(data, file, Net)
        input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, scale, real, double \
            = readinfo('{}/info.txt'.format(model_path), '{}/info.txt'.format(dircheck(data, file, Net)) if save else None)

        model = Network(Net[:Net.index('_')], input_size, output_size, embedding_size, hidden_size, z, hermi, diago,
                        restr, real, scale=scale, double=double)
        print(model)

        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # print(model.pre.t.dtype)
        # model.scale = torch.tensor(not model.scale, device=device)
        model = model.to(device)
        model.eval()

        InsulatorTest(model, device)
        roc_curve(model, z, double, scale, device)



    # Etrain = 2 * torch.rand(20000) - 1
    # Etest = 2 * torch.rand(4000) - 1
    # torch.save(Etrain, 'datasets/Ins_12_nfz/train/Es.pt')
    # torch.save(Etest, 'datasets/Ins_12_nfz/test/Es.pt')