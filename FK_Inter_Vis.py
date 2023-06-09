from FK_DMFT import *
from FK_Data import Ham, Ham2
import numpy as np
from FK_rgfnn import Network
from utils import myceil, mymkdir, LoadFKHamData, LoadFKHamDatawithSE
import matplotlib.pyplot as plt
from torch.nn.functional import softmax, tanh, relu
import sklearn.metrics as metrics
import os, warnings

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, linewidth=80, suppress=True)

threads = 8
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = str(threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
os.environ['MKL_NUM_THREADS'] = str(threads)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
torch.set_num_threads(threads)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(4396)


L = 12  # size = L ** 2
data = f'FK_{L}_QPT'
Net = 'Naive_'
T = 0.005
save = True
show = True

'''construct DMFT'''
count = 20
iota = 0.
momentum = 0.5
momDisor = 0.
maxEpoch = 200
milestone = 30
f_filling = 0.5
d_filling = None
tol_sc = 1e-6
tol_bi = 1e-7
gap = 1


def FKTest(T, model, scf):
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)
    '''construct Hamiltonians'''
    if 'QPT' in data:
        tp = torch.linspace(0., 1.3, 131)
        U = torch.ones(len(tp))
        mu = torch.zeros(len(tp))
        adjMu = torch.cat((torch.linspace(0.5, 0.2, 71), torch.linspace(0.2, 0.5, len(tp) - 71)))
        H0 = torch.stack([Ham(L, i.item(), j.item()) for i, j in zip(mu, tp)], dim=0).unsqueeze(1)
    else:
        U = torch.linspace(1., 4., 150)
        mu = U / 2.
        adjMu = None
        H0 = Ham2(L, mu).unsqueeze(1)
    PTPs = {'0.005': (0.575, 1 / np.sqrt(2)), '0.020': (0.33, 0.392), '0.100': (1.47, 1.76), '0.110': (1.63, 2.01),
            '0.120': (1.8, 2.28), '0.130': (1.96, 2.6), '0.140': (2.14, 3.03), '0.150': (2.34, 3.99),
            '0.160': (2.54, 4), '0.170': (2.74, 4), '0.180': (2.98, 4), '0.190': (3.27, 4), '0.200': (3.57, 4)}
    try:
        PTP, QMCPTP = PTPs[f'{T:.3f}']
    except KeyError:
        PTP, QMCPTP = None, None
    T = T * torch.ones(len(U))

    '''compute self-energy by DMFT'''
    bz = 150
    P = []
    if '2d' in Net:
        mymkdir(f'results/{data}/SE+OP')
        try:
            SEs = torch.load(f'results/{data}/SE+OP/SE_{T[0].item():.3f}.pt')
            OP = torch.load(f'results/{data}/SE+OP/OP_{T[0].item():.3f}.pt')
        except FileNotFoundError:
            SEs, OP = [], []
    else:
        OP = []
    for i in range(myceil(len(U) / bz)):
        H0_batch = H0[i * bz:(i + 1) * bz].to(device)
        T_batch = T[i * bz:(i + 1) * bz].to(device)
        U_batch = U[i * bz:(i + 1) * bz].to(device)
        adjMu_batch = adjMu[i * bz:(i + 1) * bz].to(device) if adjMu is not None else None
        if '2d' in Net:
            if type(SEs) is torch.Tensor:
                SE = SEs[i * bz:(i + 1) * bz].to(device)
            else:
                SE, op = scf(T_batch, H0_batch, U_batch, model=None, adjMu=adjMu_batch, reOP=True,
                             OPfuns=(op_cb, op_str),
                             prinfo=True if i == 0 else False)  # (bz, scf.count, size)
                SEs.append(SE.cpu())
                OP.append(op.cpu())
        else:
            SE, op = scf(T_batch, H0_batch, U_batch, model=model, adjMu=adjMu_batch, reOP=True, OPfuns=(op_cb, op_str),
                         prinfo=True if i == 0 else False)  # (bz, scf.count, size)
            OP.append(op.cpu())
        if Net.startswith('C') or 'sf' in Net:
            SE = SE[:, count:count + 1]  # (bz, 1, size)
            if '2d' in Net or model.z.size(0) != T_batch.shape[0]:
                model.z = T_batch.to(device=device, dtype=scf.iomega0.dtype) * scf.iomega0[0, count]  # (bz,)
            else:
                model.z = model.z[:, count, 0]  # (bz,)
        elif '2d' in Net or model.z.size(0) != T_batch.shape[0]:
            model.z = (T_batch[:, None].to(device=device, dtype=scf.iomega0.dtype) @ scf.iomega0).unsqueeze(
                -1)  # (bz, scf.count, 1)

        '''compute phase diagram'''
        H = H0_batch + torch.diag_embed(SE)
        LDOS = model(H)
        if model.out.size == 1:
            P.append(tanh(relu(LDOS / 2)).data.cpu())
        else:
            P.append(softmax(LDOS, dim=1)[:, 1].data.cpu())
    P = torch.cat(P, dim=0).numpy()
    if type(OP) is list: OP = torch.cat(OP, dim=1)
    if '2d' in Net and type(SEs) is list:
        torch.save(torch.cat(SEs, dim=0), f'results/{data}/SE+OP/SE_{T[0].item():.3f}.pt')
        torch.save(OP, f'results/{data}/SE+OP/OP_{T[0].item():.3f}.pt')
    OP = OP.numpy()
    x = tp.numpy() if 'QPT' in data else U.numpy()

    '''plot phase diagram'''
    labels = ['cb', 'stripe']
    fig, ax1 = plt.subplots()
    plt.axis([x[0], x[-1], 0., 1.])
    if 'QPT' in data:
        plt.title(f'Checkerboard VS Stripe / T={T[0].item():.3f}, L={L}')
        ax1.set_xlabel("t'")
    else:
        plt.title(f'Metal VS Insulator / T={T[0].item():.3f}, L={L}')
        ax1.set_xlabel("U")
    ax1.set_xlim([x[0], x[-1]])
    ax1.set_ylabel('P', c='r')
    ax1.set_ylim([0., 1.])
    ax1.set_yticks(0.1 * np.arange(11))
    ax1.scatter(x, P, s=10, c='r', marker='o')
    ax1.plot([x[0], x[-1]], [0.5, 0.5], 'ko--', linewidth=0.5, markersize=0.1)
    if PTP is not None: ax1.plot([PTP, PTP], [0., 1.], 'go--', linewidth=0.5, markersize=0.1)
    if QMCPTP is not None: ax1.plot([QMCPTP, QMCPTP], [0., 1.], 'yo--', linewidth=0.5, markersize=0.1)
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('OP', c='b')
    ax2.set_ylim([0., 1.])
    ax2.set_yticks(0.1 * np.arange(11))
    for i, op in enumerate(OP):
        ax2.scatter(x, op, s=10, marker='^', label=labels[i])
    ax2.legend(loc='lower right')
    ax2.tick_params(axis='y', labelcolor='b')
    if save:
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/PD_{:.3f}.jpg'.format(path, T[0].item()))
    if show: plt.show()
    plt.close()
    if save: np.save('{}/PD_{:.3f}.npy'.format(path, T[0].item()), np.concatenate((np.stack((x, P), axis=0), OP), axis=0))


def roc_curve(model, scf):
    if save:
        path = 'results/{}'.format(data)
        mymkdir(path)
    dataset = torch.load('datasets/{}/test/dataset.pt'.format(data))
    labels = torch.load('datasets/{}/test/labels.pt'.format(data))
    if Net.endswith('-'): labels = 1 - labels
    if '2d' in Net:
        testdata = LoadFKHamDatawithSE(dataset, labels, torch.load('datasets/{}/train/SE.pt'.format(data)))
    else:
        SEinit = 0.01 * (2. * torch.rand((len(dataset), scf.count, L * L)).type(scf.iomega0.dtype) - 1.)
        testdata = LoadFKHamData(dataset, labels, SEinit)
    loader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False)
    with torch.no_grad():
        y_true, y_pred = [], []
        for pkg in loader:
            y_true.append(pkg[-1][:, -1].long().numpy())
            H0 = pkg[0].to(device, non_blocking=True)
            bz = H0.size(0)
            if '2d' in Net:  # H0: (bz, scf.count, size, size)
                if 'sf' in Net:
                    H0 = H0[:, count:count + 1]  # (bz, 1, size, size)
                    model.z = pkg[1][:, 0].to(device=device, dtype=scf.iomega0.dtype, non_blocking=True) * \
                              scf.iomega0[0, count]  # (bz,)
                else:
                    model.z = (pkg[1][:, :1].to(device=device, dtype=scf.iomega0.dtype, non_blocking=True) @
                               scf.iomega0).unsqueeze(-1)  # (bz, scf.count, 1)
            else:  # H0: (bz, 1, size, size)
                SE = scf(pkg[-1][:, 1].to(device, non_blocking=True), H0,
                         pkg[-1][:, 0].to(device, non_blocking=True), model=model,
                         SEinit=pkg[1].to(device, non_blocking=True))  # (bz, scf.count, size)
                if 'sf' in Net:
                    SE = SE[:, count:count + 1]  # (bz, 1, size, size)
                    if model.z.size(0) != bz:
                        model.z = pkg[-1][:, 1].to(device=device, dtype=scf.iomega0.dtype, non_blocking=True) * \
                                  scf.iomega0[0, count]  # (bz,)
                    else:
                        model.z = model.z[:, count, 0]  # (bz,)
                elif model.z.size(0) != bz:
                    model.z = (pkg[-1][:, 1:2].to(device=device, dtype=scf.iomega0.dtype, non_blocking=True) @
                               scf.iomega0).unsqueeze(-1)  # (bz, scf.count, 1)
                H0 = H0 + torch.diag_embed(SE)  # (bz, scf.count, size, size)

            # compute output
            output = model(H0)
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
        path = '{}/{}'.format(path, Net)
        mymkdir(path)
        plt.savefig('{}/ROC_curve.jpg'.format(path))
    if show: plt.show()
    plt.close()



if __name__ == "__main__":
    scf = DMFT(count, iota, momentum, momDisor, maxEpoch, milestone, f_filling, d_filling, tol_sc, tol_bi, gap, device)

    '''construct FQNN'''
    model_path = 'models/{}/{}'.format(data, Net)
    model = Network(Net[:Net.index('_')], L ** 2, 1 if Net.startswith('C') else 2, 100, 64, None, double=True)
    checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path), map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    roc_curve(model, scf)

    FKTest(T, model, scf)