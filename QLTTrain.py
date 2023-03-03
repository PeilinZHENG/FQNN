import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import compute_grads, AverageMeter
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(0)

Net = 'Sig_0'
train = False
batchsz = 256
order = 2


class Model_relu(nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=64, output_size=2):
        super(Model_relu, self).__init__()
        self.pre = nn.Linear(input_size, embedding_size)
        self.emb = nn.Linear(embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.pre.apply(self.weights_init)
        self.emb.apply(self.weights_init)
        self.fc1.apply(self.weights_init)
        self.fc2.apply(self.weights_init)
        self.fc3.apply(self.weights_init)
        self.fc4.apply(self.weights_init)
        self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.pre(x))
        x = F.relu(self.emb(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)


class Model_sig(nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=64, output_size=2):
        super(Model_sig, self).__init__()
        self.pre = nn.Linear(input_size, embedding_size)
        self.emb = nn.Linear(embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.pre.apply(self.weights_init)
        self.emb.apply(self.weights_init)
        self.fc1.apply(self.weights_init)
        self.fc2.apply(self.weights_init)
        self.fc3.apply(self.weights_init)
        self.fc4.apply(self.weights_init)
        self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.sigmoid(self.pre(x))
        x = F.sigmoid(self.emb(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return self.out(x)


def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.
    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


def load_data_wrapper(tr_d, tr_r, te_d, te_r):
    training_data = list(zip(tr_d, [vectorized_result(y) for y in tr_r]))
    test_data = list(zip(te_d, te_r))
    return (training_data, test_data)


class Loadtrain(Dataset):
    def __init__(self, dataload):
        self.dataload = dataload

    def __getitem__(self, index):
        g = self.dataload[index]
        label = torch.tensor(np.argmax(g[-1]), dtype=torch.long)
        g = torch.from_numpy(np.squeeze(g[0])).float()
        return g, label

    def __len__(self):
        return len(self.dataload)


class Loadtest(Dataset):
    def __init__(self, dataload):
        self.dataload = dataload

    def __getitem__(self, index):
        g = self.dataload[index]
        label = torch.tensor(g[-1], dtype=torch.long)
        g = torch.from_numpy(np.squeeze(g[0])).float()
        return g, label

    def __len__(self):
        return len(self.dataload)


def generate_insulator_data(count, qlt, kappa=1.0, delta=0.25, dmax=1, L=12, autocorr=4*12*12):
    """Generate a specific number of samples of the Chern/normal model with model parameter delta = 0.5 and kappa.
    The default kappa is at the critical kappa_c. dmax controls the cut-off length scale of the operators. qlt is a list of local triangles."""
    N = int(L*L/2)
    training_data = []
    conf = -np.ones((L, L), dtype=int)
    index = np.zeros((L, L), dtype=int)
    corr=np.zeros((L,L,2*dmax+1,2*dmax+1,3),dtype=complex)
    sample = np.zeros((L,L,len(qlt)),dtype=float)
    for x in range(L):
        for y in range(L):
            if (x-y)%2 == 0:
                conf[x,y]*=-1
            index[x,y] = int((y*L+x)/2)
            """ Initial configuration. Checkerboard occupation of electrons, 1 for occupied and -1 for unoccupied.
            Index keep track the row number of the (x, y) fermion in the Slater determinant (and its supplement). """
    SD = np.zeros((N, N), dtype=complex)
    SD2= np.zeros((N, N), dtype=complex)
    """ Initialize the Slater determinant (for the occupied sites) and its supplement (for the unoccupied sites).
    Changing the configuration will switch the corresponding rows of the Slater determinant and its supplement."""
    for kx in range(L):
        for ky in range(int(L/2)):
            energy = math.sqrt(math.cos(kx*2*np.pi/L)**2 + math.cos(ky*2*np.pi/L)**2 + (2*delta*math.sin(ky*2*np.pi/L)*(1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
            for x in range(L):
                for y in range(L):
                    if conf[x,y] == 1:
                        """Occupied site, go to Slater determinant."""
                        if (y%2 == 0):
                            """First sublattice."""
                            SD[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) *  \
                            complex(math.cos(ky*2*np.pi/L), -2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L))) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                        else:
                            """Second sublattice."""
                            SD[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) * \
                            (math.cos(kx*2*np.pi/L)+energy) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                    else:
                        """Unoccupied site, go to supplement."""
                        if (y%2 == 0):
                            """First sublattice."""
                            SD2[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) *  \
                            complex(math.cos(ky*2*np.pi/L), -2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L))) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
                        else:
                            """Second sublattice."""
                            SD2[index[x,y], ky*L+kx] = complex(math.cos((kx*x+ky*y)*2*np.pi/L), math.sin((kx*x+ky*y)*2*np.pi/L)) * \
                            (math.cos(kx*2*np.pi/L)+energy) / \
                            math.sqrt((math.cos(kx*2*np.pi/L)+energy)**2 + math.cos(ky*2*np.pi/L) ** 2 + (2*delta*math.sin(ky*2*np.pi/L) * (1-kappa+kappa*math.sin(kx*2*np.pi/L)))**2)
    SDinv = np.linalg.inv(SD)
    #print(np.amax(abs(np.dot(SD, SDinv)-np.diag(np.ones(N)))))
    """Also keep the inverse of Slater determinant. Initialization complete."""
    for icount in range(count*30+10):
        for iautocorr in range(autocorr):
            """ A number of steps between measurements to ensure (approximately) independent samples."""
            while True:
                x, y = np.random.randint(L, size=2)
                if conf[x,y] == 1:
                    break
            while True:
                xp, yp = np.random.randint(L, size=2)
                if conf[xp,yp] == -1:
                    break
            if (np.random.rand()<abs(1+np.dot(SD2[index[xp,yp],:]-SD[index[x,y],:],SDinv[:, index[x,y]]))**2):
                """Calculate the update probability. Accept and update configuration."""
                conf[x,y] *=-1
                conf[xp,yp] *=-1
                """Update matrix inverse and matrices."""
                SDinv[:,index[x,y]] /= np.dot(SD2[index[xp,yp],:],SDinv[:,index[x,y]])
                tempnp = np.reshape(np.dot(SD2[index[xp,yp],:], SDinv),N)
                for i in range(N):
                    if i != index[x,y]:
                        SDinv[:,i]-=SDinv[:,index[x,y]]*tempnp[i]
                tempnp = SD[index[x,y],:]
                SD[index[x,y],:] = SD2[index[xp,yp],:]
                SD2[index[xp,yp],:] = tempnp
                tempi = index[x,y]
                index[x,y] = index[xp,yp]
                index[xp,yp] = tempi
        if(icount > 9):
            for x in range(L):
                for y in range(L):
                    for dx in range(2*dmax+1):
                        for dy in range(2*dmax+1):
                            xp = modpbc(x+dx-dmax,L)
                            yp = modpbc(y+dy-dmax,L)
                            if(conf[x,y]==1 and conf[xp,yp]==-1):
                                corr[x,y,dx,dy,(icount-10)%3] = 1+np.dot(SD2[index[xp,yp],:]-SD[index[x,y],:],SDinv[:,index[x,y]])
                            else:
                                corr[x,y,dx,dy,(icount-10)%3] = 0
            if ((icount-10)%3 == 2):
                for x in range(L):
                    for y in range(L):
                        for iqlt in range(len(qlt)):
                            xp = modpbc(x+qlt[iqlt][0],L)
                            yp = modpbc(y+qlt[iqlt][1],L)
                            xpp = modpbc(xp+qlt[iqlt][2],L)
                            ypp = modpbc(yp+qlt[iqlt][3],L)
                            sample[x,y,iqlt]+=(corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,0] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,1]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 2]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,1] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,2]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 0]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,2] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,0]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 1]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,0] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,2]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 1]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,2] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,1]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 0]).imag \
                                            + (corr[x, y, qlt[iqlt][0]+dmax, qlt[iqlt][1]+dmax,1] * corr[xp, yp, qlt[iqlt][2]+dmax, qlt[iqlt][3]+dmax,0]+corr[xpp, ypp, dmax-qlt[iqlt][0]-qlt[iqlt][2], dmax-qlt[iqlt][1]-qlt[iqlt][3], 2]).imag
            if ((icount-10)%30 == 29):
                training_data.append(np.reshape(sample, (L**2*len(qlt), 1)))
                sample = np.zeros((L,L,len(qlt)),dtype=float)
                #print(np.amax(abs(np.dot(SD, SDinv)-np.diag(np.ones(N)))))
    return training_data


def modpbc(x, L=28):
    """Ensure the coordinate is between [0, L-1] by applying periodic boundary condition. """
    if (x<0):
        return modpbc(x+L, L)
    elif (x>L-1):
        return modpbc(x-L, L)
    else:
        return x


def countriangle(dmax):
    """The list of triangles within a cut-off scale dmax."""
    qlt = []
    for dx in range(-dmax, dmax+1):
        for dy in range(dmax+1):
            for dx2 in range(-dmax, dmax+1):
                for dy2 in range(-dy, dmax+1):
                    if(abs(dx+dx2) <= dmax and abs(dy+dy2) <= dmax and (dy2*dx-dy*dx2)>0 and (dy>0 or dx>0) and (dx+dx2>0 or dy+dy2>0)):
                        qlt.append([dx, dy, dx2, dy2])
    return qlt


def plot_fig(x, p0, p1, xlabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, p0, '-b', label='p0')
    ax2 = ax.twinx()
    ax2.plot(x, p1, '-r', label='p1')
    ax.legend(loc=6)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("p0")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("p1")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc=7)
    plt.title(title)
    plt.savefig("{}.png".format(title))


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
    plt.savefig("models/QLT/{}/{}.jpg".format(Net, title))
    plt.show()
    plt.close()


def plotbargrads(x, title):
    plt.figure()
    plt.xlim(0, len(x) + 1)
    plt.bar(np.arange(1, len(x) + 1), x, width=0.8)
    plt.ylabel('Gradient')
    plt.xlabel('Layer')
    plt.title(title)
    plt.savefig("models/QLT/{}/{}.jpg".format(Net, title))
    plt.show()
    plt.close()


def accuracy(dataloader, model):
    model.eval()
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in dataloader:
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy: %.5f %%' % (100.0 * sum(class_correct) / sum(class_total)))
    for i in range(2):
        print('Accuracy of %5s : %.5f %%' % (classes[i], 100.0 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('Normal', 'Chern')
    if not os.path.exists("models/QLT/{}".format(Net)):
        os.mkdir("models/QLT/{}".format(Net))
    if train:
        if os.path.exists("datasets/Insulator_dataset.npy"):
            training_data, val_data = np.load("datasets/Insulator_dataset.npy", allow_pickle=True)
            print("Datasets loading finished!")
        else:
            train_count, test_count = 10000, 1000
            tr_d, tr_r, te_d, te_r = [], [], [], []
            tr_d += generate_insulator_data(int(train_count / 2), countriangle(1), kappa=0.1)
            tr_r += int(train_count / 2) * [0]
            tr_d += generate_insulator_data(int(train_count / 2), countriangle(1), kappa=1.0)
            tr_r += int(train_count / 2) * [1]
            te_d += generate_insulator_data(int(test_count / 2), countriangle(1), kappa=0.1)
            te_r += int(test_count / 2) * [0]
            te_d += generate_insulator_data(int(test_count / 2), countriangle(1), kappa=1.0)
            te_r += int(test_count / 2) * [1]
            training_data, val_data = load_data_wrapper(tr_d, tr_r, te_d, te_r)
            np.save("datasets/Insulator_dataset.npy", (training_data, val_data))
            print("Datasets generation finished!")
        trainset, valset = Loadtrain(training_data), Loadtest(val_data)
        trainloader = DataLoader(trainset, batch_size=batchsz, shuffle=True, num_workers=0)
        valloader = DataLoader(valset, batch_size=batchsz, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss().to(device)

        if Net.startswith('Relu'):
            model = Model_relu(training_data[0][0].shape[0], 100, 64, 2).to(device)
        else:
            model = Model_sig(training_data[0][0].shape[0], 100, 64, 2).to(device)
        print(len(trainset), len(trainloader), len(valset), len(valloader), '\n', model)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)

        Grads, Agrads, Grads1, Agrads1, Grads2, Agrads2, Grads_1, Agrads_1 = [], [],[], [], [], [], [], []
        Grads3, Agrads3, Grads4, Agrads4, Grads5, Agrads5, Grads6, Agrads6 = [], [], [], [], [], [], [], []

        best = 1e5
        for epoch in range(21):
            # train
            grads = AverageMeter('Grad', ':.4e')
            agrads = AverageMeter('AGrad', ':.4e')
            grads1 = AverageMeter('Grad1', ':.4e')
            agrads1 = AverageMeter('AGrad1', ':.4e')
            grads2 = AverageMeter('Grad2', ':.4e')
            agrads2 = AverageMeter('AGrad2', ':.4e')
            grads3 = AverageMeter('Grad3', ':.4e')
            agrads3 = AverageMeter('AGrad3', ':.4e')
            grads4 = AverageMeter('Grad4', ':.4e')
            agrads4 = AverageMeter('AGrad4', ':.4e')
            grads5 = AverageMeter('Grad5', ':.4e')
            agrads5 = AverageMeter('AGrad5', ':.4e')
            grads6 = AverageMeter('Grad6', ':.4e')
            agrads6 = AverageMeter('AGrad6', ':.4e')
            grads_1 = AverageMeter('Grad-1', ':.4e')
            agrads_1 = AverageMeter('AGrad-1', ':.4e')
            model.train()
            running_loss = 0.
            for batchidx, (x, label) in enumerate(trainloader):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_norm, avg_norm = compute_grads(model, order)
                grads.update(total_norm, x.size(0))
                agrads.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.pre, order)
                grads1.update(total_norm, x.size(0))
                agrads1.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.emb, order)
                grads2.update(total_norm, x.size(0))
                agrads2.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.fc1, order)
                grads3.update(total_norm, x.size(0))
                agrads3.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.fc2, order)
                grads4.update(total_norm, x.size(0))
                agrads4.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.fc3, order)
                grads5.update(total_norm, x.size(0))
                agrads5.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.fc4, order)
                grads6.update(total_norm, x.size(0))
                agrads6.update(avg_norm, x.size(0))
                total_norm, avg_norm = compute_grads(model.out, order)
                grads_1.update(total_norm, x.size(0))
                agrads_1.update(avg_norm, x.size(0))
                running_loss += loss.item() * len(label)
            scheduler.step()
            epoch_loss = running_loss / len(trainset)
            Grads.append(float('{:.4e}'.format(grads.avg)))
            Agrads.append(float('{:.4e}'.format(agrads.avg)))
            Grads1.append(float('{:.4e}'.format(grads1.avg)))
            Agrads1.append(float('{:.4e}'.format(agrads1.avg)))
            Grads2.append(float('{:.4e}'.format(grads2.avg)))
            Agrads2.append(float('{:.4e}'.format(agrads2.avg)))
            Grads3.append(float('{:.4e}'.format(grads3.avg)))
            Agrads3.append(float('{:.4e}'.format(agrads3.avg)))
            Grads4.append(float('{:.4e}'.format(grads4.avg)))
            Agrads4.append(float('{:.4e}'.format(agrads4.avg)))
            Grads5.append(float('{:.4e}'.format(grads5.avg)))
            Agrads5.append(float('{:.4e}'.format(agrads5.avg)))
            Grads6.append(float('{:.4e}'.format(grads6.avg)))
            Agrads6.append(float('{:.4e}'.format(agrads6.avg)))
            Grads_1.append(float('{:.4e}'.format(grads_1.avg)))
            Agrads_1.append(float('{:.4e}'.format(agrads_1.avg)))

            # validation
            model.eval()
            with torch.no_grad():
                running_loss = 0.
                for batchidx, (x, label) in enumerate(valloader):
                    x, label = x.to(device), label.to(device)
                    logits = model(x)
                    loss = criterion(logits, label)
                    running_loss += loss.item() * len(label)
                val_loss = running_loss / len(valset)
            print(epoch, 'epoch loss:', epoch_loss, 'val loss:', val_loss)

            # accuarcy
            print("Train accuarcy:")
            accuracy(trainloader, model)
            print("Validation accuarcy:")
            accuracy(valloader, model)

            # save model
            if best > val_loss:
                best = val_loss
                torch.save(model, "models/QLT/{}/model.pkl".format(Net))
        print('Finished Training')

        grads, agrads = np.array(Grads), np.array(Agrads)
        grads1, agrads1 = np.array(Grads1), np.array(Agrads1)
        grads2, agrads2 = np.array(Grads2), np.array(Agrads2)
        grads3, agrads3 = np.array(Grads3), np.array(Agrads3)
        grads4, agrads4 = np.array(Grads4), np.array(Agrads4)
        grads5, agrads5 = np.array(Grads5), np.array(Agrads5)
        grads6, agrads6 = np.array(Grads6), np.array(Agrads6)
        grads_1, agrads_1 = np.array(Grads_1), np.array(Agrads_1)
        Grads = np.stack((grads1, grads2, grads3, grads4, grads5, grads6, grads_1), axis=0)
        Agrads = np.stack((agrads1, agrads2, agrads3, agrads4, agrads5, agrads6, agrads_1), axis=0)
        np.save("models/QLT/{}/grads.npy".format(Net), grads)
        np.save("models/QLT/{}/agrads.npy".format(Net), agrads)
        np.save("models/QLT/{}/grads_layers.npy".format(Net), Grads)
        np.save("models/QLT/{}/agrads_layers.npy".format(Net), Agrads)
        plotgrads(Grads, 'QLT_Gradients', grads)
        plotgrads(Agrads, 'QLT_Average Gradients', agrads)
        Grads = np.mean(Grads[:, 1:2], axis=1)
        NGrads = Grads / np.max(Grads)
        Agrads = np.mean(Agrads[:, 1:2], axis=1)
        NAgrads = Agrads / np.max(Agrads)
        plotbargrads(NGrads, 'QLT_Normalized Gradients ({})'.format(Net[:Net.index('_')]))
        plotbargrads(NAgrads, 'QLT_Normalized Average Gradients ({})'.format(Net[:Net.index('_')]))
    else:
        # model = torch.load("models/QLT/{}/model.pkl".format(Net), map_location=device)
        if os.path.exists("models/QLT/{}/grads.npy".format(Net)):
            grads = np.load("models/QLT/{}/grads.npy".format(Net))
            agrads = np.load("models/QLT/{}/agrads.npy".format(Net))
            Grads = np.load("models/QLT/{}/grads_layers.npy".format(Net))
            Agrads = np.load("models/QLT/{}/agrads_layers.npy".format(Net))
            plotgrads(Grads, 'QLT_Gradients', grads)
            plotgrads(Agrads, 'QLT_Average Gradients', agrads)
            Grads = np.mean(Grads[:, :3], axis=1)
            NGrads = Grads / np.max(Grads)
            Agrads = np.mean(Agrads[:, :3], axis=1)
            NAgrads = Agrads / np.max(Agrads)
            NGrads, NAgrads = NGrads[:-1], NAgrads[:-1]
            plotbargrads(NGrads, 'QLT_Normalized Gradients ({})'.format(Net[:Net.index('_')]))
            plotbargrads(NAgrads, 'QLT_Normalized Average Gradients ({})'.format(Net[:Net.index('_')]))

    # # plot phase diagram
    # model.eval()
    # Kap, p0, p1, group = np.arange(0.1, 1.01, 0.05), [], [], 10
    # for k in Kap:
    #     samples = torch.from_numpy(
    #         np.concatenate(generate_insulator_data(group, countriangle(1), kappa=k, dmax=1), axis=1).T).float().to(
    #         device)
    #     output = F.softmax(model(samples), dim=1)
    #     p0.append(torch.mean(output[:, 0]).item())
    #     p1.append(torch.mean(output[:, 1]).item())
    # plot_fig(Kap, p0, p1, "Kappa", "Normal VS Topological")


