import torch


L = 12

def obc(x, y, L):
    if x < 0 or x > L - 1:
        x = None
    if y < 0 or y > L - 1:
        y = None
    return x, y


def l2c(x, y, L):
    x, y = obc(x, y, L)
    if (x is None) or (y is None):
        return None
    else:
        return x + y * L


def c2l(n, L):
    assert n < L ** 2
    return n % L, n // L


def Adj(L):
    H = torch.zeros((L ** 2, L ** 2), dtype=torch.complex64)
    for x in range(L):
        for y in range(L):
            n = l2c(x, y, L)
            nx = l2c(x + 1, y, L)
            ny = l2c(x, y + 1, L)
            n1 = l2c(x + 1, y + 1, L)
            n2 = l2c(x + 1, y - 1, L)
            if nx is not None:
                H[nx, n] = H[nx, n] + 1.
                H[n, nx] = H[n, nx] + 1.
            if ny is not None:
                H[ny, n] = H[ny, n] + 1.
                H[n, ny] = H[n, ny] + 1.
            if n1 is not None:
                H[n1, n] = H[n1, n] + 1.
                H[n, n1] = H[n, n1] + 1.
            if n2 is not None:
                H[n2, n] = H[n2, n] + 1.
                H[n, n2] = H[n, n2] + 1.
    return H


if __name__ == "__main__":
    adj = Adj(L) * 1e-2 + torch.diag_embed(torch.cat((torch.rand(8), torch.zeros(8))))
    # save
    torch.save(adj, 'datasets/AdjMat.pt')

