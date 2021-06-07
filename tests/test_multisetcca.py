import numpy as np
import matplotlib.pyplot as plt
from mvneuro.multisetcca import mcca, mcca_add_subject


def amari_d(W=None, A=None, P=None):
    if P is None:
        P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


m, p, n = 10, 2000, 100
S = np.random.randn(p, n)
A = np.random.randn(m, p, p)
N = np.random.randn(m, p, n)
powers = 0.1 * np.random.rand(m, p)
X = np.array([a.dot(S + p[:, None] * n) for p, a, n in zip(powers, A, N)])


def test_mcca_amari():
    np.random.seed(0)
    m, p, n = 10, 5, 100
    S = np.random.randn(p, n)
    A = np.random.randn(m, p, p)
    N = np.random.randn(m, p, n)
    powers = np.random.rand(m, p)
    X = np.array([a.dot(S + p[:, None] * n) for p, a, n in zip(powers, A, N)])
    n_components = 3
    W_pred, S_sum = mcca(X[1:], n_components=n_components)
    W = mcca_add_subject(X[0], S_sum)
    f, axes = plt.subplots(len(W_pred) + 1)
    for i, (w_pred, a) in enumerate(zip(W_pred, A[1:])):
        Ri = w_pred.dot(a)
        I = np.argsort(np.max(np.abs(Ri), axis=0))[-n_components:]
        axes[i].imshow(Ri)
        # print(amari_d(P=Ri[:, I]))
        assert amari_d(P=Ri[:, I]) < 0.1

    Ri = W.dot(A[0])
    I = np.argsort(np.max(np.abs(Ri), axis=0))[-n_components:]
    axes[i + 1].imshow(Ri)
    # print(amari_d(P=Ri[:, I]))
    # plt.show()
    assert amari_d(P=Ri[:, I]) < 0.1


def test_mcca_closeness():
    n_components = 50
    W_pred, S_sum = mcca(X[1:, :, : int(0.9 * n)], n_components=n_components)
    W = mcca_add_subject(X[0, :, : int(0.9 * n)], S_sum)
    W = np.array([W] + [w for w in W_pred])
    S = []
    for i in range(m):
        S.append(W[i].dot(X[i, :, int(0.9 * n) :]))

    for i in range(m):
        Si = S[i]
        Si = Si - np.mean(Si, axis=1, keepdims=True)
        Si = Si / np.linalg.norm(Si, axis=1, keepdims=True)
        S[i] = Si

    f, axes = plt.subplots(m)
    corrs = []
    for i in range(m):
        corrs.append(np.sum((S[i] * S[0])) / n_components)
        axes[i].plot(S[i].T)
    print(np.mean(corrs))
    assert np.mean(corrs) > 0.4
    # plt.show()
