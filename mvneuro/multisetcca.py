import numpy as np
from qndiag import qndiag
from sklearn.utils.extmath import randomized_svd


def mcca(
    X_list, n_components=None,
):
    """

    Parameters
    ----------
    X_list : ndarray of shape (m, k, n)
        input data

    max_iter: int
        Maximum number of iterations to perform

    tol: float
        Tolerance. The algorithm stops when the loss
        decrease is below this value.

    verbose : bool
        If True, prints information about convergence

    Returns
    -------
    W_list : ndarray of shape (m, k, k)
        Unmixing matrices
    """
    m, k, n = X_list.shape
    if n_components is None:
        n_components = k
    if n_components > k:
        raise ValueError(
            "Number of components %i is larger than number of features %i"(
                n_components, k
            )
        )
    if n_components > n:
        raise ValueError(
            "Number of components %i is larger than number of samples %i"
            % (n_components, n)
        )
    Xw = []
    Us = []
    for i in range(m):
        X = X_list[i]
        U, S, V = randomized_svd(X, n_components=k)
        Xw.append(V)
        Us.append(U * S)

    Xw = np.vstack(Xw)
    U, S, V = randomized_svd(Xw, n_components=n_components)
    A = np.array(np.split(U, m))
    W_list = [A[i].T.dot(np.linalg.pinv(Us[i])) for i in range(m)]
    W_list = np.array(W_list)

    Ds = []
    for i in range(m):
        Yi = W_list[i].dot(X_list[i])
        Di = Yi.dot(Yi.T)
        Ds.append(Di)

    B, _ = qndiag(np.array(Ds))
    W_list = np.array([B.dot(w) for w in W_list])
    S_sum = np.sum([w.dot(x) for w, x in zip(W_list, X_list)], axis=0)

    norm = np.mean(
        [
            np.std(W_list[i].dot(X_list[i]), axis=1, keepdims=True)
            for i in range(m)
        ],
        axis=0,
    )
    for i in range(m):
        W_list[i] = W_list[i] / norm
    return W_list, S_sum


def mcca_add_subject(X, S_sum):
    p, n = X.shape
    U, S, V = np.linalg.svd(X, full_matrices=False)
    A = S_sum.dot(V.T).T
    w = np.linalg.pinv(U * S).T.dot(A)
    w = w.T
    w = w / np.std(w.dot(X), axis=1, keepdims=True)
    return w
