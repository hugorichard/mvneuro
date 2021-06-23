import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from qndiag import qndiag
from sklearn.utils.extmath import randomized_svd


def mcca(X_list, n_components=None, use_jointdiag=True):
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
    basis_list : ndarray of shape (m, k, k)
        Unmixing matrices
    S_sum : ndarray of shape (k, n)
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
    basis_list = [A[i].T.dot(np.linalg.pinv(Us[i])) for i in range(m)]
    basis_list = np.array(basis_list)

    if use_jointdiag:
        Ds = []
        for i in range(m):
            Yi = basis_list[i].dot(X_list[i])
            Di = Yi.dot(Yi.T)
            Ds.append(Di)

        B, _ = qndiag(np.array(Ds))
        basis_list = np.array([B.dot(w) for w in basis_list])

    norm = np.mean(
        [
            np.std(basis_list[i].dot(X_list[i]), axis=1, keepdims=True)
            * k
            / n_components
            for i in range(m)
        ],
        axis=0,
    )
    for i in range(m):
        basis_list[i] = basis_list[i] / norm
    S_sum = np.sum([w.dot(x) for w, x in zip(basis_list, X_list)], axis=0)
    return basis_list, S_sum


def mcca_add_subject(X, S_sum):
    p, n = X.shape
    k, _ = S_sum.shape
    U, S, V = np.linalg.svd(X, full_matrices=False)
    A = S_sum.dot(V.T).T
    w = np.linalg.pinv(U * S).T.dot(A)
    w = w.T
    w = w / (np.std(w.dot(X), axis=1, keepdims=True) * p / k)
    return w


class MCCA(BaseEstimator, TransformerMixin):
    """
    Solving 

    X_list : list, length = n_pb;
            each element is an array, shape= (p, n)
            n_pb : number of problems (or languages)
            p : number of sources
            n : number of samples
    n_iter : number of iterations of the outer loop
    noise: float
        Positive float (noise level)
    """

    def __init__(self, n_components=None, use_jointdiag=False):
        self.n_components = n_components
        self.use_jointdiag = use_jointdiag

    def fit(self, X, y=None):
        """
        Fits the model
        Parameters
        ----------
        X: list of np arrays of shape (n_voxels, n_samples)
            Input data: X[i] is the data of subject i

        Attributes
        basis_list: list
            basis_list[i] is the basis of subject i
            X[i] = basis_list[i].dot(shared_response)
            Only available if temp_dir is None
        """
        if self.n_components is None:
            self.n_components = min(X[0].shape[0], X[0].shape[1])

        W, S_sum = mcca(
            np.array(X), self.n_components, use_jointdiag=self.use_jointdiag
        )
        self.basis_list = W
        self.S_sum = S_sum
        return self

    def add_subjects(self, X_list, S):
        """
        Add subjects to a fitted model
        """
        basis_list = np.array(
            [w for w in self.basis_list]
            + [mcca_add_subject(X, self.S_sum) for X in X_list]
        )
        self.basis_list = basis_list

    def transform(self, X, subjects_indexes=None):
        """
        Fits the model
        Parameters
        ----------
        X: list of np arrays of shape (n_voxels, n_samples)
        """
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.basis_list))

        return [
            self.basis_list[k].dot(X[i])
            for i, k in enumerate(subjects_indexes)
        ]

    def inverse_transform(self, S, subjects_indexes=None):
        """
        Data from shared response
        """
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.basis_list))

        return [self.basis_list[i].dot(S) for i in subjects_indexes]
