from sklearn.base import BaseEstimator, TransformerMixin
from fastsrm.identifiable_srm import IdentifiableFastSRM
import numpy as np


class NoPreproc(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        return self

    def transform(self, X, y=None, subjects_indexes=None):
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.basis_list))
        return np.array([X[i] for i in range(len(subjects_indexes))])

    def inverse_transform(self, X, subjects_indexes=None):
        return np.array([X for _ in subjects_indexes])

    def add_subjects(self, X_list, S):
        self.basis_list = [w for w in self.basis_list] + [
            np.eye(x.shape[0]) for x in X_list
        ]


class BaseMultiView(BaseEstimator, TransformerMixin):
    """
    Main class for the noise linear rosetta stone problem using ICA

    X_list : list, length = n_pb;
            each element is an array, shape= (p, n)
            n_pb : number of problems (or languages)
            p : number of sources
            n : number of samples
    n_iter : number of iterations of the outer loop
    noise: float
        Positive float (noise level)
    """

    def __init__(
        self,
        verbose=False,
        n_components=None,
        reduction="srm",
        memory=None,
        random_state=0,
    ):
        self.verbose = verbose
        self.n_components = n_components
        self.reduction = reduction
        self.memory = memory
        self.random_state = random_state

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
            X[i} = basis_list[i].dot(shared_response)
        """
        if self.n_components is None:
            self.n_components = X[0].shape[0]

        if self.reduction is None:
            self.preproc = NoPreproc()

        if self.reduction == "srm":
            self.preproc = IdentifiableFastSRM(
                n_iter_reduced=10000,
                n_components=self.n_components,
                memory=self.memory,
                tol=1e-12,
                identifiability="decorr",
                aggregate=None,
                verbose=self.verbose,
                random_state=self.random_state,
            )

        reduced_X = self.preproc.fit_transform(X)
        if len(reduced_X[0][0].shape) > 1:
            # We are in this setting where SRM is fit on
            # [[X_ij]] list (subject i session j)
            reduced_X = [
                np.concatenate(reduced_x, axis=1) for reduced_x in reduced_X
            ]
        reduced_X = np.array(reduced_X)
        res_fit = self._fit(reduced_X)
        W_list, Y_avg = res_fit

        self.y_avg = Y_avg
        self.W_list = W_list
        self.basis_list = [
            np.linalg.inv(W_list[i]).T.dot(self.preproc.basis_list[i])
            for i in range(len(W_list))
        ]
        return self

    def add_subjects(self, X_list, S):
        """
        Add subjects to a fitted model
        """
        n_basis = len(self.basis_list)
        self.preproc.add_subjects(X_list, S)
        X_transform = self.preproc.transform(
            X_list, subjects_indexes=np.arange(n_basis, n_basis + len(X_list))
        )
        X_transform = np.array(X_transform)
        W_list = self._add_subjects(X_transform, S)
        self.W_list = [w for w in self.W_list] + [w for w in W_list]
        basis_list = [
            np.linalg.inv(W_list[i]).T.dot(
                self.preproc.basis_list[n_basis + i]
            )
            for i in range(len(W_list))
        ]
        self.basis_list = self.basis_list + basis_list

    def transform(self, X, subjects_indexes=None):
        """
        Fits the model
        Parameters
        ----------
        X: list of np arrays of shape (n_voxels, n_samples)
        """
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.W_list))

        transformed_X = self.preproc.transform(
            X, subjects_indexes=subjects_indexes
        )
        if len(transformed_X[0][0].shape) > 1:
            # We are in this setting where SRM is fit on
            # [[X_ij]] list (subject i session j)
            transformed_X = [
                np.concatenate(transformed_x, axis=1)
                for transformed_x in transformed_X
            ]

        return [
            self.W_list[k].dot(transformed_X[i])
            for i, k in enumerate(subjects_indexes)
        ]

    def inverse_transform(self, S, subjects_indexes=None):
        """
        Data from shared response
        """
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.W_list))

        return [
            self.preproc.inverse_transform(
                np.linalg.inv(self.W_list[i]).dot(S), subjects_indexes=[i],
            )[0]
            for i in subjects_indexes
        ]
