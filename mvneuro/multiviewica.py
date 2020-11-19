from multiviewica import multiviewica
import numpy as np
from mvneuro.base import BaseMultiView


class MultiViewICA(BaseMultiView):
    """
    This solves the MVICA problem
    """

    def __init__(
        self,
        n_iter=100,
        noise=1.0,
        tol=1e-6,
        verbose=False,
        n_components=None,
        reduction="srm",
        memory=None,
        random_state=0,
        init="permica",
    ):
        super().__init__(
            verbose, n_components, reduction, memory, random_state
        )
        self.noise = noise
        self.tol = tol
        self.init = init
        self.n_iter = n_iter

    def _fit(self, reduced_X):
        K, W, Y = multiviewica(
            reduced_X,
            noise=self.noise,
            max_iter=self.n_iter,
            init=self.init,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
        )
        return W, Y

    def _add_subjects(self, reduced_X, S):
        W_init = [
            (S.dot(x.T)).dot(np.linalg.inv(x.dot(x.T))) for x in reduced_X
        ]
        return W_init
