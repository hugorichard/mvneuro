from multiviewica import multiviewica
import numpy as np
from mvneuro.base import BaseMultiView


class MultiViewICA(BaseMultiView):
    """
    Performs MultiViewICA.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    dimension_reduction: str, optional
        if srm: use srm to reduce the data
        if pca: use group specific pca to reduce the data
    noise : float, optional
        Gaussian noise level
    max_iter : int, optional
        Maximum number of iterations to perform
    init : str or np array of shape (n_groups, n_components, n_components)
        If permica: initialize with perm ICA, if groupica, initialize with
        group ica. Else, use the provided array to initialize.
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    verbose : bool, optional
        Print information

    Attributes
    -------
    basis_list : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
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
