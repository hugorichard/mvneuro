import os
import pytest
import numpy as np
from mvneuro.multiviewica import MultiViewICA
import tempfile


def generate_data(
    n_voxels, n_supervoxels, n_timeframes, n_components, n_subjects, datadir,
):
    """
    Generate data without noise
    Returns
    ------
    W, Ss, Xs, np.array(paths)
    W, Ss, Xs
    """
    n_sessions = len(n_timeframes)
    bigger_mix = [
        np.linalg.svd(
            np.random.rand(n_voxels, n_supervoxels), full_matrices=False,
        )[0]
        for _ in range(n_subjects)
    ]
    W = [
        np.random.rand(n_supervoxels, n_components) for _ in range(n_subjects)
    ]
    Ss = []
    for j in range(n_sessions):
        Sj = np.random.laplace(size=(n_components, n_timeframes[j]))
        Sj = Sj - np.mean(Sj, axis=1, keepdims=True)
        Ss.append(Sj)

    Xs = []
    paths = []
    for subject in range(n_subjects):
        sessions_path = []
        Xs_ = []
        for session in range(n_sessions):
            pth = "%i_%i" % (subject, session)
            X__ = bigger_mix[subject].dot(W[subject]).dot(Ss[session])
            Xs_.append(X__)

            if datadir is not None:
                np.save(os.path.join(datadir, pth), X__)
                sessions_path.append(os.path.join(datadir, pth + ".npy"))
        if datadir is not None:
            paths.append(sessions_path)
        Xs.append(Xs_)
    if datadir is not None:
        return W, Ss, Xs, np.array(paths)
    else:
        return W, Ss, Xs


def test_fit_transform_mvica():
    """
    Check that we recover same subject specific sources
    """
    n_voxels = 10
    n_timeframes = [20, 20]
    n_subjects = 5
    n_components = 2

    W, Ss, Xs = generate_data(
        n_voxels, n_components, n_timeframes, n_components, n_subjects, None
    )
    S = np.concatenate(Ss, axis=1)
    X = [np.concatenate(np.array(x), axis=1) for x in Xs]

    ica = MultiViewICA(
        reduction="srm", noise=1, n_components=n_components, n_iter=100
    )
    shared = ica.fit_transform(X)

    for i in range(len(shared)):
        np.testing.assert_almost_equal(shared[0], shared[i], 2)


def test_fit_transform_mvica():
    """
    Check that we recover same subject specific sources
    """
    n_voxels = 10
    n_timeframes = [20, 20]
    n_subjects = 5
    n_components = 2

    W, Ss, Xs = generate_data(
        n_voxels, n_components, n_timeframes, n_components, n_subjects, None
    )
    S = np.concatenate(Ss, axis=1)
    X = [np.concatenate(np.array(x), axis=1) for x in Xs]

    ica = MultiViewICA(
        reduction="srm", noise=1, n_components=n_components, n_iter=100
    )
    shared = ica.fit_transform(X)

    for i in range(len(shared)):
        np.testing.assert_almost_equal(shared[0], shared[i], 2)


@pytest.mark.parametrize("reduction, n_voxels", [("srm", 10), (None, 2)])
def test_inverse_transform(reduction, n_voxels):
    """
    Test that we can recover data after transform
    """
    n_timeframes = [10, 10]
    n_subjects = 2
    n_components = 2

    W, Ss, Xs = generate_data(
        n_voxels, n_components, n_timeframes, n_components, n_subjects, None
    )
    X = [np.concatenate(np.array(x), axis=1) for x in Xs]
    X = [x + 1e-15 * np.eye(*x.shape) for x in X]

    ica = MultiViewICA(reduction=reduction, n_components=n_components, n_iter=100)
    shared = ica.fit_transform(X)
    X_pred = ica.inverse_transform(np.mean(shared, axis=0))

    for i in range(len(X_pred)):
        np.testing.assert_array_almost_equal(X_pred[i], X[i], 2)

    shared_pred = ica.transform(X_pred)
    np.testing.assert_array_almost_equal(
        np.mean(shared, axis=0), np.mean(shared_pred, axis=0), 5
    )


@pytest.mark.parametrize("reduction, n_voxels", [("srm", 10), (None, 2)])
def test_add_subjects(reduction, n_voxels):
    """
    Test that we can recover data after transform
    """
    n_voxels = n_voxels
    n_timeframes = [10, 10]
    n_subjects = 5
    n_components = 2

    W, Ss, Xs = generate_data(
        n_voxels, n_components, n_timeframes, n_components, n_subjects, None
    )
    X = [np.concatenate(np.array(x), axis=1) for x in Xs]

    ica = MultiViewICA(reduction=reduction, n_components=n_components, n_iter=100)
    shared = ica.fit_transform(X[:4])
    ica.add_subjects([X[4]], np.mean(shared, axis=0))
    X_pred = ica.inverse_transform(np.mean(shared, axis=0))

    for i in range(len(X_pred)):
        np.testing.assert_array_almost_equal(X_pred[i], X[i], 2)

    shared_pred = ica.transform(X_pred)
    np.testing.assert_array_almost_equal(
        np.mean(shared, axis=0), np.mean(shared_pred, axis=0), 5
    )

def test_input():
    """
    Test that we can recover data after transform
    whatever input it is
    """
    n_voxels = 10
    n_timeframes = [20, 20]
    n_subjects = 5
    n_components = 2

    with tempfile.TemporaryDirectory() as datadir:
        W, Ss, Xs, paths = generate_data(
            n_voxels,
            n_components,
            n_timeframes,
            n_components,
            n_subjects,
            datadir,
        )
        np_data = [np.concatenate(np.array(x), axis=1) for x in Xs]
        ica = MultiViewICA(reduction="srm", n_components=None, n_iter=100)
        shared = ica.fit_transform(paths)

        for i in range(len(Xs)):
            X_predi = ica.inverse_transform(
                shared[i], subjects_indexes=[i]
            )[0]
            np.testing.assert_allclose(X_predi, np_data[i], 5)
