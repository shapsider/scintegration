from typing import Any, Iterable, List, Optional

import numpy as np
import scipy.sparse

from .typehint import Array

EPS = 1e-7



def prod(x: Iterable) -> Any:
    
    try:
        from math import prod  # pylint: disable=redefined-outer-name
        return prod(x)
    except ImportError:
        ans = 1
        for item in x:
            ans = ans * item
        return ans


def sigmoid(x: np.ndarray) -> np.ndarray:
    
    return 1 / (1 + np.exp(-x))



def densify(arr: Array) -> np.ndarray:
    
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def col_var(
        X: Array, Y: Optional[Array] = None, bias: bool = False
) -> np.ndarray:
    
    Y = X if Y is None else Y
    if X.shape != Y.shape:
        raise ValueError("X and Y should have the same shape!")
    bias_scaling = 1 if bias else X.shape[0] / (X.shape[0] - 1)
    if scipy.sparse.issparse(X) or scipy.sparse.issparse(Y):
        if not scipy.sparse.issparse(X):
            X, Y = Y, X  # does not affect trace
        return (
            np.asarray((X.multiply(Y)).mean(axis=0)) -
            np.asarray(X.mean(axis=0)) * np.asarray(Y.mean(axis=0))
        ).ravel() * bias_scaling
    return (
        (X * Y).mean(axis=0) - X.mean(axis=0) * Y.mean(axis=0)
    ) * bias_scaling


def col_pcc(X: Array, Y: Array) -> np.ndarray:
    
    return col_var(X, Y) / np.sqrt(col_var(X) * col_var(Y))


def col_spr(X: Array, Y: Array) -> np.ndarray:
    
    X = densify(X)
    X = np.array([
        scipy.stats.rankdata(X[:, i])
        for i in range(X.shape[1])
    ]).T
    Y = densify(Y)
    Y = np.array([
        scipy.stats.rankdata(Y[:, i])
        for i in range(Y.shape[1])
    ]).T
    return col_pcc(X, Y)


def cov_mat(
        X: Array, Y: Optional[Array] = None, bias: bool = False
) -> np.ndarray:
    
    X_mean = X.mean(axis=0) if scipy.sparse.issparse(X) \
        else X.mean(axis=0, keepdims=True)
    if Y is None:
        Y, Y_mean = X, X_mean
    else:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y should have the same number of rows!")
        Y_mean = Y.mean(axis=0) if scipy.sparse.issparse(Y) \
            else Y.mean(axis=0, keepdims=True)
    bias_scaling = 1 if bias else X.shape[0] / (X.shape[0] - 1)
    return np.asarray((X.T @ Y) / X.shape[0] - X_mean.T @ Y_mean) * bias_scaling


def pcc_mat(
        X: Array, Y: Optional[Array] = None
) -> np.ndarray:

    X = X.astype(np.float64)
    Y = Y if Y is None else Y.astype(np.float64)
    X_std = np.sqrt(col_var(X))[np.newaxis, :]
    Y_std = X_std if Y is None else np.sqrt(col_var(Y))[np.newaxis, :]
    pcc = cov_mat(X, Y) / X_std.T / Y_std
    if Y is None:
        assert (pcc - pcc.T).max() < EPS
        pcc = (pcc + pcc.T) / 2  # Remove small floating point errors
        assert np.abs(np.diag(pcc) - 1).max() < EPS
        np.fill_diagonal(pcc, 1)  # Remove small floating point errors
    overshoot_mask = pcc > 1
    if np.any(overshoot_mask):
        assert (pcc[overshoot_mask] - 1).max() < EPS
        pcc[overshoot_mask] = 1  # Remove small floating point errors
    return pcc


def spr_mat(
        X: Array, Y: Optional[Array] = None
) -> np.ndarray:

    X = densify(X)
    X = np.array([
        scipy.stats.rankdata(X[:, i])
        for i in range(X.shape[1])
    ]).T
    if Y is not None:
        Y = densify(Y)
        Y = np.array([
            scipy.stats.rankdata(Y[:, i])
            for i in range(Y.shape[1])
        ]).T
    return pcc_mat(X, Y)


def tfidf(X: Array) -> Array:

    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def prob_or(probs: List[float]) -> float:

    return 1 - (1 - np.asarray(probs)).prod()


def vertex_degrees(
        eidx: np.ndarray, ewt: np.ndarray,
        vnum: Optional[int] = None, direction: str = "both"
) -> np.ndarray:

    vnum = vnum or eidx.max() + 1
    adj = scipy.sparse.coo_matrix((ewt, (eidx[0], eidx[1])), shape=(vnum, vnum))
    if direction == "in":
        return adj.sum(axis=0).A1
    elif direction == "out":
        return adj.sum(axis=1).A1
    elif direction == "both":
        return adj.sum(axis=0).A1 + adj.sum(axis=1).A1 - adj.diagonal()
    raise ValueError("Unrecognized direction!")


def normalize_edges(
        eidx: np.ndarray, ewt: np.ndarray, method: str = "keepvar"
) -> np.ndarray:

    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    enorm = ewt
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(eidx, ewt, direction="in")
        in_normalizer = np.power(
            in_degrees[eidx[1]],
            -1 if method == "in" else -0.5
        )
        in_normalizer[~np.isfinite(in_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(eidx, ewt, direction="out")
        out_normalizer = np.power(
            out_degrees[eidx[0]],
            -1 if method == "out" else -0.5
        )
        out_normalizer[~np.isfinite(out_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * out_normalizer
    return enorm


def all_counts(x: Array) -> bool:

    if scipy.sparse.issparse(x):
        x = x.tocsr().data
    if x.min() < 0:
        return False
    return np.allclose(x, x.astype(int))
