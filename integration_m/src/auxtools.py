import os
from collections import defaultdict
from itertools import chain
from typing import Callable, List, Mapping, Optional

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import scipy.stats
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn.preprocessing import normalize
from sparse import COO
from tqdm.auto import tqdm

from . import num
from .typehint import Kws
from .utils import logged


def count_prep(adata: AnnData) -> None:
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)


def aggregate_obs(
        adata: AnnData, by: str, X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> AnnData:
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) \
        if pd.api.types.is_categorical_dtype(by) \
        else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix((
        np.ones(adata.shape[0]), (
            agg_idx.get_indexer(by),
            np.arange(adata.shape[0])
        )
    )).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy()
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({
        k: agg_method[v](adata.obs[k])
        for k, v in obs_agg.items()
    }, index=agg_idx.astype(str))
    obsm = {
        k: agg_method[v](adata.obsm[k])
        for k, v in obsm_agg.items()
    }
    layers = {
        k: agg_method[v](adata.layers[k])
        for k, v in layers_agg.items()
    }
    for c in obs:
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers,
        dtype=None if X is None else X.dtype
    )


def transfer_labels(
        ref: AnnData, query: AnnData, field: str,
        n_neighbors: int = 30, use_rep: Optional[str] = None,
        key_added: Optional[str] = None, **kwargs
) -> None:

    xrep = ref.obsm[use_rep] if use_rep else ref.X
    yrep = query.obsm[use_rep] if use_rep else query.X
    xnn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(xrep)
    ynn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(yrep)
    xx = xnn.kneighbors_graph(xrep)
    xy = ynn.kneighbors_graph(xrep)
    yx = xnn.kneighbors_graph(yrep)
    yy = ynn.kneighbors_graph(yrep)
    jaccard = (xx @ yx.T) + (xy @ yy.T)
    jaccard.data /= 4 * n_neighbors - jaccard.data
    normalized_jaccard = jaccard.multiply(1 / jaccard.sum(axis=0))
    onehot = sklearn.preprocessing.OneHotEncoder()
    xtab = onehot.fit_transform(ref.obs[[field]])
    ytab = normalized_jaccard.T @ xtab
    pred = pd.Series(
        onehot.categories_[0][ytab.argmax(axis=1).A1],
        index=query.obs_names, dtype=ref.obs[field].dtype
    )
    conf = pd.Series(
        ytab.max(axis=1).toarray().ravel(),
        index=query.obs_names
    )
    key_added = key_added or field
    query.obs[key_added] = pred
    query.obs[key_added + "_confidence"] = conf


def extract_rank_genes_groups(
        adata: AnnData, groups: Optional[List[str]] = None,
        filter_by: str = "pvals_adj < 0.01", sort_by: str = "scores",
        ascending: str = False
) -> pd.DataFrame:

    if "rank_genes_groups" not in adata.uns:
        raise ValueError("Please call `sc.tl.rank_genes_groups` first!")
    if groups is None:
        groups = adata.uns["rank_genes_groups"][sort_by].dtype.names
    df = pd.concat([
        pd.DataFrame({
            k: np.asarray(v[g])
            for k, v in adata.uns["rank_genes_groups"].items()
            if k != "params"
        }).assign(group=g)
        for g in groups
    ])
    df["group"] = pd.Categorical(df["group"], categories=groups)
    df = df.sort_values(
        sort_by, ascending=ascending
    ).drop_duplicates(
        subset=["names"], keep="first"
    ).sort_values(
        ["group", sort_by], ascending=[True, ascending]
    ).query(filter_by)
    df = df.reset_index(drop=True)
    return df


def bedmap2anndata(
        bedmap: os.PathLike, var_col: int = 3, obs_col: int = 6
) -> AnnData:

    bedmap = pd.read_table(bedmap, sep="\t", header=None, usecols=[var_col, obs_col])
    var_names = pd.Index(sorted(set(bedmap[var_col])))
    bedmap = bedmap.dropna()
    var_pool = bedmap[var_col]
    obs_pool = bedmap[obs_col].str.split(";")
    obs_names = pd.Index(sorted(set(chain.from_iterable(obs_pool))))
    X = scipy.sparse.lil_matrix((var_names.size, obs_names.size))  # Transposed
    for obs, var in tqdm(zip(obs_pool, var_pool), total=bedmap.shape[0], desc="bedmap2anndata"):
        row = obs_names.get_indexer(obs)
        col = var_names.get_loc(var)
        X.rows[col] += row.tolist()
        X.data[col] += [1] * row.size
    X = X.tocsc().T  # Transpose back
    X.sum_duplicates()
    return AnnData(
        X=X, obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
        dtype=X.dtype
    )


@logged
def estimate_balancing_weight(
        *adatas: AnnData, use_rep: str = None, use_batch: Optional[str] = None,
        resolution: float = 1.0, cutoff: float = 0.5, power: float = 4.0,
        key_added: str = "balancing_weight"
) -> None:
    if use_batch:  # Recurse per batch
        estimate_balancing_weight.logger.info("Splitting batches...")
        adatas_per_batch = defaultdict(list)
        for adata in adatas:
            groupby = adata.obs.groupby(use_batch, dropna=False)
            for b, idx in groupby.indices.items():
                adata_sub = adata[idx]
                adatas_per_batch[b].append(AnnData(
                    obs=adata_sub.obs,
                    obsm={use_rep: adata_sub.obsm[use_rep]}
                ))
        if len(set(len(items) for items in adatas_per_batch.values())) != 1:
            raise ValueError("Batches must match across datasets!")
        for b, items in adatas_per_batch.items():
            estimate_balancing_weight.logger.info("Processing batch %s...", b)
            estimate_balancing_weight(
                *items, use_rep=use_rep, use_batch=None,
                resolution=resolution, cutoff=cutoff,
                power=power, key_added=key_added
            )
        estimate_balancing_weight.logger.info("Collating batches...")
        collates = [
            pd.concat([item.obs[key_added] for item in items])
            for items in zip(*adatas_per_batch.values())
        ]
        for adata, collate in zip(adatas, collates):
            adata.obs[key_added] = collate.loc[adata.obs_names]
        return

    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    adatas_ = [
        AnnData(
            obs=adata.obs.copy(deep=False).assign(n=1),
            obsm={use_rep: adata.obsm[use_rep]}
        ) for adata in adatas
    ]  # Avoid unwanted updates to the input objects

    estimate_balancing_weight.logger.info("Clustering cells...")
    for adata_ in adatas_:
        sc.pp.neighbors(
            adata_, n_pcs=adata_.obsm[use_rep].shape[1],
            use_rep=use_rep, metric="cosine"
        )
        sc.tl.leiden(adata_, resolution=resolution)

    leidens = [
        aggregate_obs(
            adata, by="leiden", X_agg=None,
            obs_agg={"n": "sum"}, obsm_agg={use_rep: "mean"}
        ) for adata in adatas_
    ]
    us = [normalize(leiden.obsm[use_rep], norm="l2") for leiden in leidens]
    ns = [leiden.obs["n"] for leiden in leidens]

    estimate_balancing_weight.logger.info("Matching clusters...")
    cosines = []
    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1:], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = num.prod(cosines)
    estimate_balancing_weight.logger.info(
        "Matching array shape = %s...", str(joint_cosine.shape)
    )

    estimate_balancing_weight.logger.info("Estimating balancing weight...")
    for i, (adata, adata_, leiden, n) in enumerate(zip(adatas, adatas_, leidens, ns)):
        balancing = joint_cosine.sum(axis=tuple(
            k for k in range(joint_cosine.ndim) if k != i
        )).todense() / n
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs["leiden"]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        adata.obs[key_added] = balancing


@logged
def get_metacells(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        save_path: str = None,
        common: bool = True, seed: int = 0,
        agg_kws: Optional[List[Kws]] = None
) -> List[AnnData]:

    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    if n_meta is None:
        raise ValueError("Missing required argument `n_meta`!")
    adatas = [
        AnnData(
            X=adata.X,
            obs=adata.obs.set_index(adata.obs_names + f"-{i}"), var=adata.var,
            obsm=adata.obsm, varm=adata.varm, layers=adata.layers,
            dtype=None if adata.X is None else adata.X.dtype
        ) for i, adata in enumerate(adatas)
    ]  # Avoid unwanted updates to the input objects

    get_metacells.logger.info("Clustering metacells...")
    combined = ad.concat(adatas)
    sc.pp.neighbors(combined, use_rep=use_rep, metric="cosine")
    sc.tl.umap(combined)
    clust_method = sklearn.cluster.DBSCAN(eps=0.3, min_samples=10)
    combined.obs["metacell"] = [str(x) for x in clust_method.fit_predict(combined.obsm["X_umap"])]
    if save_path:
        combined.write(save_path)

    for adata in adatas:
        adata.obs["metacell"] = combined[adata.obs_names].obs["metacell"]

    return adatas


def _metacell_corr(
        *adatas: AnnData, skeleton: nx.Graph = None, method: str = "spr",
        prep_fns: Optional[List[Optional[Callable[[AnnData], None]]]] = None
) -> nx.Graph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    prep_fns = prep_fns or [None] * len(adatas)
    if not len(prep_fns) == len(adatas):
        raise ValueError("Length of `prep_fns` must match the number of datasets!")
    for adata, prep_fn in zip(adatas, prep_fns):
        if prep_fn:
            prep_fn(adata)
    adata = ad.concat(adatas, axis=1)
    edgelist = nx.to_pandas_edgelist(skeleton)
    source = adata.var_names.get_indexer(edgelist["source"])
    target = adata.var_names.get_indexer(edgelist["target"])
    X = num.densify(adata.X.T)
    if method == "spr":
        X = np.array([scipy.stats.rankdata(x) for x in X])
    elif method != "pcc":
        raise ValueError(f"Unrecognized method: {method}!")
    mean = X.mean(axis=1)
    meansq = np.square(X).mean(axis=1)
    std = np.sqrt(meansq - np.square(mean))
    edgelist["corr"] = np.array([
        ((X[s] * X[t]).mean() - mean[s] * mean[t]) / (std[s] * std[t])
        for s, t in zip(source, target)
    ])
    return nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_corr(
        *adatas: AnnData, skeleton: nx.Graph = None, method: str = "spr",
        agg_fns: Optional[List[str]] = None,
        prep_fns: Optional[List[Optional[Callable[[AnnData], None]]]] = None,
        **kwargs
) -> nx.Graph:

    adatas = get_metacells(*adatas, **kwargs, agg_kws=[
        dict(X_agg=agg_fn) for agg_fn in agg_fns
    ] if agg_fns else None)
    metacell_corr.logger.info(
        "Computing correlation on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_corr(
        *adatas, skeleton=skeleton, method=method, prep_fns=prep_fns
    )


def _metacell_regr(
        *adatas: AnnData, skeleton: nx.DiGraph = None,
        model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    if skeleton is None:
        raise ValueError("Missing required argument `skeleton`!")
    for adata in adatas:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    if set.intersection(*(set(adata.var_names) for adata in adatas)):
        raise ValueError("Overlapping features are currently not supported!")
    adata = ad.concat(adatas, axis=1)

    targets = [node for node, in_degree in skeleton.in_degree() if in_degree]
    biadj = biadjacency_matrix(
        skeleton, adata.var_names, targets, weight=None
    ).astype(bool).T.tocsr()
    X = num.densify(adata.X)
    Y = num.densify(adata[:, targets].X.T)
    coef = []
    model = getattr(sklearn.linear_model, model)
    for target, y, mask in tqdm(zip(targets, Y, biadj), total=len(targets), desc="metacell_regr"):
        X_ = X[:, mask.indices]
        lm = model(**kwargs).fit(X_, y)
        coef.append(pd.DataFrame({
            "source": adata.var_names[mask.indices],
            "target": target,
            "regr": lm.coef_
        }))
    coef = pd.concat(coef)
    return nx.from_pandas_edgelist(coef, edge_attr=True, create_using=type(skeleton))


@logged
def metacell_regr(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        skeleton: nx.DiGraph = None, model: str = "Lasso", **kwargs
) -> nx.DiGraph:
    for adata in adatas:
        if not num.all_counts(adata.X):
            raise ValueError("``.X`` must contain raw counts!")
    adatas = get_metacells(*adatas, use_rep=use_rep, n_meta=n_meta, common=True)
    metacell_regr.logger.info(
        "Computing regression on %d common metacells...",
        adatas[0].shape[0]
    )
    return _metacell_regr(*adatas, skeleton=skeleton, model=model, **kwargs)