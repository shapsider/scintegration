import h5py
import uuid
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from typing import Any, List, Mapping, Optional, Tuple
import functools
import operator
from src.num import vertex_degrees
from src.typehint import AnyArray, Array, RandomState
from src.utils import get_rs, logged, processes
from src.config import config
from src.nn import get_default_numpy_dtype

from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset
import copy

DATA_CONFIG = Mapping[str, Any]

@logged
class ArrayDataset(Sequence):

    def __init__(self, *arrays, getitem_size=1, seed=None):
        self.arrays = arrays
        self.getitem_size = getitem_size
        self.sizes = [array.shape[0] for array in arrays]
        if min(self.sizes) == 0:
            raise ValueError("Empty array is not allowed!")
        self.size = max(self.sizes)
        self.view_idx = [np.arange(s) for s in self.sizes]
        self.shuffle_idx = self.view_idx
        self.seed = seed

    def __len__(self):
        return np.ceil(self.size / self.getitem_size).astype(int)

    def __getitem__(self, idx):
        index = np.arange(idx * self.getitem_size, min((idx + 1) * self.getitem_size, self.size))
        batch = [self._to_tensor(self.arrays[i], self.shuffle_idx[i][np.mod(index, self.sizes[i])]) for i in range(len(self.arrays))]
        return batch

    def _to_tensor(self, array, index):
        if isinstance(array, tf.sparse.SparseTensor):
            return tf.sparse.to_dense(tf.gather(array, index))
        else:
            return tf.gather(array, index)
        
    def on_epoch_end(self):
        if self.seed is not None:
            rs = get_rs(self.seed)
            self.shuffle_idx = [rs.permutation(view_idx) for view_idx in self.view_idx]

    def random_split(self, fractions: List[float], random_state: RandomState = None) -> List["ArrayDataset"]:
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        subdatasets = [
            ArrayDataset(
                *self.arrays, getitem_size=self.getitem_size
            ) for _ in fractions
        ]
        for j, view_idx in enumerate(self.view_idx):
            view_idx = rs.permutation(view_idx)
            split_pos = np.round(cum_frac * view_idx.size).astype(int)
            split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
            for i, idx in enumerate(split_idx):
                subdatasets[i].sizes[j] = len(idx)
                subdatasets[i].view_idx[j] = idx
                subdatasets[i].shuffle_idx[j] = idx
        return subdatasets
    
@logged
class AnnDataset(Sequence):
    def __init__(
            self, adatas: List[AnnData], data_configs: List[DATA_CONFIG],
            mode: str = "train", getitem_size: int = 1, seed: Optional[int] = None
    ) -> None:
        self.getitem_size = getitem_size
        if mode not in ("train", "eval"):
            raise ValueError("Invalid `mode`!")
        self.mode = mode
        self.adatas = adatas
        self.data_configs = data_configs
        self.seed = seed

    @property
    def adatas(self) -> List[AnnData]:
        r"""
        Internal :class:`AnnData` objects
        """
        return self._adatas

    @property
    def data_configs(self) -> List[DATA_CONFIG]:
        r"""
        Data configuration for each dataset
        """
        return self._data_configs

    @adatas.setter
    def adatas(self, adatas: List[AnnData]) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        if min(self.sizes) == 0:
            raise ValueError("Empty dataset is not allowed!")
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs: List[DATA_CONFIG]) -> None:
        if len(data_configs) != len(self.adatas):
            raise ValueError(
                "Number of data configs must match "
                "the number of datasets!"
            )
        self.data_idx, self.extracted_data = self._extract_data(data_configs)
        self.view_idx = pd.concat(
            [data_idx.to_series() for data_idx in self.data_idx]
        ).drop_duplicates().to_numpy()
        self.size = self.view_idx.size
        self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx)
        self._data_configs = data_configs

    def _get_idx_pmsk(
            self, view_idx: np.ndarray, random_fill: bool = False,
            random_state: RandomState = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rs = get_rs(random_state) if random_fill else None
        shuffle_idx, shuffle_pmsk = [], []
        for data_idx in self.data_idx:
            idx = data_idx.get_indexer(view_idx)
            pmsk = idx >= 0
            n_true = pmsk.sum()
            n_false = pmsk.size - n_true
            idx[~pmsk] = rs.choice(idx[pmsk], n_false, replace=True) \
                if random_fill else idx[pmsk][np.mod(np.arange(n_false), n_true)]
            shuffle_idx.append(idx)
            shuffle_pmsk.append(pmsk)
        return np.stack(shuffle_idx, axis=1), np.stack(shuffle_pmsk, axis=1)
    
    
    def __len__(self):
        return np.ceil(self.size / self.getitem_size).astype(int)

    def __getitem__(self, index):
        s = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        shuffle_idx = self.shuffle_idx[s].T
        shuffle_pmsk = self.shuffle_pmsk[s]

        items = [
            tf.convert_to_tensor(self._index_array(data, idx))
            for extracted_data in self.extracted_data
            for idx, data in zip(shuffle_idx, extracted_data)
        ]
        items.append(tf.convert_to_tensor(shuffle_pmsk))
        return items
    
    def random_split(
            self, fractions: List[float], random_state: RandomState = None
    ) -> List["AnnDataset"]:

        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        view_idx = rs.permutation(self.view_idx)
        split_pos = np.round(cum_frac * view_idx.size).astype(int)
        split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
        subdatasets = []
        for idx in split_idx:
            sub = copy.copy(self)
            sub.view_idx = idx
            sub.size = idx.size
            sub.shuffle_idx, sub.shuffle_pmsk = sub._get_idx_pmsk(idx)  # pylint: disable=protected-access
            subdatasets.append(sub)
        return subdatasets


    @staticmethod
    def _index_array(arr: AnyArray, idx: np.ndarray) -> np.ndarray:
        if isinstance(arr, (h5py.Dataset, SparseDataset)):
            rank = scipy.stats.rankdata(idx, method="dense") - 1
            sorted_idx = np.empty(rank.max() + 1, dtype=int)
            sorted_idx[rank] = idx
            arr = arr[sorted_idx.tolist()][rank.tolist()]  # Convert to sequantial access and back
        else:
            arr = arr[idx]
        return arr.toarray() if scipy.sparse.issparse(arr) else arr

    def _extract_data(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)  # self.mode == "train"

    def _extract_data_train(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xrep = [
            self._extract_xrep(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = [
            self._extract_xbch(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xlbl = [
            self._extract_xlbl(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xdwt = [
            self._extract_xdwt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        return xuid, (x, xrep, xbch, xlbl, xdwt)

    def _extract_data_eval(self, data_configs: List[DATA_CONFIG]) -> Tuple[
            List[pd.Index], Tuple[
                List[AnyArray], List[AnyArray], List[AnyArray],
                List[AnyArray], List[AnyArray]
            ]
    ]:
        default_dtype = get_default_numpy_dtype()
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xrep = [
            self._extract_xrep(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            if xrep_.size else self._extract_x(adata, data_config)
            for adata, data_config, xrep_ in zip(self.adatas, data_configs, xrep)
        ]
        xbch = xlbl = [
            np.empty((adata.shape[0], 0), dtype=int)
            for adata in self.adatas
        ]
        xdwt = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            for adata in self.adatas
        ]
        return xuid, (x, xrep, xbch, xlbl, xdwt)

    def _extract_x(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            x = adata.layers[use_layer]
        else:
            x = adata.X
        if x.dtype.type is not default_dtype:
            if isinstance(x, (h5py.Dataset, SparseDataset)):
                raise RuntimeError(
                    f"User is responsible for ensuring a {default_dtype} dtype "
                    f"when using backed data!"
                )
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr()
        return x

    def _extract_xrep(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_rep = data_config["use_rep"]
        rep_dim = data_config["rep_dim"]
        if use_rep:
            if use_rep not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_rep}' "
                    f"cannot be found in input data!"
                )
            xrep = np.asarray(adata.obsm[use_rep]).astype(default_dtype)
            if xrep.shape[1] != rep_dim:
                raise ValueError(
                    f"Input representation dimensionality {xrep.shape[1]} "
                    f"does not match the configured {rep_dim}!"
                )
            return xrep
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xbch(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_batch = data_config["use_batch"]
        batches = data_config["batches"]
        if use_batch:
            if use_batch not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_batch}' "
                    f"cannot be found in input data!"
                )
            return batches.get_indexer(adata.obs[use_batch])
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xlbl(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_cell_type = data_config["use_cell_type"]
        cell_types = data_config["cell_types"]
        if use_cell_type:
            if use_cell_type not in adata.obs:
                raise ValueError(
                    f"Configured cell type '{use_cell_type}' "
                    f"cannot be found in input data!"
                )
            return cell_types.get_indexer(adata.obs[use_cell_type])
        return -np.ones(adata.shape[0], dtype=int)

    def _extract_xdwt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_dsc_weight = data_config["use_dsc_weight"]
        if use_dsc_weight:
            if use_dsc_weight not in adata.obs:
                raise ValueError(
                    f"Configured discriminator sample weight '{use_dsc_weight}' "
                    f"cannot be found in input data!"
                )
            xdwt = adata.obs[use_dsc_weight].to_numpy().astype(default_dtype)
            xdwt /= xdwt.sum() / xdwt.size
        else:
            xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        return xdwt

    def _extract_xuid(self, adata: AnnData, data_config: DATA_CONFIG) -> pd.Index:
        if data_config["use_obs_names"]:
            xuid = adata.obs_names.to_numpy()
        else:  # NOTE: Assuming random UUIDs never collapse with anything
            self.logger.debug("Generating random xuid...")
            xuid = np.array([uuid.uuid4().hex for _ in range(adata.shape[0])])
        if len(set(xuid)) != xuid.size:
            raise ValueError("Non-unique cell ID!")
        return pd.Index(xuid)
    
    def on_epoch_end(self):
        if self.seed is not None:
            rs = get_rs(self.seed)
            self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx, random_fill=True, random_state=rs)


@logged
class GraphDataset(Sequence):

    def __init__(
            self, graph: nx.Graph, vertices: pd.Index,
            neg_samples: int = 1, weighted_sampling: bool = True,
            deemphasize_loops: bool = True, getitem_size: int = 1, seed: Optional[int] = None
    ) -> None:
        self.getitem_size = getitem_size
        self.eidx, self.ewt, self.esgn = \
            self.graph2triplet(graph, vertices)
        self.eset = {
            (i, j, s) for (i, j), s in
            zip(self.eidx.T, self.esgn)
        }

        self.vnum = self.eidx.max() + 1
        if weighted_sampling:
            if deemphasize_loops:
                non_loop = self.eidx[0] != self.eidx[1]
                eidx = self.eidx[:, non_loop]
                ewt = self.ewt[non_loop]
            else:
                eidx = self.eidx
                ewt = self.ewt
            degree = vertex_degrees(eidx, ewt, vnum=self.vnum, direction="both")
        else:
            degree = np.ones(self.vnum, dtype=self.ewt.dtype)
        degree_sum = degree.sum()
        if degree_sum:
            self.vprob = degree / degree_sum  # Vertex sampling probability
        else:  # Possible when `deemphasize_loops` is set on a loop-only graph
            self.vprob = np.ones(self.vnum, dtype=self.ewt.dtype) / self.vnum

        effective_enum = self.ewt.sum()
        self.eprob = self.ewt / effective_enum  # Edge sampling probability
        self.effective_enum = round(effective_enum)

        self.neg_samples = neg_samples
        self.size = self.effective_enum * (1 + self.neg_samples)
        self.seed = seed
        self.samp_eidx, self.samp_ewt, self.samp_esgn = self.propose_shuffle(self.seed)

    def graph2triplet(
            self, graph: nx.Graph, vertices: pd.Index,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        graph = nx.MultiDiGraph(graph)  

        default_dtype = get_default_numpy_dtype()
        i, j, w, s = [], [], [], []
        for k, v in dict(graph.edges).items():
            i.append(k[0])
            j.append(k[1])
            w.append(v["weight"])
            s.append(v["sign"])
        eidx = np.stack([
            vertices.get_indexer(i),
            vertices.get_indexer(j)
        ]).astype(np.int64)
        if eidx.min() < 0:
            raise ValueError("Missing vertices!")
        ewt = np.asarray(w).astype(default_dtype)
        if ewt.min() <= 0 or ewt.max() > 1:
            raise ValueError("Invalid edge weight!")
        esgn = np.asarray(s).astype(default_dtype)
        if set(esgn).difference({-1, 1}):
            raise ValueError("Invalid edge sign!")
        return eidx, ewt, esgn

    def __len__(self) -> int:
        return np.ceil(self.size / self.getitem_size).astype(int)

    def __getitem__(self, index: int):
        s = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        return [
            tf.convert_to_tensor(self.samp_eidx[:, s]),
            tf.convert_to_tensor(self.samp_ewt[s]),
            tf.convert_to_tensor(self.samp_esgn[s])
        ]
    

    def propose_shuffle(
            self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (pi, pj), pw, ps = self.eidx, self.ewt, self.esgn
        rs = get_rs(seed)
        psamp = rs.choice(self.ewt.size, self.effective_enum, replace=True, p=self.eprob)
        pi_, pj_, pw_, ps_ = pi[psamp], pj[psamp], pw[psamp], ps[psamp]
        pw_ = np.ones_like(pw_)
        ni_ = np.tile(pi_, self.neg_samples)
        nw_ = np.zeros(pw_.size * self.neg_samples, dtype=pw_.dtype)
        ns_ = np.tile(ps_, self.neg_samples)
        nj_ = rs.choice(self.vnum, pj_.size * self.neg_samples, replace=True, p=self.vprob)

        remain = np.where([
            item in self.eset
            for item in zip(ni_, nj_, ns_)
        ])[0]
        while remain.size:  # NOTE: Potential infinite loop if graph too dense
            newnj = rs.choice(self.vnum, remain.size, replace=True, p=self.vprob)
            nj_[remain] = newnj
            remain = remain[[
                item in self.eset
                for item in zip(ni_[remain], newnj, ns_[remain])
            ]]
        idx = np.stack([np.concatenate([pi_, ni_]), np.concatenate([pj_, nj_])])
        w = np.concatenate([pw_, nw_])
        s = np.concatenate([ps_, ns_])
        perm = rs.permutation(idx.shape[1])
        return idx[:, perm], w[perm], s[perm]
    
    def on_epoch_end(self):
        if self.seed is not None:
            self.samp_eidx, self.samp_ewt, self.samp_esgn = self.propose_shuffle(self.seed)

import itertools

class ParallelDataLoader(Sequence):
    def __init__(self, *sequences, cycle_flags=None, batch_size=1):
        if cycle_flags is None:
            cycle_flags = [False] * len(sequences)

        if len(cycle_flags) != len(sequences):
            raise ValueError("Length of cycle_flags must match number of sequences")

        self.sequences = sequences
        self.cycle_flags = cycle_flags
        self.lengths = [len(seq) for seq in sequences]
        self.max_length = max(self.lengths)
        self.batch_size = batch_size

    def __len__(self):

        if self.max_length % self.batch_size == 0:

            return self.max_length // self.batch_size
        else:

            return (self.max_length // self.batch_size)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch = []

        for seq_idx, seq in enumerate(self.sequences):
            seq_batch = []
            for i in range(start_idx, end_idx):
                if i < self.lengths[seq_idx]:
                    seq_batch.append(seq[i])
                elif self.cycle_flags[seq_idx]:
                    cycle_index = i % self.lengths[seq_idx]
                    seq_batch.append(seq[cycle_index])
                else:
                    raise StopIteration
            
            if isinstance(seq, GraphDataset):
                eidx, ewt, esgn = zip(*seq_batch)
                eidx = tf.concat(eidx, axis=1)
                ewt = tf.concat(ewt, axis=0)
                esgn = tf.concat(esgn, axis=0)
                seq_batch = [eidx, ewt, esgn]
            else:
                seq_batch = [tf.concat(items, 0) for items in zip(*seq_batch)]

            batch.append(seq_batch)

        return list(itertools.chain(*batch))

                
