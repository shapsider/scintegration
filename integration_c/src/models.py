from tensorflow.keras.callbacks import ReduceLROnPlateau
import copy
import os
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple, Union
from tensorflow.keras.utils import Progbar
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
D = tfp.distributions

from anndata import AnnData

from src.num import normalize_edges
from src.config import config
from src.utils import AUTO, get_chained_attr, logged
from src.data import AnnDataset, ArrayDataset, GraphDataset, ParallelDataLoader

import src.sc as sc

from src.loss import SelfEntropyLoss, DDCLoss
from tensorflow.keras.layers import Input

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)

DataTensors = Tuple[
    Mapping[str, tf.Tensor],  # x (data)
    Mapping[str, tf.Tensor],  # xrep (alternative input data)
    Mapping[str, tf.Tensor],  # xbch (data batch)
    Mapping[str, tf.Tensor],  # xlbl (data label)
    Mapping[str, tf.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, tf.Tensor],  # xflag (modality indicator)
    tf.Tensor,  # eidx (edge index)
    tf.Tensor,  # ewt (edge weight)
    tf.Tensor  # esgn (edge sign)
]  


class CustomDict(dict):
    def __init__(self, data):
        super().__init__(data)

@logged
class CoVELModel(tf.keras.Model):

    GRAPH_BATCHES: int = 32  
    ALIGN_BURNIN_PRG: float = 8.0 
    MAX_EPOCHS_PRG: float = 48.0  
    PATIENCE_PRG: float = 4.0  
    REDUCE_LR_PATIENCE_PRG: float = 2.0  
    BURNIN_NOISE_EXAG: float = 1.5 

    def __init__(self, adatas, vertices, latent_dim=50, h_depth=2, h_dim=256, dropout=0.2, shared_batches=False, random_seed=0):
        super(CoVELModel, self).__init__()

        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)


        self.g2v = sc.GraphEncoder(self.vertices.size, latent_dim, name="g2v")

        self.v2g = sc.GraphDecoder(name="v2g")

        modalities, idx = {}, {}
        self.x2u, self.u2x,  = {}, {}
        all_ct = set()
        
        regist_k = []
        
        for k, adata in adatas.items():
            regist_k.append(k)
            num = regist_k.count(k)
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = tf.convert_to_tensor(idx[k])
            self.x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout, name=f"{k}_{num}_x2u"
            )

            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            self.u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                len(data_config["features"]),
                n_batches=max(data_config["batches"].size, 1), name=f"{k}_{num}_u2x"
            )

            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            modalities[k] = copy.deepcopy(data_config)
        self.modalities = CustomDict(modalities)
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        self.du = sc.Discriminator(
            latent_dim, len(self.modalities), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout, name="sc"
        )

        if not set(self.x2u.keys()) == set(self.u2x.keys()) == set(idx.keys()) != set():
            raise ValueError(
                "`x2u`, `u2x`, `idx` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(idx.keys())  # Keeps a specific order

        self.key_idx = idx

        
        self.u2c = None if all_ct.empty else tf.keras.layers.Dense(all_ct.size, name="u2c")
        self.u2c_outdim = None if all_ct.empty else all_ct.size

        self.epoch = 0
        self.freeze_u = False

    def adopt_pretrained_model(target, source, submodule=None):
        if submodule:
            source_layer = source.get_layer(submodule)
            target_layer = target.get_layer(submodule)
            source_weights = source_layer.get_weights()
            target_layer.set_weights(source_weights)
        else:
            # Iterate over the layers of the source model
            sub_model = ["g2v", "v2g", "x2u", "u2x", "du"]
            for submodel in sub_model:
                try:
                    getattr(target, submodel).set_weights(getattr(source, submodel).get_weights())
                except Exception as e:
                    dict = getattr(source, submodel)
                    for key in dict.keys():
                        getattr(target, submodel)[key].set_weights(dict[key].get_weights())

    def compile(  
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.keys}

        self.lam_sup = lam_sup
        ##
        self.required_losses = ["g_nll", "g_kl", "g_elbo"]
        for k in self.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.required_losses += ["dsc_loss", "vae_loss", "gen_loss"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        if min(modality_weight.values()) < 0:
            raise ValueError("Modality weight must be non-negative!")
        normalizer = sum(modality_weight.values()) / len(modality_weight)
        self.modality_weight = {k: v / normalizer for k, v in modality_weight.items()}

        self.lr = lr
        self.vae_optim = tf.keras.optimizers.RMSprop(learning_rate=self.lr, rho=0.99, momentum=0, epsilon=1e-08)
        self.dsc_optim = tf.keras.optimizers.RMSprop(learning_rate=self.lr, rho=0.99, momentum=0, epsilon=1e-08)
        self.normalize_u = normalize_u


        if self.u2c:
            print("use loss")
            self.required_losses.append("sup_loss")
            loss_weights=[1.0, 1.0]
            self.ddc = DDCLoss(self.u2c_outdim, loss_weights[1])


    def fit(
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: str = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None,
            callbacks=None
    ) -> None:
        data = AnnDataset(
            [adatas[key] for key in self.keys],
            [self.modalities[key] for key in self.keys],
            mode="train"
        )
        graph = GraphDataset(
            graph, self.vertices, neg_samples=neg_samples,
            weighted_sampling=True, deemphasize_loops=True
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            print("graph_batch_size = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            print("align_burnin = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            print("max_epochs = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            print("patience = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            print("reduce_lr_patience = %d", reduce_lr_patience)

        if self.freeze_u:
            print("Cell embeddings are frozen")

        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )

        self.enorm = tf.constant(normalize_edges(graph.eidx, graph.ewt))
        self.esgn = tf.constant(graph.esgn)
        self.eidx = tf.constant(graph.eidx)

        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        graph.getitem_size = max(1, round(graph_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=self.random_seed)


        train_loader = ParallelDataLoader(data_train, graph, cycle_flags=[False, True], batch_size=4)


        self.align_burnin = align_burnin


        best_loss = float('inf')
        patience_counter = 0
        factor = 0.1
        patience = patience
        min_lr = 2e-5
        patience_counter = 0

        for epoch in range(max_epochs):
            train_progbar = Progbar(target=len(train_loader))
            if epoch+1 % 20 == 0:
                tf.keras.backend.set_value(self.vae_optim.lr, self.vae_optim*0.1)
                tf.keras.backend.set_value(self.dsc_optim.lr, self.dsc_optim*0.1)
            
            print(f"Epoch {epoch+1}/{max_epochs}")
            self.epoch = epoch + 1
            losses = {}
            avg_losses = {}
            
            val_losses = {}
            val_avg_losses = {}
            for step,d in enumerate(train_loader):
                
                loss = self.train_step(d)
                for key, value in loss.items():
                    losses[key] = losses.get(key, 0) + value.numpy()
                    avg_losses[key] = losses[key] / (step + 1)
                train_progbar.update(step + 1, values=[(key, value) for key, value in avg_losses.items()])
            val_loader = ParallelDataLoader(data_val, graph, cycle_flags=[False, True], batch_size=4)

            try:
                for step,d in enumerate(val_loader):
                    val_loss = self.test_step(d)
                    for key, value in val_loss.items():
                        val_losses[key] = val_losses.get(key, 0) + value.numpy()
                        val_avg_losses[key] = val_losses[key] / (step + 1)

                print()
            except Exception as e:
                pass
            print("[val loss]: ")
            for key, value in val_avg_losses.items():
                print(f"{key}: {value:.4f} ", end="")
            print()

            if val_avg_losses[self.earlystop_loss] < best_loss:
                best_loss = val_avg_losses[self.earlystop_loss]
                patience_counter = 0
            else:
                reduce_lr_patience = 1
                patience_counter += reduce_lr_patience


            if patience_counter >= patience:
                new_lr_vae = self.vae_optim.lr * factor
                new_lr_dsc = self.dsc_optim.lr * factor

                tf.keras.backend.set_value(self.vae_optim.lr, new_lr_vae)
                tf.keras.backend.set_value(self.dsc_optim.lr, new_lr_dsc)
                print(f"Epoch {epoch}: reducing learning rate of group 0 to {self.vae_optim.lr.numpy().item()}.")
                patience_counter = 0

            if self.vae_optim.lr < min_lr:
                print(f"early stop at epoch {epoch}")
                break


    def train_step(self, data):
        data = self.format_data(data)

        if self.freeze_u:
            pass
        else:
            with tf.GradientTape() as tape1:
                losses = self.compute_losses(data, self.epoch, dsc_only=True, training=True)
                dsc_loss = losses["dsc_loss"]
        dsc_gradients = tape1.gradient(dsc_loss, self.du.trainable_variables)
        self.dsc_optim.apply_gradients(zip(dsc_gradients, self.du.trainable_variables))


        with tf.GradientTape() as tape2:
            losses = self.compute_losses(data, self.epoch, training=True)
            vae_loss = losses["gen_loss"]

        trainable_variables = (
            self.g2v.trainable_variables + 
            self.v2g.trainable_variables + 
            self.x2u.trainable_variables + 
            self.u2x.trainable_variables
        )
        vae_gradients = tape2.gradient(vae_loss, trainable_variables)

        self.vae_optim.apply_gradients(zip(vae_gradients, trainable_variables))
        return losses


    def test_step(self, data):
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                data = self.format_data(data)
                losses = self.compute_losses(data, self.epoch)
                return losses


    def get_losses( 
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        data = AnnDataset(
            [adatas[key] for key in self.keys],
            [self.modalities[key] for key in self.keys],
            mode="train"
        )
        graph = GraphDataset(
            graph, self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True
        )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            print("graph_batch_size = %d", graph_batch_size)
        return super().get_losses(
            data, graph, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    def encode_graph(
            self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        graph = GraphDataset(graph, self.vertices)
        enorm = tf.constant(normalize_edges(graph.eidx, graph.ewt))
        esgn = tf.constant(graph.esgn)
        eidx = tf.constant(graph.eidx)

        v = self.g2v(eidx, enorm, esgn)
        return v.mean().numpy()

    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:

        encoder = self.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=1
        )
        result = []
        for x, xrep, *_ in data:
            u = encoder(
                x,
                xrep,
                lazy_normalizer=True
            )[0]
            result.append(u.mean().numpy())
        return tf.constant(result).numpy().reshape(-1, result[0].shape[-1])

    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData, graph: nx.Graph,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = self.device
        self.eval()

        u = self.encode_data(source_key, adata, batch_size=1)
        v = self.encode_graph(graph)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=1)
        decoder = self.u2x[target_key]

        result = []
        for u_, b_, l_ in data:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean().numpy())
        return tf.constant(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False, training=False
    ) -> Mapping[str, tf.Tensor]:
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        for k in self.keys:
            u[k], l[k] = self.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only, training=training)
        usamp = {k: u[k].sample() for k in self.keys}
        self.normalize_u = True
        if self.normalize_u:
            usamp = {k:  tf.linalg.normalize(usamp[k], axis=1)[0] for k in self.keys}
        prior = sc.create_normal_distribution()

        u_cat = tf.concat([u[k].mean() for k in self.keys], 0)
        xbch_cat = tf.concat([xbch[k] for k in self.keys], 0)
        xdwt_cat = tf.concat([xdwt[k] for k in self.keys], 0)
        xflag_cat = tf.concat([xflag[k] for k in self.keys], 0)
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, tf.math.reduce_std(u_cat, 0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        dsc_loss = cross_entropy(xflag_cat, self.du(u_cat, xbch_cat))
        num_elements = tf.size(xdwt_cat).numpy().item()
        dsc_loss = tf.reduce_sum(dsc_loss * xdwt_cat) / num_elements
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        if self.u2c:

            xlbl_cat = tf.concat([xlbl[k] for k in self.keys], 0)
            lmsk = xlbl_cat >= 0
            
            # sce_loss = self.sce(self.u2c(u_cat[lmsk]))
            ddc_loss = self.ddc(u_cat[lmsk], self.u2c(u_cat[lmsk]))
            # print("sce: {}, ddc: {}".format(sce_loss.item(), ddc_loss.item()))
            sup_loss = ddc_loss

        else:
            sup_loss = tf.constant(0.0, dtype=tf.float32)

        v = self.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.sample()

        g_nll = -self.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = tf.cast(tf.not_equal(ewt, 0), tf.int32)
        n_pos = pos_mask.numpy().sum()
        num_elements = tf.size(pos_mask).numpy().item()
        n_neg = num_elements - n_pos
        g_nll_pn = tf.zeros([2], dtype=g_nll.dtype)
        pos_mask = tf.expand_dims(pos_mask, -1)
        g_nll_pn = tf.tensor_scatter_nd_add(g_nll_pn, pos_mask, g_nll)


        avgc = int(n_pos > 0) + int(n_neg > 0)
        g_nll = (g_nll_pn[0] / tf.maximum(n_neg, 1).numpy() + g_nll_pn[1] / tf.maximum(n_pos, 1).numpy()) / avgc

        g_kl = tf.reduce_mean(tf.reduce_sum(tfp.distributions.kl_divergence(v, prior), 1)) / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        x_nll = {}
        for k in self.keys:
            prob = self.u2x[k](usamp[k], tf.gather(vsamp, self.key_idx[k]), xbch[k], l[k])
            prob = prob.log_prob(x[k])
            x_nll[k] = -tf.reduce_mean(prob)

        x_kl = {
            k: tf.reduce_mean(tf.reduce_sum(tfp.distributions.kl_divergence(u[k], prior), 1)) / x[k].shape[1]
            for k in self.keys
        }
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in self.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in self.keys)

        # vae_loss = self.lam_data * x_elbo_sum + self.lam_graph * len(self.keys) * g_elbo + self.lam_sup * sup_loss
        vae_loss = self.lam_data * x_elbo_sum + self.lam_graph * len(self.keys) * g_elbo

        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {"dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss, "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo}
        for k in self.keys:
            losses.update({f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]})
        if self.u2c:
            losses["sup_loss"] = sup_loss

        return losses
    
    def format_data(self, data):
        keys = self.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], \
            data[5*K+1:]
        
        x = {k: x[i] for i, k in enumerate(keys)}
        xrep = {k: xrep[i] for i, k in enumerate(keys)}
        xbch = {k: xbch[i] for i, k in enumerate(keys)}
        xlbl = {k: xlbl[i] for i, k in enumerate(keys)}
        xdwt = {k: xdwt[i] for i, k in enumerate(keys)}
        xflag = {k: tf.fill([tf.shape(x[k])[0]], i) for i, k in enumerate(keys)}

        return x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn

    def for_call(self, adatas, graph):
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                data = AnnDataset(
                    [adatas[key] for key in self.keys],
                    [self.modalities[key] for key in self.keys],
                    mode="train"
                )
                graph = GraphDataset(
                    graph, self.vertices, neg_samples=10,
                    weighted_sampling=True, deemphasize_loops=True
                )


                self.enorm = tf.constant(normalize_edges(graph.eidx, graph.ewt))
                self.esgn = tf.constant(graph.esgn)
                self.eidx = tf.constant(graph.eidx)

                data.getitem_size = 32
                graph.getitem_size = 32
                data_train, data_val = data.random_split([1 - 0.1, 0.1], random_state=self.random_seed)
                self.align_burnin = 1

                train_loader = ParallelDataLoader(data_train, graph, cycle_flags=[False, True], batch_size=4)

                for step,d in enumerate(train_loader):
                    d = self.format_data(d)
                    losses = self.compute_losses(d, 0)
                    break

if __name__ == "__main__":
    print(1)