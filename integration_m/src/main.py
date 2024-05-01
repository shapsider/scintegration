import anndata
import networkx as nx
import scanpy as sc
from anndata import AnnData
import torch
from typing import Any, List, Mapping, Optional
import pandas as pd
import os
import numpy as np
import seaborn as sns

from .config import configure_dataset
from .train import covel_train


def  integration(
    adatas: List[AnnData],
    modal_names: List[str], 
    graph: nx.Graph,
    prob: List[str],
    save_path = "./ckpt"
):
    vertices = sorted(graph.nodes)
    for idx, adata in enumerate(adatas):
        configure_dataset(adata, prob[idx])

    data = dict(zip(modal_names, adatas))

    covel = covel_train(
        data, 
        graph,
        fit_kws={"directory": save_path}
    )

    print("Integration data")

    for modal_name in modal_names:
        data[modal_name].obsm['embedding'] = covel.encode_data(modal_name, data[modal_name])
    
    combined = anndata.AnnData(
        obs=pd.concat([adata.obs for adata in adatas], join="inner"),
        obsm={"embedding": np.concatenate([
            adata.obsm["embedding"] for adata in adatas
        ])}
    )

    combined.obs["domain"] = pd.Categorical(
        combined.obs["domain"],
        # categories=modal_names
    )

    combined.uns["domain_colors"] = list(sns.color_palette(n_colors=len(modal_names)).as_hex())

    feature_embeddings = covel.encode_graph(graph)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=covel.vertices)

    for adata in adatas:
        adata.varm["embedding"] = feature_embeddings.reindex(adata.var_names).to_numpy()

    print("UMAP vis integration data, in combined.h5ad X_umap")

    sc.pp.neighbors(combined, n_pcs=50, use_rep="embedding", metric="dice")
    sc.tl.umap(combined)

    print("Save data")
    for modal_name in modal_names:
        data[modal_name].obsm["joint_umap"] = combined[data[modal_name].obs_names, :].obsm["X_umap"]
        data[modal_name].write("{}/{}.h5ad".format(save_path, modal_name), compression="gzip")

    combined.write("{}/combined.h5ad".format(save_path), compression="gzip")