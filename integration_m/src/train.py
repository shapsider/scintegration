import anndata
import networkx as nx
import scanpy as sc
from anndata import AnnData
import torch
from typing import Any, List, Mapping, Optional
import pandas as pd
import os
import numpy as np

from pathlib import Path

from .typehint import Kws
from .models import CoVELModel
from .config import config
from .auxtools import get_metacells
from .models_pretrain import CoVELModel_P
from .config import configure_dataset

def covel_train(
        adatas: List[AnnData], graph: nx.Graph, 
        model: type = CoVELModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None,
        config = None,
        result_path = None
) -> CoVELModel:
    [modal_names, prob, rep, cell_type] = config

    for idx, adata in enumerate(adatas):
        configure_dataset(adata, prob[idx], 
                      use_highly_variable=True,
                      use_rep=rep[idx],
                      use_cell_type=cell_type[idx])

    adatas = dict(zip(modal_names, adatas))

    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}

    print("Pretraining...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")

    pretrain = CoVELModel_P(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    print("Clustering cells 1...")
    for k, adata in adatas.items():
        adata.obsm["X_emb_tmp"] = pretrain.encode_data(k, adata)
    _ = get_metacells(*adatas.values(), use_rep="X_emb_tmp", n_meta=10,
                                save_path=f"{result_path}/combined1.h5ad")


    print("Fine-tuning stage 1...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune stage 1")

    finetune_1 = CoVELModel(adatas, sorted(graph.nodes), **init_kws)
    finetune_1.adopt_pretrained_model(pretrain)
    finetune_1.compile(**compile_kws)
    print("Increasing random seed by 1 to prevent idential data order...")
    finetune_1.random_seed += 1
    finetune_1.fit(adatas, graph, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune_1.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    print("Clustering cells 2...")
    for k, adata in adatas.items():
        adata.obsm["X_emb_tmp"] = finetune_1.encode_data(k, adata)
    adatas_list = get_metacells(*adatas.values(), use_rep="X_emb_tmp", n_meta=10,
                                save_path=f"{result_path}/combined2.h5ad")

    for adata in adatas_list:
        adata.obs["cell_type"] = adata.obs["metacell"]
        print(k, adata.obs.cell_type.value_counts())

    for idx, adata in enumerate(adatas_list):
        configure_dataset(adata, prob[idx], 
                      use_highly_variable=True,
                      use_rep=rep[idx],
                      use_cell_type=cell_type[idx])

    adatas = dict(zip(modal_names, adatas_list))

    
    print("Fine-tuning stage 2...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune stage 2")

    finetune_2 = CoVELModel(adatas, sorted(graph.nodes), **init_kws)
    finetune_2.adopt_pretrained_model(finetune_1)
    finetune_2.compile(**compile_kws)
    print("Increasing random seed by 1 to prevent idential data order...")
    finetune_2.random_seed += 1
    finetune_2.fit(adatas, graph, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune_2.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))
    
    return finetune_2