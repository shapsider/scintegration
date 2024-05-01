import anndata
import networkx as nx
import scanpy as sc
from anndata import AnnData
from typing import Any, List, Mapping, Optional
import pandas as pd
import os
import numpy as np

from pathlib import Path

from src.typehint import Kws
from src.models import CoVELModel
from src.config import config
from src.auxtools import estimate_balancing_weight


def save_model(model: CoVELModel, path: str):
    sub_models = ["g2v", "v2g", "x2u", "u2x", "du"]
    for submodel in sub_models:
        try:
            getattr(model, submodel).save_weights(os.path.join(path, f"{submodel}.h5"))
        except Exception as e:
            dict = getattr(model, submodel)
            for key in dict.keys():
                dict[key].save_weights(os.path.join(path, f"{submodel}_{key}.h5"))

def load_model(model: CoVELModel, path: str):
    sub_models = ["g2v", "v2g", "x2u", "u2x", "du"]
    for submodel in sub_models:
        try:
            getattr(model, submodel).load_weights(os.path.join(path, f"{submodel}.h5"))
        except Exception as e:
            dict = getattr(model, submodel)
            for key in dict.keys():
                dict[key].load_weights(os.path.join(path, f"{submodel}_{key}.h5"))


def covel_train(
        adatas: Mapping[str, AnnData], graph: nx.Graph, 
        model: type = CoVELModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None
) -> CoVELModel:

    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}

    print("Pretraining...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.for_call(adatas, graph)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    print("Fine-tuning stage 1...")
    finetune_fit_kws = fit_kws.copy()
    finetune_1 = model(adatas, sorted(graph.nodes), **init_kws)
    finetune_1.compile(**compile_kws)
    finetune_1.for_call(adatas, graph)
    finetune_1.adopt_pretrained_model(pretrain)
    print("Increasing random seed by 1 to prevent idential data order...")
    finetune_1.random_seed += 1
    finetune_1.fit(adatas, graph, **finetune_fit_kws)

    return finetune_1