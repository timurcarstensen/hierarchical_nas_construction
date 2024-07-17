import logging
import os
import random
import time
from functools import partial
from pathlib import Path

import hydra
import neps
import numpy as np
import torch

# importing objectives
from benchmarks.objectives.addNIST import AddNISTObjective
from benchmarks.objectives.cifar_activation import (
    CIFAR10ActivationObjective,
)
from benchmarks.objectives.cifarTile import CifarTileObjective
from benchmarks.objectives.darts_cnn import DARTSCnn
from benchmarks.objectives.hierarchical_nb201 import NB201Pipeline

# importing search spaces
from benchmarks.search_spaces.activation_function_search.graph import (
    ActivationSpace,
)
from benchmarks.search_spaces.darts_cnn.graph import DARTSSpace
from benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from experiments.zero_cost_rank_correlation import ZeroCost, evaluate
from neps.optimizers.bayesian_optimization.acquisition_samplers import (
    EvolutionSampler,
)
from neps.optimizers.bayesian_optimization.kernels import (
    GraphKernelMapping,
)
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)
from neps.search_spaces.search_space import SearchSpace
from omegaconf import DictConfig

SearchSpaceMapping = {
    "nb201": NB201Spaces,
    "act": partial(ActivationSpace, base_architecture="resnet20"),
    "darts": DARTSSpace,
}

hierarchies_considered_in_search_space = {**NB201_HIERARCHIES_CONSIDERED}
hierarchies_considered_in_search_space["act_cifar10"] = [0, 1, 2]
hierarchies_considered_in_search_space["darts"] = [0]


def run_debug_pipeline(architecture):
    start = time.time()
    model = architecture.to_pytorch()
    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(1.5e7 - number_of_params)
    end = time.time()
    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


ObjectiveMapping = {
    "nb201_addNIST": AddNISTObjective,
    "nb201_cifarTile": CifarTileObjective,
    "nb201_cifar10": partial(NB201Pipeline, dataset="cifar10"),
    "nb201_cifar100": partial(NB201Pipeline, dataset="cifar100"),
    "nb201_ImageNet16-120": partial(
        NB201Pipeline, dataset="ImageNet16-120",
    ),
    "act_cifar10": partial(CIFAR10ActivationObjective, dataset="cifar10"),
    "act_cifar100": partial(
        CIFAR10ActivationObjective, dataset="cifar100",
    ),
    "darts": DARTSCnn,
    "debug": run_debug_pipeline,
}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="base",
)
def main(cfg: DictConfig):
    base_directory = Path(__file__).parent.parent.resolve()
    working_directory = os.path.join(
        base_directory,
        f"{cfg.experiment.working_directory}/{cfg.experiment.searcher}",
    )
    data_path = os.path.join(base_directory, cfg.experiment.data_path)

    if "bayesian_optimization" in cfg.experiment.searcher:
        working_directory += f"_{cfg.experiment.surrogate_model}"
        working_directory += f"_{cfg.experiment.pool_strategy}"
        working_directory += f"_pool{cfg.experiment.pool_size}"
    working_directory = os.path.join(
        working_directory, f"{cfg.experiment.seed}",
    )

    if (
        "nb201_" in cfg.experiment.objective
        or "act_" in cfg.experiment.objective
    ):
        # Gets the run pipeline and calls it with the given arguments
        run_pipeline_fn = ObjectiveMapping[cfg.experiment.objective](
            data_path=data_path,
            seed=cfg.experiment.seed,
            log_scale=cfg.experiment.log,
        )

        # Gets the dataset from the objective name
        idx = cfg.experiment.search_space.find("_")
        dataset = cfg.experiment.objective[
            cfg.experiment.objective.find("_") + 1 :
        ]
        # TODO: this spits out nb20 for nb201
        search_space_key = cfg.experiment.search_space[:idx]

        # adjust params an only be max or None -> could also be a boolean flag
        if (
            cfg.experiment.adjust_params is not None
            and "nb201_" in cfg.experiment.objective
        ):
            assert cfg.experiment.adjust_params in ["max"]

            # Creates a pipeline space with the neps SearchSpace class
            pipeline_space = {
                "architecture": SearchSpaceMapping[search_space_key](
                    space="fixed_1_none",
                    dataset=dataset,
                    adjust_params=None,
                ),
            }
            pipeline_space = SearchSpace(**pipeline_space)
            if cfg.experiment.adjust_params == "max":
                identifier = (
                    "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS conv3x3)"
                    " (OPS conv3x3) (OPS conv3x3) (OPS conv3x3))"
                )
            else:
                raise NotImplementedError

            # Loads the pipeline space from the identifier
            pipeline_space.load_from({"architecture": identifier})
            model = pipeline_space.hyperparameters[
                "architecture"
            ].to_pytorch()

            # check whether the model takes the correct input shape
            if dataset in ["cifar10", "cifar100"]:
                _ = model(torch.rand(1, 3, 32, 32))
            elif dataset == "ImageNet16-120":
                _ = model(torch.rand(1, 3, 16, 16))
            elif dataset == "addNIST":
                _ = model(torch.rand(1, 3, 28, 28))
            elif dataset == "cifarTile":
                _ = model(torch.rand(1, 3, 64, 64))
            else:
                raise NotImplementedError

            cfg.experiment.adjust_params = sum(
                p.numel() for p in model.parameters()
            )

        return_graph_per_hierarchy = (
            cfg.experiment.surrogate_model
            in ("gpwl_hierarchical", "gpwl", "gp_nasbot")
        )
        if "nb201_" in cfg.experiment.objective:
            search_space = SearchSpaceMapping[search_space_key](
                space=cfg.experiment.search_space[idx + 1 :],
                dataset=dataset,
                # adjust_params=cfg.experiment.adjust_params,
                return_graph_per_hierarchy=return_graph_per_hierarchy,
            )
        elif "act_" in cfg.experiment.objective:
            search_space = SearchSpaceMapping[search_space_key](
                dataset=dataset,
                return_graph_per_hierarchy=return_graph_per_hierarchy
                if cfg.experiment.surrogate_model
                in ("gpwl_hierarchical", "gpwl", "gp_nasbot")
                else False,
            )

    elif cfg.experiment.objective == "darts":
        run_pipeline_fn = ObjectiveMapping[cfg.experiment.objective](
            data_path=data_path,
            seed=cfg.experiment.seed,
            log_scale=cfg.experiment.log,
        )
        search_space = {
            "normal": SearchSpaceMapping["darts"](),
            "reduce": SearchSpaceMapping["darts"](),
        }
        cfg.experiment.pool_strategy = partial(
            EvolutionSampler, p_crossover=0.0, patience=10,
        )
    # run the debug pipeline
    elif cfg.experiment.objective == "debug":
        run_pipeline_fn = ObjectiveMapping[cfg.experiment.objective]
        idx = cfg.experiment.search_space.find("_")
        dataset = cfg.experiment.objective[
            cfg.experiment.objective.find("_") + 1 :
        ]

        search_space = SearchSpaceMapping[
            cfg.experiment.search_space[:idx]
        ](space=cfg.experiment.search_space[idx + 1 :], dataset="cifar10")
    else:
        raise NotImplementedError(
            f"Objective {cfg.experiment.objective} not implemented",
        )

    match cfg.experiment.surrogate_model:
        case "gpwl_hierarchical":
            hierarchy_considered = hierarchies_considered_in_search_space[
                cfg.experiment.search_space
            ]
            graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
            wl_h = [2, 1] + [2] * len(hierarchy_considered)
            graph_kernels = [
                GraphKernelMapping[kernel](
                    h=wl_h[j],
                    oa=False,
                    se_kernel=None,
                )
                for j, kernel in enumerate(graph_kernels)
            ]
            surrogate_model = ComprehensiveGPHierarchy
            surrogate_model_args = {
                "graph_kernels": graph_kernels,
                "hp_kernels": [],
                "verbose": False,
                "hierarchy_consider": hierarchy_considered,
                "d_graph_features": 0,
                "vectorial_features": None,
            }
        case "gpwl":
            hierarchy_considered = (
                None if cfg.experiment.objective == "darts" else []
            )
            if cfg.experiment.objective == "darts":
                graph_kernels = ["wl", "wl"]
                wl_h = [2, 2]
            else:
                graph_kernels = ["wl"]
                wl_h = [2]
            graph_kernels = [
                GraphKernelMapping[kernel](
                    h=wl_h[j],
                    oa=False,
                    se_kernel=None,
                )
                for j, kernel in enumerate(graph_kernels)
            ]
            surrogate_model = ComprehensiveGPHierarchy
            surrogate_model_args = {
                "graph_kernels": graph_kernels,
                "hp_kernels": [],
                "verbose": False,
                "hierarchy_consider": hierarchy_considered,
                "d_graph_features": 0,
                "vectorial_features": None,
            }

        case "gp_nasbot":
            hierarchy_considered = []
            graph_kernels = ["nasbot"]
            if (
                cfg.experiment.search_space
                == "nb201_variable_multi_multi"
            ):
                include_op_list = [
                    "id",
                    "zero",
                    "avg_pool",
                    "conv3x3o",
                    "conv1x1o",
                    "dconv3x3o",
                    "batch",
                    "instance",
                    "layer",
                    "relu",
                    "hardswish",
                    "mish",
                    "resBlock",
                ]
                exclude_op_list = ["input", "output"]
            else:
                raise NotImplementedError
            graph_kernels = [
                GraphKernelMapping[kernel](
                    include_op_list=include_op_list,
                    exclude_op_list=exclude_op_list,
                )
                for kernel in graph_kernels
            ]
            surrogate_model = ComprehensiveGPHierarchy
            surrogate_model_args = {
                "graph_kernels": graph_kernels,
                "hp_kernels": [],
                "verbose": False,
                "hierarchy_consider": hierarchy_considered,
                "d_graph_features": 0,
                "vectorial_features": None,
            }

        case _:
            raise NotImplementedError

    if cfg.experiment.seed is not None:
        if hasattr(run_pipeline_fn, "set_seed"):
            run_pipeline_fn.set_seed(cfg.experiment.seed)
        np.random.seed(cfg.experiment.seed)
        random.seed(cfg.experiment.seed)
        torch.manual_seed(cfg.experiment.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(cfg.experiment.seed)

    logging.basicConfig(level=logging.INFO)

    if not isinstance(search_space, dict) and not isinstance(
        search_space, SearchSpace,
    ):
        search_space = {"architecture": search_space}

    match cfg.experiment.searcher:
        case "bayesian_optimization":
            patience = (
                10
                if "fixed_1_none" in cfg.experiment.search_space
                else 100
            )
            neps.run(
                run_pipeline=run_pipeline_fn,
                pipeline_space=search_space,
                working_directory=working_directory,
                max_evaluations_total=cfg.experiment.max_evaluations_total,
                searcher=cfg.experiment.searcher,
                acquisition=cfg.experiment.acquisition,
                acquisition_sampler=cfg.experiment.pool_strategy,
                surrogate_model=surrogate_model,
                surrogate_model_args=surrogate_model_args,
                initial_design_size=cfg.experiment.n_init,
                patience=patience,
            )

        # when using AREA as described in https://arxiv.org/pdf/2006.04647
        case "assisted_regularized_evolution":
            zc_proxy = ZeroCost(
                method_type="nwot",
                n_classes=run_pipeline_fn.num_classes,
                loss_fn=None,
            )
            extract_model = lambda x: x["architecture"].to_pytorch()
            zc_proxy_evaluation = partial(
                evaluate,
                zc_proxy=zc_proxy,
                loader=run_pipeline_fn.get_train_loader(),
                extract_model=extract_model,
            )
            neps.run(
                run_pipeline=run_pipeline_fn,
                pipeline_space=search_space,
                working_directory=working_directory,
                max_evaluations_total=cfg.experiment.max_evaluations_total,
                searcher=cfg.experiment.searcher,
                assisted_zero_cost_proxy=zc_proxy_evaluation,
                assisted_init_population_dir=Path(working_directory)
                / "assisted_init_population",
                initial_design_size=cfg.experiment.n_init,
            )
        case _:
            neps.run(
                run_pipeline=run_pipeline_fn,
                pipeline_space=search_space,
                working_directory=working_directory,
                max_evaluations_total=cfg.experiment.max_evaluations_total,
                searcher=cfg.experiment.searcher,
                initial_design_size=cfg.experiment.n_init,
            )


if __name__ == "__main__":
    main()
