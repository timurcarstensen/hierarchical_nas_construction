import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import torch
import torch.distributed as dist
from benchmarks.evaluation.objective import Objective
from benchmarks.objectives.custom_nb201.config_utils import load_config
from benchmarks.objectives.custom_nb201.custom_augmentations import (
    CUTOUT,
)
from benchmarks.objectives.custom_nb201.DownsampledImageNet import (
    ImageNet16,
)
from benchmarks.objectives.custom_nb201.evaluate_utils import (
    AverageMeter,
    get_optim_scheduler,
    obtain_accuracy,
    prepare_seed,
)
from neps.search_spaces.graph_grammar.graph import Graph
from neps.search_spaces.search_space import SearchSpace
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
}


# TODO: still needs integrating with DDP
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# TODO: still needs integrating with DDP
def cleanup():
    dist.destroy_process_group()


def get_dataset(
    name: str,
    root: str,
    cutout: int | None = -1,
    use_trivial_augment: bool | None = False,
):
    """This function loads and preprocesses various image datasets, applying
    appropriate data augmentation and normalization techniques. It supports
    datasets such as CIFAR-10, CIFAR-100, ImageNet, and custom ImageNet16 variants.

    Args:
    ----
        name: The name of the dataset to load. Supported options include
            'cifar10', 'cifar100', 'imagenet-1k', 'ImageNet16', 'ImageNet16-120',
            'ImageNet16-150', and 'ImageNet16-200'.
        root: The root directory where the dataset is stored or will be
            downloaded to.
        cutout: The length of the cutout to apply during data augmentation.
            If <= 0, no cutout is applied. Defaults to -1.
        use_trivial_augment: If True, uses TrivialAugment for data augmentation.
            Note: This option is not yet implemented and will raise a
            NotImplementedError if set to True. Defaults to False.

    Returns:
    -------
        A tuple containing four elements:
            - train_data: The training dataset with applied transformations.
            - test_data: The test/validation dataset with applied transformations.
            - xshape: The shape of a single data sample (C, H, W).
            - class_num: The number of classes in the dataset.

    Raises:
    ------
        TypeError: If an unknown dataset name is provided.
        NotImplementedError: If use_trivial_augment is set to True.

    Note:
    ----
        - The function applies different normalization parameters and
          augmentation techniques based on the dataset.
        - For CIFAR and ImageNet16 datasets, it applies random horizontal flip,
          random crop, and optional cutout as augmentation techniques.
        - The function asserts the correct number of samples for each dataset
          to ensure data integrity.

    """
    # normalizing the data
    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith("imagenet-1k"):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith("ImageNet16"):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError(f"Unknow dataset : {name}")

    # augmentation
    if name in ("cifar10", "cifar100"):
        if use_trivial_augment:
            raise NotImplementedError(
                "Trivial augment impl. has to be added here!",
            )
        else:
            lists = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if cutout > 0:
                lists += [CUTOUT(cutout)]
            train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)],
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("ImageNet16"):
        if use_trivial_augment:
            raise NotImplementedError(
                "Trivial augment impl. has to be added here!",
            )
        else:
            lists = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(16, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if cutout > 0:
                lists += [CUTOUT(cutout)]
            train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)],
        )
        xshape = (1, 3, 16, 16)

    # load datasets using torchvision
    if name == "cifar10":
        train_data = dset.CIFAR10(
            root=root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_data = dset.CIFAR10(
            root=root,
            train=False,
            transform=test_transform,
            download=True,
        )
        assert len(train_data) == 50000
        assert len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root=root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_data = dset.CIFAR100(
            root=root,
            train=False,
            transform=test_transform,
            download=True,
        )
        assert len(train_data) == 50000
        assert len(test_data) == 10000
    elif name.startswith("imagenet-1k"):
        train_data = dset.ImageFolder(
            os.path.join(root, "train"),
            train_transform,
        )
        test_data = dset.ImageFolder(
            os.path.join(root, "val"),
            test_transform,
        )
        assert (
            len(train_data) == 1281167 and len(test_data) == 50000
        ), f"invalid number of images : {len(train_data)} & {len(test_data)} vs {1281167} & {50000}"
    elif name == "ImageNet16":
        train_data = ImageNet16(root=root, train=True, transform=train_transform)
        test_data = ImageNet16(root=root, train=False, transform=test_transform)
        assert len(train_data) == 1281167
        assert len(test_data) == 50000
    elif name == "ImageNet16-120":
        train_data = ImageNet16(
            root=root, train=True, transform=train_transform, use_num_of_class_only=120
        )
        test_data = ImageNet16(
            root=root, train=False, transform=test_transform, use_num_of_class_only=120
        )
        assert len(train_data) == 151700
        assert len(test_data) == 6000
    elif name == "ImageNet16-150":
        train_data = ImageNet16(
            root=root, train=True, transform=train_transform, use_num_of_class_only=150
        )
        test_data = ImageNet16(
            root=root, train=False, transform=test_transform, use_num_of_class_only=150
        )
        assert len(train_data) == 190272
        assert len(test_data) == 7500
    elif name == "ImageNet16-200":
        train_data = ImageNet16(
            root=root, train=True, transform=train_transform, use_num_of_class_only=200
        )
        test_data = ImageNet16(
            root=root, train=False, transform=test_transform, use_num_of_class_only=200
        )
        assert len(train_data) == 254775
        assert len(test_data) == 10000
    else:
        raise TypeError(f"Unknow dataset : {name}")

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_config_and_split_info(
    dataset: Literal["cifar10", "cifar100", "ImageNet16-120"],
    dir_path: Path,
    use_less: bool,
) -> tuple[Path, NamedTuple]:
    if dataset in ("cifar10", "cifar100"):
        config_path = dir_path / ("LESS.config" if use_less else "CIFAR.config")
        split_info_path = dir_path / "cifar-split.txt"
    elif dataset.startswith("ImageNet16"):
        config_path = dir_path / ("LESS.config" if use_less else "ImageNet-16.config")
        split_info_path = dir_path / f"{dataset}-split.txt"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    split_info = load_config(split_info_path, None)

    return config_path, split_info


def get_dataloaders(
    dataset: Literal["cifar10", "cifar100", "ImageNet16-120"],
    root: str,
    epochs: int,
    gradient_accumulations: Optional[int] = 1,
    workers: Optional[int] = 4,
    use_less: Optional[bool] = False,
    use_trivial_augment: Optional[bool] = False,
    eval_mode: Optional[bool] = False,
):
    """Prepare and return data loaders for the specified dataset (CIFAR-10, CIFAR-100, or
    ImageNet16). Sets up training and validation/testing loaders based on given parameters
     and configurations.
    """

    train_data, valid_data, xshape, class_num = get_dataset(
        name=dataset,
        root=root,
        cutout=-1,
        use_trivial_augment=use_trivial_augment,
    )

    # TODO: actually use this
    if torch.cuda.device_count() > 1:
        torch.utils.data.distributed.DistributedSampler(train_data)
        torch.utils.data.distributed.DistributedSampler(valid_data)

    dir_path = (
        Path(os.path.dirname(os.path.realpath(__file__))) / "custom_nb201/configs"
    )
    config_path, split_info = get_config_and_split_info(
        dataset=dataset, dir_path=dir_path, use_less=use_less
    )
    # load config, this returns a NamedTuple
    config = load_config(
        path=config_path,
        extra={
            "class_num": class_num,
            "xshape": xshape,
            "epochs": epochs,
        },
    )
    # check whether use the split validation set
    if dataset == "cifar10" and not eval_mode:
        val_loaders = {
            # ori-test is the *original* test set (here: validation set)
            "ori-test": DataLoader(
                dataset=valid_data,
                batch_size=config.batch_size // gradient_accumulations,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )
        }

        # assert that the splits (train and valid) have the same length as the train
        # dataset split in cifar10
        assert len(train_data) == len(split_info.train) + len(
            split_info.valid,
        ), (
            f"invalid length : {len(train_data)} vs {len(split_info.train)} +"
            f" {len(split_info.valid)}"
        )
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2

        # TODO: if we want to use the DistributedSampler, we have to split the training
        #  dataset using the indices in split_info.train and split_info.valid and then pass it!!!!
        # data loader
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size // gradient_accumulations,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                split_info.train,
            ),
            num_workers=workers,
            pin_memory=True,
        )

        # this is again the training data but now with the validation split
        # TODO: why don't we actually use the `train_data` again but with the
        #  `valid_data.transform`?
        valid_loader = DataLoader(
            valid_data,
            batch_size=config.batch_size // gradient_accumulations,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                split_info.valid,
            ),
            num_workers=workers,
            pin_memory=True,
        )
        val_loaders["x-valid"] = valid_loader
    else:
        # data loader
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=config.batch_size // gradient_accumulations,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=config.batch_size // gradient_accumulations,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        if dataset == "cifar10":
            val_loaders = {"ori-test": valid_loader}
        elif dataset == "cifar100":
            cifar100_splits = load_config(
                path=dir_path / "cifar100-test-split.txt",
                extra=None,
            )
            val_loaders = {
                "ori-test": valid_loader,
                "x-valid": DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xvalid,
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
                "x-test": DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xtest,
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
            }
        elif dataset == "ImageNet16-120":
            imagenet16_splits = load_config(
                dir_path / "custom_nb201/configs/imagenet-16-120-test-split.txt",
                None,
            )
            val_loaders = {
                "ori-test": valid_loader,
                "x-valid": DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xvalid,
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
                "x-test": DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xtest,
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
            }
        else:
            raise ValueError(f"invalid dataset : {dataset}")

    return config, train_loader, val_loaders


def procedure(
    dataloader: DataLoader,
    network: nn.Module,
    criterion,
    scheduler,
    mode: Literal["train", "valid"],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_accumulations: Optional[int] = None,
):
    match mode:
        case "train":
            network.train()
            network.zero_grad()
        case "valid":
            network.eval()
            top1, top5 = AverageMeter(), AverageMeter()
        case _:
            raise ValueError(f"The mode is not right: {mode}")

    for i, (inputs, targets) in enumerate(dataloader):
        if mode == "train":
            scheduler.update(None, 1.0 * i / len(dataloader))

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        # forward
        with torch.cuda.amp.autocast():
            logits = network(inputs)
            loss = criterion(logits, targets)

        match mode:
            case "train":
                # backward
                scaler.scale(loss / gradient_accumulations).backward()
                if (i + 1) % gradient_accumulations == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    network.zero_grad()

            case "valid":
                # record loss and accuracy
                prec1, prec5 = obtain_accuracy(
                    output=logits.data,
                    target=targets.data,
                    topk=(1, 5),
                )
                top1.update(val=prec1.item(), n=inputs.size(0))
                top5.update(val=prec5.item(), n=inputs.size(0))

    if mode == "valid":
        return top1.avg, top5.avg
    else:
        return None


def evaluate_for_seed(
    model: nn.Module,
    config: NamedTuple,
    train_dataloader: DataLoader,
    val_dataloaders: dict[str, DataLoader],
    gradient_accumulations: int,
    workers: int,
    working_directory: str | Path | None = None,
    previous_working_directory: str | None = None,
) -> dict[str, Any]:
    # get optimizer, scheduler, criterion
    optimizer, scheduler, criterion = get_optim_scheduler(
        parameters=model.parameters(),
        config=config,
    )
    scaler = torch.cuda.amp.GradScaler()
    if workers > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    start_epoch = 0
    total_epochs = config.epochs + config.warmup
    if previous_working_directory is not None:
        checkpoint = torch.load(
            os.path.join(previous_working_directory, "checkpoint.pth"),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epochs_trained"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion = model.to(device), criterion.to(device)
    for epoch in range(start_epoch, total_epochs):
        logging.info(f"Epoch {epoch}")
        scheduler.update(epoch, 0.0)
        _ = procedure(
            dataloader=train_dataloader,
            network=model,
            criterion=criterion,
            scheduler=scheduler,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulations=gradient_accumulations,
            mode="train",
        )

    # evaluate
    with torch.no_grad():
        out_dict = {}
        for key, dataloader in val_dataloaders.items():
            valid_acc1, valid_acc5 = procedure(
                dataloader=dataloader,
                network=model,
                criterion=criterion,
                scheduler=None,
                optimizer=None,
                scaler=None,
                gradient_accumulations=None,
                mode="valid",
            )
            out_dict[f"{key}_1"] = valid_acc1
            out_dict[f"{key}_5"] = valid_acc5

    if working_directory is not None:
        model.train()
        # save checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epochs_trained": total_epochs,
            },
            os.path.join(working_directory, "checkpoint.pth"),
            _use_new_zipfile_serialization=False,
        )

    # clean up
    del model
    del criterion
    del scheduler
    del optimizer

    return out_dict


class NB201Pipeline(Objective):
    gradient_accumulations = 1  # 2
    workers = 4

    def __init__(
        self,
        dataset: str,
        data_path,
        seed: int,
        n_epochs: Optional[int] = 12,
        log_scale: Optional[bool] = True,
        negative: Optional[bool] = False,
        eval_mode: Optional[bool] = False,
        is_fidelity: Optional[bool] = False,  # TODO: the fuck does this mean
    ) -> None:
        assert seed in [555, 666, 777, 888, 999]
        super().__init__(seed, log_scale, negative)
        self.dataset = dataset
        self.data_path = data_path
        self.failed_runs = 0

        self.eval_mode = eval_mode

        self.n_epochs = n_epochs
        if self.eval_mode:
            self.n_epochs = 200

        self.is_fidelity = is_fidelity

        match self.dataset:
            case "cifar10":
                self.num_classes = 10
            case "cifar100":
                self.num_classes = 100
            case "ImageNet16-120":
                self.num_classes = 120
            case _:
                raise NotImplementedError(
                    f"Dataset '{self.dataset}' is not implemented"
                )

    # noinspection PyMethodOverriding
    def __call__(
        self,
        working_directory: str | Path,
        previous_working_directory: str | Path,
        architecture: Graph | nn.Module,
        **hp,
    ):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gradient_accumulations = self.gradient_accumulations
        while gradient_accumulations < 16:
            try:
                start = time.time()
                config, train_loader, val_loaders = get_dataloaders(
                    dataset=self.dataset,
                    root=self.data_path,
                    epochs=self.n_epochs,
                    gradient_accumulations=gradient_accumulations,
                    workers=self.workers,
                    use_trivial_augment=hp.get("trivial_augment", False),
                    eval_mode=self.eval_mode,
                    use_less=True,
                )

                # fix seed for reproducibility
                prepare_seed(self.seed, self.workers)

                if hasattr(architecture, "to_pytorch"):
                    model = architecture.to_pytorch()
                else:
                    assert isinstance(architecture, nn.Module)
                    model = architecture

                if self.is_fidelity:
                    out_dict = evaluate_for_seed(
                        model=model,
                        config=config,
                        train_dataloader=train_loader,
                        val_dataloaders=val_loaders,
                        gradient_accumulations=gradient_accumulations,
                        workers=self.workers,
                        working_directory=working_directory,
                        previous_working_directory=previous_working_directory,
                    )
                else:
                    out_dict = evaluate_for_seed(
                        model=model,
                        config=config,
                        train_dataloader=train_loader,
                        val_dataloaders=val_loaders,
                        gradient_accumulations=gradient_accumulations,
                        workers=self.workers,
                    )
                end = time.time()
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gradient_accumulations *= 2
                    torch.cuda.empty_cache()
                else:
                    raise e

        # prepare result_dict
        if self.eval_mode and "x-valid_1" not in out_dict:
            val_err = 1
        else:
            val_err = 1 - out_dict["x-valid_1"] / 100

        nof_parameters = sum(p.numel() for p in model.parameters())
        results = {
            "loss": self.transform(val_err),
            "info_dict": {
                **out_dict,
                "train_time": end - start,
                "timestamp": end,
                "number_of_parameters": nof_parameters,
            },
        }

        del model
        del train_loader
        del val_loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def get_train_loader(self):
        # NOTE: only used for AREA (i.e., NAS with ZCPs)
        _, train_loader, _ = get_dataloaders(
            self.dataset,
            self.data_path,
            epochs=self.n_epochs,
            gradient_accumulations=1,
            workers=self.workers,
            use_trivial_augment=False,
            eval_mode=self.eval_mode,
        )
        return train_loader


if __name__ == "__main__":
    import argparse
    import re
    from functools import partial

    from benchmarks.objectives.apis.nasbench201 import NAS201
    from benchmarks.objectives.custom_nb201.genotypes import (
        Structure as CellStructure,
    )
    from benchmarks.objectives.custom_nb201.tiny_network import (
        TinyNetwork,
    )
    from benchmarks.objectives.nasbench201 import NasBench201Objective
    from benchmarks.search_spaces.hierarchical_nb201.graph import (
        NB201Spaces,
    )

    # pylint: disable=ungrouped-imports
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    # pylint: enable=ungrouped-imports

    def convert_identifier_to_str(
        identifier: str,
        terminals_to_nb201: dict,
    ) -> str:
        """Converts identifier to string representation."""
        start_indices = [m.start() for m in re.finditer("(OPS*)", identifier)]
        op_edge_list = []
        counter = 0
        for i, _ in enumerate(start_indices):
            start_idx = start_indices[i]
            end_idx = start_indices[i + 1] if i < len(start_indices) - 1 else -1
            substring = identifier[start_idx:end_idx]
            for k in terminals_to_nb201:
                if k in substring:
                    op_edge_list.append(
                        f"{terminals_to_nb201[k]}~{counter}",
                    )
                    break
            if i in (0, 2):
                counter = 0
            else:
                counter += 1

        return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--dataset",
        default="cifar-10",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="Path to folder with data or where data should be saved to if downloaded.",
        type=str,
    )
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--write_graph", action="store_true")
    parser.add_argument("--best_archs", action="store_true")
    parser.add_argument("--nb201_model_backend", action="store_true")
    args = parser.parse_args()

    pipeline_space = {
        "architecture": NB201Spaces(
            space="variable_multi_multi",
            dataset="cifar10",
            use_prior=True,
            adjust_params=False,
        ),
    }
    pipeline_space = SearchSpace(**pipeline_space)
    pipeline_space = pipeline_space.sample(user_priors=True)
    run_pipeline_fn = NB201Pipeline(
        dataset=args.dataset,
        data_path=args.data_path,
        seed=args.seed,
        eval_mode=True,
    )
    res = run_pipeline_fn(
        "",
        "",
        pipeline_space.hyperparameters["architecture"],
    )

    pipeline_space = {
        "architecture": NB201Spaces(
            space="fixed_1_none",
            dataset=args.dataset,
            adjust_params=False,
        ),
    }
    pipeline_space = SearchSpace(**pipeline_space)
    sampled_pipeline_space = pipeline_space.sample()

    # cell_shared = original NB201 space
    pipeline_space = {
        "architecture": NB201Spaces(
            space="variable_multi_multi",
            dataset=args.dataset,
            adjust_params=False,
        ),
    }
    pipeline_space = SearchSpace(**pipeline_space)
    sampled_pipeline_space = pipeline_space.sample()
    identifier = {
        "architecture": "(D2 Linear3 (D1 Linear3 (C Diamond2 (CELL Cell (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv1x1o) (NORM layer))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv3x3o) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM layer)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv1x1o) (NORM instance))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv1x1o) (NORM instance))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV dconv3x3o) (NORM batch)))) (CELL Cell (OPS zero) (OPS zero) (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM layer) (ACT relu)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT relu))) (OPS zero) (OPS avg_pool) (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT mish))))) (C Diamond2 (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT hardswish))) (OPS zero) (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV dconv3x3o) (NORM layer))) (OPS zero)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM layer))) (OPS id)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT relu) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT relu) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT mish)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv3x3o) (NORM layer))) (OPS avg_pool) (OPS id) (OPS zero))) (DOWN Residual2 (CELL Cell (OPS zero) (OPS zero) (OPS id) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM batch))) (OPS zero)) resBlock resBlock)) (D1 Linear3 (C Linear2 (CELL Cell (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT relu))) (OPS zero)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS id) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM batch) (ACT relu))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS avg_pool))) (C Linear2 (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT mish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT relu))) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT relu))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT mish) (NORM layer)))) (CELL Cell (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS id) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT hardswish))) (OPS zero))) (DOWN Residual2 (CELL Cell (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT relu) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT relu) (NORM layer))) (OPS zero) (OPS avg_pool)) resBlock resBlock)) (D0 Residual3 (C Residual2 (CELL Cell (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv1x1o) (NORM batch))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM instance))) (OPS id) (OPS avg_pool)) (CELL Cell (OPS avg_pool) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM batch) (ACT relu))) (OPS avg_pool) (OPS avg_pool)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM layer) (ACT hardswish))) (OPS avg_pool) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT mish))) (OPS zero))) (C Residual2 (CELL Cell (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS zero) (OPS zero) (OPS avg_pool)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM layer))) (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM instance)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM batch))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT relu))) (OPS id))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT hardswish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM batch) (ACT hardswish))) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM instance))) (OPS zero)) (CELL Cell (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv3x3o) (NORM instance))) (OPS zero))))",
    }
    sampled_pipeline_space.load_from(identifier)
    run_pipeline_fn = NB201Pipeline(
        dataset=args.dataset,
        data_path=args.data_path,
        seed=args.seed,
        eval_mode=True,
    )
    res = run_pipeline_fn(
        "",
        "",
        sampled_pipeline_space.hyperparameters["architecture"],
    )

    if args.write_graph:
        writer = SummaryWriter("results/hierarchical_nb201")
        net = sampled_pipeline_space.hyperparameters["architecture"].to_pytorch()
        images = torch.randn((8, 3, 32, 32))
        _ = sampled_pipeline_space.hyperparameters["architecture"](images)
        _ = net(images)
        writer.add_graph(net, images)
        writer.close()

    terminals_to_nb201 = {
        "avg_pool": "avg_pool_3x3",
        "conv1x1": "nor_conv_1x1",
        "conv3x3": "nor_conv_3x3",
        "id": "skip_connect",
        "zero": "none",
    }
    identifier_to_str_mapping = partial(
        convert_identifier_to_str,
        terminals_to_nb201=terminals_to_nb201,
    )
    api = NAS201(
        os.path.dirname(args.data_path),
        negative=False,
        seed=args.seed,
        task=f"{args.dataset}-valid" if args.dataset == "cifar10" else args.dataset,
        log_scale=True,
        identifier_to_str_mapping=identifier_to_str_mapping,
    )
    run_pipeline_fn = NasBench201Objective(api)

    if args.best_archs:
        generator = (
            sampled_pipeline_space.hyperparameters["architecture"]
            .grammars[0]
            .generate()
        )
        archs = list(generator)
        identifier = sampled_pipeline_space.serialize()["architecture"]
        vals = {}
        for arch in tqdm(archs):
            start_idx = 0
            new_identifier = identifier
            for ops in arch[1:]:
                starting_idx = new_identifier.find("OPS", start_idx)
                empty_idx = new_identifier.find(" ", starting_idx)
                closing_idx = new_identifier.find(")", starting_idx)
                new_identifier = (
                    new_identifier[: empty_idx + 1] + ops + new_identifier[closing_idx:]
                )
                start_idx = new_identifier.find(")", starting_idx)

            try:
                sampled_pipeline_space.load_from(
                    {"architecture": new_identifier},
                )
                res_api = run_pipeline_fn(
                    sampled_pipeline_space.hyperparameters["architecture"],
                )
                vals[new_identifier] = 100 * (1 - res_api["info_dict"]["val_score"])
            except Exception:
                pass

        results = sorted(
            vals.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]
    else:
        # seed 777
        identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS conv3x3) (OPS id) (OPS conv3x3) (OPS conv1x1))"
        # identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS zero) (OPS id) (OPS conv3x3) (OPS conv1x1))"
        # identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS avg_pool) (OPS id) (OPS conv3x3) (OPS conv3x3))"
        if args.nb201_model_backend:
            nb201_identifier = identifier_to_str_mapping(identifier)
            genotype = CellStructure.str2structure(nb201_identifier)
            arch_config = {"channel": 16, "num_cells": 5}
            if args.dataset == "cifar10":
                n_classes = 10
            elif args.dataset == "cifar100":
                n_classes = 100
            elif args.dataset == "ImageNet16-120":
                n_classes = 120
            else:
                raise NotImplementedError
            tiny_net = TinyNetwork(
                arch_config["channel"],
                arch_config["num_cells"],
                genotype,
                n_classes,
            )

        sampled_pipeline_space.load_from({"architecture": identifier})

        if args.nb201_model_backend:
            our_model = sampled_pipeline_space.hyperparameters[
                "architecture"
            ].to_pytorch()

            tiny_net_total_params = sum(p.numel() for p in tiny_net.parameters())
            our_model_total_params = sum(p.numel() for p in our_model.parameters())

            new_state_dict = {k: None for k in our_model.state_dict()}
            our_model_values = our_model.state_dict().values()
            for (_k_tiny, v_tiny), (k_our, _v_our) in zip(
                tiny_net.state_dict().items(),
                our_model.state_dict().items(),
                strict=False,
            ):
                new_state_dict[k_our] = v_tiny
            our_model.load_state_dict(new_state_dict)

            input_img = torch.randn((8, 3, 32, 32))
            output_tiny = tiny_net(input_img)
            output_our = our_model(input_img)

        res_api = run_pipeline_fn(
            sampled_pipeline_space.hyperparameters["architecture"],
        )
        res_api_val = 100 * (1 - res_api["info_dict"]["val_score"])
        res_api_test = 100 * (1 - res_api["info_dict"]["test_score"])
        del run_pipeline_fn
        del api

        run_pipeline_fn = NB201Pipeline(
            dataset=args.dataset,
            data_path=args.data_path,
            seed=args.seed,
            eval_mode=True,
        )
        if args.nb201_model_backend:
            res = run_pipeline_fn("", "", tiny_net)
        else:
            res = run_pipeline_fn(
                "",
                "",
                sampled_pipeline_space.hyperparameters["architecture"],
            )
        res_val = res["info_dict"]["valid_acc1es"][-1]
        res_test = res["info_dict"]["test_acc1es"][-1]
