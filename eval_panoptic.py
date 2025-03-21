#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from config import add_panoptic_swiftnet_config
from data import register_mvd
from data.build import build_detection_train_loader
from data.class_uniform_sampling import ClassUniformCrop
from data.dataset_mapper import PanopticDeeplabDatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from model.psn import PanopticSwiftNet
from util import SemSegEvaluator
import matplotlib.pyplot as plt
import numpy as np



def get_metric_value(result, metric):
    """
    Devuelve el valor de la métrica buscada en el diccionario de resultados.
    Primero busca en 'panoptic_seg', luego en 'sem_seg', y por último en 'segm'.
    """
    for key in ['panoptic_seg', 'sem_seg', 'segm']:
        if metric in result.get(key, {}):
            return result[key][metric]
    return None

def plot_results_with_inference(results, inf_times, metric_keys, test_labels=None, colors=None, inf_scale=1000):
    """
    Plots performance metrics and inference times.

    Args:
        results: List of dictionaries with evaluation metrics (e.g., each result has keys 'panoptic_seg', 'sem_seg', etc.).
        inf_times: List of inference times (in seconds) for each test.
        metric_keys: List of metric names to plot for performance (e.g., ['PQ', 'SQ', 'RQ', 'IoU']).
        test_labels: List of labels for each test (e.g., resolutions like ['1024x2048', '512x1024', ...]).
        colors: Optional list of colors for the tests.
        inf_scale: Scale factor for inference times (default multiplies seconds by 1000 to get ms).
    """
    # Ensure results is a list
    if not isinstance(results, (list, tuple)):
        results = [results]

    n_tests = len(results)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)  # x positions for metrics

    if colors is None:
        colors = plt.cm.Set1(np.linspace(0, 1, n_tests))
    if test_labels is None:
        test_labels = [f"Test {i+1}" for i in range(n_tests)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot performance metrics in ax1:
    for i, metric in enumerate(metric_keys):
        # Extract the values for each test
        values = [get_metric_value(result, metric) for result in results]
        # Sort tests by value so that the highest is plotted first (behind)
        sorted_indices = np.argsort(values)[::-1]
        for j, idx in enumerate(sorted_indices):
            ax1.bar(x[i], values[idx], color=colors[idx], width=0.6,
                    zorder=j, alpha=0.7,
                    label=test_labels[idx] if i == 0 else "")
    ax1.set_ylabel("Performance Metrics")
    ax1.set_title("Performance Metrics Comparison")
    ax1.legend()

    # Plot inference times in ax2:
    # Convert inference times from seconds to milliseconds using inf_scale
    inf_ms = [t * inf_scale for t in inf_times]
    # Draw a separate bar for each test; here we assume one bar per test
    ax2.bar(np.arange(n_tests), inf_ms, color=colors, alpha=0.7)
    ax2.set_ylabel("Inference Time (ms)")
    ax2.set_title("Inference Times")
    ax2.set_xticks(np.arange(n_tests))
    ax2.set_xticklabels(test_labels)
    
    # Set x-axis labels for the entire figure using metric keys (for the top plot)
    plt.xticks(x, metric_keys)
    plt.xlabel("Metrics")
    plt.tight_layout()
    plt.show()

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if not cfg.INPUT.CLASS_BALANCED_CROPS and cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    elif cfg.INPUT.CLASS_BALANCED_CROPS and cfg.INPUT.CROP.ENABLED:
        augs.append(ClassUniformCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if "mvd" in dataset_name:
            evaluator_list = [
                SemSegEvaluator(dataset_name, False, "/home/jsaric/temp/"),
                COCOPanopticEvaluator(dataset_name, "/home/jsaric/temp/")
            ]
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = build_sem_seg_train_aug(cfg)
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=augs)
        loader = build_detection_train_loader(cfg, mapper=mapper)
        if isinstance(augs[1], ClassUniformCrop):
            augs[1].set_cat_repeat_factors(loader.dataset.dataset.sampler._cat_repeat_factors)
        return loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    add_panoptic_swiftnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res, infs = Trainer.test(cfg, model)
        plot_results_with_inference(res, infs, ['PQ', 'SQ', 'RQ','IoU'], test_labels=['1024x2048', '512x1024', '342x683','256x512'])
        print("res: ",res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
