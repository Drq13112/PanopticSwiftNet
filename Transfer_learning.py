#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified Panoptic-DeepLab Training Script for Transfer Learning.
This script freezes the lower layers of the backbone and reconfigures the final 
segmentation head to detect only 8 classes:
  - Carretera
  - Peatones
  - Señales de tráfico
  - Coches
  - Camiones
  - Buses
  - Calzada
  - Motos
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
from detectron2.engine import HookBase
import csv


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


def to_float(x):
    """Convert a tensor, tuple, or number to a float."""
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, (list, tuple)):
        # If it's a tuple or list, we try to convert the first element.
        return to_float(x[0])
    else:
        return float(x)

class CSVWriterHook(HookBase):
    def __init__(self, output_file):
        self.output_file = output_file
        # Write header
        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "loss_total", "loss_sem_seg", "loss_center", "loss_offset"])

    def after_step(self):
        storage = self.trainer.storage
        iteration = self.trainer.iter

        # Retrieve the losses from storage
        # Use default value 0 if a key does not exist.
        loss_total = storage.latest().get("total_loss", 0)
        loss_sem_seg = storage.latest().get("loss_sem_seg", 0)
        loss_center = storage.latest().get("loss_center", 0)
        loss_offset = storage.latest().get("loss_offset", 0)

        # Convert values to float
        loss_total = to_float(loss_total)
        loss_sem_seg = to_float(loss_sem_seg)
        loss_center = to_float(loss_center)
        loss_offset = to_float(loss_offset)

        # Append the current iteration losses to the CSV file.
        with open(self.output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, loss_total, loss_sem_seg, loss_center, loss_offset])
class Trainer(DefaultTrainer):
    """
    Modified trainer for transfer learning:
    - Freezes lower layers of the backbone.
    - Replaces the final segmentation layer for 8 classes.
    """

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg).to(cfg.MODEL.DEVICE)


        # # Freeze lower layers using cfg.MODEL.BACKBONE.FREEZE_AT.
        # # (Este ejemplo asume un backbone basado en ResNet con capas nombradas "layer1", "layer2", etc.)
        # freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        # for name, param in model.backbone.named_parameters():
        #     # Ejemplo: si freeze_at>=2, congelamos layer1
        #     if freeze_at >= 2 and "layer1" in name:
        #         param.requires_grad = False
        #     # Si freeze_at>=3, se podría congelar también layer2, y así sucesivamente.


        # Congelar parámetros del backbone        # for name, param in model.backbone.named_parameters():
        #     param.requires_grad = False

        # # Congelar parámetros del decoder
        # for name, param in model.decoder.named_parameters():
        #     param.requires_grad = False


        # Verificar qué partes se actualizarán:
        print("Parámetros que se actualizarán:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        
        # Reemplazar la última capa de clasificación.
        if hasattr(model.sem_seg_head, "classifier"):
            if "sem_seg" in model.sem_seg_head.classifier:
                old_conv = model.sem_seg_head.classifier["sem_seg"][2]  # Capa final actual
                new_conv = torch.nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES,  # 8 clases
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding
                )
                # Traslada explícitamente la nueva capa a GPU:
                new_conv = new_conv.to(next(model.parameters()).device)
                model.sem_seg_head.classifier["sem_seg"][2] = new_conv
                print(f"Replaced final segmentation layer: {old_conv.out_channels} -> {cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES}")
        return model


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
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
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
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
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    add_panoptic_swiftnet_config(cfg)
    args.config_file = "configs/Cityscapes-PanopticSegmentation/panoptic_swiftnet_R_18_cityscapes_D256_baol.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Ajustes para Transfer Learning:
    cfg.MODEL.BACKBONE.FREEZE_AT = 5  # Congela las capas inferiores; ajústalo según tus necesidades.
    cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES = 8  # 8 clases deseadas
    cfg.SOLVER.BASE_LR = 1e-3 
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.DEVICE = "cuda"
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res
    trainer = Trainer(cfg)
    trainer.register_hooks([CSVWriterHook(output_file="training_progress.csv")])
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
