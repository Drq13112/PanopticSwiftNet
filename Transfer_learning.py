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

from detectron2.engine import HookBase



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.cuda.empty_cache()


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


class LossLogger(HookBase):
    def after_step(self):
        # Always print something to verify the hook is called
        print(f"[LossLogger] After step at iter {self.trainer.iter}")
        storage = self.trainer.storage
        # Try to print all available keys
        metrics = storage.latest_with_smoothing_hint()
        print(f"[LossLogger] Available metrics: {metrics.keys()}")
        
        # If you have specific keys, try printing them:
        loss_sem_seg = metrics.get("loss_sem_seg", None)
        loss_center = metrics.get("loss_center", None)
        loss_offset = metrics.get("loss_offset", None)
        if loss_sem_seg is not None:
            print(
                f"Iter {self.trainer.iter}: loss_sem_seg={loss_sem_seg:.4f}, "
                f"loss_center={loss_center:.4f}, loss_offset={loss_offset:.4f}"
            )

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._data_loader_iter = iter(self.data_loader)  # Initialize data loader iterator

        # Ensure the model uses float32 to avoid dtype mismatch
        #self.model = self.model.float()

        # Move the model to GPU (cuda)
        self.model = self.model.to("cuda")

    # def run_step(self):
    #     """ Custom training step with proper data loader initialization """
    #     # assert self.model.training, "[Trainer] Model is in eval mode!"
        
    #     try:
    #         data = next(self._data_loader_iter)
    #     except StopIteration:
    #         self._data_loader_iter = iter(self.data_loader)
    #         data = next(self._data_loader_iter)

    #     # # Move all input tensors to float32 and GPU
    #     # for d in data:
    #     #     for k in d:
    #     #         if isinstance(d[k], torch.Tensor):
    #     #             d[k] = d[k].float().to("cuda")
    #     #     # Convert segmentation target to long (if available)
    #     #     if "sem_seg" in d and isinstance(d["sem_seg"], torch.Tensor):
    #     #         d["sem_seg"] = d["sem_seg"].long().to("cuda")

    #     self.optimizer.zero_grad()
    #     loss_dict = self.model(data)
    #     print("Loss dict:", {k: v.item() for k, v in loss_dict.items()})
    #     losses = sum(loss_dict.values())
    #     losses.backward()
    #     self.optimizer.step()


    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # Ensure segmentation head exists
        if hasattr(model.sem_seg_head, "classifier"):
            if "sem_seg" in model.sem_seg_head.classifier:
                old_conv = model.sem_seg_head.classifier["sem_seg"][2]  # Get final Conv2D layer
                
                # Replace with a new Conv2D layer for 7 classes
                new_conv = torch.nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES,  # 7 classes
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding
                )

                # # Convert new weights and bias to float16 if using mixed precision
                # if cfg.SOLVER.AMP.ENABLED:
                #     model = model.half()  # Convert entire model to float16
                #     new_conv.weight.data = new_conv.weight.data.half()  # Convert weights to fp16
                #     if new_conv.bias is not None:
                #         new_conv.bias.data = new_conv.bias.data.half()  # Convert bias to fp16

                model.sem_seg_head.classifier["sem_seg"][2] = new_conv  # Assign new Conv2D

                print(f"✅ Replaced final segmentation layer: {old_conv.out_channels} → {cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES}")

        return model

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
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = "pretrained_models/psn-r18-city.pth"  # pre-trained weights
    cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Freeze early layers
    cfg.merge_from_list(args.opts)
    # 🚀 Reduce image resolution to lower VRAM usage
    cfg.INPUT.MIN_SIZE_TRAIN = (256,)  
    cfg.INPUT.MAX_SIZE_TRAIN = 768  

    # 🚀 Ensure better memory allocation for upsampling
    cfg.MODEL.PANOPTIC_SWIFTNET.SIZE_DIVISIBILITY = 64  
    cfg.MODEL.PANOPTIC_SWIFTNET.FINAL_UP_ALIGN_CORNERS = False  

    cfg.SOLVER.IMS_PER_BATCH = 1  # Reduce batch size to prevent OOM
    cfg.SOLVER.BASE_LR = 0.00005  # Reduce LR for stability

    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.DEVICE = "cuda"

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    #  Ensure correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Build the model and move it to the correct device
    model = Trainer.build_model(cfg).to(device)

    #  Load pre-trained weights properly
    checkpoint_path = cfg.MODEL.WEIGHTS  # Ensure correct path
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint to correct device

    print("Checkpoint keys:", checkpoint.keys())

    #  Convert weights to float32 if necessary
    state_dict = {k: v.float().to(device) for k, v in checkpoint.items()}  # Move everything to same device

    #  Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f" Checkpoint loaded successfully!\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")

    #  Initialize Trainer and start training
    trainer = Trainer(cfg)
    # trainer.register_hooks([LossLogger()])
    trainer.resume_or_load(resume=args.resume)  # Resume from last checkpoint if available
    trainer.train()  # 🚀 Start training!


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