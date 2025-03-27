# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Josip Saric.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from pathlib import Path
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torch.nn.functional as F
from detectron2.structures import (
    BoxMode
)
from .target_generator import PanopticDeepLabTargetGenerator
import os
from pathlib import Path

__all__ = ["PanopticDeeplabDatasetMapper"]

def map_labels(label):
    """
    Extrae el category_id (asumiendo formato panoptic si max > 1000)
    y remapea los IDs originales de Cityscapes a los índices contiguos de las 8 clases de interés.
    Se asigna 255 a los píxeles que no corresponden a estas 8 clases.
    """
    if label.max() > 1000:
        category_ids = label // 1000
    else:
        category_ids = label
    mapping = {
        7: 0,    # road
        24: 1,   # person
        20: 2,   # traffic sign
        26: 3,   # car
        27: 4,   # truck
        28: 5,   # bus
        8: 6,    # sidewalk (calzada)
        32: 7    # motorcycle
    }
    mapped = np.full_like(category_ids, 255)
    for orig, new in mapping.items():
        mapped[category_ids == orig] = new
    return mapped

def colorize_segmentation(seg, colormap):
    """
    Convierte un mapa de etiquetas (H x W) en una imagen RGB usando el diccionario colormap.
    """
    H, W = seg.shape
    color_seg = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colormap.items():
        color_seg[seg == label] = np.array(color, dtype=np.uint8)
    return color_seg

# Definir un colormap para depuración:
CUSTOM_COLORMAP = {
    0: (128, 64, 128),   # road
    1: (220, 20, 60),    # person
    2: (220, 220, 0),    # traffic sign
    3: (0, 0, 142),      # car
    4: (0, 0, 70),       # truck
    5: (0, 60, 100),     # bus
    6: (244, 35, 232),   # sidewalk
    7: (0, 0, 230),      # motorcycle
    255: (0, 0, 0)       # ignore (negro)
}

class ExtendedAugInput(T.AugInput):
    def __init__(self, image, boxes=None, sem_seg=None, box_labels=None, baol_offset_weights=None):
        super(ExtendedAugInput, self).__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.box_labels = box_labels
        self.baol_offset_weights = baol_offset_weights

    def transform(self, tfm):
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)
        if self.baol_offset_weights is not None:
            self.baol_offset_weights = tfm.apply_segmentation(self.baol_offset_weights)


class PanopticDeeplabDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        panoptic_target_generator: Callable,
        load_offset_weights=False,
        offset_weights_folder_path="",
        visualize_debug=False  # Flag para activar visualización de debug
    ):
        """
        NOTE: this interface is experimental.
        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator
        self.load_offset_weights = load_offset_weights
        self.offset_weights_folder_path = offset_weights_folder_path
        self.visualize_debug = visualize_debug
        # Define la ruta del escritorio y la carpeta "debug"
        self.debug_output_dir = os.path.expanduser("/home/david/Documents/Panoptic_Seg/panoptic-swiftnet/debug")


    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "load_offset_weights": cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.ENABLED,
            "offset_weights_folder_path": cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.WEIGHTS_PATH
        }
        return ret

    # def __call__(self, dataset_dict):
    #     """
    #     Args:
    #         dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    #     Returns:
    #         dict: a format that builtin models in detectron2 accept
    #     """
    #     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    #     # Load image.
    #     image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
    #     utils.check_image_size(dataset_dict, image)
    #     # Panoptic label is encoded in RGB image.
    #     pan_seg_file_name = dataset_dict.pop("pan_seg_file_name")

    #     pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
    #     baol_offset_weights = None
    #     if self.load_offset_weights:
    #         try:
    #             baol_offset_weights = utils.read_image(f"{self.offset_weights_folder_path}/{Path(pan_seg_file_name).stem}.png")
    #         except Exception:
    #             print("Could not load baol offset weights for image: ", pan_seg_file_name)
    #             baol_offset_weights = np.ones(pan_seg_gt.shape[:2])


    #     # Reuses semantic transform for panoptic labels.
    #     boxes, box_labels = [], []
    #     for obj in dataset_dict["segments_info"]:
    #         boxes.append(obj["bbox"])
    #         box_labels.append(obj["category_id"])
    #     if len(boxes) != 0:
    #         boxes = np.array(boxes)
    #         box_labels = np.array(box_labels)
    #         boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    #         boxes = torch.as_tensor(boxes)
    #         box_labels = torch.as_tensor(box_labels)
    #     else:
    #         boxes = None
    #         box_labels = None

    #     aug_input = ExtendedAugInput(
    #         image,
    #         sem_seg=pan_seg_gt,
    #         boxes=boxes,
    #         box_labels=box_labels,
    #         baol_offset_weights=baol_offset_weights
    #     )
    #     _ = self.augmentations(aug_input)
    #     image, pan_seg_gt = aug_input.image, aug_input.sem_seg
    #     # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    #     # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    #     # Therefore it's important to use torch.Tensor.
    #     dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    #     # Generates training targets for Panoptic-DeepLab.

    #     targets = self.panoptic_target_generator(
    #         rgb2id(pan_seg_gt),
    #         dataset_dict["segments_info"],
    #         same_pallet_ids=dataset_dict.get("same-pallet-ids", None)
    #     )
    #     if self.load_offset_weights:
    #         baol_offset_weights = torch.from_numpy(np.array(aug_input.baol_offset_weights)).unsqueeze(0)
    #         targets["offset_weights"] = targets["offset_weights"] * baol_offset_weights

    #     dataset_dict.update(targets)

    #     return dataset_dict

    # def __call__(self, dataset_dict):
    #     dataset_dict = copy.deepcopy(dataset_dict)
    #     # Cargar imagen original y comprobar tamaño.
    #     image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
    #     utils.check_image_size(dataset_dict, image)
        
    #     # Cargar anotación panoptic (archivo RGB)
    #     pan_seg_file_name = dataset_dict.pop("pan_seg_file_name")
    #     pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
        
    #     # Cargar offset weights si se requieren.
    #     baol_offset_weights = None
    #     if self.load_offset_weights:
    #         try:
    #             baol_offset_weights = utils.read_image(
    #                 f"{self.offset_weights_folder_path}/{Path(pan_seg_file_name).stem}.png"
    #             )
    #         except Exception:
    #             print("Could not load baol offset weights for image: ", pan_seg_file_name)
    #             baol_offset_weights = np.ones(pan_seg_gt.shape[:2])
        
    #     # Procesar cajas y etiquetas de los objetos.
    #     boxes, box_labels = [], []
    #     for obj in dataset_dict["segments_info"]:
    #         boxes.append(obj["bbox"])
    #         box_labels.append(obj["category_id"])
    #     if len(boxes) != 0:
    #         boxes = np.array(boxes)
    #         box_labels = np.array(box_labels)
    #         boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    #         boxes = torch.as_tensor(boxes)
    #         box_labels = torch.as_tensor(box_labels)
    #     else:
    #         boxes = None
    #         box_labels = None

    #     # Aplicar augmentations.
    #     aug_input = ExtendedAugInput(
    #         image,
    #         sem_seg=pan_seg_gt,
    #         boxes=boxes,
    #         box_labels=box_labels,
    #         baol_offset_weights=baol_offset_weights
    #     )
    #     _ = self.augmentations(aug_input)
    #     image, pan_seg_gt = aug_input.image, aug_input.sem_seg
    #     dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
    #     # --- GENERACIÓN DE TARGETS ---
    #     # Convertir anotación panoptic a IDs y aplicar remapeo.
    #     aug_pan_seg_ids = rgb2id(pan_seg_gt)
    #     # Imprimir estadísticas de la anotación original:
    #     unique_orig = np.unique(aug_pan_seg_ids)
    #     print("Unique target values before mapping:", unique_orig)
    #     aug_pan_seg_ids = map_labels(aug_pan_seg_ids)
    #     unique_mapped = np.unique(aug_pan_seg_ids)
    #     print("Unique target values after mapping:", unique_mapped)
        
    #     # Calcular y mostrar porcentaje de píxeles válidos:
    #     valid_pixels = np.sum(aug_pan_seg_ids != 255)
    #     total_pixels = aug_pan_seg_ids.size
    #     print(f"Pixels valid: {valid_pixels} / {total_pixels} ({valid_pixels/total_pixels*100:.2f}%)")
    #     # Si se solicita visualización de depuración, generar imagen colorizada.
    #     if self.visualize_debug:
    #         color_seg = colorize_segmentation(aug_pan_seg_ids, CUSTOM_COLORMAP)
    #         vis_path = os.path.join(debug_output_dir, f"vis_{Path(pan_seg_file_name).stem}.png")
    #         from PIL import Image
    #         Image.fromarray(color_seg).save(vis_path)
    #         print("Saved visualization at:", vis_path)
        
    #     targets = self.panoptic_target_generator(
    #         aug_pan_seg_ids,
    #         dataset_dict["segments_info"],
    #         same_pallet_ids=dataset_dict.get("same-pallet-ids", None)
    #     )
        
    #     if self.load_offset_weights:
    #         baol_offset_weights = torch.from_numpy(np.array(aug_input.baol_offset_weights)).unsqueeze(0)
    #         targets["offset_weights"] = targets["offset_weights"] * baol_offset_weights

    #     dataset_dict.update(targets)
    #     return dataset_dict

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        pan_seg_file_name = dataset_dict.pop("pan_seg_file_name")
        pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
        
        # Cargar offset weights si se requieren
        baol_offset_weights = None
        if self.load_offset_weights:
            try:
                baol_offset_weights = utils.read_image(
                    f"{self.offset_weights_folder_path}/{Path(pan_seg_file_name).stem}.png"
                )
            except Exception:
                print("Could not load baol offset weights for image: ", pan_seg_file_name)
                baol_offset_weights = np.ones(pan_seg_gt.shape[:2])
        
        # Procesar cajas y etiquetas de objetos
        boxes, box_labels = [], []
        for obj in dataset_dict["segments_info"]:
            boxes.append(obj["bbox"])
            box_labels.append(obj["category_id"])
        if len(boxes) != 0:
            boxes = np.array(boxes)
            box_labels = np.array(box_labels)
            boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            boxes = torch.as_tensor(boxes)
            box_labels = torch.as_tensor(box_labels)
        else:
            boxes = None
            box_labels = None

        # Aplicar augmentations a imagen, anotación y offset weights
        aug_input = ExtendedAugInput(
            image,
            sem_seg=pan_seg_gt,
            boxes=boxes,
            box_labels=box_labels,
            baol_offset_weights=baol_offset_weights
        )
        _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        # Generar el target panoptic completo (con separación de instancias)
        panoptic_ids = rgb2id(pan_seg_gt)  # Conserva la información de instancia
        # Para la rama semántica, extraer el category_id y remapear a [0,7] (el resto a 255)
        semantic_target = map_labels(panoptic_ids)
        
        # print("Unique target values before mapping:", np.unique(panoptic_ids))
        # print("Unique target values after mapping:", np.unique(semantic_target))
        valid_pixels = np.sum(semantic_target != 255)
        total_pixels = semantic_target.size
        # print(f"Pixels valid: {valid_pixels} / {total_pixels} ({(valid_pixels/total_pixels)*100:.2f}%)")
        
        # Guardar visualización de debug si se activa
        if self.visualize_debug:
            CUSTOM_COLORMAP = {
                0: (128, 64, 128),    # road
                1: (220, 20, 60),     # person
                2: (220, 220, 0),     # traffic sign
                3: (0, 0, 142),       # car
                4: (0, 0, 70),        # truck
                5: (0, 60, 100),      # bus
                6: (244, 35, 232),    # sidewalk (calzada)
                7: (0, 0, 230),       # motorcycle
                255: (0, 0, 0)        # background
            }
            from PIL import Image
            colorized = colorize_segmentation(semantic_target, CUSTOM_COLORMAP)
            vis_path = os.path.join(self.debug_output_dir, f"vis_{Path(pan_seg_file_name).stem}.png")
            Image.fromarray(colorized).save(vis_path)
            print("Saved visualization at:", vis_path)
        
        # Generar targets para entrenamiento usando el panoptic target generator
        targets = self.panoptic_target_generator(
            panoptic_ids,
            dataset_dict["segments_info"],
            same_pallet_ids=dataset_dict.get("same-pallet-ids", None)
        )
        
        # Agregar además el target semántico para la rama sem_seg (opcional para debug o para losses separadas)
        dataset_dict["semantic_target"] = torch.as_tensor(semantic_target)
        dataset_dict.update(targets)
        return dataset_dict