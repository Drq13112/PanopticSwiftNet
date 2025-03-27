import argparse

import numpy as np
from PIL import Image
from glob import glob
import cv2
from pathlib import Path
import os
from panopticapi.utils import rgb2id
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(description='Script to create offset weights for the boundary offset loss.')
    parser.add_argument('--input',
                        help='Path to the input folder',
                        required=True)
    parser.add_argument('--output',
                        help='Path to the output folder',
                        default='./offset_weights/',
                        required=False)
    parser.add_argument('--ext',
                        help='Image extension in the input folder. Default: .png',
                        default='.png',
                        required=False)
    args = parser.parse_args()
    return args


class OffsetWeightsTransform:
    def __init__(self,
                 ignore_id,
                 bins=(4, 16, 64, 128),
                 alphas=(8., 6., 4., 2., 1.),
                 size_relative=False,
                 load_cached=False):
        self.bins = bins
        self.alphas = alphas
        self.ignore_id = ignore_id
        self.size_relative = size_relative
        self.load_cached = load_cached

    def str_transform_code(self):
        alphas = "_".join([str(x) for x in self.alphas])
        alphas = alphas.replace(".", "_")
        if self.size_relative:
            s = f"SR_{alphas}"
        else:
            bins = "_".join([str(x) for x in self.bins])
            s = f"/SR_{alphas}_B{bins}"
        return s

    def compute_offset_weight(self, labels):
        present_classes = np.unique(labels)
        distances = np.zeros(labels.shape, dtype=np.float32) - 1.
        label_distance_alphas = np.zeros(distances.shape, dtype=np.float32)
        for i in present_classes:
            if i == self.ignore_id:
                continue
            class_mask = labels == i
            _dist = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
            distances[class_mask] = _dist
            bins = self.bins
            if self.size_relative:
                bins = np.linspace(0, _dist.max(), len(self.alphas))[1:]
            label_distance_bins = np.digitize(distances, bins)
            for idx, alpha in enumerate(self.alphas):
                label_distance_alphas[np.logical_and(label_distance_bins == idx, class_mask)] = alpha

        return label_distance_alphas, distances
    # def compute_offset_weight(self, labels):
    #     present_classes = np.unique(labels)
    #     distances = np.zeros(labels.shape, dtype=np.float32) - 1.
    #     label_distance_alphas = np.zeros(distances.shape, dtype=np.float32)
    #     for i in present_classes:
    #         if i == self.ignore_id:
    #             continue
    #         class_mask = labels == i
    #         dt = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)
    #         _dist = dt[class_mask].reshape(-1)  # Aseguramos que sea 1D
    #         distances[class_mask] = _dist
    #         bins = self.bins
    #         if self.size_relative:
    #             bins = np.linspace(0, _dist.max(), len(self.alphas))[1:]
    #         label_distance_bins = np.digitize(distances, bins)
    #         for idx, alpha in enumerate(self.alphas):
    #             label_distance_alphas[np.logical_and(label_distance_bins == idx, class_mask)] = alpha

    #     return label_distance_alphas, distances


def rgb2id_image(img):
    # img: numpy array of shape (H, W, 3)
    return img[..., 0].astype(np.int32) + 256 * img[..., 1].astype(np.int32) + 256 * 256 * img[..., 2].astype(np.int32)

if __name__ == '__main__':
    args = parse_args()
    ext = args.ext
    input_folder = args.input
    output_folder = args.output
    ext = ".png"
    workers = 32
    transform = OffsetWeightsTransform(
        ignore_id=0,
        alphas=[8., 4., 2., 1.],
        size_relative=True
    )
    print(f"Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Finding all files with extension {ext} from the input folder: {input_folder}.")
    #paths = list(glob(f"{input_folder}/**/*gtFine_trainIds{ext}", recursive=True))# + \
        #list(glob(f"{input_folder}/**/*gtFine_color{ext}", recursive=True))
    paths = list(glob(f"{input_folder}/**/*{ext}", recursive=True))
    # Excluir archivos que estén en la carpeta de salida:
    paths = [p for p in paths if output_folder not in p]
    print(f"Found {len(paths)} total labels.")
    # print("paths:", paths)

    def f(path):
        path = Path(path)
        # print("started:", path)
        img = Image.open(path)
        img_np = np.array(img)
        # Si la imagen es en escala de grises, se usa directamente
        if img_np.ndim == 2:
            labels = img_np.astype(np.int32)
        # Si la imagen tiene 3 canales, se aplica la conversión
        elif img_np.ndim == 3:
            labels = rgb2id(img_np).squeeze()
        else:
            raise ValueError("Formato de imagen desconocido")
        
        offset_weights, distances = transform.compute_offset_weight(labels)
        offset_weights = offset_weights.squeeze().astype(np.uint8)
        offset_weights_img = Image.fromarray(offset_weights)
        offset_weights_img.save(f"{output_folder}/{path.name}")
        print("path done!!!:", path)

    # def f(path):
    #     path = Path(path)
    #     img = np.array(Image.open(path))
    #     labels = rgb2id_image(img).squeeze()
    #     offset_weights, distances = transform.compute_offset_weight(labels)
    #     offset_weights = offset_weights.squeeze().astype(np.uint8)
    #     offset_weights_img = Image.fromarray(offset_weights)
    #     offset_weights_img.save(f"{output_folder}/{path.name}")
    #     print(path)

    print(f"Initializing multiprocessing pool with {workers} workers.")
    with Pool(workers) as p:
        print(p.map(f, paths))