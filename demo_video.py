import argparse
import torch
import os
import time
import cv2
from torch.amp import autocast
from train_net import setup, Trainer, DetectionCheckpointer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import add_panoptic_swiftnet_config
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.visualizer import Visualizer
from numba import njit

MIN_AREA_THRESHOLD = 1000
factor = 4

cityscapes_class_names = np.array([
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle', 'Backgorund'])
def parse_args():
    parser = argparse.ArgumentParser(description='Demo inference on a camera stream or video file.')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input', help='Path to input video file or camera device ID', required=True)
    parser.add_argument('--output', help='Path to output video file or folder', default='output-preds', required=False)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

# def panoptic_labels_to_color(label, colormap, label_divisor):
#     colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
#     taken_colors = set([0, 0, 0])

#     def _random_color(base, max_dist=50):
#         new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
#         return tuple(np.maximum(0, np.minimum(255, new_color)))

#     for lab in np.unique(label):
#         mask = label == lab
#         base_color = colormap[lab // label_divisor]
#         if tuple(base_color) not in taken_colors:
#             taken_colors.add(tuple(base_color))
#             color = base_color
#         else:
#             while True:
#                 color = _random_color(base_color)
#                 if color not in taken_colors:
#                     taken_colors.add(color)
#                     break
#         colored_label[mask] = color
#     return colored_label


@njit
def panoptic_labels_to_color_numba(label, lookup):
    h, w = label.shape
    colored = np.empty((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            colored[i, j] = lookup[label[i, j]]
    return colored

# Fuera de la función, se puede crear la tabla lookup de forma vectorizada:
def create_lookup(colormap, label_divisor, max_label):
    lookup = np.empty((max_label + 1, 3), dtype=np.uint8)
    for lab in range(max_label + 1):
        # Simplemente asigna el color base (sin evitar duplicados)
        lookup[lab] = colormap[lab // label_divisor]
    return lookup

# Uso:
def panoptic_labels_to_color(label, colormap, label_divisor):
    max_label = label.max()
    lookup = create_lookup(colormap, label_divisor, max_label)
    return panoptic_labels_to_color_numba(label, lookup)



def process_frame(frame, transform, model, colormap, meta, start_time, frame_width, frame_height):
    h, w, _ = frame.shape
    img = transform.get_transform(frame).apply_image(frame)
    model_input = [{
        "image": torch.from_numpy(img).permute(2, 0, 1).to('cuda'),
        "width": w,
        "height": h
    }]
    with torch.no_grad():
        with autocast("cuda", enabled= True, dtype=torch.float16, cache_enabled= True):  # Mixed precision
            out = model(model_input)[0]
            
    # Extraer el mapa de segmentación (panoptic_seg) y la imagen de colores
    
    seg_map = out["panoptic_seg"][0].squeeze().cpu().numpy()
    seg_map = cv2.resize(seg_map, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    torch.cuda.synchronize()
    inference_time_seg = (time.time() - start_time)*1000
    print(f"Inference time seg map: {inference_time_seg:.4f} ms")

    preds_color = panoptic_labels_to_color(seg_map, colormap, meta.label_divisor)
    torch.cuda.synchronize()
    inference_time_color = (time.time() - start_time)*1000
    print(f"Inference time color: {inference_time_color:.4f} ms")

    # Obtener confianza desde sem_seg
    sem_seg_logits = out["sem_seg"]
    probabilities = F.softmax(sem_seg_logits, dim=0)
    confidence_values, _ = torch.max(probabilities, dim=0)
    confidence_map = confidence_values.cpu().numpy()
    torch.cuda.synchronize()
    inference_time_conf = (time.time() - start_time)*1000
    print(f"Inference time conf: {inference_time_conf:.4f} ms")
    confidence_map = cv2.resize(confidence_map, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)


    return preds_color, seg_map, confidence_map

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
    torch.backends.cudnn.benchmark = True
    dataset_name = cfg.DATASETS.TEST[0]
    meta = MetadataCatalog.get(dataset_name)
    if "coco" in dataset_name or "cityscapes" in dataset_name:
        colormap = np.array(meta.stuff_colors + meta.thing_colors)
    else:
        colormap = np.array(meta.colors)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to('cuda')  # Move model to GPU

    transform = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

    # Configurar video de entrada y salida
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(f'{args.output}/output_video.avi', fourcc, fps, (width, height))

    inference_times = []  # Lista para almacenar tiempos de inferencia

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        torch.cuda.synchronize()
        start_time = time.time()
        frame = cv2.resize(frame, (int(width/factor), int(height/factor)), interpolation=cv2.INTER_NEAREST)
        preds_color, seg_map,confidence_map = process_frame(frame, transform, model, colormap, meta, start_time, width, height)
        torch.cuda.synchronize()
        inference_time = (time.time() - start_time)*1000
        inference_times.append(inference_time)
        print(f"Inference time: {inference_time:.4f} ms")

        # Calcular el centroide de cada región y escribir el label
        unique_labels = np.unique(seg_map)
        for lab in unique_labels:
            mask = (seg_map == lab).astype(np.uint8)
            if np.count_nonzero(mask) == 0:
                continue
            M = cv2.moments(mask)
            # Filtrar regiones pequeñas por área (M["m00"] es el área)
            if M["m00"] < MIN_AREA_THRESHOLD:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            semantic_id = lab // meta.label_divisor
            label_text = cityscapes_class_names[semantic_id] if semantic_id < len(cityscapes_class_names) else str(semantic_id)
            confidence = confidence_map[cY, cX]
            
            # Mostrar la etiqueta y la confianza en porcentaje
            cv2.putText(preds_color, f"{label_text} ({confidence*100:.1f}%)", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out_frame = cv2.cvtColor(preds_color, cv2.COLOR_RGB2BGR)
        out_video.write(out_frame)
        cv2.imshow("Frame", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Calcular estadísticas de inferencia
    inference_times = np.array(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    print(f"Min inference time: {min_time:.4f} s")
    print(f"Max inference time: {max_time:.4f} s")
    print(f"Std of inference time: {std_time:.4f} s")

    # Graficar los tiempos de inferencia
    plt.figure(figsize=(10, 5))
    plt.plot(inference_times, label="Inference time per frame")
    plt.xlabel("Frame index")
    plt.ylabel("Time (s)")
    plt.title("Inference Time per Frame")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    main(args)
