"""Litter detection inference using the trained Faster R-CNN frozen graph."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "litter_inference_graph", "frozen_inference_graph.pb")

DEFAULT_THRESHOLD = 0.65
MAX_BOXES = 30
MAX_DIMENSION = 640

CATEGORY_INDEX = {
    1: {"id": 1, "name": "litter"},
}

RANK_LABELS = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    annotated_image: Image.Image
    detections: List[Detection]
    litter_count: int
    severity_rank: int
    severity_label: str
    inference_seconds: float
    summary: str


class LitterDetector:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._graph = None
        self._session = None
        self._image_tensor = None
        self._tensor_dict = None

    def load(self) -> None:
        if self._session is not None:
            return

        if not os.path.exists(self.model_path) or _is_lfs_pointer(self.model_path):
            from download_model import download_model

            download_model()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run: bash download_model.sh"
            )

        if os.path.getsize(self.model_path) < 1_000_000:
            raise RuntimeError(
                "Model file is incomplete (Git LFS placeholder).\n"
                "Run: bash download_model.sh"
            )

        tf.compat.v1.disable_eager_execution()
        self._graph = tf.Graph()
        with self._graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, "rb") as graph_file:
                graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def, name="")

            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in (
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
            ):
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name
                    )

            self._tensor_dict = tensor_dict
            self._image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "image_tensor:0"
            )
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.compat.v1.Session(graph=self._graph, config=config)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._graph = None

    def detect(self, image: Image.Image) -> DetectionResult:
        self.load()
        assert self._session is not None
        assert self._tensor_dict is not None
        assert self._image_tensor is not None

        start = time.time()
        resized = _resize_if_needed(image.convert("RGB"))
        image_np = np.array(resized)
        output_dict = self._session.run(
            self._tensor_dict,
            feed_dict={self._image_tensor: np.expand_dims(image_np, axis=0)},
        )

        boxes = output_dict["detection_boxes"][0]
        scores = output_dict["detection_scores"][0]
        classes = output_dict["detection_classes"][0].astype(np.int32)

        detections: List[Detection] = []
        for box, score, class_id in zip(boxes, scores, classes):
            if score < self.threshold:
                continue
            label = CATEGORY_INDEX.get(int(class_id), {}).get("name", "litter")
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    box=tuple(float(v) for v in box),
                )
            )
            if len(detections) >= MAX_BOXES:
                break

        litter_count = len(detections)
        severity_rank = _severity_rank(litter_count)
        severity_label = RANK_LABELS[severity_rank]
        annotated = _draw_detections(resized.copy(), detections)
        elapsed = time.time() - start

        summary_lines = [
            f"Litter items detected: {litter_count}",
            f"Severity: {severity_label} (rank {severity_rank}/4)",
            f"Inference time: {elapsed:.2f}s",
            "",
            "Detections:",
        ]
        if detections:
            for index, det in enumerate(detections, start=1):
                ymin, xmin, ymax, xmax = det.box
                summary_lines.append(
                    f"{index}. {det.label} — {det.score * 100:.1f}% "
                    f"[box: ({xmin:.2f}, {ymin:.2f}) to ({xmax:.2f}, {ymax:.2f})]"
                )
        else:
            summary_lines.append("No litter detected above the confidence threshold.")

        return DetectionResult(
            annotated_image=annotated,
            detections=detections,
            litter_count=litter_count,
            severity_rank=severity_rank,
            severity_label=severity_label,
            inference_seconds=elapsed,
            summary="\n".join(summary_lines),
        )

    def detect_path(self, image_path: str) -> DetectionResult:
        return self.detect(Image.open(image_path))


def _is_lfs_pointer(path: str) -> bool:
    if os.path.getsize(path) >= 1024:
        return False
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(40).startswith("version https://git-lfs.github.com/spec/v1")


def _resize_if_needed(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
        return image

    aspect_ratio = width / height
    area = MAX_DIMENSION * (MAX_DIMENSION * aspect_ratio)
    new_height = math.sqrt(area / aspect_ratio)
    new_width = new_height * aspect_ratio
    return image.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)


def _severity_rank(count: int) -> int:
    if count < 3:
        return 1
    if count < 6:
        return 2
    if count < 9:
        return 3
    return 4


def _draw_detections(image: Image.Image, detections: List[Detection]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    for det in detections:
        ymin, xmin, ymax, xmax = det.box
        left = xmin * width
        top = ymin * height
        right = xmax * width
        bottom = ymax * height

        draw.rectangle([left, top, right, bottom], outline="#00FF00", width=3)
        label = f"{det.label}: {det.score * 100:.1f}%"
        text_bbox = draw.textbbox((left, top), label, font=label_font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        text_top = max(0, top - text_height - 4)
        draw.rectangle(
            [left, text_top, left + text_width + 6, text_top + text_height + 4],
            fill="#00FF00",
        )
        draw.text((left + 3, text_top + 2), label, fill="black", font=label_font)

    rank = _severity_rank(len(detections))
    banner = f"Detected: {len(detections)} | Severity: {RANK_LABELS[rank]}"
    banner_bbox = draw.textbbox((10, 10), banner, font=title_font)
    draw.rectangle(
        [
            banner_bbox[0] - 6,
            banner_bbox[1] - 4,
            banner_bbox[2] + 6,
            banner_bbox[3] + 4,
        ],
        fill="#111111",
    )
    draw.text((10, 10), banner, fill="#FFFFFF", font=title_font)
    return image


if __name__ == "__main__":
    detector = LitterDetector()
    test_image = os.path.join(PROJECT_ROOT, "test_images", "image12.jpg")
    result = detector.detect_path(test_image)
    print(result.summary)
    output_path = os.path.join(PROJECT_ROOT, "output", "sample_result.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.annotated_image.save(output_path)
    print(f"\nSaved annotated image to: {output_path}")
