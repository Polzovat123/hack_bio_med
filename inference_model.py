from typing import Dict, Any, List, Optional
import numpy as np
import pydicom
from pathlib import Path
import torch
import time
from segment_anything import sam_model_registry, SamPredictor

# Конфигурация
SAM_CHECKPOINT = Path("models/sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"

# Глобальный predictor (будет инициализирован один раз)
_sam_predictor: Optional[SamPredictor] = None


def hu_to_rgb_uint8(img_hu: np.ndarray) -> np.ndarray:
    """
    Преобразует HU-изображение в RGB uint8 для SAM.
    """
    # Применяем окно лёгких
    img_windowed = np.clip(img_hu, -1350, 150)
    img_normalized = (img_windowed - (-1350)) / (150 - (-1350))  # [0, 1]
    img_uint8 = (img_normalized * 255).astype(np.uint8)  # [0, 255], shape (H, W)

    # Преобразуем в RGB: копируем канал 3 раза
    img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)  # (H, W, 3)
    return img_rgb


def init_sam():
    """Инициализирует SAM один раз при старте."""
    global _sam_predictor
    if _sam_predictor is None:
        if not SAM_CHECKPOINT.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {SAM_CHECKPOINT}")
        print(f"Loading SAM model from {SAM_CHECKPOINT}")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def load_dicom_with_metadata(dcm_path: Path) -> tuple[np.ndarray, dict]:
    """Загружает DICOM и возвращает (изображение в HU, метаданные)."""
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    metadata = {
        "study_uid": str(ds.get("StudyInstanceUID", "")),
        "series_uid": str(ds.get("SeriesInstanceUID", ""))
    }
    return img, metadata


def auto_generate_prompts_from_hu(img_hu: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """
    Автоматически генерирует подсказки на основе HU:
    - Ищет связные компоненты с HU от -600 до +100 (подозрительные узелки/инфильтраты)
    - Берёт центроид каждой крупной области (>50 пикселей)
    """
    from scipy import ndimage

    # Маска подозрительных тканей: узелки, инфильтраты
    mask = (img_hu >= -600) & (img_hu <= 100)

    # Убираем очень маленькие области
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None

    prompts = {"points": [], "labels": []}
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        if region_mask.sum() < 50:  # меньше 50 пикселей — игнорируем
            continue
        # Центроид области
        coords = np.argwhere(region_mask)
        centroid = coords.mean(axis=0).astype(int)  # [row, col]
        # SAM ожидает [x, y] → [col, row]
        prompts["points"].append([int(centroid[1]), int(centroid[0])])
        prompts["labels"].append(1)

    if not prompts["points"]:
        return None

    return {
        "point_coords": np.array(prompts["points"]),
        "point_labels": np.array(prompts["labels"])
    }


def run_sam_on_image(img_rgb: np.ndarray, prompts: Optional[Dict] = None) -> tuple[bool, float]:
    predictor = _sam_predictor
    if predictor is None:
        raise RuntimeError("SAM not initialized")

    predictor.set_image(img_rgb)  # ← (H, W, 3) uint8

    if prompts is None:
        return False, 0.0

    try:
        masks, scores, _ = predictor.predict(
            point_coords=prompts["point_coords"],
            point_labels=prompts["point_labels"],
            multimask_output=True
        )
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        mask_area = masks[best_idx].sum()
        img_area = masks[best_idx].size
        is_abnormal = mask_area > (img_area * 0.005)
        return is_abnormal, best_score
    except Exception as e:
        print(f"SAM inference error: {e}")
        return False, 0.0


def process_single_dicom(dcm_path: Path, original_filename: str) -> Dict[str, Any]:
    start_time = time.time()

    img_hu, metadata = load_dicom_with_metadata(dcm_path)
    img_rgb = hu_to_rgb_uint8(img_hu)
    prompts = auto_generate_prompts_from_hu(img_hu)

    is_abnormal_np, confidence_np = run_sam_on_image(img_rgb, prompts)

    is_abnormal = bool(is_abnormal_np)
    confidence = float(confidence_np)
    processing_time = time.time() - start_time

    return {
        "probability_of_pathology": confidence,
        "pathology": is_abnormal,
        "most_dangerous_pathology_type": "nodule" if is_abnormal else None,
        "pathology_localization": "lung" if is_abnormal else None,
        "path_to_study": original_filename,  # ← вот здесь!
        "study_uid": str(metadata["study_uid"]),
        "series_uid": str(metadata["series_uid"]),
        "processing_status": "completed",
        "time_of_processing": round(float(processing_time), 3)
    }