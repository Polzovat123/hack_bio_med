import numpy as np
import pydicom
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

_model = None
MODEL_PATH = Path("lung_abnormality_model.pth")

class SimpleLungClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # предполагаем вход 256x256 → после 2 пулингов: 64x64
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def load_and_preprocess_dicom(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept

    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400.0  # [0, 1]

    return img


def preprocess_for_model(img: np.ndarray, target_size=(256, 256)) -> torch.Tensor:
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    resized = torch.nn.functional.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized  # shape: [1, 1, 256, 256]


def get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            print(f"Загружаем модель из {MODEL_PATH}")
            _model = SimpleLungClassifier()
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            _model.eval()
        else:
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}. Обучите и сохраните её сначала!")
    return _model


def save_model(model: nn.Module, path: Path = MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в {path}")


def run_inference(dcm_path: Path) -> dict:
    """
    Выполняет инференс модели на DICOM-изображении.
    Возвращает: {"is_abnormal": bool, "confidence": float}
    """
    # Загрузка и предобработка изображения
    img = load_and_preprocess_dicom(dcm_path)
    input_tensor = preprocess_for_model(img)

    # Загрузка модели
    model = get_model()

    # Инференс
    with torch.no_grad():
        output = model(input_tensor)
        confidence = output.item()  # вероятность класса "патология"

    is_abnormal = confidence > 0.5

    return {
        "is_abnormal": is_abnormal,
        "confidence": float(confidence)
    }

