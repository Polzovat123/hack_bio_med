from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
from inference_model import run_inference

app = FastAPI(
    title="Lung CT Abnormality Classifier",
    description="Classifies chest CT DICOM slices as normal/abnormal",
    version="0.1.0"
)

@app.get("/health")
async def healthcheck():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает один DICOM-файл (.dcm), возвращает результат классификации.
    """
    # Проверка расширения
    if not file.filename.lower().endswith(('.dcm', '.dicom')):
        raise HTTPException(status_code=400, detail="Only .dcm or .dicom files allowed")

    try:
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Запуск инференса
        result = run_inference(tmp_path)

        # Удаляем временный файл
        tmp_path.unlink()

        return JSONResponse(content=result)

    except Exception as e:
        # Удаляем временный файл даже при ошибке
        if 'tmp_path' in locals():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")