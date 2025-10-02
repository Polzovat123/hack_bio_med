from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

# Импортируем функции и инициализацию
from inference_model import init_sam, process_single_dicom


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает SAM при старте сервера."""
    init_sam()
    yield
    # Очистка (если нужно)


app = FastAPI(
    title="Lung CT Abnormality Segmenter (SAM-based)",
    description="Segments suspicious regions in chest CT DICOM using SAM",
    version="0.2.0",
    lifespan=lifespan  # ← инициализация при старте
)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.dcm', '.dicom')):
        raise HTTPException(status_code=400, detail="Only .dcm or .dicom files allowed")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Передаём оригинальное имя файла
        result = process_single_dicom(tmp_path, original_filename=file.filename)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()