import os
import base64
import traceback
import importlib.util
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



# DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DEBUG = "true"

detectors_path = './detectors'

app = FastAPI(
    title="FastAPI template for YOLO",
    description="",
    version="0.1",
)

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    if DEBUG:
        # Форматируем traceback
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        
        # Возвращаем его клиенту в JSON-ответе
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal Server Error",
                "traceback": tb_str.splitlines()  # Разбиваем на строки для удобства,
            },
        )
        # return PlainTextResponse(
        #     status_code=500,
        #     content=f"Error: Internal Server Error\nTraceback:\n{tb_str}"
        # )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


# Функция для динамической загрузки детектора
def load_detector(detector_name):
    detector_filename = f"{detector_name}.py"
    detector_path = os.path.join(detectors_path, detector_filename)
    
    if not os.path.exists(detector_path):
        raise HTTPException(status_code=404, detail="Detector not found")
    
    spec = importlib.util.spec_from_file_location(detector_name, detector_path)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    
    return detector


class PredictRequestModel(BaseModel):
    detector_name: str
    image: str  # base64 encoded image
    
@app.post("/run_predict")
async def run_predict(request: PredictRequestModel):
    """
    Распознать изображение с помощью выбранного детектора

    Parameters:        
        detector_name: str Имя детектора.
        image: str Изображение в формате base64.

    Returns:
        Ответ детектора в формате JSON.
    """
    function_name = "predict"
    
    detector = load_detector(request.detector_name)

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    try:
        image_data = base64.b64decode(request.image)
        if len(image_data) > 5 * 1024 * 1024:  # Ограничение размера изображения до 5 МБ
            raise HTTPException(status_code=400, detail="Image size exceeds the 5MB limit")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(request.image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)


class TrainRequestModel(BaseModel):
    detector_name: str
    dataset_path: str

@app.post("/run_train")
async def run_train(request: TrainRequestModel):
    """
    Запустить функцию обучения выбранного детектора.

    Parameters:
        detector_name: str  Имя детектора.
        dataset_path: str  Пусть к датасету.

    Returns:
        Ответ детектора в формате JSON.
    """
    
    function_name = "train"
    
    detector = load_detector(request.detector_name)

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(request.detector_name, request.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)


@app.get("/get_list")
async def get_list():
    """
    Получить список всех доступных детекторов
    
    Returns:
        Список в формате JSON.
    """
    
    # Получение списка файлов .py
    py_files = [f for f in os.listdir(detectors_path) if f.endswith('.py')]

    # Удаление расширения из имен файлов
    detectors = [os.path.splitext(f)[0] for f in py_files]

    return JSONResponse(detectors)


class MetadataRequestModel(BaseModel):
    detector_name: str
    
@app.post("/get_metadata")
async def get_metadata(request: MetadataRequestModel):
    """
    Получить метаданные детектора

    Parameters:
        detector_name: str Имя детектора.

    Returns:
        Метаданные детектора формате JSON.
    """
    
    function_name = "get_metadata"
    
    detector = load_detector(request.detector_name)

    if not hasattr(detector, function_name):
        raise HTTPException(status_code=404, detail="Function not found in detector")

    function = getattr(detector, function_name)
    
    # Выполнение функции детектора с переданными параметрами
    try:
        result = function(request.detector_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while executing detector function: {str(e)}")

    return JSONResponse(result)



