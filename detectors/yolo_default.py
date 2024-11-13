import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Описане детектора
metadata = {
    "type": "yolo8s",
    "model": "/models/yolo_v8_model.pt",
    "dataset": "/datasets/yolo_v8_model/",
    # "version": "1.0",
    # "uploaded_by": "developer1", # Добавлять автоматически. Либо  настроив CI/CD на Gitlab, либо при коммите
    # "comment": "text",
}


def predict(base64_image_str):
    # Пример для YOLO v8
    
    model = YOLO("yolov8n.pt")
    
    # Декодируем картинку base64 -> PIL -> np.array
    img_data = base64.b64decode(base64_image_str)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    image_np = np.array(image)
    
    # Распознаем
    results = model(image_np)
    

    # Вытаскиваем рузельтаты распознавания
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": model.names[class_id],
            })
    
    return detections
    
    
def train(detector_name, dataset_path): # del dataset_path
    # Тут должен быть код на котором тренировали модель
    
    result = {"detector_name": detector_name, "dataset_path": dataset_path}
    return result


def get_metadata(detector_name):
    return {'name': detector_name, **metadata}
