import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO


metadata = {
    "name": "yolo_v8_model", # убрать и вставлять имя файла при ответе
    "type": "yolo8s",
    "model": "/models/yolo_v8_model.pt",
    "dataset": "/datasets/yolo_v8_model/",
    "version": "1.0",
    "uploaded_by": "developer1", # Добавлять автоматически. Либо  настроив CI/CD на Gitlab, либо при коммите
    "comment": "text",
}


def predict(base64_image_str):
    # Example for YOLO v8
    
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    
    # Decode base64 image string
    img_data = base64.b64decode(base64_image_str)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Perform detection
    results = model(image_np)
    

    # Extract detection results
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": model.names[class_id],
            })
    
    # Return results as JSON
    return detections
    
    
    
    
    
    
    
def train(image, epochs=10):
    # Пример тренировки
    return {"status": "training_started", "epochs": epochs}

def info():
    # Пример информации о модели
    return {"description": "This is a YOLO object detector"}
