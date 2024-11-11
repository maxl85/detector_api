from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.pt")

# Run inference
results = model.predict("bus.jpg")
print(results)