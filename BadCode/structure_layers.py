import torch
from torchsummary import summary
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("Weights/yolov8x.pt")

# Вывод структуры модели
# summary(model, input_size=(3, 640, 640))
model.info(True)

