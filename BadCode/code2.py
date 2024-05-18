import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import time

# Загрузка видеофайла
cap = cv2.VideoCapture("cars.mp4")

# Загрузка модели YOLO для детекции объектов
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Список классов объектов для детекции
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Загрузка маски для области интереса (ROI)
mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Указываем границы области подсчета объектов
limits1 = [410, 300, 700, 300]
limits2 = [400, 297, 673, 297, 673, 400, 400, 400]

# Словарь для хранения времени, проведенного каждым объектом в области limits2
object_times2 = {}

# Список для хранения идентификаторов объектов, пересекших limits1
object_ids1 = []


video_finished = False
# Основной цикл обработки видеопотока
while not video_finished:
    success, img = cap.read()  # Захват кадра из видео
    
    if not success:
        video_finished = True 
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Применение маски кадра для области интереса (ROI)
    imgRegion = cv2.bitwise_and(img, cv2.resize(mask, (img.shape[1], img.shape[0])))

    # Предсказание объектов на кадре с помощью модели YOLO
    results = model(imgRegion, stream=True)

    # Создание пустого массива для хранения детекций
    detections = np.empty((0, 5))

    # Обработка результатов детекции объектов
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Получение координат и размеров ограничивающего прямоугольника
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Уровень уверенности детекции
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Класс объекта
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Проверка наличия объекта в списке интересующих нас классов и достаточного уровня уверенности
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Добавление детекции в массив
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Обновление трекера объектов с использованием полученных детекций
    resultsTracker = tracker.update(detections)

    # Отображение границ областей подсчета объектов
    cv2.polylines(img, [np.array(limits1).reshape((-1, 2))], True, (0, 255, 0), 2)
    cv2.polylines(img, [np.array(limits2).reshape((-1, 2))], True, (0, 0, 255), 2)

    # Обработка треков объектов и отображение информации на кадре
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Отображение идентификатора трека
        cv2.putText(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Отображение центра объекта
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Подсчет объектов, пересекающих границу области limits1
        if limits1[0] < cx < limits1[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            if id not in object_ids1:
                object_ids1.append(id)

        # Подсчет времени, проведенного в области limits2 для существующего объекта
        if limits2[0] < cx < limits2[4] and limits2[1] < cy < limits2[7]:
            if id in object_times2:
                if time.time() - object_times2[id] > 10:
                    print(f"Warning: Object ID {id} spent more than 10 seconds in limits2!")
                    cv2.putText(img, f'Warning: Object ID {id} spent more than 10 seconds in limits2!',
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                object_times2[id] = time.time()

    # Отображение количества объектов, пересекших limits1 на кадре
    cv2.putText(img, f"Limits1: {len(object_ids1)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Отображение кадра
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()  
cv2.destroyAllWindows()
