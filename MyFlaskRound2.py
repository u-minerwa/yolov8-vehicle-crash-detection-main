import flask
from flask import Flask, request, jsonify
import cv2, pyautogui
import pandas as pd
import numpy as np 
from datetime import datetime
from ultralytics import YOLO
import cvzone
from sort import Sort
import math, time

app = Flask(__name__)

# Функция обработки видео для первой нейронной сети
def process_network_1(myVideoUse):
    # Здесь должна быть реализация вашей нейронной сети
    # В этом примере просто выводим длительность видео
    
    yoloModel = "best.pt"
    myVideoUse = "cr.mp4"
    myFileUse = "coco1.txt"
    
    model = YOLO(yoloModel) 

    def update_statistics_window(statistics):
        # Создаём изображение для отображения статистики
        stat_image = 255 * np.ones((200, 300, 3), dtype=np.uint8)  # Белое изображение размером 200x300 

        # Добавляем текст статистики на изображение
        cv2.putText(stat_image, f"Accident: {statistics['Accident']}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        cv2.putText(stat_image, f"TrafficLight: {statistics['TrafficLight']}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        cv2.putText(stat_image, f"Car: {statistics['Car']}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        cv2.putText(stat_image, f"Sign: {statistics['Sign']}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2) 

        # Отображаем изображение статистики
        cv2.imshow("Statistics", stat_image)
        cv2.waitKey(1)


    def WindowVideo(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE :  
            point = [x, y]
            print(point)


    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", WindowVideo)
    cap = cv2.VideoCapture(myVideoUse) 

    my_file = open(myFileUse, 'r')
    data = my_file.read()
    class_list = data.split("\n")

    count = 0 
    dtp_count = 0  # Переменная для подсчёта количества ДТП 

    # Инициализируем статистику
    statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0}

    # Создаем окно для статистики
    cv2.namedWindow("Statistics") 

    video_finished = False

    while not video_finished:    
        ret, frame = cap.read()
        
        if not ret:
            video_finished = True 
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        count += 1
        if count % 3 != 0:
            continue
        
        frame = cv2.resize(frame,(1020,500))
        results = model.predict(frame)          # a=results[0].boxes.data
        aa = results[0].boxes.data
        a = aa.cpu().detach().numpy()
        px = pd.DataFrame(a).astype("float")

        # Считаем количество объектов
        statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0}
        for index, row in px.iterrows():
            d = int(row[5])
            c = class_list[d]
            statistics[c] += 1

        # Отображаем статистику в окне
        update_statistics_window(statistics)

        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            
            c=class_list[d]
            
            if "Accident" in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)        #cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, f'{c}', (x1, y1), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA) 
                
                # Увеличиваем счетчик ДТП
                dtp_count += 1
                
                # Получаем текущую дату и время
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                
                # Сохраняем кадр с ДТП в файл с датой и временем в названии
                # cv2.imwrite(f"AccidentFrames/accident_frame_{dt_string}_{dtp_count}.png", frame)
                
                # Сохраняем кадр с ДТП в файл
                # cv2.imwrite(f"AccidentFrames/accident_frame_{dtp_count}.png", frame)
                # cv2.imwrite("AccidentFrames/accident_frame.png", frame)
                
            if "TrafficLight" in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(17,249,249),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
                
            if "Car" in c: 
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                
            if "Sign" in c: 
                cv2.rectangle(frame,(x1,y1),(x2,y2),(230,240,100),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
            
        cv2.imshow("Video", frame)
        if cv2.waitKey(1)&0xFF==27:
            break
        
        
    cap.release()  
    cv2.destroyAllWindows()

    return len(myVideoUse)

# По аналогии добавляем функции для остальных нейронных сетей...


def process_network_2(myVideoUse): 
    
    myVideoUse = "cars.mp4"
    yoloModel = "yolov8l.pt"
    maskPng = "mask.png"
    
    # Загрузка видеофайла
    cap = cv2.VideoCapture(myVideoUse)

    # Загрузка модели YOLO для детекции объектов
    model = YOLO(yoloModel)

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
    mask = cv2.imread(maskPng)

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Указываем границы области подсчета объектов
    limits1 = [410, 300, 700, 300]
    limits2 = [400, 297, 673, 297, 673, 400, 400, 400]

    # Словарь для хранения времени, проведенного каждым объектом в области limits2
    object_times2 = {}

    # Список для хранения идентификаторов объектов, пересекших limits1
    object_ids1 = []


    # Основной цикл обработки видеопотока
    video_finished = False
    
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
    return len(myVideoUse)


def process_network_3(myVideoUse):
    yoloModel = "best2.pt"
    myVideoUse = "Med.mp4"
    myFileUse = "coco2.txt"
    model = YOLO(yoloModel) 

    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE :  
            point = [x, y]
            print(point)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture(myVideoUse) 

    my_file = open(myFileUse, 'r')
    data = my_file.read()
    class_list = data.split("\n") 

    count = 0
    video_finished = False

    while not video_finished:
        ret, frame = cap.read()
        
        if not ret:
            video_finished = True
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        aa = results[0].boxes.data
        a = aa.cpu().detach().numpy()
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            
            c = class_list[d]
                
            if "Knife" in c: 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                
            if "Pistol" in c: 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 240, 100), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  
    cv2.destroyAllWindows()
    
    return len(myVideoUse)

#---------------------------------------------APP ROUTES--------------------------------------------------------------------#

@app.route('/lesha', methods=['POST'])
def lesha():
    myVideo = "Med.mp4"
    
    # Обработка видео каждой из нейронных сетей
    result_3 = process_network_3(myVideo)
    # Аналогично для остальных нейронных сетей...

    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result_3': result_3,
        # Добавьте результаты для остальных нейронных сетей...
    })


@app.route('/vlad', methods=['POST'])
def vlad():
    myVideo = "cars.mp4"
    
    # Обработка видео каждой из нейронных сетей
    result_2 = process_network_2(myVideo)
    # Аналогично для остальных нейронных сетей...

    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result_2': result_2,
        # Добавьте результаты для остальных нейронных сетей...
    })


@app.route('/process_video', methods=['POST'])
def process_video():
    myVideo = "cr.mp4"
    #if myVideo not in request.files:
    #    return jsonify({'error': 'No video file provided aaaa!'}), 400
    
    #myVideoUse = request.files[myVideo]
    
    # Обработка видео каждой из нейронных сетей
    result_1 = process_network_1(myVideo)
    # Аналогично для остальных нейронных сетей...

    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result_1': result_1,
        # Добавьте результаты для остальных нейронных сетей...
    })


@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

