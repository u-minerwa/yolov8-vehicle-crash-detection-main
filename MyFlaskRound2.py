import flask
from torchvision import datasets, transforms, models
from flask import Flask, request, jsonify
import cv2, pyautogui
import pandas as pd
import numpy as np 
import datetime, json 
from datetime import datetime
from ultralytics import YOLO
import cvzone
from sort import Sort
import math, time, os
from torchfusion_utils.models import load_model, save_model
import torch
from torch.autograd import Variable
from PIL import Image
from deepstack_sdk import ServerConfig, Detection

app = Flask(__name__)

# Функция обработки видео для первой нейронной сети
def process_network_1(myVideoUse):
    # Здесь должна быть реализация вашей нейронной сети
    # В этом примере просто выводим длительность видео
        
    yoloModel = "best.pt"
    myVideoUse = "cr.mp4"
    myFileUse = "coco1.txt"
    model = YOLO(yoloModel) 

    def WindowVideo(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:  
            point = [x, y]
            print(point)
            
        if chr(event & 0xFF) == 'q':
            cap.release()  # Выключаем видео
            cv2.destroyAllWindows()


    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", WindowVideo)
    cap = cv2.VideoCapture(myVideoUse) 

    my_file = open(myFileUse, 'r')
    data = my_file.read()
    class_list = data.split("\n")

    waitKeyKoef = 60 
    count = 0 
    accidCount = 0
    dtp_count = 0  # Переменная для подсчёта количества ДТП 
    total_accident_frames = 0  # Общее количество кадров с авариями

    # Инициализируем статистику:
    statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0} 
    video_finished = False
    accidents_data = []  # Список для хранения данных о каждой аварии

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
        results = model.predict(frame)
        aa = results[0].boxes.data
        a = aa.cpu().detach().numpy()
        px = pd.DataFrame(a).astype("float")

        # Считаем количество объектов
        statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0} 
        for index, row in px.iterrows():
            d = int(row[5])
            c = class_list[d]
            statistics[c] += 1

        # Отображаем статистику в окне
        stats_text = f"Accident: {statistics['Accident']}, TrafficLight: {statistics['TrafficLight']}, Car: {statistics['Car']}, Sign: {statistics['Sign']}, TotalAccidents: {total_accident_frames}" 
        cv2.putText(frame, stats_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Получаем текущую дату и время:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y, Time: %H:%M:%S") 
        # Добавляем текущую дату и время на кадр: 
        cv2.putText(frame, f'Date: {dt_string}', (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        
        has_accident = False  # Переменная для отслеживания наличия аварии в текущем кадре

        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            
            c=class_list[d]
            
            if "Accident" in c:
                has_accident = True
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2) 
                #cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, f'{c}', (x1, y1), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA) 
                
                # Увеличиваем счётчик ДТП:
                dtp_count += 1
                accidCount += 1 
                if accidCount==1:
                    total_accident_frames += 1
                
                if accidCount==2:
                    # Создаём словарь с данными об аварии: 
                    accident_data = {
                        "Statistics": {
                            "Accident": statistics['Accident'],
                            "TrafficLight": statistics['TrafficLight'],
                            "Car": statistics['Car'],
                            "Sign": statistics['Sign'],
                            "TotalAccidents": total_accident_frames
                        },
                        "DateTime": dt_string
                    }
                    # Добавляем данные об аварии в список accidents_data
                    accidents_data.append(accident_data)
                
                #if accidCount==3:
                #    cv2.imshow("Accident Frame "+f"{total_accident_frames}", frame)
                #    cv2.waitKey(waitKeyKoef) 
                
            if "TrafficLight" in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(17,249,249),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
                
            if "Car" in c: 
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                
            if "Sign" in c: 
                cv2.rectangle(frame,(x1,y1),(x2,y2),(230,240,100),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
            
        # Если в текущем кадре не обнаружено аварии, обнуляем счётчик: 
        if not has_accident:
            accidCount = 0
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(waitKeyKoef) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()
    return len(myVideoUse), accidents_data  # Возвращаем длину видео и данные об авариях

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

            # Подсчёт объектов, пересекающих границу области limits1 
            if limits1[0] < cx < limits1[2] and limits1[1] - 15 < cy < limits1[1] + 15:
                if id not in object_ids1:
                    object_ids1.append(id)

            # Подсчёт времени, проведенного в области limits2 для существующего объекта
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
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
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
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()  
    cv2.destroyAllWindows()
    
    return len(myVideoUse)


def process_network_4(myVideo):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def initialize_model(model_path):
        return YOLO(model_path)

    def process_video(video_path, model):
        cap = cv2.VideoCapture(video_path)
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                timestamp = datetime.now().strftime("12:05:01")
                print(f"Processing frame at {timestamp}")

                results = model.track(frame, persist=True, conf=0.5)
                annotated_frame = results[0].plot()

                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def main():
        myVideo = "fall.mp4"
        myModel = "fall_det_1.pt"
        
        model_path = myModel
        video_path = myVideo
        
        model = initialize_model(model_path)
        process_video(video_path, model)

    if __name__ == "__main__":
        main()

    return len(myVideo)


def process_network_5(myVideo):
    class FireDetection:
        def __init__(self, model_path):
            # self.model = torch.load(model_path)
            # self.model.eval()
            # self.model = self.model.cuda()
            self.model = YOLO(model_path)
            self.transformer = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5])])

        def detect(self, img):

            orig = img.copy()
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
            print("Max: " + str(maxVal))
            print("Min: " + str(minVal))

            if maxVal > 200 and minVal < 10:
                print("Fire detected")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype('uint8'))
            orig = img.copy()
            img_processed = self.transformer(img).unsqueeze(0)
            img_var = Variable(img_processed, requires_grad=False)
            img_var = img_var.cuda()
            logp = self.model(img_var)
            expp = torch.softmax(logp, dim=1)
            confidence, clas = expp.topk(1, dim=1)

            co = confidence.item() * 100

            class_no = str(clas).split(',')[0]
            class_no = class_no.split('(')
            class_no = class_no[1].rstrip(']]')
            class_no = class_no.lstrip('[[')

            orig = np.array(orig)
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig = cv2.resize(orig, (800, 500))

            if class_no == '1':
                label = "Neutral: " + str(co) + "%"
                cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif class_no == '2':
                label = "Smoke: " + str(co) + "%"
                cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif class_no == '0':
                label = "Fire: " + str(co) + "%"
                cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            return orig, class_no, co

        def detect2(self, img): 

            objects = self.model.predict(img, classes=[0], verbose=False, conf=0.5)
            fireObjects = objects[0].boxes.data

            tmpBB = []

            for i in range(len(fireObjects)):
                tmpBB.append(fireObjects[i].tolist())
                # print(fireObjects[i].tolist())

            return tmpBB
        
    def humanDetection(model, image):
        # objects = self.model(frame).xyxy[0]
        objects = model.predict(image, classes=[0], verbose=False)

        humanObjects = objects[0].boxes.data

        # print(humanObjects)

        tmpBB = []
        conf = []
        tmp = []

        for obj in humanObjects:
            if obj[5] == 0:
                tmp.append(obj)

        for obj in tmp:
            bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
            tmpBB.append(bbox)

        return tmpBB, conf


    if __name__ == "__main__":
        
        myModelYolo = "yolov8x.pt"
        myModelFire = "fire-yolov8.pt"
        myVideo = "Fire2.mp4"
        
        model_path = myModelFire
        fire_detection = FireDetection(model_path=model_path)
        model = YOLO(myModelYolo)

        video_path = myVideo
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            human_bb, conf = humanDetection(model, img)

            if len(human_bb) > 0:
                for bb in human_bb:
                    cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
                    
                    new_bb = [bb[0] - 30, bb[1] - 30, bb[2] + 30, bb[3] + 30]

                    if new_bb[0] < 0:
                        new_bb[0] = 0
                    if new_bb[1] < 0:
                        new_bb[1] = 0
                    if new_bb[2] > img.shape[1]:
                        new_bb[2] = img.shape[1]
                    if new_bb[3] > img.shape[0]:
                        new_bb[3] = img.shape[0]

                    crop = img[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]]

                    bb = fire_detection.detect2(crop)

                    cv2.rectangle(img, (int(new_bb[0]), int(new_bb[1])), (int(new_bb[2]), int(new_bb[3])), (255, 0, 0), 2)

                    for i in range(len(bb)):
                        bb[i][0] = bb[i][0] + new_bb[0]
                        bb[i][1] = bb[i][1] + new_bb[1]
                        bb[i][2] = bb[i][2] + new_bb[0]
                        bb[i][3] = bb[i][3] + new_bb[1]
                        cv2.rectangle(img, (int(bb[i][0]), int(bb[i][1])), (int(bb[i][2]), int(bb[i][3])), (0, 255, 0), 2)
                        cv2.putText(img, str(bb[i][4]), (int(bb[i][0]), int(bb[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            frame_count += 1
            if frame_count % 100 == 0:
                print(frame_count)
            cv2.imshow("img", img)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        cv2.destroyAllWindows()
        
    return len(myVideo) 

#---------------------------------------------APP ROUTES--------------------------------------------------------------------#

@app.route('/leshafire', methods=['POST'])
def leshafire():
    myVideo = "Fire2.mp4"
    
    # Обработка видео каждой из нейронных сетей
    result_5 = process_network_5(myVideo)
    # Аналогично для остальных нейронных сетей...

    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result_5': result_5,
        # Добавьте результаты для остальных нейронных сетей...
    })


@app.route('/leshafall', methods=['POST'])
def leshafall():
    myVideo = "fall.mp4"
    
    # Обработка видео каждой из нейронных сетей
    result_4 = process_network_4(myVideo)
    # Аналогично для остальных нейронных сетей...

    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result_4': result_4,
        # Добавьте результаты для остальных нейронных сетей...
    })


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
    myVideoUse = "cr.mp4"
    # myVideoUse = request.files['video']
    # Обработка видео каждой из нейронных сетей
    result_1, accidents_data = process_network_1(myVideoUse)
    # Возвращаем результаты обработки в формате JSON
    data = {
        'result_1': result_1,
        'accidents_data': accidents_data  # Добавляем данные об авариях в ответ
        # Добавьте результаты для остальных нейронных сетей...
    }
    return jsonify(data)


@app.route('/crash_car', methods=['POST'])
def crash_car():
    # Получаем файл видео из запроса
    myVideoUse = request.files['video']
    
    # Получаем другие данные из запроса
    form_data = request.form
    param1 = form_data.get('param1')
    param2 = form_data.get('param2')
    # Получите остальные данные, если есть...
    # Обрабатываем видео и данные
    # Возвращаем результаты обработки в формате JSON
    return jsonify({
        'result': 'success'
    })


@app.route('/')
def index():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)

