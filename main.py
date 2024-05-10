import cv2
import pandas as pd
import numpy as np 
from datetime import datetime
from ultralytics import YOLO
import cvzone

model = YOLO("best.pt") 

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
cap = cv2.VideoCapture("cr.mp4") 

my_file = open("coco1.txt", 'r')
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

