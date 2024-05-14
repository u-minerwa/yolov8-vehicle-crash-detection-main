import cv2
import pandas as pd
import numpy as np 
from datetime import datetime
from ultralytics import YOLO
import cvzone
import upd_stat_window

model = YOLO("best.pt") 

def WindowVideo(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)
        
    if event == cv2.EVENT_KEYDOWN and chr(event & 0xFF) == 'q':
        cap.release()  # Выключаем видео
        cv2.destroyAllWindows()


cv2.namedWindow("Video")
cv2.setMouseCallback("Video", WindowVideo)
cap = cv2.VideoCapture("cr.mp4") 

my_file = open("coco1.txt", 'r')
data = my_file.read()
class_list = data.split("\n")

count = 0 
accidCount = 0
dtp_count = 0  # Переменная для подсчёта количества ДТП 
total_accident_frames = 0  # Общее количество кадров с авариями

# Инициализируем статистику:
statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0} 
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
    cv2.putText(frame, stats_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
            accidCount +=1
            print("Accident count:", accidCount)
            if accidCount==1:
                total_accident_frames += 1
            
            # Получаем текущую дату и время
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        
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
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()  
print("Total accident frames:", total_accident_frames)
cv2.destroyAllWindows()

