# DATABASE: 
'''
!pip install mysql-connector-python


CREATE DATABASE incident_db;
USE incident_db;

CREATE TABLE incidents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    accident_count INT,
    traffic_light_count INT,
    car_count INT,
    sign_count INT,
    total_accidents INT,
    datetime_recorded DATETIME
);
'''

import cv2
import pandas as pd
import numpy as np 
from datetime import datetime
from ultralytics import YOLO
import cvzone
import json, os 
import mysql.connector

yoloModel = "Weights/best.pt"
myVideoUse = "Videos/cr.mp4"
myFileUse = "TxtFiles/coco1.txt"
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
class_list = ["Car", "TrafficLight", "Sign", "Accident"] 

waitKeyKoef = 60
count = 0 
accidCount = 0
dtp_count = 0
total_accident_frames = 0

statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0} 
video_finished = False
accidents_data = []

# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="incident_db"
)
cursor = db.cursor()

# Create directory to save images
if not os.path.exists("incident_images"):
    os.makedirs("incident_images")

while not video_finished:    
    ret, frame = cap.read()
    
    if not ret:
        video_finished = True 
        continue
    
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    aa = results[0].boxes.data
    a = aa.cpu().detach().numpy()
    px = pd.DataFrame(a).astype("float")

    statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0} 
    for index, row in px.iterrows():
        d = int(row[5])
        c = class_list[d]
        statistics[c] += 1

    stats_text = f"Accident: {statistics['Accident']}, TrafficLight: {statistics['TrafficLight']}, Car: {statistics['Car']}, Sign: {statistics['Sign']}, TotalAccidents: {total_accident_frames}" 
    cv2.putText(frame, stats_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y, Time: %H:%M:%S")
    cv2.putText(frame, f'Date: {dt_string}', (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    has_accident = False

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
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, f'{c}', (x1, y1), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA) 
            
            dtp_count += 1
            accidCount += 1
            print("Accident count:", accidCount)
            if accidCount==1:
                total_accident_frames += 1
                
            if accidCount==2:
                save_folder = "AccidentJsons"
                file_counter = 1
                def generate_file_name():
                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    return os.path.join(save_folder, f"data_{dt_string}_count_{file_counter}.json") 

                data_to_save = {
                    "Statistics": {
                        "Accident": statistics['Accident'],
                        "TrafficLight": statistics['TrafficLight'],
                        "Car": statistics['Car'],
                        "Sign": statistics['Sign'],
                        "TotalAccidents": total_accident_frames
                    },
                    "DateTime": dt_string
                }

                file_name = generate_file_name()

                with open(file_name, 'w') as json_file:
                    json.dump(data_to_save, json_file)

                print("Json file saved:", file_name)
                file_counter += 1
                
            if accidCount==3:
                cv2.imshow("Accident Frame "+f"{total_accident_frames}", frame)
                cv2.waitKey(waitKeyKoef)
                
        if "TrafficLight" in c:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(17,249,249),2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1) 
            
        if "Car" in c: 
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
            
        if "Sign" in c: 
            cv2.rectangle(frame,(x1,y1),(x2,y2),(230,240,100),2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
        
    if not has_accident:
        accidCount = 0
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(waitKeyKoef) & 0xFF == ord("q"):
        break

    if has_accident:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save the frame as an image file
        #image_path = os.path.join("incident_images", f"incident_{total_accident_frames}.jpg")
        #cv2.imwrite(image_path, frame)
        
        image_path = os.path.join("incident_images", f"incident_{total_accident_frames}.jpg")
        cv2.imwrite(image_path, frame)

        # Store incident data along with the image path
        incident_data = (
            statistics['Accident'],
            statistics['TrafficLight'],
            statistics['Car'],
            statistics['Sign'],
            total_accident_frames,
            dt_string,
            image_path
        )
        
        cursor.execute(
            "INSERT INTO incidents (accident_count, traffic_light_count, car_count, sign_count, total_accidents, datetime_recorded, image_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            incident_data
        )
        db.commit()

cap.release()  
print("Total accident frames:", total_accident_frames)
cv2.destroyAllWindows()
db.close()


# ПОЯСНЕНИЯ: 
'''
1. Подключение к БД: 

db = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="incident_db"
)
cursor = db.cursor()
'''

'''
2. Вставка данных в таблицу: 
    
if has_accident:
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    incident_data = (
        statistics['Accident'],
        statistics['TrafficLight'],
        statistics['Car'],
        statistics['Sign'],
        total_accident_frames,
        dt_string
    )
    
    cursor.execute(
        "INSERT INTO incidents (accident_count, traffic_light_count, car_count, sign_count, total_accidents, datetime_recorded) VALUES (%s, %s, %s, %s, %s, %s)",
        incident_data
    )
    db.commit()
'''

'''
Этот код вставляет данные об авариях в таблицу incidents в базе данных incident_db каждый раз, 
когда обнаруживается авария. Убедитесь, что значения host, user, password и database соответствуют 
вашим настройкам MySQL.
'''

'''
1. Создание директории для сохранения изображений:

if not os.path.exists("incident_images"):
    os.makedirs("incident_images")


2. Сохранение кадров в виде изображений и получение пути к файлу: 

image_path = os.path.join("incident_images", f"incident_{total_accident_frames}.jpg")
cv2.imwrite(image_path, frame)


3. Добавление пути к изображению в БД: 

incident_data = (
    statistics['Accident'],
    statistics['TrafficLight'],
    statistics['Car'],
    statistics['Sign'],
    total_accident_frames,
    dt_string,
    image_path
)

cursor.execute(
    "INSERT INTO incidents (accident_count, traffic_light_count, car_count, sign_count, total_accidents, datetime_recorded, image_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
    incident_data
)
db.commit()

'''
