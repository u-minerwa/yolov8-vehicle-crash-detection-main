import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import cvzone
import json, os
import mysql.connector

# DATABASE: 
'''
CREATE DATABASE accidents_db;
USE accidents_db;

CREATE TABLE accidents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name_of_neuro VARCHAR(50),
    datetime_recorded DATETIME,
    camera_id INT,
    image_path VARCHAR(255)
);
'''

# CODE: 
yoloModel = "Weights/best.pt"
myVideoUse = "Videos/cr.mp4"
model = YOLO(yoloModel)
name_of_neuro = "neuro_car_crash_1"
cam_id = "1"

def WindowVideo(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)
        
    if chr(event & 0xFF) == 'q':
        cap.release()
        cv2.destroyAllWindows()

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", WindowVideo)
cap = cv2.VideoCapture(myVideoUse)
class_list = ["Car", "TrafficLight", "Sign", "Accident"]

waitKeyKoef = 60
count = 0
accidCount = 0
dtp_count = 0
total_accident_frames = 0
width = 1020
height = 500

statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0, 'TotalAccidents': 0}
video_finished = False
accidents_data = []

# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="accidents_db"
)
cursor = db.cursor()

# Create directory to save images
if not os.path.exists("car_crash_1_images"):
    os.makedirs("car_crash_1_images")

# Счётчик подряд идущих кадров с инцидентами
incident_frame_count = 0

while not video_finished:
    ret, frame = cap.read()
    
    if not ret:
        video_finished = True
        continue
    
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (width, height))
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

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
        
        if "Accident" in c:
            has_accident = True
            frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 15) # red frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, f'{c}', (x1, y1), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            dtp_count += 1
            accidCount += 1
            print("Accident count:", accidCount)
            if accidCount == 1:
                total_accident_frames += 1
                
        if "TrafficLight" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (17, 249, 249), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        if "Car" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        if "Sign" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 240, 100), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        
    if not has_accident:
        accidCount = 0
        incident_frame_count = 0  # Обнуляем счётчик кадров с инцидентами
    else:
        if incident_frame_count < 10:
            incident_frame_count += 1
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            dt_string_file = now.strftime("%Y-%m-%d_%H-%M-%S")
            
            # Сохраняем кадр с ДТП в файл с датой и временем в названии
            image_path = os.path.join("car_crash_1_images", f"{name_of_neuro}_{dt_string_file}_{dtp_count}_{total_accident_frames}.jpg")
            cv2.imwrite(image_path, frame)

            # Store incident data along with the image path
            incident_data = (
                name_of_neuro,
                dt_string,
                cam_id,
                image_path
            )
            
            cursor.execute(
                "INSERT INTO accidents (name_of_neuro, datetime_recorded, camera_id, image_path) VALUES (%s, %s, %s, %s)",
                incident_data
            )
            db.commit()
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(waitKeyKoef) & 0xFF == ord("q"):
        break

cap.release()
print("Total accident frames:", total_accident_frames)
cv2.destroyAllWindows()
db.close()

