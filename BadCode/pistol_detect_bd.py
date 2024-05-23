import cv2
import os
import pandas as pd
import cvzone
from ultralytics import YOLO
from datetime import datetime
import mysql.connector
import time

yoloModel = "Weights/best2.pt"
myVideoUse = "Videos/Med.mp4"
name_of_neuro = "neuro_pistol"
cam_id = "3"
model = YOLO(yoloModel) 
width = 1020 
height = 500

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(myVideoUse) 

class_list = ["Knife", "Pistol"] 

count = 0
video_finished = False

# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="accidents_db"
)
cursor = db.cursor()

# Create directory to save images
if not os.path.exists("pistol_images"):
    os.makedirs("pistol_images")

last_saved_time = time.time()  # Initialize last saved time
frames_saved = 0

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

    has_accident = False

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
            
        if "Knife" in c or "Pistol" in c: 
            has_accident = True
            frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 15) # red frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0 if "Knife" in c else 250), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    
    current_time = time.time()
    if has_accident and (current_time - last_saved_time < 1):
        if frames_saved < 5:
            frames_saved += 1

            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            dt_string_file = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Сохраняем кадр с пистолетом/ножом в файл с датой и временем в названии: 
            image_path = os.path.join("pistol_images", f"neuro_{name_of_neuro}_and_datetime_{dt_string_file}.jpg")
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
    else:
        last_saved_time = current_time  # Reset last saved time
        frames_saved = 0  # Reset frames saved count
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"): 
        break

cap.release()  
cv2.destroyAllWindows()
db.close()

