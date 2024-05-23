import cv2
import pandas as pd
from datetime import datetime
import mysql.connector
from ultralytics import YOLO
import cvzone, os, time

yoloModel = "Weights/Violbest.pt"
myVideoUse = "Videos/Fight_M.mp4"
name_of_neuro = "fight_neuro_detect"
accid_count = 0
cam_id = 6
width = 1020
height = 500

# Инициализация подключения к базе данных
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="accidents_db"
)
cursor = db.cursor()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(myVideoUse) 
class_list = ['normal', 'fight']

model = YOLO(yoloModel) 

count = 0
video_finished = False

# Create directory to save images
if not os.path.exists("fight_frames_images"):
    os.makedirs("fight_frames_images")

last_saved_time = time.time()  # Initialize last saved time

while not video_finished:
    ret, frame = cap.read()
    
    if not ret:
        video_finished = True
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (width, height))
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
            
        if "normal" in c: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        if "fight" in c: 
            has_accident = True
            if accid_count <= 1:
                accid_count += 1
            else:
                accid_count = 1
            
            frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 15) # red frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 250), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
    current_time = time.time()
    if has_accident and (current_time - last_saved_time >= 0.5):
        last_saved_time = current_time  # Update last saved time

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        dt_string_file = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Сохраняем кадр в файл с датой и временем в названии: 
        image_path = os.path.join("fight_frames_images", f"neuro_{name_of_neuro}_and_datetime_{dt_string_file}_{accid_count}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Выполнение действий при обнаружении боя
        # Добавляем данные в базу данных: 
        incident_data = (name_of_neuro, dt_string, cam_id, image_path)  # camera_id - временное решение
        cursor.execute(
            "INSERT INTO accidents (name_of_neuro, datetime_recorded, camera_id, image_path) VALUES (%s, %s, %s, %s)",
            incident_data
        )
        db.commit()
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()  
cv2.destroyAllWindows()
db.close()

