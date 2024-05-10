import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import cvzone

model = YOLO("best.pt") 

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture("cr.mp4") 

my_file = open("coco1.txt", 'r')
data = my_file.read()
class_list = data.split("\n") 

count = 0
dtp_count = 0  # Переменная для подсчета количества ДТП

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    aa = results[0].boxes.data
    a = aa.cpu().detach().numpy()
    px = pd.DataFrame(a).astype("float")

    # Считаем количество объектов
    statistics = {'Accident': 0, 'TrafficLight': 0, 'Car': 0, 'Sign': 0}
    for index, row in px.iterrows():
        d = int(row[5])
        c = class_list[d]
        statistics[c] += 1

    # Отображаем статистику на кадре
    stats_text = f"Accident: {statistics['Accident']}, TrafficLight: {statistics['TrafficLight']}, Car: {statistics['Car']}, Sign: {statistics['Sign']}"
    cv2.putText(frame, stats_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
        
        if "Accident" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1) 
            
            # Увеличиваем счетчик ДТП
            # dtp_count += 1
            # Получаем текущую дату и время
            # now = datetime.now()
            # dt_string = now.strftime("%Y%m%d_%H%M%S")
            # Сохраняем кадр с ДТП в файл с датой и временем в названии
            # cv2.imwrite(f"accident_frame_{dt_string}_{dtp_count}.png", frame)

        if "TrafficLight" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (17, 249, 249), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1) 
            
        if "Car" in c: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        if "Sign" in c: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 240, 100), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  
cv2.destroyAllWindows()

