import cv2
import pandas as pd
import pyautogui
import pygetwindow as gw
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
            
            # Получаем текущее активное окно
            active_window = gw.getWindowsWithTitle("RGB")[0]
            # Получаем координаты левого верхнего угла и размеры окна
            x, y, width, height = active_window.left, active_window.top, active_window.width, active_window.height
            # Делаем скриншот текущего окна с видео
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot.save("screenshot.png")
            # Здесь можно добавить код для вывода уведомления с изображением скриншота

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
