import cv2
import pandas as pd
import pyautogui
from ultralytics import YOLO
import cvzone

yoloModel = "Violbest.pt"
myVideoUse = "Fight_M.mp4"
myFileUse = "cocov.txt"

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
            
        if "normal" in c: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
        if "fight" in c: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 250), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()  
cv2.destroyAllWindows()
