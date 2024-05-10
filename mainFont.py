import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

# Initialize YOLO model
model = YOLO("best.pt") 

# Mouse event callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

# Create a named window and set mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video file
cap = cv2.VideoCapture("cr.mp4") 

# Read class labels
my_file = open("coco1.txt", 'r')
data = my_file.read()
class_list = data.split("\n") 

# Initialize frame count
count = 0

# Main loop for video processing
while True:    
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Process every 3rd frame
    count += 1
    if count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using YOLO model
    results = model.predict(frame)
    aa = results[0].boxes.data
    a = aa.cpu().detach().numpy()
    px = pd.DataFrame(a).astype("float")

    # Iterate over detected objects
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        # Get class label
        c = class_list[d]
        
        # Draw rectangle around the object
        if "Accident" in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Change font to Hershey Simplex
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Put text on the rectangle
            cv2.putText(frame, f'{c}', (x1, y1), font, 1, (0, 0, 255), 2, cv2.LINE_AA) 

        # Other classes can be handled similarly
        
    # Show the frame
    cv2.imshow("RGB", frame)
    
    # Check for escape key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture object and close all windows
cap.release()  
cv2.destroyAllWindows()
