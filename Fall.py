from ultralytics import YOLO
import cv2
import os
import datetime

from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def initialize_model(model_path):
    return YOLO(model_path)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp = datetime.now().strftime("12:05:01")
            print(f"Processing frame at {timestamp}")

            results = model.track(frame, persist=True, conf=0.5)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    myVideo = "pm.mp4"
    myModel = "fall_det_1.pt"
    
    model_path = myModel
    video_path = myVideo
    
    model = initialize_model(model_path)
    process_video(video_path, model)

if __name__ == "__main__":
    main()
