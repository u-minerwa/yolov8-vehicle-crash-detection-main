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
    myVideo = "Videos/fall.mp4"
    myModel = "Weights/fall_det_1.pt"
    cam_id = 5
    name_of_neuro = "fall_neuro_detect"
    
    model_path = myModel
    video_path = myVideo
    
    model = initialize_model(model_path)
    process_video(video_path, model)

if __name__ == "__main__":
    main()

