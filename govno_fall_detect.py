import os
import cv2
import mysql.connector
from datetime import datetime

from ultralytics import YOLO

# Установите путь к вашей папке для сохранения изображений
IMAGE_FOLDER = "fall_frames_images"

# Создайте папку, если она не существует
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Инициализация подключения к базе данных
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="accidents_db"
)
cursor = db.cursor()

def initialize_model(model_path):
    return YOLO(model_path)


def process_video(video_path, model, camera_id, name_of_neuro):
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, persist=True, conf=0.5)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Получаем текущее дата и время
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if "Fall-Detected" in frame:
                # Сохраняем кадр в папке
                image_filename = f"{name_of_neuro}_{camera_id}_{dt_string}.jpg"
                image_path = os.path.join(IMAGE_FOLDER, image_filename)
                cv2.imwrite(image_path, frame)

                # Добавляем данные в базу данных
                incident_data = (name_of_neuro, dt_string, camera_id, image_path)
                cursor.execute(
                    "INSERT INTO accidents (name_of_neuro, datetime_recorded, camera_id, image_path) VALUES (%s, %s, %s, %s)",
                    incident_data
                )
                db.commit()

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
    process_video(video_path, model, cam_id, name_of_neuro)

if __name__ == "__main__":
    main()

