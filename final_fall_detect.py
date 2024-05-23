from ultralytics import YOLO
import cv2
import os
import datetime
import mysql.connector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def initialize_model(model_path):
    return YOLO(model_path)

def process_video(video_path, model, output_dir, db, cursor, name_of_neuro, cam_id):
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, persist=True, conf=0.5)

            for obj in results.xyxy[0]:
                if obj[5] == 0 and obj[4] > 0.8:  # Class 0 (fall) with confidence > 0.8
                    # Save frame with fall detection
                    filename = save_frame_with_fall(frame, output_dir)

                    # Store incident data in MySQL database
                    dt_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    incident_data = (name_of_neuro, dt_string, cam_id, filename)
                    cursor.execute(
                        "INSERT INTO accidents (name_of_neuro, datetime_recorded, camera_id, image_path) VALUES (%s, %s, %s, %s)",
                        incident_data
                    )
                    db.commit()

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def save_frame_with_fall(frame, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f'fall_frame_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
    cv2.imwrite(filename, frame)
    print(f'Saved frame with fall detection to {filename}')
    return filename

def main():
    myVideo = "Videos/fall.mp4"
    myModel = "Weights/fall_det_1.pt"
    cam_id = 5
    name_of_neuro = "fall_neuro_detect"
    output_dir = "fall_frames_images"

    model_path = myModel
    video_path = myVideo
    
    # Connect to MySQL database
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="accidents_db"
    )
    cursor = db.cursor()

    model = initialize_model(model_path)
    process_video(video_path, model, output_dir, db, cursor, name_of_neuro, cam_id)

    # Close MySQL database connection
    db.close()

if __name__ == "__main__":
    main()

