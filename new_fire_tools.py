import os
import torch
from torchvision import transforms
from torch.autograd import Variable
import cv2
from PIL import Image
from ultralytics import YOLO

class FireDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.transformer = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def detect2(self, img):
        objects = self.model.predict(img, classes=[0], verbose=False, conf=0.5)
        fire_objects = objects[0].boxes.data
        return [obj.tolist() for obj in fire_objects]

def humanDetection(model, image):
    objects = model.predict(image, classes=[0], verbose=False)
    human_objects = objects[0].boxes.data
    human_bb = [obj[:4].int().tolist() for obj in human_objects if obj[5] == 0]
    return human_bb

def save_frame_with_fire(frame, frame_count, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
    cv2.imwrite(filename, frame)
    print(f'Saved frame {frame_count} with fire detection to {filename}')

if __name__ == "__main__":
    myModelYolo = "Weights/yolov8x.pt"
    myModelFire = "Weights/fire-yolov8.pt"
    myVideo = "Videos/Fire2.mp4"
    output_dir = "fire_frames_images"

    fire_detection = FireDetection(model_path=myModelFire)
    model = YOLO(myModelYolo)

    cap = cv2.VideoCapture(myVideo)
    frame_count = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        human_bb = humanDetection(model, img)

        fire_detected = False  # Flag to check if fire is detected in the frame
        
        for bb in human_bb:
            # Draw original bounding box around human
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
            
            # Define expanded bounding box
            new_bb = [max(0, bb[0] - 30), max(0, bb[1] - 30), min(img.shape[1], bb[2] + 30), min(img.shape[0], bb[3] + 30)]
            cv2.rectangle(img, (new_bb[0], new_bb[1]), (new_bb[2], new_bb[3]), (0, 255, 0), 2)

            # Crop the region defined by the expanded bounding box
            crop = img[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]]
            
            # Detect fire within the cropped region
            fire_bb = fire_detection.detect2(crop)

            for fb in fire_bb:
                # Adjust fire bounding box coordinates to the original image
                adjusted_fb = [int(coord + offset) for coord, offset in zip(fb[:4], new_bb[:2] * 2)]
                cv2.rectangle(img, (adjusted_fb[0], adjusted_fb[1]), (adjusted_fb[2], adjusted_fb[3]), (0, 255, 0), 2)
                cv2.putText(img, f'burning {fb[4]:.2f}', (adjusted_fb[0], adjusted_fb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                fire_detected = True

        if fire_detected:
            # Draw red frame around the image to indicate fire detection
            img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 10)
            # Save frame with fire detection
            save_frame_with_fire(img, frame_count, output_dir)
            # Print fire detection info
            print(f'Fire detected in frame {frame_count}')

        frame_count += 1
        if frame_count % 100 == 0:
            print(f'Processed {frame_count} frames')

        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

