import re

from ultralytics import YOLO
import cv2
from src.visualization.tools import convert_output_image, add_text2image, TextPosition
import time
import os
import pandas as pd
from LPRNet import LPRNet  # import your LPRNet model
import torch
from torchvision import transforms
import numpy as np
from src.config.config import get_cfg_defaults
from src.tools.utils import decode_function, BeamDecoder


cfg = get_cfg_defaults()


# Load models
def load_model():
    model = YOLO("models/best.pt")  # load a custom model for license plate detection
    coco_model = YOLO('yolov8n.pt')  # load a pre-trained model for car detection

    lpr_model = LPRNet(class_num=len(cfg.CHARS.LIST),
                        dropout_prob=0,
                        out_indices=cfg.LPRNet.OUT_INDEXES)  # build your LPRNet model
    lpr_model.eval()
    state_dict = torch.load("models/LPRNet_Ep_BEST_model.ckpt")
    lpr_model.load_state_dict(state_dict["net_state_dict"])

    names = model.names


cap = cv2.VideoCapture("videos/2.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)


# Video writer
video_writer = cv2.VideoWriter("object_cropping_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (w, h))

idx = 0
vehicles = [2,3,5]  # vehicle class IDs from the COCO dataset
vehicle_bounding_boxes = []

# Create a DataFrame to store the results
df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "license_plate"])

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Car detection
    car_detections = coco_model.track(im0, persist=True)[0]
    for detection in car_detections.boxes.data.tolist():
        try:
            x1, y1, x2, y2, track_id, score, class_id = detection
        except ValueError:
            continue

        if int(class_id) in vehicles and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])

            # Crop the car area
            car_area = im0[int(y1):int(y2), int(x1):int(x2)]

            # License plate detection on the car area
            results = model.predict(car_area, show=False)

            since = time.time()

            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            if boxes:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    cv2.rectangle(car_area, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(car_area, names[int(cls)], (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                    crop_obj = car_area[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                    # Define the transformation
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        # these are the standard ImageNet mean and std
                    ])

                    # Resize the image
                    crop_obj = cv2.resize(crop_obj, (94, 24))

                    # Apply the transformation
                    crop_obj = transform(crop_obj)

                    # Add an extra dimension for the batch
                    crop_obj = crop_obj.unsqueeze(0)

                    # Use LPRNet model to recognize the license plate
                    license_plate_logits = lpr_model(crop_obj)

                    # Convert the logits to probabilities
                    license_plate_probs = torch.nn.functional.softmax(license_plate_logits, dim=-1)

                    # Get the predicted classes
                    license_plate_pred = torch.argmax(license_plate_probs, dim=-1)

                    # Decode the predicted classes using Beam Search
                    try:
                        predictions = license_plate_logits.cpu().detach().numpy()  # (1, 68, 18)
                        labels, prob, pred_labels = decode_function(predictions, cfg.CHARS.LIST, BeamDecoder)
                        print("model inference in {:2.3f} seconds".format(time.time() - since))

                    except TypeError:
                        print("An error occurred while decoding the license plate. Skipping this frame.")
                        continue

                    # Add the result to the DataFrame
                    license_plate = labels[0]  # Assuming labels[0] contains the recognized license plate

                    # Check if the license plate matches the format
                    if re.match(r'^[A-Z]\d{3}[A-Z]{2}\d{2,3}$', license_plate):
                        df = pd.DataFrame([{"frame": idx, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                                            "license_plate": license_plate}], index=[0])

                        # Save the result to the CSV file
                        df.to_csv("license_plate_recognition_results.csv", mode='a', header=False, index=False)
    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

