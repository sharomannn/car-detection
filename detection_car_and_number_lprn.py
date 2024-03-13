from ultralytics import YOLO
import cv2
import os
import pandas as pd
from LPRNet import LPRNet  # import your LPRNet model
import torch
from torchvision import transforms
import numpy as np

class Decoder:
    """Interface for sequence decoding"""
    def decode(self, predicted_seq, chars_list):
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def decode(self, predicted_seq, chars_list):
        full_pred_labels = []
        labels = []
        # predicted_seq.shape = [batch, len(chars_list), len_seq]
        for i in range(predicted_seq.shape[0]):
            single_prediction = predicted_seq[i, :, :]
            predicted_labels = []
            for j in range(single_prediction.shape[1]):
                predicted_labels.append(np.argmax(single_prediction[:, j], axis=0))

            without_repeating = []
            current_char = predicted_labels[0]
            if current_char != len(chars_list) - 1:
                without_repeating.append(current_char)
            for c in predicted_labels:
                if (current_char == c) or (c == len(chars_list) - 1):
                    if c == len(chars_list) - 1:
                        current_char = c
                    continue
                without_repeating.append(c)
                current_char = c

            full_pred_labels.append(without_repeating)

        for i, label in enumerate(full_pred_labels):
            decoded_label = ''
            for j in label:
                decoded_label += chars_list[j]
            labels.append(decoded_label)

        return labels, full_pred_labels

class BeamDecoder(Decoder):
    def decode(self, predicted_seq, chars_list):

        labels = []
        final_labels = []
        final_prob = []
        k = 1
        for i in range(predicted_seq.shape[0]):
            sequences = [[list(), 0.0]]
            all_seq = []
            single_prediction = predicted_seq[i, :, :]
            for j in range(single_prediction.shape[1]):
                single_seq = []
                for char in single_prediction[:, j]:
                    single_seq.append(char)
                all_seq.append(single_seq)

            for row in all_seq:
                all_candidates = []
                for i in range(len(sequences)):
                    seq, score = sequences[i]
                    for j in range(len(row)):
                        candidate = [seq + [j], score - row[j]]

                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                # select k best
                sequences = ordered[:k]

            full_pred_labels = []
            probs = []
            for i in sequences:

                predicted_labels = i[0]
                without_repeating = []
                current_char = predicted_labels[0]
                if current_char != len(chars_list) - 1:
                    without_repeating.append(current_char)
                for c in predicted_labels:
                    if (current_char == c) or (c == len(chars_list) - 1):
                        if c == len(chars_list) - 1:
                            current_char = c
                        continue
                    without_repeating.append(c)
                    current_char = c

                full_pred_labels.append(without_repeating)
                probs.append(i[1])
            for i, label in enumerate(full_pred_labels):
                decoded_label = ''
                for j in label:
                    decoded_label += chars_list[j]
                labels.append(decoded_label)
                final_prob.append(probs[i])
                final_labels.append(full_pred_labels[i])

        return labels, final_prob, final_labels


def decode_function(predicted_seq, chars_list, decoder=GreedyDecoder):
    return decoder().decode(predicted_seq, chars_list)



LIST = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

# Load models
model = YOLO("models/best.pt")  # load a custom model for license plate detection
coco_model = YOLO('yolov8n.pt')  # load a pre-trained model for car detection
lpr_model = LPRNet(class_num=23, dropout_prob=0, out_indices=(2, 6, 13, 22))  # build your LPRNet model
# Load the state dict
state_dict = torch.load("models/LPRNet_Ep_BEST_model.ckpt")

# Load the weights into the model
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
                        labels, prob, pred_labels = decode_function(predictions, LIST, BeamDecoder)
                        print(f"Labels: {labels}, Probabilities: {prob}, Predicted Labels: {pred_labels}")

                    except TypeError:
                        print("An error occurred while decoding the license plate. Skipping this frame.")
                        continue

                    # Convert the predicted classes to characters
                    license_plate = ''.join(LIST[i] for i in pred_labels[0])

                    # Add the result to the DataFrame
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

# Save the results to a CSV file
df.to_csv("license_plate_recognition_results.csv", index=False)