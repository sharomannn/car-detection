import os
import re
import time

import cv2
import pandas as pd
import torch
from torchvision import transforms
from ultralytics import YOLO

from LPRNet import LPRNet  # import your LPRNet model
from src.config.config import get_cfg_defaults
from src.tools.utils import decode_function, BeamDecoder

cfg = get_cfg_defaults()


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

    return model, coco_model, lpr_model, names


def load_video(video):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("object_cropping_output.avi",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (w, h))
    return cap, video_writer


VEHICLE_CLASS_IDS = [2, 3, 5]


def detect_cars(im0, coco_model):
    # Car detection
    car_detections = coco_model.track(im0, persist=True)[0]
    vehicle_bounding_boxes = []
    for detection in car_detections.boxes.data.tolist():
        try:
            x1, y1, x2, y2, track_id, score, class_id = detection
        except ValueError:
            continue

        if int(class_id) in VEHICLE_CLASS_IDS and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
    return vehicle_bounding_boxes


def draw_rectangle_and_label(car_area, box, cls, names):
    # Рисуем прямоугольник и метку
    cv2.rectangle(car_area, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(car_area, names[int(cls)], (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)


def transform_and_resize(crop_obj):
    # Определяем преобразование
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # это стандартные средние значения и стандартные отклонения ImageNet
    ])
    crop_obj = cv2.resize(crop_obj, (94, 24))
    crop_obj = transform(crop_obj)
    crop_obj = crop_obj.unsqueeze(0)

    return crop_obj


def recognize_license_plate(car_area, model, lpr_model, names, idx):
    # Обнаружение номерного знака на области автомобиля
    results = model.predict(car_area, show=False)
    since = time.time()
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    if boxes:
        for box, cls in zip(boxes, clss):
            idx += 1
            draw_rectangle_and_label(car_area, box, cls, names)

            crop_obj = car_area[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            crop_obj = transform_and_resize(crop_obj)

            license_plate_logits = lpr_model(crop_obj)

            # Декодируем предсказанные классы с помощью Beam Search
            try:
                predictions = license_plate_logits.cpu().detach().numpy()  # (1, 68, 18)
                labels, prob, pred_labels = decode_function(predictions, cfg.CHARS.LIST, BeamDecoder)
                print(f"Время вывода модели {time.time() - since:.3f} секунд")

            except TypeError:
                print("Произошла ошибка при декодировании номерного знака. Пропускаем этот кадр.")
                continue

            # Добавляем результат в DataFrame
            license_plate = labels[0]  # Предполагаем, что labels[0] содержит распознанный номерной знак

            # Проверяем, соответствует ли номерной знак формату
            if re.match(r'^[A-Z]\d{3}[A-Z]{2}\d{2,3}$', license_plate):
                df = pd.DataFrame([{"frame": idx, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                                    "license_plate": license_plate}], index=[0])

                # Сохраняем результат в CSV-файл
                df.to_csv("license_plate_recognition_results.csv", mode='a', header=False, index=False)


def main():
    cap, video_writer = load_video('videos/2.mp4')
    model, coco_model, lpr_model, names = load_model()
    idx = 0

    # Create a DataFrame to store the results
    df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "license_plate"])

    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            vehicle_bounding_boxes = detect_cars(im0, coco_model)

            for box in vehicle_bounding_boxes:
                x1, y1, x2, y2, track_id, score = box
                # Crop the car area
                car_area = im0[int(y1):int(y2), int(x1):int(x2)]
                recognize_license_plate(car_area, model, lpr_model, names, idx)

            cv2.imshow("ultralytics", im0)
            video_writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


main()
