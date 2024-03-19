import re
import time

import cv2
import pandas as pd

from src.config.config import get_cfg_defaults
from src.tools.utils import decode_function, BeamDecoder
from main import load_model, detect_cars, draw_rectangle_and_label,transform_and_resize

cfg = get_cfg_defaults()


def recognize_license_plate(car_area, model, lpr_model, names, idx):
    # Check if the car area has zero height or width
    if car_area.shape[0] == 0 or car_area.shape[1] == 0:
        print("Car area has zero height or width. Skipping...")
        return car_area, None  # Возвращаем изображение и None вместо номерного знака

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

                return car_area, license_plate

    return car_area, None  # Возвращаем изображение и None, если номерной знак не обнаружен


def main():
    image_path = '/home/roman/cv/car-detection/videos/2.jpeg'  # Замените на путь к вашему изображению
    im0 = cv2.imread(image_path)

    model, coco_model, lpr_model, names = load_model()
    idx = 0

    # Create a DataFrame to store the results
    df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "license_plate"])

    vehicle_bounding_boxes = detect_cars(im0, coco_model)

    for box in vehicle_bounding_boxes:
        x1, y1, x2, y2, track_id, score = box
        # Crop the car area
        car_area = im0[int(y1):int(y2), int(x1):int(x2)]
        _, license_plate = recognize_license_plate(car_area, model, lpr_model, names, idx)
        print(license_plate)

    cv2.imshow("ultralytics", im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()