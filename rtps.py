import cv2
import subprocess
import pandas as pd
import argparse
from main import load_model, load_video, detect_cars, recognize_license_plate
import time
import streamlit as st
from src.config.config import get_cfg_defaults

cfg = get_cfg_defaults()
df = pd.read_csv('allowed_numbers.csv')
last_call_time = 0
last_image = None
parser = argparse.ArgumentParser(description='Process RTSP URL.')
parser.add_argument('rtsp_url', type=str, help='The RTSP URL to process.')

# Разберите аргументы
args = parser.parse_args()

# Используйте переданный URL RTSP
rtsp_url = args.rtsp_url
n = 4


def call():
    command = ['adb', 'shell', 'am', 'start', '-a', 'android.intent.action.CALL', '-d', 'tel:89631333413']
    subprocess.run(command, check=True)

cap = cv2.VideoCapture(rtsp_url)
model, coco_model, lpr_model, names = load_model()
idx = 0
frame_counter = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    if frame_counter % n == 0:
        vehicle_bounding_boxes = detect_cars(im0, coco_model)

        for box in vehicle_bounding_boxes:
            x1, y1, x2, y2, track_id, score = box
            # Crop the car area
            car_area = im0[int(y1):int(y2), int(x1):int(x2)]
            result = recognize_license_plate(car_area, model, lpr_model, names, idx)

            if result is not None:
                car_image, license_plate = result
                if car_image is not None:
                    if last_image is not None:
                        del last_image
                    last_image = car_image
                    st.image(last_image, caption=f'Processed Frame with license plate: {license_plate}',
                                use_column_width=True)
                    print(license_plate)

                    # Если номер есть в списке разрешенных, сделайте звонок
                    if license_plate in df['license_plate'].values:
                        current_time = time.time()
                        if current_time - last_call_time >= 60:  # 60 seconds = 1 minute
                            call()
                            last_call_time = current_time

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break