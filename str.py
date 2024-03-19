import tempfile
import cv2
import pandas as pd
import streamlit as st
from main import load_model, load_video, detect_cars, recognize_license_plate
import time
import re
from src.config.config import get_cfg_defaults
from src.tools.utils import decode_function, BeamDecoder

cfg = get_cfg_defaults()

st.title('Автоматическое распознавание номерных знаков')

page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Детекция по видео", "Детекция по фото"])

if page == "Главная":
    st.header("Главная страница")
    st.write("Выберите тип детекции, используя боковую панель.")

elif page == "Детекция по видео":
    st.header("Детекция по видео")

    video_file = st.file_uploader("Загрузите видео", type=['mp4', 'avi'])

    start_time = time.time()
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        cap, video_writer = load_video(tfile.name)
        model, coco_model, lpr_model, names = load_model()
        idx = 0
        total_time = 0
        frame_counter = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                st.write("Обработка видео завершена.")
                break

            # Обрабатываем только каждый четвертый кадр
            if frame_counter % 4 == 0:
                vehicle_bounding_boxes = detect_cars(im0, coco_model)

                for box in vehicle_bounding_boxes:
                    x1, y1, x2, y2, track_id, score = box
                    # Crop the car area
                    car_area = im0[int(y1):int(y2), int(x1):int(x2)]
                    result = recognize_license_plate(car_area, model, lpr_model, names, idx)

                    if result is not None:
                        car_image, license_plate = result
                        if car_image is not None and license_plate is not None:
                            st.image(car_image, caption=f'Processed Frame with license plate: {license_plate}', use_column_width=True)

            frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    end_time = time.time()
    total_time = end_time - start_time
    st.write(f'Общее время обработки: {total_time} секунд')

elif page == "Детекция по фото":
    st.header("Детекция по фото")

    image_file = st.file_uploader("Загрузите изображение", type=['jpg', 'png'])

    if image_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(image_file.read())

        im0 = cv2.imread(tfile.name)

        model, coco_model, lpr_model, names = load_model()
        idx = 0

        vehicle_bounding_boxes = detect_cars(im0, coco_model)

        for box in vehicle_bounding_boxes:
            x1, y1, x2, y2, track_id, score = box
            # Crop the car area
            car_area = im0[int(y1):int(y2), int(x1):int(x2)]
            result = recognize_license_plate(car_area, model, lpr_model, names, idx)
            if result is not None:
                _, license_plate = result
                if license_plate is not None:
                    st.image(car_area, caption=f'Processed Image with license plate: {license_plate}', use_column_width=True)