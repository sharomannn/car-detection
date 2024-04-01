import glob
import os
import tempfile
import cv2
import subprocess
import pandas as pd
import streamlit as st
from main import load_model, load_video, detect_cars, recognize_license_plate
import time
from src.config.config import get_cfg_defaults
import sqlite3
from datetime import datetime, timedelta
import streamlit as st

cfg = get_cfg_defaults()
df = pd.read_csv('allowed_numbers.csv')
last_call_time = 0
last_image = None
conn = sqlite3.connect('recognition_history.db')
c = conn.cursor()



st.title('Автоматическое распознавание номерных знаков')

page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Детекция по видео", "Детекция по фото", "Детекция с камеры", "История", "Очистить историю"])


def call():
    command = ['adb', 'shell', 'am', 'start', '-a', 'android.intent.action.CALL', '-d', 'tel:89631333413']
    subprocess.run(command, check=True)


if page == "Главная":
    st.header("Главная страница")
    st.write("Выберите тип детекции, используя боковую панель.")

    if st.button("Открыть шлакбаум"):
        call()

elif page == "История":
    st.header("История распознавания")

    # Получаем историю распознавания из базы данных
    c.execute("SELECT * FROM recognition_history")
    rows = c.fetchall()

    # Отображаем историю в Streamlit
    for row in rows:
        timestamp, license_plate, image_path = row
        st.write(f"{timestamp}: {license_plate}")
        st.image(image_path)


elif page == "Очистить историю":
    st.header("Очистить историю")

    # Удаляем записи, которые старше суток
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute("DELETE FROM recognition_history WHERE timestamp < ?", (yesterday,))
    conn.commit()
    st.write("История распознавания очищена.")
    #Удаляем файлы изображений с диска
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

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

elif page == "Детекция с камеры":
    st.header("Детекция с камеры")
    stop_button = st.button("Стоп")

    rtsp_url = st.text_input("Введите адрес RTSP")

    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        model, coco_model, lpr_model, names = load_model()
        idx = 0
        frame_counter = 0

        while cap.isOpened() and not stop_button:
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
                        if car_image is not None:
                            if last_image is not None:
                                del last_image
                            last_image = car_image
                            st.image(last_image, caption=f'Processed Frame with license plate: {license_plate}',
                                     use_column_width=True)

                            # Если номер есть в списке разрешенных, сделайте звонок
                            if license_plate in df['license_plate'].values:
                                current_time = time.time()
                                if current_time - last_call_time >= 60:  # 60 seconds = 1 minute
                                    call()
                                    last_call_time = current_time

            frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()