import tempfile
import cv2
import pandas as pd
import streamlit as st
from main import load_model, load_video, detect_cars, recognize_license_plate

st.title('Автоматическое распознавание номерных знаков')

video_file = st.file_uploader("Загрузите видео", type=['mp4', 'avi'])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    st.video(tfile.name)

    cap, video_writer = load_video(tfile.name)
    model, coco_model, lpr_model, names = load_model()
    idx = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            st.write("Обработка видео завершена.")
            break

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()