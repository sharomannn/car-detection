from ultralytics import YOLO
import cv2
import os

# Load models
model = YOLO("models/best.pt")  # load a custom model for license plate detection
coco_model = YOLO('yolov8n.pt')  # load a pre-trained model for car detection
names = model.names

cap = cv2.VideoCapture("videos/1.mp4")
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

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    cv2.rectangle(car_area, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(car_area, names[int(cls)], (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    crop_obj = car_area[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                    cv2.imwrite(os.path.join(crop_dir_name, str(idx)+".png"), crop_obj)

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()