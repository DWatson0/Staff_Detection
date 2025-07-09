from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

model = YOLO("custom_model.pt")

source = "./sample.mp4"
vid = cv2.VideoCapture(source)
bounding_box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
label_annotator = sv.LabelAnnotator()
enlarge = 20
while (True):
    ret, frame = vid.read()
    if not ret:
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections.xyxy) == 0:
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    expanded_boxes = []
    coordinates_label = []
    for i, box in enumerate(detections.xyxy):
        conf = detections.confidence[i]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(x1 - enlarge, 0)
        y1 = max(y1 - enlarge, 0)
        x2 = min(x2 + enlarge, frame.shape[1])
        y2 = min(y2 + enlarge, frame.shape[0])
        expanded_boxes.append([x1, y1, x2, y2])
        coordinates_label.append(f"staff {conf:.2f} ({x1}, {y1}) ({x2}, {y2})")

    detections.xyxy = np.array(expanded_boxes)
    annotated_image = bounding_box_annotator.annotate(scene=frame,detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image,detections = detections,labels = coordinates_label)
    cv2.imshow('Detection',annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()