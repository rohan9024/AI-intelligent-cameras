from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

def process_video(yolo_coco_model_path, yolo_license_plate_model_path, video_path, density_threshold):
    results = {}
    mot_tracker = Sort()

    coco_model = YOLO(yolo_coco_model_path)
    license_plate_detector = YOLO(yolo_license_plate_model_path)

    cap = cv2.VideoCapture(video_path)
    vehicles_and_person = [0, 2, 3, 5, 7]
    frame_nmr = -1
    ret = True
    vehicle_count = 0
    person_count = 0
    prev_vehicle_count = 0
    prev_person_count = 0
    
    detected_persons = {}
    detected_vehicles = {}

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles_and_person :
                    detections_.append([x1, y1, x2, y2, score,int(class_id)])
                    color = (0,255,0)
                    if int(class_id) == 0 :
                        color =(0,0,255)
                        person_count += 1
                        if (x1, y1, x2, y2) not in detected_persons:
                            detected_persons[(x1, y1, x2, y2)] = True
                            
                    elif int(class_id) == 2 or int(class_id) == 5:
                        vehicle_count += 1
                        if (x1, y1, x2, y2) not in detected_vehicles:
                            detected_vehicles[(x1, y1, x2, y2)] = True
                                        
                    label = coco_model.names[int(class_id)]
                    label_text = f"{label}: {score:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

            # person_count = len(detected_persons)
            # vehicle_count = len(detected_vehicles)

            if vehicle_count > density_threshold:
                # density = vehicle_count / 2
                # if density > density_threshold:
                print(f"Vehicle density increased! Vehicle count: {vehicle_count}")
            
            if person_count > density_threshold:
                # density = person_count / 2
                # if density > density_threshold:
                print(f"People density increased! Person count: {person_count}")
                    

            try:            
                track_ids = mot_tracker.update(np.asarray(detections_))
            except :
                continue
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    cv2.rectangle(frame, (int (x1),int (y1)),(int (x2),int (y2)), (250,0,0),2) 
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    license_plate_crop_gray = cv2.cvtColor(
                        license_plate_crop, cv2.COLOR_BGR2GRAY
                    )
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                    )

                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop_thresh
                    )

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "bbox": [x1, y1, x2, y2],
                                "text": license_plate_text,
                                "bbox_score": score,
                                "text_score": license_plate_text_score,
                            },
                        }
            
            cv2.imshow("license_plate", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_vehicle_count = vehicle_count
            prev_person_count = person_count
        else:
            print("Empty frame encountered. Skipping...")
    
    print("Processing complete.")
    write_csv(results, "./test.csv")

yolo_coco_model_path = "./yolov8n.pt"
yolo_license_plate_model_path = "./indian_license_plate_detector.pt"
video_path = "./indian_roadsample.mp4"
density_threshold = 20
process_video(yolo_coco_model_path, yolo_license_plate_model_path, video_path, density_threshold)