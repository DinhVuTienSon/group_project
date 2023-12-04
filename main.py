from ultralytics import YOLO
import cv2
from tracker import *
import numpy as np
from sort.sort import *
from util import get_vehicle, read_license_plate, extractValue, maximizeContrast, write_csv

# # train model
# model = YOLO("yolov8n.yaml")
# result = model.train(data='C:/test_detection/data.yaml', epochs = 100)

# using model
vehicles_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('C:/test_detection/runs/detect/train5/weights/best.pt') 

# using Sort library for tracking
tracker = Sort()

video = cv2.VideoCapture('C:/test_detection/sample.mp4')

output = {}
vehicles = [2, 3, 5, 7]

# read frames
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = video.read()
    if ret:
        output[frame_number] = {}
        
        # detect vehicles
        detections = vehicles_detector(frame)[0]
        detections_list = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, accuracy, class_id = detection 
            if int(class_id) in vehicles:
                detections_list.append([x1, y1, x2, y2, accuracy])
                
        # track vehicles
        vehicles_id = tracker.update(np.asarray(detections_list))
        
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, accuracy, class_id = license_plate
            
            # assign license plate to vehicle
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_id = get_vehicle(license_plate, vehicles_id)
            
            if vehicle_id != -1:

                # crop license plate
                crop_plate = frame[int(y1):int(y2), int(x1): int(x2), :]
                
                # process license plate
                # grayscale = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
                
                grayscale = extractValue(crop_plate)
                
                max_contrast = maximizeContrast(grayscale)
                # # cv2.imshow('gray', max_contrast)
                # # cv2.waitKey(0)
                height, width = grayscale.shape
                gaussian = np.zeros((height, width, 1), np.uint8)
                gaussian = cv2.GaussianBlur(max_contrast, (3, 3), 0)
                
                # thresholding = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
                _, thresholding = cv2.threshold(gaussian, 64, 255, cv2.THRESH_BINARY_INV)

                # cv2.imshow('origin', thresholding)
                # cv2.imshow('gray', thresholding)
                # cv2.waitKey(0)
                # print('------', frame_number, '---------')
                # read license plate number
                license_plate_value, license_plate_value_accuracy = read_license_plate(thresholding)
                # print('------', frame_number, '---------')
                # print(license_plate_value,'-------', license_plate_value_accuracy, '-----------')
                if license_plate_value is not None:
                    # print('------', frame_number, '---------')
                    # print(license_plate_value,'-------', license_plate_value_accuracy, '-----------')
                    output[frame_number][vehicle_id] = {'vehicle': {'bounding_box': [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2]},
                                                  'license_plate': {'bounding_box': [x1, y1, x2, y2],
                                                                    'text': license_plate_value,
                                                                    'bounding_box_accuracy': accuracy,
                                                                    'text_accuracy': license_plate_value_accuracy}}

# write output
write_csv(output, './output.csv')