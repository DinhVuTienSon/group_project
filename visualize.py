import ast

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./output_interpolated.csv')

# load video
video_path = 'C:/test_detection/sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./output.mp4', fourcc, fps, (width, height))

license_plate = {}
for vehicle_id in np.unique(results['vehicle_id']):
    max_ = np.amax(results[results['vehicle_id'] == vehicle_id]['license_number_accuracy'])
    license_plate[vehicle_id] = {'license_plate_number': results[(results['vehicle_id'] == vehicle_id) &
                                                             (results['license_number_accuracy'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['vehicle_id'] == vehicle_id) &
                                             (results['license_number_accuracy'] == max_)]['frame_number'].iloc[0])
    ret, frame = cap.read()

    # 'license_crop': None,
    # x1, y1, x2, y2 = ast.literal_eval(results[(results['vehicle_id'] == vehicle_id) &
    #                                           (results['license_number_accuracy'] == max_)]['license_plate_bounding_box'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    # license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    # license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    # license_plate[vehicle_id]['license_crop'] = license_crop


frame_number = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_number += 1
    if ret:
        df_ = results[results['frame_number'] == frame_number]
        for row_index in range(len(df_)):
            # draw car
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = ast.literal_eval(df_.iloc[row_index]['vehicle_bounding_box'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(vehicle_x1), int(vehicle_y1)), (int(vehicle_x2), int(vehicle_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)
            # cv2.rectangle(frame, (int(vehicle_x1), int(vehicle_y1)), (int(vehicle_x2), int(vehicle_y2)), (0, 255, 0), 25)
                        

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_index]['license_plate_bounding_box'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
            # license_crop = license_plate[df_.iloc[row_index]['vehicle_id']]['license_crop']

            # H, W, _ = license_crop.shape

            cv2.putText(frame, license_plate[df_.iloc[row_index]['vehicle_id']]['license_plate_number'], (int(vehicle_x1), int(vehicle_y1) - 18), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)

            # try:
            #     frame[int(vehicle_y1) - H - 100:int(vehicle_y1) - 100,
            #           int((vehicle_x2 + vehicle_x1 - W) / 2):int((vehicle_x2 + vehicle_x1 + W) / 2), :] = license_crop

            #     # frame[int(vehicle_y1) - H - 50:int(vehicle_y1) - 50,
            #     #       int((vehicle_x2 + vehicle_x1 - W) / 2):int((vehicle_x2 + vehicle_x1 + W) / 2), :] = (255, 255, 255)
            
            #     frame[int(vehicle_y1) - H - 400:int(vehicle_y1) - H - 100,
            #           int((vehicle_x2 + vehicle_x1 - W) / 2):int((vehicle_x2 + vehicle_x1 + W) / 2), :] = (255, 255, 255)

            #     (text_width, text_height), _ = cv2.getTextSize(
            #         license_plate[df_.iloc[row_index]['vehicle_id']]['license_plate_number'],
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         4.3,
            #         17)

            #     cv2.putText(frame,
            #                 license_plate[df_.iloc[row_index]['vehicle_id']]['license_plate_number'],
            #                 (int((vehicle_x2 + vehicle_x1 - text_width) / 2), int(vehicle_y1 - H - 250 + (text_height / 2))),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 4.3,
            #                 (0, 255, 0),
            #                 17)
            #     # print(license_plate[df_.iloc[row_index]['vehicle_id']]['license_plate_number'])
            # except:
            #     pass

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

out.release()
cap.release()