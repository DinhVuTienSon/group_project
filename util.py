import string
import easyocr
import numpy as np
import cv2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}

dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# write result to csv file
def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'vehicle_id', 'vehicle_bounding_box',
                                                'license_plate_bounding_box', 'license_plate_bounding_box_accuracy', 'license_number',
                                                'license_number_accuracy'))

        for frame_number in results.keys():
            for vehicle_id in results[frame_number].keys():
                print(results[frame_number][vehicle_id])
                if 'vehicle' in results[frame_number][vehicle_id].keys() and \
                   'license_plate' in results[frame_number][vehicle_id].keys() and \
                   'text' in results[frame_number][vehicle_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                            vehicle_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][vehicle_id]['vehicle']['bounding_box'][0],
                                                                results[frame_number][vehicle_id]['vehicle']['bounding_box'][1],
                                                                results[frame_number][vehicle_id]['vehicle']['bounding_box'][2],
                                                                results[frame_number][vehicle_id]['vehicle']['bounding_box'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][vehicle_id]['license_plate']['bounding_box'][0],
                                                                results[frame_number][vehicle_id]['license_plate']['bounding_box'][1],
                                                                results[frame_number][vehicle_id]['license_plate']['bounding_box'][2],
                                                                results[frame_number][vehicle_id]['license_plate']['bounding_box'][3]),
                                                            results[frame_number][vehicle_id]['license_plate']['bounding_box_accuracy'],
                                                            results[frame_number][vehicle_id]['license_plate']['text'],
                                                            results[frame_number][vehicle_id]['license_plate']['text_accuracy'])
                            )
        f.close()

# verify license plate format
def license_complies_format(text):
    if len(text) != 8:
        return False

    if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        return True
    else:
        return False

# declare license plate format
def format_license(text):

    license_plate_ = ''
    mapping = {0: dict_char_to_int, 1: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int,
               2: dict_int_to_char}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

# read license plate
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bounding_box, text, accuracy = detection

        text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(')', '').replace('}', '')

        if license_complies_format(text):
            # print('----------', text, '--------------')
            return format_license(text), accuracy

    return None, None

def extractValue(license_plate_crop):
    height, width, numChannels = license_plate_crop.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu" 
    return imgValue

def maximizeContrast(license_plate_crop_gray):
    #Làm cho độ tương phản lớn nhất 
    height, width = license_plate_crop_gray.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(license_plate_crop_gray, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(license_plate_crop_gray, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(license_plate_crop_gray, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat

# assign vehicle id to its license plate
def get_vehicle(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, accuracy, class_id = license_plate

    found_vehicle = False
    for j in range(len(vehicle_track_ids)):
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_id = vehicle_track_ids[j]

        if x1 > vehicle_x1 and y1 > vehicle_y1 and x2 < vehicle_x2 and y2 < vehicle_y2:
            vehicle_index = j
            found_vehicle = True
            break

    if found_vehicle:
        return vehicle_track_ids[vehicle_index]

    return -1, -1, -1, -1, -1