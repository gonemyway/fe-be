import os
from mtcnn import MTCNN
import cv2
import sys
import dlib
from imutils import face_utils
import pickle

detect = MTCNN()
landmark_detector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
### Đọc thư mục face_data

raw_folder = "face_shape_data/face_data"

landmark_list = []
label_list = []

for folder in os.listdir(raw_folder):
    if folder[0] != ".":
        print("Process folder", folder)

        for file in os.listdir(os.path.join(raw_folder, folder)):
            print("Process file", file)

            # Phát hiện khuôn mặt (MTCNN)
            pix_file = os.path.join(raw_folder, folder, file)
            image = cv2.imread(pix_file)
            results = detect.detect_faces(image)

            if len(results) > 0:
                result = results[0]
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2 = x1 + width
                y2 = y1 + height

                face = image[y1:y2, x1:x2]
               # cv2.imshow("", face)
               # cv2.waitKey()
               # sys.exit()

                # Trích xuất landmark bằng dlib
                landmark = landmark_detector(image, dlib.rectangle(x1, y1, x2, y2))

                # Chuyển kết quả đầu ra từ hàm landmark_detector thành dạng numpy
                landmark = face_utils.shape_to_np(landmark)

                # 68 điểm và mỗi điểm sẽ có 2 tọa độ. Cần reshape về thành 1 vector
                landmark = landmark.reshape(68*2)

                # Thêm landmark vào landmark list
                landmark_list.append(landmark)
                label_list.append(folder)

print(len(landmark_list))

# Chuyển sang numpy array
import numpy as np
landmark_list = np.array(landmark_list)
label_list = np.array(label_list)

# Write file landmark.pkl
file = open("landmarks.pkl", 'wb')
pickle.dump(landmark_list, file)
file.close()

# Write file label.pkl
file = open("labels.pkl", 'wb')
pickle.dump(label_list, file)
file.close()

### Mỗi khuôn mặt được ánh xạ thành 1 vector có độ dài là 136


