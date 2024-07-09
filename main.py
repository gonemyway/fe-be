import os
from mtcnn import MTCNN
import cv2
import sys


detect = MTCNN()
### Đọc thư mục face_data

raw_folder = "face_shape_data/face_data"

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

### Liệt kê các folder

### Đọc các file trong folder

### Append vào 1 list

### Write list vào file pickle

