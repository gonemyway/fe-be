from flask import Flask, render_template, request
import os
import cv2
from mtcnn.mtcnn import MTCNN
import dlib
from imutils import face_utils
import numpy as np
import pickle
import pandas as pd

# Khởi tạo model
file = open("model.sav", 'rb')
model = pickle.load(file)
file.close()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "static"  # File static chứa ảnh trên giao diện và ảnh mà người dùng upload lên

# Đọc file CSV với dấu phân tách là |
csv_file_path = "face_desc.csv"
face_shape_info_df = pd.read_csv(csv_file_path, delimiter='|', header=None, names=['shape', 'title', 'description'])
face_shape_info = face_shape_info_df.set_index('shape')[['title', 'description']].to_dict('index')

detect = MTCNN()
landmark_detector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Xử lý các request
@app.route("/", methods=['POST', 'GET'])
def home():
    # Nếu là GET request:
    if request.method == "GET":
        return render_template("index.html")
    else:
        # POST
        # Lấy file mà client gửi lên
        image_file = request.files['file']  # Do trong tệp index.html name của file input là "file"
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)  # Lưu vào file static
        image_file.save(path_to_save)

        # Xử lý ảnh mà client gửi lên giống như trong file make_data.py
        # Convert image to dest size tensor
        frame = cv2.imread(path_to_save)

        results = detect.detect_faces(frame)

        if len(results) != 0:
            for result in results:
                x1, y1, width, height = result['box']

                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = frame[y1:y2, x1:x2]

                # Extract dlib
                landmark = landmark_detector(frame, dlib.rectangle(x1, y1, x2, y2))
                landmark = face_utils.shape_to_np(landmark)
                landmark = landmark.reshape(68 * 2)

                # Lấy landmark vector đưa vào model
                face_shape = model.predict([landmark])
                face_shape_name = face_shape[0]  # Lấy tên hình dạng khuôn mặt
                face_shape_title = face_shape_info[face_shape_name]['title']
                face_shape_description = face_shape_info[face_shape_name]['description']

                cv2.imwrite(path_to_save, face)

        else:
            return render_template('index.html', msg="Không nhận diện được khuôn mặt")

        # Trả thông tin về cho index.html để hiển thị
        return render_template("index.html", user_image=image_file.filename, msg="Tải file lên thành công",
                               face_shape=face_shape_title, face_shape_description=face_shape_description, hasface=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
