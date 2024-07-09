import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dữ liệu từ 2 file pickle
file = open('landmarks.pkl', 'rb')
landmark_list = pickle.load(file)
file.close()

file = open('labels.pkl', 'rb')
label_list = pickle.load(file)
file.close()

# print(len(label_list))

svm = svm.SVC(kernel='linear') # SVM param gridsearch
svm.fit(landmark_list, label_list) # Train

# Thử predict mặt đầu tiên
predict = svm.predict([landmark_list[0]])
print("Kết quả dự đoán là: ", predict)
print("Kết quả thực tế là: ", label_list[0])

y_pred = svm.predict(landmark_list)
accuracy = accuracy_score(label_list, y_pred)
print("Accuracy:", accuracy * 100, "%")

# Lưu model
model_file = "model.sav"
file = open(model_file, 'wb')
pickle.dump(svm, file)
file.close()