import math
from sklearn import neighbors
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    얼굴 인식을 위한 k-최근접 이웃 분류기를 학습시킵니다.

    :param train_dir: 각 알려진 사람에 대한 하위 디렉토리를 포함하는 디렉토리 경로.
    :param model_save_path: (선택 사항) 디스크에 모델을 저장할 경로.
    :param n_neighbors: (선택 사항) 분류에서 고려할 이웃의 수. 지정하지 않으면 자동으로 선택됩니다.
    :param knn_algo: (선택 사항) knn을 지원하는 기본 데이터 구조. 기본값은 'ball_tree'입니다.
    :param verbose: 학습 과정의 상세 출력 여부.
    :return: 학습된 knn 분류기.
    """
    X = []
    y = []

    # 학습 세트의 각 사람을 루프
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 현재 사람에 대한 각 학습 이미지를 루프
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 학습 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                if verbose:
                    print("이미지 {}는 학습에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
            else:
                # 현재 이미지의 얼굴 인코딩을 학습 세트에 추가
                face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                X.append(face_encoding)
                y.append(class_dir)

    # knn 분류기에서 가중치를 적용할 이웃의 수를 결정
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("자동으로 선택된 n_neighbors:", n_neighbors)

    # knn 분류기 생성 및 학습
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 학습된 knn 분류기 저장
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

# 출석 파일을 읽어 이미 출석된 사람의 이름 반환
def load_attendance(file_path):
    if not os.path.exists(file_path):
        return set()
    
    with open(file_path, "r") as f:
        return set(line.strip() for line in f.readlines())

# 출석 파일에 이름 기록
def mark_attendance(name, file_path, attendance_set):
    if name not in attendance_set:
        with open(file_path, "a") as f:
            f.write(f"{name}\n")
        attendance_set.add(name)

# 학습 디렉토리 설정
train_dir = "C:\\Users\\kjoon\\Downloads\\yongan\\yongan\\yongan"
attendance_file = "attendance.txt"

# KNN 모델 학습
knn_clf = train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False)

# 출석 파일에서 이미 출석된 사람들의 이름 로드
attendance_set = load_attendance(attendance_file)

# 비디오 처리 변수 초기화
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# 웹캠에서 비디오 캡처
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        # 입력된 얼굴 인코딩과 가장 가까운 이웃 사이의 거리 계산
        for face_encoding in face_encodings:
            closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
            is_recognized = closest_distances[0][0][0] <= 0.5  # 거리가 임계값 이하일 때만 True
            if is_recognized:
                name = knn_clf.predict([face_encoding])[0]
                face_names.append(name)
                mark_attendance(name, attendance_file, attendance_set)
            # 학습되지 않은 얼굴인 경우
            else:
                face_names.append("Unknown")

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 출석된 사람들의 이름을 화면 모퉁이에 표시
    y0, dy = 50, 20
    cv2.putText(frame, "Attendance", (5, y0 - dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(attendance_set):
        y = y0 + i * dy
        cv2.putText(frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
