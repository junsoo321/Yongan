import math
from sklearn import neighbors
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np
import time

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class TrainFaceData:
    def __init__(self, train_dir, model_save_path, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        self.train_dir = train_dir
        self.model_save_path = model_save_path
        self.n_neighbors = n_neighbors
        self.knn_algo = knn_algo
        self.verbose = verbose
        self.knn_clf = None

    def train(self):
        X = []
        y = []

        # 학습 세트의 각 사람을 루프
        for class_dir in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                continue

            # 현재 사람에 대한 각 학습 이미지를 루프
            for img_path in image_files_in_folder(os.path.join(self.train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # 학습 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                    if self.verbose:
                        print("이미지 {}는 학습에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
                else:
                    # 현재 이미지의 얼굴 인코딩을 학습 세트에 추가
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                    X.append(face_encoding)
                    y.append(class_dir)

        # knn 분류기에서 가중치를 적용할 이웃의 수를 결정
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(X))))
            if self.verbose:
                print("자동으로 선택된 n_neighbors:", self.n_neighbors)

        # knn 분류기 생성 및 학습
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # 학습된 knn 분류기 저장
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf

class StudentAttendance:
    def __init__(self, file_path):
        self.file_path = file_path

    # 출석 파일을 읽어 이미 출석된 사람의 이름 반환
    def load_attendance(self):
        if not os.path.exists(self.file_path):
            return set()
        
        with open(self.file_path, "r") as f:
            return set(line.strip() for line in f.readlines())

    # 출석 파일에 이름 기록
    def mark_attendance(self, name):
        #attendance_set = self.load_attendance()  # 기존 출석 이름 로드
        if name not in attendance_set:
            with open(self.file_path, "a") as f:
                f.write(f"{name}\n")
            attendance_set.add(name)

    def reset_attendance(self):
        with open(self.file_path, 'w') as file:
            file.write('')  # 파일 내용 비우기
        return set()

class FaceRecognition:
    def __init__(self) -> None:
        pass

    def recognize_face(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distance = []
        #face_recognition_frames = {}

        # 입력된 얼굴 인코딩과 가장 가까운 이웃 사이의 거리 계산
        for face_encoding in face_encodings:
            closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
            face_distance = closest_distances[0][0][0]
            is_recognized = closest_distances[0][0][0] <= 0.4  # 거리가 임계값 이하일 때만 True
            if is_recognized:
                name = knn_clf.predict([face_encoding])[0]
                face_names.append(name)
                #face_recognition_frames[name] = face_recognition_frames.get(name, 0) + 1
                #attendance.mark_attendance(name)
            # 학습되지 않은 얼굴인 경우
            else:
                face_names.append("Unknown")
        
        return face_locations, face_names, face_distance#, face_recognition_frames

# 학습 디렉토리 설정
#train_dir = "/Users/yewon/Desktop/linearAlgebra2_face_detection_datasets/team1"
attendance_file = "attendance.txt"

with open('C:/Users/kjoon/Downloads/model.h5', 'rb') as f:
    knn_clf = pickle.load(f) 

# KNN 모델 학습
#train_data = TrainFaceData(train_dir, model_save_path="model.h5", n_neighbors=None, knn_algo='ball_tree', verbose=False)
#knn_clf = train_data.train()

face_recognition_times = {}
already_attended = set()

# 출석 파일에서 이미 출석된 사람들의 이름 로드
attendance = StudentAttendance(attendance_file)
attendance.reset_attendance()
attendance_set = attendance.load_attendance()

process_this_frame = True

attendance_request={} # name, request_time을 쌍으로 저장
current_request=None # 출석 메시지가 떠있는지 판단

frame_threshold = 90  # 3초 동안의 프레임 수 (30fps 기준)

# 웹캠에서 비디오 캡처
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1) #좌우반전

    if process_this_frame:
        recognization = FaceRecognition()
        face_locations, face_names, distance = recognization.recognize_face(frame)
        # 최초로 인식된 얼굴의 시간 기록
        current_time = time.time()
        for name in face_names:
            if name != "Unknown":
                if name not in face_recognition_times:
                    face_recognition_times[name] = current_time
                elapsed_time = current_time - face_recognition_times[name]
                if elapsed_time >= 10:  # 10초 이상 경과한 경우
                    del face_recognition_times[name]  # 딕셔너리에서 기록 제거
                if elapsed_time >= 3:  # 3초 이상 인식된 경우
                    if name not in already_attended:
                        current_request=name
                        attendance_request[name]=current_time
                        # attendance.mark_attendance(name)
                        # already_attended.add(name)  # 이미 출석부에 기록된 얼굴은 다시 기록하지 않음
            else:
                # 얼굴이 인식되지 않은 경우 시간값을 기록하지 않음
                pass

    process_this_frame = not process_this_frame

    # 출석 요청 메시지 표시 및 입력 대기
    if current_request:
        for name, request_time in list(attendance_request.items()):
            frame_height, frame_width = frame.shape[:2] # 프레임의 높이와 너비만 저장
            text_size, baseline = cv2.getTextSize(f"{current_request}님 출석하시겠습니까?", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            # 화면 중앙 좌표 계산
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.putText(frame, f"{current_request}님 출석하시겠습니까?", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

            # 키 입력받기
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                attendance.mark_attendance(name)
                already_attended.add(name)  # 이미 출석부에 기록된 얼굴은 다시 기록하지 않음
                del attendance_request[name] 
                current_request=None
            elif key == ord('n'):
                del attendance_request[name]  
                current_request=None
            if key == ord('q'):
                break

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f'{name} {(1-(round(distance,2)))*100}%', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

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
