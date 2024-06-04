import os
import mysql.connector

# MySQL 데이터베이스 연결
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="face_recognition"
cursor = conn.cursor()

# 데이터베이스에서 사용자 정보 불러오기
cursor.execute("SELECT name, image_path FROM face_data")
users = cursor.fetchall()

# 사용자 정보를 담을 리스트 초기화
known_face_encodings = []
known_face_names = []

# 데이터베이스에서 불러온 사용자 정보를 사용하여 얼굴 인식에 사용할 이미지를 로드합니다.
for user in users:
    name, image_path = user
    # 이미지를 로드하기 위해 경로에서 파일을 읽어옵니다.
    img = cv2.imread(image_path)
    # OpenCV는 이미지를 BGR 형식으로 읽어오므로 RGB 형식으로 변환합니다.
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 얼굴 인식에 사용할 수 있도록 얼굴 인코딩을 수행합니다.
    encoding = face_recognition.face_encodings(rgb_img)
    if encoding:  
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)
