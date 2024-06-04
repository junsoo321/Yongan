import os
import mysql.connector

conn = mysql.connector.connect( #이부분은 mysql서버와 연결하는 부분이라서 mysql설치 및 테이블 생성 시 값을 입력해야합니다.
    host="localhost", #호스트 이름
    user="root", #아이디
    password="1234", # 비번
    database="face_recognition" #테이블 생성한 데이터베이스
)
cursor = conn.cursor()

# 이미지 파일이 저장된 폴더 경로
f_path = "C:/Users/수정필요"

def datebase_store_code(current_folder): #폴더를 재귀적으로 탐색하여 데이터를 추출하는 함수 정의
    for item in os.listdir(current_folder): #current_folder 경로의 모든 데이터를 나열
        item_path = os.path.join(current_folder, item) #데이터별 경로 생성
        if os.path.isdir(item_path): #current_folder 경로의 데이터가 폴더인지 확인
            datebase_store_code(item_path) #폴더일경우 재귀함수로 폴더 내 데이터 추출
        else: #폴더가 아닐경우 = 사진일경우
            if item.endswith(('.jpg')):  # .jpg파일 필터링(.DS.Store 파일 등 걸러내는 용도)
                user_name = item.split("_")[1]  # 첫번째 _부터 두번째 _사이의 사용자 이름 추출
                insert_query = "INSERT INTO face_data (name, image_path) VALUES (%s, %s)" #SQL 쿼리 정의
                cursor.execute(insert_query, (user_name, item_path)) #데이터베이스에 넣기

datebase_store_code(f_path) #위에서 정의한 함수로 (f_path)부터 탐색 시작

conn.commit()# 변경 사항 저장

#연결 종료
cursor.close()
conn.close()
