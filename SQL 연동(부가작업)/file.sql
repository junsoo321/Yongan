create database face_recognition;
-- 데이터베이스 생성 코드

use face-recognition;
-- 데이터베이스 사용

create table face_data(
	id int primary	key auto_increment,
	forder_name varchar(255),
	file_name varchar(255)
);

-- 위 코드는 테이블 생성하는 코드입니다.

SELECT * FROM face_data;
-- 이 코드는 현재 users 테이블에 들어가 있는 데이터 리스트를 출력하는 코드입니다.

drop table face_data;
-- 이 코드는 users 테이블 자체를 삭제하는 코드로 이후 첫번째 create코드로 테이블을 다시 생성해야합니다.

TRUNCATE table face_data;
-- 이 코드는 users 테이블의 형태는 유지하고 내부 데이터만 삭제하는 코드입니다.
