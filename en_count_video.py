
import yt_dlp
from ultralytics import YOLOv10  # YOLO를 YOLOv10으로 변경
import cv2
import os
import time

def download_and_process_video(url):  # model_name 파라미터 제거
    """
    YouTube 영상을 다운로드하고 객체를 감지하는 함수
    
    Args:
        url (str): YouTube 영상 URL
    """
    try:
        # 임시 파일명 생성 (timestamp 사용)
        timestamp = int(time.time())
        video_path = f'video_{timestamp}.mp4'
        
        # 영상 다운로드 설정
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': video_path,
            'quiet': True
        }
        
        print("Downloading video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print("Loading YOLOv10 model...")  # 메시지 수정
        model = YOLOv10.from_pretrained('jameslahm/yolov10n')  # YOLOv10 모델 로드
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Cannot open video file")
        
        # 윈도우 생성 및 크기 조정
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # 객체 감지 수행 (person=0, car=2, motorcycle=3)
            results = model.predict(source=frame, classes=[0, 2, 3], conf=0.3)  # car 클래스 추가
            
            # 객체 수 계산
            boxes = results[0].boxes
            person_count = len([box for box in boxes if box.cls == 0])
            car_count = len([box for box in boxes if box.cls == 2])      # 자동차 카운트 추가
            motorcycle_count = len([box for box in boxes if box.cls == 3])
            
            # 결과 시각화
            annotated_frame = results[0].plot()
            
            # 정보 표시를 위한 반투명 배경
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 115), (0, 0, 0), -1)  # 높이 증가
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # 텍스트 추가
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f'Person: {person_count}', 
                       (10, 35), font, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Car: {car_count}', 
                       (10, 70), font, 1, (255, 255, 0), 2)  # 노란색
            cv2.putText(annotated_frame, f'Bike: {motorcycle_count}', 
                       (10, 105), font, 1, (0, 255, 255), 2)
            
            # 화면에 표시
            cv2.imshow("Object Detection", annotated_frame)
            
            # 'q' 키나 ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()
        
        # 임시 비디오 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        # 에러 발생 시에도 임시 파일 삭제 시도
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=0zGm62U8hzA"
    download_and_process_video(url)
