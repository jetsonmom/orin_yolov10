import yt_dlp
from ultralytics import YOLO
import cv2
import os
import time

def download_and_process_video(url, model_name='yolov8n.pt'):
    """
    YouTube 영상을 다운로드하고 객체를 감지하는 함수
    
    Args:
        url (str): YouTube 영상 URL
        model_name (str): 사용할 YOLO 모델 이름
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
        
        print("Loading YOLO model...")
        model = YOLO(model_name)
        
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
            
            # 객체 감지 수행
            results = model.predict(source=frame, classes=[0, 3], conf=0.3)  # person=0, motorcycle=3
            
            # 객체 수 계산
            boxes = results[0].boxes
            person_count = len([box for box in boxes if box.cls == 0])
            motorcycle_count = len([box for box in boxes if box.cls == 3])
            
            # 결과 시각화
            annotated_frame = results[0].plot()
            
            # 정보 표시를 위한 반투명 배경
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # 텍스트 추가 (영문으로 변경)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f'Person: {person_count}', 
                       (10, 35), font, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Bike: {motorcycle_count}', 
                       (10, 70), font, 1, (0, 255, 255), 2)
            
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
