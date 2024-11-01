# 한글로 카운트, 영상 저장, 실행후 저장한 영상 삭제, 결과 영상 보존
import yt_dlp
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_korean_text(img, text, position, font_size=32, font_color=(255,255,255)):
    """한글 텍스트를 이미지에 추가하는 함수"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Windows
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        try:
            # Ubuntu
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
        except:
            try:
                # macOS
                font = ImageFont.truetype("/Library/Fonts/AppleGothic.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=font_color)
    return np.array(img_pil)

def download_and_process_video(url, model_name='yolov8n.pt'):
    """
    YouTube 영상을 다운로드하고 객체를 감지하는 함수
    
    Args:
        url (str): YouTube 영상 URL
        model_name (str): 사용할 YOLO 모델 이름
    """
    try:
        # video 폴더 생성
        if not os.path.exists('video'):
            os.makedirs('video')
            
        # 임시 파일명과 출력 파일명 생성
        timestamp = int(time.time())
        video_path = f'video/video_{timestamp}.mp4'
        output_path = 'video/output.mp4'
        
        # 영상 다운로드 설정
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': video_path,
            'quiet': True
        }
        
        print("영상 다운로드 중...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print("YOLO 모델 로딩 중...")
        model = YOLO(model_name)
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("비디오를 열 수 없습니다.")
        
        # 비디오 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # VideoWriter 객체 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # 윈도우 생성 및 크기 조정
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 객체 감지 수행 (person=0, car=2, motorcycle=3)
            results = model.predict(source=frame_rgb, classes=[0, 2, 3], conf=0.3)
            
            # 객체 수 계산
            boxes = results[0].boxes
            person_count = len([box for box in boxes if box.cls == 0])
            car_count = len([box for box in boxes if box.cls == 2])
            motorcycle_count = len([box for box in boxes if box.cls == 3])
            
            # 결과 시각화
            annotated_frame = results[0].plot()
            
            # RGB에서 BGR로 다시 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # 정보 표시를 위한 반투명 배경 (높이 증가)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 125), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # 한글 텍스트 추가
            annotated_frame = put_korean_text(
                annotated_frame, 
                f"사람: {person_count}명", 
                (10, 15), 
                font_size=30, 
                font_color=(0, 255, 0)
            )
            
            annotated_frame = put_korean_text(
                annotated_frame, 
                f"자동차: {car_count}대", 
                (10, 50), 
                font_size=30, 
                font_color=(255, 255, 0)  # 노란색
            )
            
            annotated_frame = put_korean_text(
                annotated_frame, 
                f"오토바이: {motorcycle_count}대", 
                (10, 85), 
                font_size=30, 
                font_color=(0, 255, 255)
            )
            
            # 처리된 프레임 저장
            out.write(annotated_frame)
            
            # 화면에 표시
            cv2.imshow("Object Detection", annotated_frame)
            
            # 'q' 키나 ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        # 자원 해제
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"처리된 영상이 저장되었습니다: {output_path}")
        print("영상을 실행합니다...")
        
        # 저장된 영상 실행
        os.system(f'start {output_path}' if os.name == 'nt' else f'xdg-open {output_path}')
        
        # 임시 다운로드 파일만 삭제
        if os.path.exists(video_path):
            os.remove(video_path)
            print("다운로드된 원본 영상이 삭제되었습니다.")
        print(f"처리된 영상이 {output_path}에 저장되었습니다.")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        # 에러 발생 시 임시 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=0zGm62U8hzA"
    download_and_process_video(url)
