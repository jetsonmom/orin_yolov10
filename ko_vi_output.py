#한글로 카운트 결과 파일은  output_filename = f'detected_{now.strftime("%Y%m%d_%H%M%S")}.mp4'
import yt_dlp
from ultralytics import YOLOv10  # YOLO를 YOLOv10으로 변경
import cv2
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def put_text(img, text, position, font_size=32, font_color=(255,255,255)):
    """나눔고딕 폰트로 한글 텍스트를 이미지에 추가하는 함수"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    font_paths = [
        "NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        "나눔고딕.ttf"
    ]
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except:
            continue
    
    if font is None:
        print("나눔고딕 폰트를 찾을 수 없습니다. 폰트를 설치해주세요.")
        return img
    
    draw.text(position, text, font=font, fill=font_color)
    return np.array(img_pil)

def download_and_process_video(url, save_video=True):  # model_name 파라미터 제거
    """
    YouTube 영상을 다운로드하고 객체를 감지하는 함수
    
    Args:
        url (str): YouTube 영상 URL
        save_video (bool): 처리된 영상 저장 여부
    """
    try:
        # 임시 파일명 생성
        timestamp = int(time.time())
        video_path = f'video_{timestamp}.mp4'
        
        # 저장할 비디오 파일명 생성 (현재 날짜시간 사용)
        now = datetime.now()
        output_filename = f'detected_{now.strftime("%Y%m%d_%H%M%S")}.mp4'
        
        # 영상 다운로드 설정
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': video_path,
            'quiet': True
        }
        
        print("영상 다운로드 중...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print("YOLOv10 모델 로딩 중...")  # 메시지 수정
        model = YOLOv10.from_pretrained('jameslahm/yolov10n')  # YOLOv10 모델 로드
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("비디오를 열 수 없습니다.")
        
        # 비디오 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # VideoWriter 객체 생성
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            print(f"처리된 영상을 {output_filename}로 저장합니다.")
        
        # 윈도우 생성
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
            
            # BGR에서 RGB로 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 정보 표시를 위한 반투명 배경
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 125), (0, 0, 0), -1)  # 높이 증가
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # 한글 텍스트 추가
            annotated_frame = put_text(
                annotated_frame,
                f"사람: {person_count}명",
                (10, 15),
                font_size=30,
                font_color=(0, 255, 0)
            )
            
            # 자동차 카운트 텍스트 추가
            annotated_frame = put_text(
                annotated_frame,
                f"자동차: {car_count}대",
                (10, 50),
                font_size=30,
                font_color=(255, 255, 0)  # 노란색
            )
            
            annotated_frame = put_text(
                annotated_frame,
                f"오토바이: {motorcycle_count}대",
                (10, 85),
                font_size=30,
                font_color=(0, 255, 255)
            )
            
            # RGB에서 BGR로 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # 처리된 프레임 저장
            if save_video:
                out.write(annotated_frame)
            
            # 화면에 표시
            cv2.imshow("Object Detection", annotated_frame)
            
            # 'q' 키나 ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        # 자원 해제
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        # 임시 비디오 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)
            
        if save_video:
            print(f"\n처리된 영상이 저장되었습니다: {output_filename}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        # 에러 발생 시에도 임시 파일 삭제 시도
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=0zGm62U8hzA"
    download_and_process_video(url, save_video=True)
