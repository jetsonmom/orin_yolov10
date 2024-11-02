# orin_yolov10
<b>  ARM64용 아나콘다를 설치해야 합니다.

1. ARM64용 아나콘다 설치 파일을 다운로드합니다:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-aarch64.sh
```

2. 실행 권한을 부여합니다:
```bash
chmod +x Anaconda3-2024.02-1-Linux-aarch64.sh
```

3. 설치를 실행합니다:
```bash
./Anaconda3-2024.02-1-Linux-aarch64.sh
```

enter
yes
enter
-->  
[/home/orin/anaconda3] >>> 
PREFIX=/home/orin/anaconda3
Unpacking payload ...
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes


source ~/.bashrc
conda --version
yolov10 환경을 생성하고 base에서 변경하는 방법을 알려드리겠습니다:

4. 먼저 yolov10 환경을 생성합니다:
```bash
conda create -n yolov10 python=3.9
```

5. 생성 후 'yolov10' 환경으로 전환하려면:
```bash
conda activate yolov10
```

실행하면 프롬프트가 (base)에서 (yolov10)으로 바뀔 것입니다.

참고: 나중에 다시 base로 돌아가고 싶으시다면:
```bash
conda activate base
```

또는 환경을 완전히 비활성화하고 싶으시다면:
```bash
conda deactivate
```

requirements.txt 파일이 현재 디렉토리에 없어서 발생한 오류입니다. YOLOv10을 설치하시려는 것 같은데, 먼저 프로젝트 디렉토리를 만들고 필요한 파일들을 준비해야 합니다.

6. 먼저 프로젝트 디렉토리를 만들고 이동합니다:
```bash
mkdir yolov10
cd yolov10
```

7. 필요한 의존성 패키지들을 직접 설치합니다 (requirements.txt 파일 대신): 
```bash
pip install ultralytics
pip install torch torchvision 
pip install opencv-python
pip install numpy pandas
pip install pytest
pip install psutil
pip install PyYAML
pip install tqdm matplotlib seaborn
```


네, YouTube 영상을 다운로드하고 사람을 인식하고 카운트하는 코드를 만들어보겠습니다.

8. 먼저 YouTube 영상 다운로드를 위한 패키지를 설치합니다:
```bash
pip install pytube
pip install --upgrade pytube
pip install yt-dlp
pip install huggingface_hub
pip install transformers
pip install pillow
pip install ffpyplayer
# Ubuntu의 경우
sudo apt-get install fonts-nanum


```

9. 다음 코드를 실행하면 됩니다:

```python
from pytube import YouTube
from ultralytics import YOLO
import cv2
import numpy as np

# YouTube 영상 다운로드 함수
def download_youtube_video(url, output_path='video.mp4'):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        stream.download(filename=output_path)
        print("다운로드 완료!")
        return output_path
    except Exception as e:
        print("다운로드 중 에러 발생:", e)
        return None

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# YouTube 영상 URL을 여기에 입력하세요
video_url = "여기에 YouTube URL을 넣으세요"
video_path = download_youtube_video(video_url)

if video_path:
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # YOLO로 객체 감지
            results = model(frame, classes=[0])  # class 0은 사람
            
            # 감지된 사람 수 계산
            person_count = len(results[0].boxes)
            
            # 결과 시각화
            annotated_frame = results[0].plot()
            
            # 화면에 사람 수 표시
            cv2.putText(annotated_frame, f'People Count: {person_count}', 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow("Detection", annotated_frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()
```

사용 방법:
1. video_url = "여기에 YouTube URL을 넣으세요" 부분에 분석하고 싶은 YouTube 영상의 URL을 넣으세요.
2. 코드를 실행하면:
   - YouTube 영상을 다운로드합니다
   - 영상에서 사람을 감지합니다
   - 화면 좌상단에 감지된 사람 수를 표시합니다
   - 'q' 키를 누르면 종료됩니다



(base) orin@orin-desktop:~$ conda activate yolov10
(yolov10) orin@orin-desktop:~$ python3 video1.py
