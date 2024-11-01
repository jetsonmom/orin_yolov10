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

1. 먼저 yolov10 환경을 생성합니다:
```bash
conda create -n yolov10 python=3.9
```

2. 생성 후 'yolov10' 환경으로 전환하려면:
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

를 사용하시면 됩니다.

