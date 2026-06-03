# PROMPTS.md — AgriFM 실험 재현 프롬프트 로그
본 파일은 AgriFM 논문의 실험 파트를 AI 코딩 툴(Claude Code, Google Colab)을 활용하여 재현하는 과정에서 사용한 프롬프트와 그 결과를 시간순으로 기록한 것이다.
- 논문: AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Agriculture Mapping (Remote Sensing of Environment, 2026)
- 원본 코드: https://github.com/flyakon/AgriFM
- 사용 도구: Claude Code (로컬 코드 분석), Google Colab T4 (환경 구축 및 학습·평가·추론)
- 작성자: 정유진
- GitHub: https://github.com/YujinJung-Lab/agrifm-capstone
---
## 1. 과제 시작 및 코드 분석 환경 준비 (Claude Code, 로컬)
### Prompt 1-1. 작업 디렉토리 클론
```text
이 폴더에서 작업할 거야. 먼저 git으로 https://github.com/flyakon/AgriFM.git
를 ./AgriFM 에 clone 해줘. (코드 분석만 할 거라 학습/실행은 안 함)
```
Purpose:
- AgriFM repo를 로컬에 받아 코드 구조를 정적 분석.
- Colab에서 무거운 실험을 돌리기 전 모델 구조를 먼저 이해.
Assistant actions:
- `git clone https://github.com/flyakon/AgriFM.git ./AgriFM` 실행.
- 폴더 구조 확인.
```bash
git clone https://github.com/flyakon/AgriFM.git ./AgriFM
cd AgriFM
ls
```
Result:
- `AgriFM/`(모델 코드), `configs/`(설정), `train.py`, `test.py`, `inference.py`, `example_dataset/` 등이 존재함을 확인.
---
### Prompt 1-2. config 전체 분석
```text
configs/cropland_mapping.py 전체를 읽고 Encoder / Neck / Head 설정을 표로
정리한 뒤, 각 파라미터가 논문 AgriFM의 어느 구조에 대응되는지 한국어로
설명해줘.
```
Purpose:
- 모델 구성 요소(Encoder/Neck/Head)를 식별.
- 각 하이퍼파라미터가 논문의 어느 부분에 해당하는지 확인.
Assistant actions:
- `configs/cropland_mapping.py`를 읽고 다음 구조 확인:
| 구성 요소 | 클래스 | 주요 파라미터 |
| --- | --- | --- |
| Encoder | `PretrainingSwinTransformer3DEncoder` | embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=(8,7,7), patch_size=(4,2,2) |
| Neck | `MultiFusionNeck` | embed_dim=1024, out_size=(256,256) |
| Head | `CropFCNHead` | num_classes=2 (작물/비작물) |
Result:
- AgriFM은 Multi-source Encoder + U-Net 식 Neck + FCN Head 구조임을 확인.
- `init_cfg`의 사전학습 가중치 경로(`pretrained=...`)가 placeholder이므로 Colab 단계에서 수정 필요.
---
### Prompt 1-3. modified Video Swin 구현 위치 확인
```text
모델 코드(AgriFM/ 폴더)에서 논문 핵심인 "temporal down-sampling을 spatial
down-sampling과 동기화한 modified Video Swin Transformer"가 실제로 구현된
파일/클래스/함수를 찾아서, 원본 Video Swin 대비 뭐가 바뀌었는지 짚어줘.
```
Purpose:
- 논문의 핵심 기여(시간축-공간축 동기 다운샘플)가 실제 코드 어디서 구현되는지 파악.
Assistant actions:
- `AgriFM/models/video_swin_transformer.py`를 검색·분석.
- `PatchMerging` 클래스(L297~358)에서 다음 차이 확인:
```python
# 원본 Video Swin: 시간축(D)은 유지하고 spatial(H, W)만 stride 2 다운샘플
# x[:, :, 0::2, 0::2, :]
# AgriFM modified: 시간축도 D_step으로 stride 다운샘플 + mean_frame_down으로 평균
# x[:, ::D_step, ...]
```
Result:
- 핵심 변경 위치는 `PatchMerging`의 `D_step` 인자와 `mean_frame_down` 옵션.
- `BasicLayer`(~L378), `SwinTransformer3D`(~L754)에서 이를 호출.
- 즉, "temporal을 spatial과 동기 다운샘플"이 논문 컨트리뷰션의 코드 단 실체.
---
### Prompt 1-4. 입력 T, C 파라미터 확정
```text
S2 입력의 channel 수 C와 시퀀스 길이 T가 코드 어디서 강제되는지 찾아줘.
config의 num_frames=32와 patch_emd_cfg의 in_chans=10이 실제 데이터 shape와
맞는지 확인이 필요해.
```
Purpose:
- 데이터셋의 실제 shape와 config가 일치하는지 사전 검증.
Assistant actions:
- 4곳 교차 검증으로 `C=10`(`in_chans=10`) 확정.
- `T=32`는 Neck의 `in_fusion_key_list`가 모두 1024 채널이 되려면 stage별 D=4·2·1로 줄어야 하므로 역산하여 32로 추정 (이후 Colab에서 h5로 실측 검증 예정).
```python
# patch_emd_cfg
in_chans = 10
patch_size = (4, 2, 2)  # (T-patch, H-patch, W-patch)
# num_frames = 32 → 첫 PatchEmbed 후 D = 32/4 = 8
# downsample_steps 4단 적용 → D 8→4→2→1
```
Result:
- C=10, T=32 코드상 확정.
- 다음 단계에서 Colab의 실제 h5 데이터로 검증할 것.
---
### Prompt 1-5. train/test/inference 스크립트 흐름 요약
```text
test.py / train.py / inference.py 흐름을 각각 5줄 이내로 요약해줘.
```
Purpose:
- 어떤 entry point를 어떤 순서로 실행해야 하는지 파악.
Assistant actions:
- 세 스크립트의 entry point 구조를 분석:
```text
train.py     : Config 로드 → Runner(epoch loop) → Val → best checkpoint 저장
test.py      : Config 로드 → load_checkpoint → TestLoop → metric 출력
inference.py : Config 로드 → load_checkpoint → DataLoader → 예측맵 png 저장
```
Result:
- Colab에서는 train → test → inference 순서로 실행.
- 각 단계의 출력물: checkpoint(.pth), metric 로그, 예측 png.
---
## 2. Colab 환경 구축 (실행 환경)
### Prompt 2-1. 초기 설치 시도
```text
Colab에서 AgriFM 돌리려고 해. README 보니까 mmcv 2.1.0 깔라고 적혀있어.
런타임은 GPU T4고 Python은 3.12야. 일단 그대로 깔아볼게.
```
Purpose:
- README 권장 절차대로 설치 시도.
Assistant actions:
```bash
pip install -U openmim
mim install mmcv==2.1.0
```
Result:
- `AttributeError: pkgutil has no ImpImporter` 발생.
- 원인: Python 3.12에서 `pkgutil.ImpImporter`가 제거됨. openmim 0.3.9가 이를 참조하여 import 단계에서 실패.
---
### Prompt 2-2. condacolab으로 우회 시도
```text
mmcv 2.1.0이 Python 3.12에서 안 깔려. condacolab으로 Python 3.9 환경
만들어서 conda env 그대로 재현해볼게.
```
Purpose:
- 논문 권장 환경(Python 3.9 + mmcv 2.1)을 conda로 재현하여 호환성 문제 우회.
Assistant actions:
```python
!pip install -q condacolab
import condacolab
condacolab.install()
```
```bash
conda env create -f environment.yml
```
Result:
- `ModuleNotFoundError: distutils` 발생 → `pip install setuptools<60` 선설치로 우회 후 condacolab 설치 성공.
- 그러나 `conda env create -f environment.yml` 약 17분 후 무료 Colab 세션 타임아웃으로 전체 환경 소실.
- 결론: conda 경로는 무료 Colab에서 비현실적. 휠 직접 설치로 전환 결정.
---
### Prompt 2-3. 휠 직접 설치로 전환
```text
conda는 시간 초과로 안 되겠어. torch / torchvision / mmcv 휠을 cu121 기준
직접 다운로드 설치하는 방향으로 가자. cp312 휠이 있는 가장 가까운 안정 조합
찾아줘.
```
Purpose:
- 환경 재현을 포기하고 의존성 chain의 버전을 정합 가능한 조합으로 새로 구성.
Assistant actions:
- `torch==2.1.1`은 cp312 휠 없음 확인 → `torch==2.2.2`부터 cp312 휠 존재.
- mmcv는 openmmlab 휠 인덱스에서 torch 2.2 + cu121용 2.2.0 휠 존재 확인.
```bash
pip install -q numpy==1.26.4
pip install -q torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -q "mmcv==2.2.0" -f \
    https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install -q mmengine mmsegmentation
```
```python
import torch, mmcv, mmengine
print(torch.__version__, torch.cuda.is_available())
print(mmcv.__version__, mmengine.__version__)
```
Result:
```
2.2.2+cu121 True
2.2.0 0.10.7
```
- 설치 성공. pip dependency resolver 충돌 경고가 일부 출력되었으나 import는 정상이므로 무시.
---
### Prompt 2-4. MMCV 상한 패치
```text
모델 빌드 단계에서 "MMCV==2.2.0 is incompatible" 에러가 나. mmseg가
MMCV_MAX를 너무 낮게 잡고 있는 것 같아. 어디 고치면 돼?
```
Purpose:
- mmseg 패키지의 mmcv 버전 검사를 통과시키기 위한 소스 패치.
Assistant actions:
- `mmseg/__init__.py`의 `MMCV_MAX` 상수를 검색·수정.
```python
import re, pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/mmseg/__init__.py")
txt = p.read_text()
txt = re.sub(r"MMCV_MAX\s*=\s*'2\.2\.0'", "MMCV_MAX = '2.3.0'", txt)
p.write_text(txt)
print("MMCV_MAX 패치 완료")
```
Result:
- mmcv 2.2.0 import 후 incompatible 에러 사라짐.
---
## 3. 데이터셋 준비
### Prompt 3-1. Google Drive mount 및 자산 확인
```text
사전학습 가중치 AgriFM.pth와 example_dataset.zip을 Google Drive에 올려뒀어.
Drive mount하고 두 파일 확인하는 코드 줘.
```
Purpose:
- 학습에 필요한 사전학습 가중치와 example 데이터를 Colab에 가져오기.
Assistant actions:
```python
from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/MyDrive/ | grep -E "AgriFM|example"
```
Result:
- `AgriFM.pth`, `example_dataset.zip` 두 파일 존재 확인.
---
### Prompt 3-2. example_dataset 압축 해제
```text
example_dataset.zip을 /content/AgriFM/example_dataset 아래에 풀어줘.
```
Purpose:
- 학습/검증/테스트 분할 텍스트와 h5 샘플 파일을 사용 가능한 경로에 배치.
Assistant actions:
```bash
mkdir -p /content/AgriFM/example_dataset
unzip -q /content/drive/MyDrive/example_dataset.zip -d /content/AgriFM/example_dataset
ls /content/AgriFM/example_dataset/example_dataset
```
Result:
- 이중 폴더(`example_dataset/example_dataset/`) 구조로 압축 해제됨.
- 내부에 `data_list/{train,val,test}.txt`, `h5_samples/*.h5` 존재.
- h5 파일 총 335개 확인.
---
### Prompt 3-3. h5 shape 검증
```text
h5 파일 하나 열어서 S2의 shape를 찍어봐. 1단계에서 역산한 (T=32, C=10,
H=256, W=256)이 맞는지 검증하고 싶어.
```
Purpose:
- 1단계 코드 분석에서 추정한 T, C 값을 실제 데이터로 검증.
Assistant actions:
```python
import h5py, glob
fp = glob.glob('/content/AgriFM/example_dataset/example_dataset/h5_samples/*.h5')[0]
with h5py.File(fp, 'r') as f:
    print("keys:", list(f.keys()))
    print("S2 shape:", f['S2'].shape)
    print("label shape:", f['label'].shape)
```
Result:
```
keys: ['S2', 'label']
S2 shape: (32, 10, 256, 256)
label shape: (256, 256)
```
- 1단계 역산값 T=32, C=10이 실제 데이터와 일치함을 확인.
---
## 4. config 수정
### Prompt 4-1. 경로 및 학습 파라미터 수정
```text
원본 configs/cropland_mapping.py를 복사해서 cropland_mapping_colab.py 만들고
다음 항목 수정해줘:
- pretrained 경로를 /content/drive/MyDrive/AgriFM.pth로
- data_path / data_list_path를 위에서 푼 경로로
- 무료 GPU 제약이라 max_epochs 50→1, batch_size 4·8→1, num_workers→2
```
Purpose:
- 원본 config의 placeholder 경로 3개를 실제 환경에 맞게 교체.
- 무료 Colab T4(약 15GB VRAM)에서 돌아가도록 학습 규모 축소.
Assistant actions:
```bash
cp configs/cropland_mapping.py configs/cropland_mapping_colab.py
```
```python
import re, pathlib
p = pathlib.Path("configs/cropland_mapping_colab.py")
txt = p.read_text()
txt = txt.replace("'/path/to/AgriFM.pth'",
                  "'/content/drive/MyDrive/AgriFM.pth'")
txt = txt.replace("'/path/to/data'",
                  "'/content/AgriFM/example_dataset/example_dataset/h5_samples'")
txt = txt.replace("'/path/to/data_list'",
                  "'/content/AgriFM/example_dataset/example_dataset/data_list'")
txt = re.sub(r"max_epochs=\d+", "max_epochs=1", txt)
txt = re.sub(r"batch_size=(4|8)", "batch_size=1", txt)
txt = re.sub(r"num_workers=\d+", "num_workers=2", txt)
p.write_text(txt)
```
Result:
- placeholder 3개 교체 완료.
- max_epochs=1, batch_size=1, num_workers=2로 축소.
---
### Prompt 4-2. param_scheduler begin/end 수정
```text
config 그대로 두니까 학습 시작하자마자 "ValueError: end should be larger
than begin, got begin=1000 end=44" 에러나. max_epochs를 줄였으니까
scheduler도 그에 맞게 줄여야 할 듯.
```
Purpose:
- max_epochs 축소(50→1)에 따라 param_scheduler의 begin/end 값을 조정.
Assistant actions:
```python
p = pathlib.Path("configs/cropland_mapping_colab.py")
txt = p.read_text()
txt = txt.replace("end=1000", "end=20").replace("begin=1000", "begin=20")
p.write_text(txt)
```
Result:
- scheduler 에러 해소.
---
## 5. 학습 실행
### Prompt 5-1. 1 epoch 학습 실행
```text
이제 train.py 실행해줘. config는 위에서 수정한 cropland_mapping_colab.py 사용.
```
Purpose:
- 1 epoch fine-tune으로 파이프라인이 정상 동작하는지 검증.
Assistant actions:
```bash
cd /content/AgriFM
python train.py configs/cropland_mapping_colab.py --work-dir work_dirs/colab
```
Result (학습 로그 일부):
```text
Epoch(train) [1][50/88]  lr: 3.7254e-05  loss: 0.7019  crop_ce_loss: 0.7019
Saving checkpoint at 1 epochs
Epoch(val) [1][51/51]    aAcc: 60.1000  mFscore: 52.0300  mIoU: 37.6000
The best checkpoint with 52.0300 mFscore at 1 epoch is saved to best_mFscore_epoch_1.pth
```
- train loss 0.7019, val mFscore 52.03 / mIoU 37.60 / aAcc 60.10.
---
### Prompt 5-2. CUDA OOM 해결
```text
batch_size 4로 시작하니까 "CUDA out of memory" 떠. T4 GPU에서 돌릴 수
있는 한도로 줄여줘.
```
Purpose:
- T4의 VRAM 제약 안에서 학습이 끝까지 돌도록 메모리 사용 조정.
Assistant actions:
```python
import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
```
```python
# config에서 batch_size를 1로 조정 (이미 4-1에서 수정 완료)
```
Result:
- batch_size=1 + expandable_segments 적용 후 OOM 없이 1 epoch 학습 완주.
- GPU 메모리 사용량 약 11.5 GB 수준.
---
## 6. 테스트 실행
### Prompt 6-1. 테스트셋 평가
```text
저장된 best_mFscore_epoch_1.pth로 test.py 돌려서 테스트셋 196장 평가해줘.
```
Purpose:
- 학습된 가중치의 테스트셋 일반화 성능 측정.
Assistant actions:
```bash
python test.py configs/cropland_mapping_colab.py \
    work_dirs/colab/best_mFscore_epoch_1.pth
```
Result (테스트 로그):
```text
per class results:
+-------+--------+-----------+--------+-------+-------+
| Class | Fscore | Precision | Recall |  IoU  |  Acc  |
+-------+--------+-----------+--------+-------+-------+
|   0   | 24.23  |   79.73   | 14.29  | 13.79 | 14.29 |
|   1   | 73.25  |   58.79   | 97.12  | 57.79 | 97.12 |
+-------+--------+-----------+--------+-------+-------+
Epoch(test) [196/196]    aAcc: 60.4600  mFscore: 48.7400  mIoU: 35.7900
```
- 테스트셋(196장) mFscore 48.74 / mIoU 35.79 / aAcc 60.46.
- 작물 클래스(Class 1) Recall 97.12로 작물 영역 검출 성능은 양호.
---
## 7. 추론 및 시각화
### Prompt 7-1. inference.py 실행 - 첫 시도
```text
inference.py 돌려서 예측 png 뽑아보자.
```
Purpose:
- 학습된 모델로 테스트 이미지에 대한 segmentation 예측맵 생성.
Assistant actions:
```bash
python inference.py configs/cropland_mapping_colab.py \
    work_dirs/colab/best_mFscore_epoch_1.pth ./vis_results
```
Result:
- `KeyError: 'result_path'` → `AttributeError: 'ConfigDict' object has no attribute 'result_path'`.
- 원인: `inference.py`가 `cfg.result_path`를 읽는데 config에 해당 키가 없음.
---
### Prompt 7-2. result_path 키 추가
```text
config에 result_path가 없어서 에러나는 듯해. config 맨 아래에 추가해줘.
```
Purpose:
- inference 스크립트가 요구하는 누락 키를 config에 보강.
Assistant actions:
```python
with open('configs/cropland_mapping_colab.py', 'a') as f:
    f.write("\nresult_path = './vis_results'\n")
import os
os.makedirs('./vis_results', exist_ok=True)
```
```bash
python inference.py configs/cropland_mapping_colab.py \
    work_dirs/colab/best_mFscore_epoch_1.pth ./vis_results
```
Result:
- 추론 성공: `100% 49/49 [02:52<00:00, 3.52s/it]`.
- `./vis_results/` 폴더에 49개 `*_pred.png` 파일 생성.
---
### Prompt 7-3. 결과 png 시각적 확인
```text
inference 끝났는데 결과 png 열어보니까 데이터가 잘못된 것 같아. 어떤 유형의 문제인지 확인하고 싶어.
```
Purpose:
- 생성된 예측 png가 의도대로 보이는지 확인.
Assistant actions:
```python
from PIL import Image
import numpy as np
import glob
p = sorted(glob.glob('./vis_results/*_pred.png'))[0]
arr = np.array(Image.open(p))
print("픽셀 고유값:", np.unique(arr))
print("shape:", arr.shape)
```
Result:
```
픽셀 고유값: [0 1]
shape: (256, 256)
```
- `inference.py`가 클래스 라벨(0/1)을 그대로 픽셀값으로 저장하여, 0(검정)과 1(거의 검정)로 시각적 구분이 안 되는 상황.
- 데이터 자체는 정상. 시각화만 문제.
---
### Prompt 7-4. inference.py 소스 확인
```text
inference.py가 어떻게 저장하는지 코드 자체를 보고싶어. cat 해줘.
```
Purpose:
- 흑백 저장의 원인이 되는 코드 위치를 정확히 파악.
Assistant actions:
```bash
cat /content/AgriFM/inference.py
```
Result (핵심 부분):
```python
cls_pred = torch.argmax(logits[i], dim=0)
cls_pred = cls_pred.cpu().numpy()
pred_mask_path = os.path.join(resutl_path, '%s_pred.png' % tile_name)
io.imsave(pred_mask_path, cls_pred.astype(np.uint8), check_contrast=False)
```
- 원인 확인: `cls_pred`가 `uint8`로 0/1 값을 그대로 저장 → 8-bit 이미지에서 0과 1은 모두 거의 검정.
- 해결 방향: 재추론 없이, 저장된 png를 후처리로 컬러 팔레트 적용.
---
### Prompt 7-5. 컬러 팔레트 후처리
```text
재추론 없이 저장된 흑백 png들에 컬러 팔레트 입혀줘. 비작물=검정, 작물=녹색
으로 보이게. 처음 3장 미리보기로 띄워줘.
```
Purpose:
- 0/1 라벨을 시각적으로 구분 가능한 색으로 변환하여 결과 검증 및 보고서 자료로 활용.
Assistant actions:
```python
import glob, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
vis_dir = '/content/AgriFM/vis_results'
preds = sorted(glob.glob(f'{vis_dir}/*_pred.png'))
print(f"예측맵 {len(preds)}개 발견")
palette = np.array([[0, 0, 0], [0, 200, 0]], dtype=np.uint8)  # 0=흑, 1=녹
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, p in zip(axes, preds[:3]):
    m = np.array(Image.open(p))
    m = (m > 0).astype(int)
    ax.imshow(palette[m])
    ax.set_title(f'{os.path.basename(p)}\ncrop={100*m.mean():.1f}%', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('/content/AgriFM/prediction_colored.png', dpi=120, bbox_inches='tight')
plt.show()
```
Result:
- 3장 미리보기에서 작물 영역이 녹색, 비작물이 검정으로 명확히 구분됨.
- 객체별 작물 비율(crop=100.0%, 100.0%, 97.0%)이 제목에 함께 표시되어 정량 확인 가능.
- `prediction_colored.png` 저장 완료.
---
## 8. 최종 결과 정리
### Prompt 8-1. 학습-검증-테스트-추론 통합 요약
```text
지금까지 결과 다 정리해서 표로 만들어줘. 학습 / 검증 / 테스트 / 추론 단계별로.
```
Purpose:
- 보고서에 사용할 정량 결과 정리.
Assistant actions:
| 단계 | 데이터 | 주요 결과 |
| --- | --- | --- |
| Train | example_dataset / 1 epoch | loss 0.7019, best_mFscore_epoch_1.pth 저장 |
| Val | 51 samples | mFscore 52.03 / mIoU 37.60 / aAcc 60.10 |
| Test | 196 samples | mFscore 48.74 / mIoU 35.79 / aAcc 60.46 |
| Inference | 49 samples | 49개 예측 png 생성 + 컬러 팔레트 후처리 |
클래스별 (Test):
| Class | Fscore | Precision | Recall | IoU |
| --- | --- | --- | --- | --- |
| 0 (비작물) | 24.23 | 79.73 | 14.29 | 13.79 |
| 1 (작물) | 73.25 | 58.79 | 97.12 | 57.79 |
Result:
- 학습-검증-테스트-추론 전 과정이 정상 동작함을 확인.
- example_dataset만으로 1 epoch만 학습한 제약상 논문 보고치(70%+) 대비 낮으나, 파이프라인 자체의 재현은 완료.
---
## 9. 마주친 오류 요약
| 단계 | 오류 | 해결 |
| --- | --- | --- |
| 환경 구축 | `mim install mmcv==2.1.0` → `AttributeError: pkgutil has no ImpImporter` | Python 3.12 비호환. openmim 우회하고 휠 직접 설치 |
| 환경 구축 | condacolab → `ModuleNotFoundError: distutils` | `pip install setuptools<60` 선설치로 우회 |
| 환경 구축 | conda env create 17분 후 무료 세션 타임아웃 | conda 포기, torch 2.2.2 + mmcv 2.2.0 휠 직접 설치 |
| 환경 구축 | torch 2.1.1 cp312 휠 부재 | torch 2.2.2(+cu121)로 상향, 의존성 chain 재정합 |
| 환경 구축 | `MMCV==2.2.0 is incompatible` (mmseg 상한) | `mmseg/__init__.py`의 `MMCV_MAX`를 2.2 → 2.3으로 패치 |
| 학습 | `ValueError: end should be larger than begin, got begin=1000 end=44` | param_scheduler의 `end`를 1000 → 20으로 수정 (max_epochs 축소 반영) |
| 학습 | `CUDA out of memory` (batch 4) | `batch_size=1`, `num_workers=2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 추론 | `KeyError: 'result_path'` | config 맨 끝에 `result_path = './vis_results'` 추가 |
| 추론 | 저장된 예측 png가 거의 새까맣게 보임 | 0/1 라벨 그대로 저장된 것이 원인. 재추론 없이 컬러 팔레트(작물=녹색) 후처리로 시각화 |
---
## 10. 실행 환경 요약
- OS: Google Colab (Ubuntu)
- GPU: NVIDIA Tesla T4 (~15 GB VRAM)
- CUDA Runtime: 12.1
- Python: 3.12.13
- PyTorch: 2.2.2 + cu121
- TorchVision: 0.17.2 + cu121
- MMEngine: 0.10.7
- MMCV: 2.2.0
- MMSegmentation: 1.x (MMCV_MAX 패치 적용)
---
## 11. 제출물
- 본 PROMPTS.md (프롬프트 로그)
- `AgriFM_분석_결과.ipynb` (Colab 실행 노트북: 환경 구축 → 학습 → 검증 → 테스트 → 추론 → 시각화)
- 별도 보고서 (메일 첨부)
---
## 12. 참고 링크
- 논문: AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Agriculture Mapping, Remote Sensing of Environment, 2026
- 원본 코드: https://github.com/flyakon/AgriFM
- 본 과제 repo: https://github.com/YujinJung-Lab/agrifm-capstone
