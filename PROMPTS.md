# PROMPTS LOG

## 1. 과제 목표

- 타겟 논문: AgriFM (*Remote Sensing of Environment*, 2026)
- 목표: AI 코딩 툴로 논문의 cropland mapping 실험 구현 및 분석

## 2. 사용 도구

- Claude Code: 코드 구조 분석
- Google Colab (T4): 환경 구축, 학습·평가·추론
- GitHub: github.com/YujinJung-Lab/agrifm-capstone

## 3. 코드 분석 (Claude Code)

- repo 클론 후 config(Encoder/Neck/Head) 분석
- modified Video Swin의 temporal-spatial 동기 다운샘플 구현 위치 확인
- 입력 S2 채널 C=10, 시퀀스 길이 T=32 확인

사용 프롬프트(요약):

> AgriFM repo 클론 후 `configs/cropland_mapping.py` 의 Encoder/Neck/Head, modified Video Swin 구현 위치, train/test/inference 흐름 분석.

> S2 입력의 T, C 확정 (T=32, C=10 가정 검증).

## 4. 환경 구축 (Colab)

- torch 2.2.2 + mmcv 2.2.0 휠 직접 설치
- mmseg MMCV_MAX 상한 패치 (2.2 → 2.3)
- example_dataset 압축 해제, h5 shape (32,10,256,256) 확인

## 5. 실험 실행 (Colab)

- config 경로/epoch/scheduler/batch 수정 후 1 epoch fine-tune
- train.py → 학습·검증, test.py → 테스트, inference.py → 예측맵 생성

## 6. 오류와 해결

- openmim ImpImporter (py3.12) → 휠 직접 설치로 우회
- conda 환경 세션 타임아웃 → pip 휠 설치로 전환
- MMCV 2.2.0 incompatible → mmseg MMCV_MAX 상향
- scheduler begin/end 충돌 → end 20으로 수정
- CUDA OOM → batch_size 1, expandable_segments 적용
- inference.py 예측맵 0/1 저장으로 흑백 → 컬러 팔레트 후처리

## 7. 결과

- Train: loss 0.7019, best_mFscore_epoch_1.pth 저장
- Val (51): mFscore 52.03 / mIoU 37.60 / aAcc 60.10
- Test (196): mFscore 48.74 / mIoU 35.79 / aAcc 60.46
- Inference (49): 예측맵 생성 및 시각화 완료
- example_dataset 1 epoch 제약으로 논문 정확도(70%+) 대비 낮으나 전 과정 정상 동작 확인
