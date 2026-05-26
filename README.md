# AgriFM 실험 구현 — 스마트팩토리 캡스톤

## 개요

AI 코딩 툴(Claude Code, Google Colab)을 활용하여 논문 AgriFM의 cropland mapping 실험을 구현·재현한 과제입니다.

- 논문: AgriFM: A Multi-source Temporal Remote Sensing Foundation Model for Agriculture Mapping (Remote Sensing of Environment, 2026)
- 원본 코드: github.com/flyakon/AgriFM
- 사용 도구: Claude Code(코드 분석), Google Colab T4(학습·평가·추론)

## 파일 구성

- AgriFM_실험_결과.ipynb : Colab 실행 노트북 (환경구축 → 학습 → 평가 → 추론)
- PROMPTS.md : 사용한 프롬프트 및 수행 기록
- report/01_code_analysis.md : 코드 구조 분석 결과

## 실험 결과 (example_dataset, 1 epoch fine-tune)

- Val (51): mFscore 52.03 / mIoU 37.60 / aAcc 60.10
- Test (196): mFscore 48.74 / mIoU 35.79 / aAcc 60.46
- Inference (49): 예측맵 생성 및 시각화 완료

## 재현 방법

1. Google Colab에서 런타임 GPU(T4) 선택
2. AgriFM_실험_결과.ipynb 를 위에서부터 순차 실행
3. AgriFM.pth(사전학습 가중치)와 example_dataset 을 Google Drive에 위치

전체 데이터가 아닌 example_dataset으로 1 epoch만 학습한 제약이 있어 논문 정확도(70%+)에는 미치지 못하나, 학습-검증-테스트-추론 전 과정의 정상 동작을 확인하였습니다.
