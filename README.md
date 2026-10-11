# Gaussian-Mixture-Model

This repository is for testing a simple Gaussian Mixture Model (GMM). The goal is to understand the basic concepts and implementation of GMM through a simple experiment.

숫자 발음 데이터셋을 GMM으로 분류해보는 과제이다.

## Dataset

- 출처: [free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset/tree/master/recordings)
- 숫자 0~9를 발음한 `.wav` 파일 (클래스 10개)
- 파일명 형식: `{digit}_{speaker}_{index}.wav`
- Split: `index < 5`는 test(300개), 나머지는 train
- 샘플링 레이트: 16 kHz (mono)

## Method

1. **Feature extraction** 
   - MFCC: 13차원, `(T, 13)`
   - Mel-spectrogram: 40 mel bands + `power_to_db`, `(T, 40)`
2. **Training**: 클래스(숫자)마다 GMM 하나를 EM 알고리즘으로 학습
   - 컴포넌트 수 8, iteration 10
   - 대각 공분산(diagonal covariance) 가정
   - E-step에서 responsibility 계산, M-step에서 `mu`, `std`, `phi` 갱신
3. **Classification**: 테스트 발음의 모든 프레임에 대한 로그우도를 클래스별 GMM으로 계산하고, 가장 높은 클래스를 예측


## Results

| Feature | Dim | Test ACC |
|---|---|---|
| MFCC | 13 | 94.33% |
| Mel-spectrogram | 40 | 78.00% |

- 시드를 고정하여 실험함 (`np.random.seed(0)`)

## Discussion

- Mel-spectrogram이 MFCC보다 낮은 이유는 무엇일까?