# Bank Marketing Campaign Optimization

은행 텔레마케팅 캠페인 데이터를 활용하여 고객의 정기예금 가입 여부를 예측하고,  
마케팅 비용 대비 효율을 극대화하는 타겟팅 전략을 설계한 머신러닝 프로젝트입니다.

단순 예측 모델이 아니라,  
**실제 캠페인 적용 가능한 의사결정 구조까지 포함한 분석 시스템**

---

## 핵심 문제 정의

> "어떤 고객에게 연락해야 전환율을 높이면서 비용을 줄일 수 있는가?"

---

## 데이터셋

- 출처: UCI Bank Marketing Dataset
- 데이터 크기: 11,162 rows × 17 columns
- 타겟 변수: `deposit` (yes / no → 1 / 0)

### 주요 변수

- 고객 정보: age, job, marital, education
- 금융 정보: balance, housing, loan
- 캠페인 정보: campaign, pdays, poutcome
- 연락 정보: contact, duration

---

## 프로젝트 구조
```
bank-marketing-campaign-optimization/
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ └── evaluate.py
├── config/
│ └── config.yaml
├── outputs/
│ ├── models/
│ └── figures/
└── README.md
```
---

## 데이터 전처리

### 주요 처리 로직

- `deposit` → binary encoding (yes=1, no=0)
- `pdays = -1` → NaN 처리 (미접촉 의미 분리)
- 중복 데이터 제거
- 결측치 탐지 및 로그 출력

---

## Feature Engineering

도메인 기반 마케팅 변수 생성

### 생성 피처

- **campaign_bucket**
  - 접촉 횟수 구간화 (1회 / 2회 / 3회 / 4-5회 / 6회+)

- **prev_success**
  - 이전 캠페인 성공 여부

- **was_contacted_before**
  - 과거 접촉 여부

- **balance_segment**
  - 고객 잔액 구간화 (negative / low / mid / high)

- 범주형 변수 One-Hot Encoding

---

## 모델링 전략

### 핵심 설계: 데이터 누수 통제 실험

`duration` 변수는 통화 종료 후에만 알 수 있는 값 → 실제 캠페인에서는 사용 불가

따라서 두 가지 버전 비교

### Version A (Upper Bound)
- duration 포함
- 모델 성능 상한선 확인 목적

### Version B (Real-world)
- duration 제외
- 실제 타겟팅 적용 모델

---

## 모델

- Logistic Regression (baseline)
- Random Forest
- XGBoost

---

## 학습 및 검증 전략

- Stratified Train/Test Split
- Stratified K-Fold Cross Validation
- 평가 지표:
  - ROC-AUC
  - Average Precision
  - Classification Report

---

## 핵심 평가 결과

### ROC-AUC 비교

| Model | With Duration | Without Duration |
|------|---------------|------------------|
| Logistic Regression | 0.852 | 0.656 |
| Random Forest | **0.876** | **0.688** |
| XGBoost | 0.859 | 0.682 |

---

## 마케팅 시뮬레이션 (A/B Test)

### 실험 설계

- Control: 전체 고객
- Treatment: 예측 확률 상위 30%

### 결과

| Group | Conversion Rate | Sample Size |
|------|----------------|-------------|
| Control | 47.4% | 2,233 |
| Treatment | 70.3% | 669 |

### 효과

- 접촉 비용 절감: **70%**
- 전환율 개선: **+48.3%**

---

## 핵심 인사이트

### 1. 데이터 누수 영향

- duration 포함 시 성능 과대평가 발생
- 실제 환경에서는 사용할 수 없는 정보

→ **모델 평가 시 시점 기반 feature 검증 필수**

---

### 2. 타겟팅 전략 효과

- 상위 확률 고객만 타겟팅 시
  - 비용 감소
  - 전환율 증가

→ 단순 예측이 아닌 **ranking 문제로 접근해야 함**

---

### 3. 중요한 변수

- previous campaign outcome (`poutcome`)
- contact 횟수 (`campaign`)
- 고객 잔액 (`balance`)

→ 행동 기반 변수 > 인구통계 변수

---

## 실행 방법

```bash
pip install -r requirements.txt

# 데이터 전처리
python src/preprocessing.py

# 모델 학습
python src/train.py

# 평가 및 시뮬레이션
python src/evaluate.py

출력 결과
모델: outputs/models/
ROC Curve: outputs/figures/roc_curves.png
Feature Importance: outputs/figures/feature_importance.png
```
---
## 한계 및 개선 방향
### 한계
- 정적 데이터 기반 분석
- 고객 행동 로그 부족
- 실제 비용 구조 미반영
### 개선 방향
- 시계열 기반 고객 행동 모델링
- LTV (Customer Lifetime Value) 예측
- Uplift Modeling 적용 (진짜 캠페인 효과 측정)

## 한 줄 요약
"예측 모델이 아니라, 돈이 되는 타겟팅 시스템을 설계한 프로젝트"
