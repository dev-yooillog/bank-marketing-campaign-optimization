# Campaign Response Optimization System

은행 텔레마케팅 캠페인 데이터를 활용해 고객의 정기예금(deposit) 가입 여부를 예측하고, 예측 확률 기반 타겟팅으로 마케팅 비용 대비 효율을 개선하는 프로젝트

## 문제 정의

모든 고객에게 캠페인을 진행하면 비용이 낭비되고, 실제로 반응하는 고객은 일부에 불과합니다.

> **핵심 질문: "어떤 고객이 캠페인에 반응하는가?"**

고객 속성, 접촉 이력, 이전 캠페인 결과를 바탕으로 가입 확률을 예측하는 분류 모델을 구축하고, 예측 확률 상위 고객만 타겟팅했을 때의 비즈니스 효과를 시뮬레이션함

## 데이터

- 출처: Bank Marketing Dataset (포르투갈 은행 텔레마케팅 캠페인 기록)
- 규모: 11,162건, 17개 변수
- 주요 변수: 고객 정보(age, job, marital, education), 접촉 정보(contact, campaign, day, month), 이전 캠페인 이력(pdays, previous, poutcome)
- 타겟: `deposit` (정기예금 가입 여부, yes/no)
- 클래스 분포: No 52.6% / Yes 47.4% (비교적 균형 잡힌 분포)

## 핵심 이슈: 데이터 누수(Data Leakage)

EDA 과정에서 `duration`(통화 시간)이 타겟과 가장 높은 상관관계(0.45)를 보였습니다. 하지만 통화 시간은 캠페인 통화가 끝난 후에만 알 수 있는 정보로, 캠페인을 시작하기 전 타겟팅 의사결정에는 사용할 수 없는 변수

이에 따라 두 가지 버전의 모델을 모두 구축해 비교함

| 버전 | 설명 | 용도 |
|---|---|---|
| with_duration | duration 포함 | 성능 상한선 참고용 |
| no_duration | duration 제외 | 실제 캠페인 사전 타겟팅에 사용 가능한 모델 |

## 분석 프로세스

### 1. EDA
- 타겟 클래스 분포 확인
- 직군(job)별 캠페인 반응률 분석 → student, retired 그룹이 가장 높은 전환율
- 접촉 횟수(campaign)와 전환율의 관계 → 3회 초과 시 전환율 급격히 하락
- 이전 캠페인 결과(poutcome)별 전환율 → success 고객은 91.3%로 압도적으로 높음

### 2. Feature Engineering
- `campaign_bucket`: 접촉 횟수 구간화 (1회 / 2회 / 3회 / 4-5회 / 6회+)
- `prev_success`: 이전 캠페인 성공 여부 플래그
- `was_contacted_before`: 이전 접촉 여부 플래그 (`pdays`의 -1 값을 분리 처리)
- `balance_segment`: 잔액 기반 고객 세그먼트 (negative / low / mid / high)
- 범주형 변수 원-핫 인코딩 (17개 → 52개 컬럼)

### 3. 모델링
- Baseline: Logistic Regression (class_weight='balanced')
- Advanced: Random Forest, XGBoost (클래스 불균형 보정 적용)
- 5-Fold Stratified Cross Validation으로 검증

### 4. 평가 지표
ROC-AUC를 기본 지표로 사용하고, 마케팅 비용 절감 관점에서 Precision도 함께 확인했습니다.

## 모델 성능

| 모델 | ROC-AUC (with duration) | ROC-AUC (no duration) |
|---|---|---|
| Logistic Regression | 0.852 | 0.656 |
| Random Forest | **0.876** | **0.688** |
| XGBoost | 0.859 | 0.682 |

duration 포함 시 모든 모델의 AUC가 0.15~0.20 가량 상승하는데, 이는 모델 성능이 아니라 데이터 누수에 의한 결과입니다. 실전 배포 기준 모델은 **Random Forest (no_duration, AUC 0.688)** 입니다.

## 비즈니스 해석

- 이전 캠페인에서 성공(`poutcome=success`)했던 고객은 재반응 확률이 매우 높음 → 재마케팅 우선순위 1순위
- 접촉 횟수 3회를 넘어가면 전환율이 떨어짐 → 무리한 반복 접촉은 효율을 오히려 낮춤
- student, retired 직군은 상대적으로 캠페인에 우호적 → 세그먼트별 메시징 차별화 여지

## A/B 테스트 시뮬레이션

**가설**: "예측 확률 상위 30% 고객만 타겟팅하면 ROI가 증가할 것이다"

**설계**
- Control: 전체 고객 (2,233명)
- Treatment: Random Forest(no_duration) 예측 확률 상위 30% 고객 (669명)

**결과**

| 구분 | 대상 수 | 전환율 |
|---|---|---|
| Control (전체) | 2,233명 | 47.4% |
| Treatment (상위 30%) | 669명 | 70.3% |

- 접촉 비용 절감: **70.0%**
- 전환율 개선: **48.3%**

동일한 예산으로 더 적은 고객에게 접촉하면서도 전환율을 크게 끌어올릴 수 있음을 확인했습니다. 단, 본 결과는 보유 데이터셋 기반 시뮬레이션이며, 실제 적용 전 온라인 A/B 테스트를 통한 검증이 필요합니다.

## 폴더 구조
bank-marketing-project/
│
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/               # 전처리/피처 엔지니어링 결과
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── config/
│   └── config.yaml
│
├── outputs/
│   ├── models/                 
│   ├── figures/                 
│   └── reports/
│
├── README.md
└── requirements.txt

## 실행 방법

```bash
# 1. 데이터 전처리
python src/preprocessing.py

# 2. 피처 엔지니어링 (notebooks/02_feature_engineering.ipynb 실행)

# 3. 모델 학습 및 평가
python src/train.py
python src/evaluate.py
```

## 기술 스택

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib

## 한계 및 향후 개선 방향

- 캠페인 시점의 거시경제 지표(금리, 고용률 등) 미반영
- 시간에 따른 모델 성능 변화(Concept Drift) 미검증
- 실제 비용/수익 데이터를 반영한 정밀 ROI 계산 필요
- 온라인 A/B 테스트를 통한 실측 검증 필요
