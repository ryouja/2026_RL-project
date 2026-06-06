# Reinforcement Learning for Molecular Property Search

DQN을 활용한 QM9 데이터셋 내 지능적 분자 탐색 전략 학습


## 1. 프로젝트 개요

QM9 데이터셋(20,000개 서브셋)에서 목표 Dipole Moment 범위(2.0~3.0 Debye)를 갖는 분자를 효율적으로 탐색하는 강화학습 에이전트를 개발하였다.

새로운 분자를 생성하는 대신, 기존 분자 데이터셋 내에서 목표 물성 영역에 도달하는 탐색 정책(Policy)을 학습하는 것을 목표로 한다.

---

## 2. 데이터셋

### QM9
- 약 134,000개의 유기 분자
- DFT 계산 기반 물성 제공
- SMILES 구조 포함

#### 사용 물성:
- Dipole Moment (μ)

#### 목표 범위:
```text
2.0 ≤ Dipole Moment ≤ 3.0 Debye
```
실험에서는 계산 효율을 위해 20,000개 분자 서브셋을 사용.

---

## 3. 방법

### (1) Surrogate Model

SMILES를 Morgan Fingerprint(2048-bit)로 변환한 뒤 Random Forest를 이용해 Dipole Moment를 예측.
```text
SMILES -> Morgan Fingerprint -> Random Forest -> Dipole Moment Prediction
```
성능:
- R²: 0.7185
- RMSE: 0.8142 Debye
- MAE: 0.5313 Debye

---

### (2) 강화학습 환경

#### State
현재 분자의 Morgan Fingerprint (2048-bit)
#### Action
현재 분자와 유사한 이웃 분자 10개 중 하나 선택
#### Reward
* 목표 범위(2~3 D) 안에 있으면 양의 보상
* 범위 밖이면 패널티
* 2.5 Debye에 가까울수록 높은 보상
#### Episode
* 랜덤 분자에서 시작
* 최대 50 step

---

### (3) DQN

#### 네트워크 구조
```text
2048 -> FC(256) + ReLU -> FC(128) + ReLU -> 10 Q-values
```
#### 주요 학습 설정
- Episodes      = 500    
- Batch Size    = 64     
- Learning Rate = 0.0005 
- Gamma         = 0.99   
- Replay Buffer = 10000  

---

## 4. 실험 결과

| Method               | Target Hits / 50 step |
| -------------------- | --------------------- |
| Random Search        | 12.87 ± 3.74          |
| DQN Agent            | 26.58 ± 7.96          |
| Greedy (Upper Bound) | 49.42 ± 0.85          |

DQN은 Random Search 대비 약 2배 높은 탐색 성능을 보였다.

---

## 5. 실행 방법

### 환경 생성

```bash
conda create -n rl_chem python=3.10 -y
conda activate rl_chem

pip install rdkit scikit-learn pandas numpy deepchem joblib gymnasium torch matplotlib
```

### 실행 순서

```bash
# Surrogate Model 학습
python train_surrogate.py

# DQN 학습
python train_dqn.py

# 평가
python evaluate.py
```

---

## 6. 파일 구조

```
├── train_surrogate.py   # Surrogate Model 학습
├── chem_env.py          # Gymnasium 환경
├── train_dqn.py         # DQN 에이전트 학습
├── evaluate.py          # 평가 및 비교
├── models/              # 학습된 모델 (Google Drive에서 다운로드)
│   ├── dqn_molecule_model.pth
│   └── surrogate_rf.pkl
├── data/                # 전처리된 데이터 (train_surrogate.py 실행 시 자동 생성)
│   ├── fps.npy
│   └── targets.npy
└── results/             # 실험 결과 그래프 (evaluate.py 실행 시 자동 생성)
    ├── training_metrics.png
    ├── comparison_steps.png
    └── comparison_episodes.png
```

| File | Link |
|------|------|
| `dqn_molecule_model.pth` | [Download](https://drive.google.com/file/d/16R9wMRxLr_PjS-jo_eGOirhBRijmKtYj/view?usp=sharing) |
| `surrogate_rf.pkl` | [Download](https://drive.google.com/file/d/1XBH2rr0X_gxsV5Ek_QcMFZ6PlXfBfuYC/view?usp=sharing) |

---

## 7. 한계점

* QM9 내부 분자만 탐색 가능
* 새로운 분자 생성 불가
* Surrogate Model 오차 존재

---

## 8. 향후 개선 방향

* Graph Neural Network 기반 물성 예측
* PPO, Double DQN 등 RL 알고리즘 비교
* 분자 생성 모델과 결합
* 생성형 분자 역설계 문제로 확장

---

## Author
류정아 (A70052)  
서강대학교 AI/SW 대학원  
Reinforcement Learning Project (2026)
