# 선택 모델과 학습 기록

## 최종 모델

| 항목 | 값 |
|---|---|
| 파일 | `models/final_model.pth` |
| 데이터 계열 | annotation 20개 → Mimic 성공 생성 499개 |
| source checkpoint | `20261028105120/models/model_epoch_2000.pth` |
| 선택 기준 | 사용자 지정, 고정 epoch 2000 |
| 학습 seed | 101 |
| 정책 | BC-RNN, LSTM hidden 400·2 layers, GMM 5 modes |
| batch / sequence length | 100 / 10 |
| optimizer | Adam, 학습률 0.001 |
| epoch당 학습 step 설정 | 100 |
| 가중치 수 | 저장소에 선택 모델 1개만 포함 |

모델 내부 config가 `toy2box_agibot_mimic_n20_500.hdf5`를 학습 입력으로 사용했음을 확인했다. 실제 성공 episode는 499개다. [모델 manifest](../models/model_manifest.json)와 [학습 config](../configs/final_train_config.json)를 제공한다.

Dataset A 계열 config는 [참고 설정](../configs/dataset_a_reference_train_config.json)으로 보관하며 가중치는 추가하지 않았다. Dataset C은 선택 모델의 학습 원본이 아니다.

## 실제 입력

```text
eef_pos                  3
eef_quat                 4
gripper_pos              2
toy_truck_positions      3
toy_truck_orientations   4
box_positions            3
box_orientations         4
총 입력                 23차원
```

HDF5 전체 관측과 실제 policy 입력을 구분한다. 저장된 관절 상태는 이 config의 입력 목록에 없다.

## 평가와 분석

선택 모델로 로드·추론을 실행한 [부분 확인 기록](../submission/model_smoke_check.json)을 제공한다. 실행을 마친 일부 rollout만 포함하므로 100회 정식 성공률이나 학습 seed 2개의 평균·표준편차로 사용하지 않는다.

[평가 명령어](../submission/evaluate.txt)는 `--num_rollouts 100 --horizon 800 --seed 42`를 따른다. 이 검증 seed와 조교의 비공개 평가 seed를 구분한다.

추가 확인 항목은 학습 seed 111, 접근·파지·운반·배치 실패 분류, 진단한 원인 하나에 대한 통제 실험이다. 수집기의 성공률과 정책 평가 성공률을 혼동하지 않는다.
