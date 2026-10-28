# BC 학습 설정과 확인 범위

정리 시점: 2026-10-28. 아래 내용은 실제 로컬 config와 체크포인트 파일 목록을 기준으로 작성했다. config에 설정한 목표 epoch와 실제 완료 epoch를 구분한다.

## 보관한 설정

| 항목 | 100개 계열 | 이전 20개 계열 |
|---|---|---|
| config | [bc_auto100_mimic500.json](../configs/bc_auto100_mimic500.json) | [bc_prior20_mimic499.json](../configs/bc_prior20_mimic499.json) |
| 학습 입력 | `toy2box_agibot_mimic_n100_500.hdf5` | `toy2box_agibot_mimic_n20_500.hdf5` |
| 실제 생성 시범 수 | 500 | 499 |
| 학습 seed | 101 | 101 |
| 목표 epoch | 2,000 | 2,000 |
| batch size | 100 | 100 |
| sequence length | 10 | 10 |
| RNN | LSTM, hidden 400, 2 layers | LSTM, hidden 400, 2 layers |
| 행동 분포 | GMM, 5 modes | GMM, 5 modes |
| optimizer | Adam, 학습률 0.001 | Adam, 학습률 0.001 |
| epoch당 학습 step 설정 | 100 | 100 |
| 파일 목록 확인 시 최대 checkpoint epoch | 2,000 | 1,800 |

학습이 별도 프로세스에서 진행 중일 수 있으므로 파일 목록은 스냅샷이다. [정리 시점의 목록](../reports/training_snapshot.json)을 참고한다. 모델 가중치는 이번 데이터·영상 아카이브에 포함하지 않았다.

## 입력 관측

두 config는 다음 7개 low-dimensional 관측을 사용한다.

```text
eef_pos                  3
eef_quat                 4
gripper_pos              2
toy_truck_positions      3
toy_truck_orientations   4
box_positions            3
box_orientations        4
총 입력 차원            23
```

HDF5에 저장된 관측 전체와 학습 config에서 실제 선택한 입력은 다르다. 저장 데이터에 관절 상태가 있다고 해서 해당 학습이 모든 관절 상태를 입력받았다고 설명하지 않는다.

## 현재 확정할 수 없는 결과

- 학습 seed 101·111 두 조건을 모두 완료한 공정 비교 결과는 확인하지 않았다.
- 평가 seed, checkpoint 선택 규칙과 연결된 최종 100회 BC rollout 성공률은 이 아카이브에서 확정하지 않는다.
- 기존 정책 참고 영상 두 개만으로 성공률을 계산하지 않는다.
- 100개·최근 20개 수집기의 시도 성공률을 BC 성공률로 바꾸어 쓰지 않는다.
- 이전 20개에서 만든 Mimic 학습을 최신 idle 목표 21% 데이터의 학습이라고 설명하지 않는다.

## 다음 실험을 진행할 때

1. [연구주제](research_topics.md) 하나와 baseline 실패 진단을 연결한다.
2. 학습·검증 데이터를 원본 episode 계열 기준으로 분리한다.
3. 학습 seed 101·111과 동일한 학습·checkpoint 선택 기준을 적용한다.
4. 검증 seed 42로 100회, horizon 800 평가를 진행한다.
5. 성공률 평균·표준편차와 접근·파지·운반·배치 실패 분포를 보고한다.
