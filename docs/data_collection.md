# Toy2Box 데이터 수집 기록

## 공개 데이터

| 이름 | 파일 | 시범 수 | 구성 |
|---|---|---:|---|
| Dataset A | `toy2box_dataset_a.hdf5` | 100 | 기준 동작 |
| Dataset B | `toy2box_dataset_b.hdf5` | 20 | 속도 변화·분산 멈춤 |

공개 파일명과 수집 당시 파일명, 바이트 크기, SHA256은 [manifest](../data/manifest.json)에 기록했다. 이름만 바꾸었으며 HDF5 내부 데이터는 변경하지 않았다.

## 과업과 수집

Agibot 오른팔로 트럭을 집어 박스 안에 놓는다. 상대 이동 3축, 회전 3축, 그리퍼 개폐의 7차원 action을 20 Hz로 기록한다. 두 공개 데이터는 피드백 제어에 의한 자동 수집이며 원본 실행 기록을 보존한다.

트럭 바닥이 박스 상단보다 충분히 높아진 다음 수평 운반하고, 박스 입구 정렬을 확인한 뒤 그리퍼를 연다. 성공한 시범만 HDF5에 저장하며 성공·실패 후 장면과 기록 버퍼를 리셋한다. 원본에 RGB는 없으며 영상은 실제 시뮬레이터 재생으로 생성했다.

## 통계

| 항목 | Dataset A | Dataset B |
|---|---:|---:|
| 성공 저장 / 시도 | 100 / 278 | 20 / 120 |
| 수집 시도 성공률 | 35.97% | 16.67% |
| 저장 프레임 | 29,229 | 8,406 |
| 평균 길이 | 14.6145초 | 21.015초 |
| episode별 idle 평균 | 0.6969% | 19.7918% |
| idle 최소~최대 | 0.4988~0.8511% | 18.4758~20.8925% |
| 그리퍼 전환 | 모두 2회 | 모두 2회 |
| 수집 seed | 42 | 101 |

`idle = mean(all(actions[:, :6] == 0, axis=1))`이며 그리퍼 값은 제외한다. 작은 비0 명령은 idle에 포함되지 않는다. 수집 성공률과 BC 정책 평가 성공률은 다르다.

## Dataset B 동작 구성

- 기존 명령 크기의 약 45~100% 범위에서 속도 변화.
- 접근·운반 중 최대 8 mm 위치 보정, 목표까지 마지막 4 cm에서는 추가 보정 제거.
- 접근 중 최대 0.025 rad 회전 보정.
- 누적 idle을 계산해 약 0.25~0.65초 멈춤을 분산.
- 멈춤 중 그리퍼 명령 유지, 기존 성공 판정·리셋 로직 유지.

## 수집 중단 이유

| 이유 | Dataset A | Dataset B |
|---|---:|---:|
| approach 정체 | 119 | 91 |
| descend 정체 | 37 | 6 |
| lift 정체 | 4 | 2 |
| transfer 정체 | 7 | 0 |
| 트럭을 함께 들어 올리지 못함 | 2 | 1 |
| 릴리즈 전 입구 이탈 | 5 | 0 |
| wait 시간 초과 | 4 | 0 |
| 성공 | 100 | 20 |

이 표는 수집기 중단 사유이며 학습 정책의 실패 단계 분류로 대신 사용하지 않는다.

## 모델 학습 입력과 증강 기록

선택 모델의 실제 학습 입력은 `toy2box_agibot_mimic_n20_500.hdf5`이며 성공 episode는 499개다. 학습 입력은 공개 원본 두 파일과 구분해 관리한다. 실제 파일명·원본 관계는 [모델 manifest](../models/model_manifest.json)와 [내용 해시 연결 기록](../reports/dataset_lineage.json)에 보관한다.

참고 annotation 파일 `annotated_100`의 실제 개수는 99개이며 원본 `demo_39`가 포함되지 않는다. 생성 파일명의 숫자만으로 실제 episode 수나 생성 성공률을 판단하지 않는다. 실패 생성 파일은 학습용에 포함하지 않았다.

## 근거

- [전체 요약](../reports/dataset_summary.json) · [내용 해시 연결](../reports/dataset_lineage.json)
- [Dataset A CSV](../reports/toy2box_dataset_a.episodes.csv) · [Dataset B CSV](../reports/toy2box_dataset_b.episodes.csv)
- [Dataset A 실행 기록](../reports/collection/a.run.json) · [Dataset B 실행 기록](../reports/collection/b.run.json)
- [공개 원본 manifest](../data/manifest.json) · [관련 로컬 파일 manifest](../reports/related_local_datasets.json)
