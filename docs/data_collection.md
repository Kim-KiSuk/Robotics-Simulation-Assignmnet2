# Toy2Box 데이터 수집 기록

## 실험 이름

| 이름 | 공개 파일명 / 로컬 원본 | 의미 |
|---|---|---|
| Dataset A | `toy2box_dataset_a.hdf5` | 표준 자동 피드백 수집 100개 |
| Dataset B | `toy2box_agibot_auto_n20.hdf5` | 속도 변화·짧은 멈춤 자동 수집 20개 |
| Dataset C | `toy2box_dataset_c.hdf5` | idle 목표 21% 자동 수집 20개 |

공개 파일명은 실험 이름에 맞게 정리했다. HDF5 내부 배열과 원본 바이트는 변경하지 않았다. 원본 파일명과 SHA256은 `data/manifest.json`에 기록했다. 수집 당시 실행 JSON의 경로는 역사적 원본 기록으로 보존한다.

## 과업과 수집

Agibot 오른팔로 트럭을 집어 박스 안에 놓는다. 상대 이동 3축, 회전 3축, 그리퍼 개폐의 7차원 action을 20 Hz로 기록한다. 카메라 RGB는 원본에 없고 시뮬레이터 재생으로 별도 영상을 생성했다.

수집기는 성공 참조 시범의 파지 변환과 현재 관측을 사용한다. `접근 → 하강 → 닫기 → 수직 상승 → 수평 운반 → 하강 → 열기` 순서이며, 성공·실패 후 기록 버퍼와 장면을 리셋한다. 성공 조건을 충족한 시범만 저장한다.

## 수집 통계

| 항목 | Dataset A | Dataset B | Dataset C |
|---|---:|---:|---:|
| 성공 저장 / 시도 | 100 / 278 | 20 / 120 | 20 / 120 |
| 수집 시도 성공률 | 35.97% | 16.67% | 16.67% |
| 저장 프레임 | 29,229 | 7,070 | 8,406 |
| 평균 길이 | 14.6145초 | 17.675초 | 21.015초 |
| episode별 idle 평균 | 0.6969% | 4.7779% | 19.7918% |
| idle 최소~최대 | 0.4988~0.8511% | 1.4245~8.8068% | 18.4758~20.8925% |
| 그리퍼 전환 | 모두 2회 | 모두 2회 | 모두 2회 |
| 수집 seed | 42 | 101 | 101 |
| 실제 방식 | 자동 | 자동 | 자동 |

`idle = mean(all(actions[:, :6] == 0, axis=1))`이다. 매우 작은 비0 명령은 idle이 아니다. 그리퍼 전환 횟수만으로 파지·배치 성공을 판단하지 않는다. 이 표는 수집 통계이며 BC 성능의 통제 실험 결과가 아니다.

## Dataset C 동작 구성

- 기존 명령 크기의 약 45~100% 범위에서 부드럽게 속도 변화.
- 접근·운반 중 최대 8 mm 중간 위치 보정. 목표와의 마지막 4 cm에서는 추가 보정 제거.
- 접근 중 최대 0.025 rad 회전 보정.
- 누적 idle을 계산해 이동 중 약 0.25~0.65초 멈춤을 분산.
- 멈춤 중 그리퍼 명령 유지. 의도적인 멈춤을 단계 정체 시간에서 제외.
- 기존 과업 성공 판정과 성공·실패 후 리셋 로직 유지.

## 수집기 중단 이유

| 기록된 이유 | Dataset A | Dataset C |
|---|---:|---:|
| approach 정체 | 119 | 91 |
| descend 정체 | 37 | 6 |
| lift 정체 | 4 | 2 |
| transfer 정체 | 7 | 0 |
| 트럭을 함께 들어 올리지 못함 | 2 | 1 |
| 릴리즈 전 입구 이탈 | 5 | 0 |
| wait 시간 초과 | 4 | 0 |
| 성공 | 100 | 20 |

이는 자동 수집기의 중단 이유다. 학습 정책의 실패 원인 분류로 재사용하지 않는다.

## Annotation·Mimic 연결

| 계열 | 원본 수 | annotation 실제 수 | Mimic 성공 데이터 수 | 연결 |
|---|---:|---:|---:|---|
| Dataset A | 100 | 99 | 500 | 참고 학습 |
| Dataset B | 20 | 20 | 499 | 최종 모델 학습 |
| Dataset C | 20 | 미포함 | 미포함 | 수집 데이터 비교 |

파일명 `annotated_100`에는 99개가 있다. action·손끝 위치의 내용 해시 비교에서 원본 `demo_39`는 포함되지 않았다. 제외 원인은 확인되지 않았다.

Dataset B annotation은 Dataset B의 20개와 모두 일치하며 Dataset C과는 일치하지 않는다. 파일명 `mimic_n20_500`의 실제 성공 episode는 499개다. 생성 시도 수나 생성 성공률을 파일명만으로 계산하지 않는다.

공개 HDF5는 Dataset A과 Dataset C 두 개다. Dataset B·annotation·Mimic의 HDF5 원본은 공개하지 않고 통계·연결 관계만 보관한다. 실패 생성 파일은 학습용에 포함하지 않았다.

## 근거 파일

- [전체 요약](../reports/dataset_summary.json) · [내용 해시 연결](../reports/dataset_lineage.json)
- [Dataset A CSV](../reports/toy2box_dataset_a.episodes.csv) · [Dataset C CSV](../reports/toy2box_dataset_c.episodes.csv)
- [Dataset A 실행 기록](../reports/collection/a.run.json)
- [Dataset B 실행 기록](../reports/collection/b.run.json)
- [Dataset C 실행 기록](../reports/collection/c.run.json)
- [공개 원본 manifest](../data/manifest.json) · [관련 로컬 파일 manifest](../reports/related_local_datasets.json)
