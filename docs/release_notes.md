# Toy2Box datasets — Dataset A / Dataset B

두 데이터 세트를 고정된 실험 이름으로 제공한다. 파일명만 정리했으며 HDF5 내부 데이터와 SHA256은 수집 원본과 동일하다.

| 실험 | 다운로드 파일 | 성공 시범 | 프레임 | idle 평균 |
|---|---|---:|---:|---:|
| Dataset A | `toy2box_dataset_a.hdf5` | 100 | 29,229 | 0.70% |
| Dataset B | `toy2box_dataset_b.hdf5` | 20 | 8,406 | 19.79% |

수집 방식은 자동 피드백 제어다. Dataset B에는 속도 변화·작은 수정·분산 멈춤이 포함된다. [파일명 대응과 SHA256](https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2/blob/main/data/manifest.json)을 확인할 수 있다.

저장소에는 환경·학습 코드, Mimic499로 학습한 선택 모델 가중치 1개, 평가 명령어, 발표 초안과 재생 영상을 제공한다. 학습 참조·annotation·Mimic 원본 5개는 이 Release에 포함하지 않으며 통계와 연결 관계만 보관한다.

Dataset A·Dataset B의 대표 시범 재생 6개는 기존 성공 조건을 10 step 연속 충족했다. 이 결과는 학습 정책의 100회 평가 성공률이 아니다.
