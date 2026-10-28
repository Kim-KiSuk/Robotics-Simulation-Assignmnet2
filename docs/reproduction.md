# 다운로드와 재현 명령

## 1. 저장소와 원본 데이터 받기

```bash
git clone https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2.git
cd Robotics-Simulation-Assignmnet2
python tools/download_data.py
# HDF5 7개 전체 받기
python tools/download_data.py --all
```

기본 다운로드는 100개 원본과 최신 20개 원본이다. 전체 다운로드는 이전 20개·annotation·Mimic 성공 데이터도 포함한다. 다운로드 후 SHA256을 자동 검사한다.

## 2. 통계 재생성

```bash
conda activate lerobot-arena
python tools/analyze_datasets.py --dataset_dir ./data
```

분석 도구는 원본을 읽기 전용으로 열며 episode별 CSV, 요약 JSON, 연결 관계와 체크섬 manifest를 생성한다. 7개 파일 전체가 필요하다. `reports/*.run.json`은 수집 당시의 별도 기록이며 HDF5에서 복원되는 정보가 아니다. 도구는 데이터 폴더의 실행 로그를 우선 사용하고, 없으면 저장소에 보관한 실행 로그를 사용한다.

## 3. 대표 시범을 실제 시뮬레이터에서 MP4로 재생

이 저장소는 Isaac Lab 전체나 로봇·물체 자산을 포함하지 않는다. 다음 명령은 Agibot task가 설치된 `IsaacLab_RS`와 `lerobot-arena` 환경을 전제로 한다. `REPO`를 저장소 절대 경로로 수정한다.

```bash
conda activate lerobot-arena
cd /home/kisuk/IsaacLab_RS
REPO=/home/kisuk/IsaacLab_RS/Robotics-Simulation-Assignmnet2

./isaaclab.sh -p "$REPO/tools/replay_dataset_video.py" \
  --dataset_files \
    "$REPO/data/toy2box_agibot_auto_100.hdf5" \
    "$REPO/data/toy2box_agibot_idle21_n20.hdf5" \
  --output_dir /tmp/toy2box_replay_new \
  --samples_per_dataset 3 \
  --headless
```

도구는 각 파일에서 처음·중간·마지막 episode를 선택하고, 초기 상태 복원 후 action을 실행한다. 카메라를 활성화하고 렌더링을 직접 수행한다. 출력 폴더에 같은 영상이 있으면 덮어쓰지 않는다. 전체 episode를 재생하려면 `--samples_per_dataset 100`으로 지정한다. 20개 파일은 실제 episode 수까지만 선택한다.

이 아카이브의 6개 영상은 다음 방식으로 생성했다. Conda C++ 라이브러리 충돌을 피하려고 환경의 `lib` 경로를 우선 적용했다.

```bash
LD_LIBRARY_PATH=/home/kisuk/miniforge3/envs/lerobot-arena/lib \
  /home/kisuk/miniforge3/envs/lerobot-arena/bin/python \
  "$REPO/tools/replay_dataset_video.py" \
  --dataset_files \
    /home/kisuk/IsaacLab_RS/datasets/toy2box_agibot_auto_100.hdf5 \
    /home/kisuk/IsaacLab_RS/datasets/toy2box_agibot_idle21_n20.hdf5 \
  --output_dir /tmp/toy2box_replay_new \
  --samples_per_dataset 3 --headless
```

원본 HDF5에는 RGB 프레임이 없으므로 카메라 프레임 변환 전용인 `hdf5_to_mp4.py`로 바로 영상을 만들 수 없다. 위 명령은 초기 상태와 action을 시뮬레이터에서 실제로 재생해 영상을 만든다.

## 4. 최신 설정으로 새 자동 데이터 수집

원본 경로를 자신의 환경에 맞게 변경한다. 참조 데이터는 이 저장소에 포함하지 않았다. 출력은 새 파일명을 사용한다.

```bash
conda activate lerobot-arena
cd /home/kisuk/IsaacLab_RS
REPO=/home/kisuk/IsaacLab_RS/Robotics-Simulation-Assignmnet2

./isaaclab.sh -p "$REPO/tools/collection/record_toy2box_auto.py" \
  --reference_dataset /home/kisuk/toy/toy2box_agibot.hdf5 \
  --dataset_file ./datasets/toy2box_new_varied_20.hdf5 \
  --num_demos 20 \
  --motion_style varied \
  --motion_variation_strength 1.0 \
  --target_idle_fraction 0.21 \
  --seed 101
```

이 명령은 새로운 자동 수집이며 과거 데이터를 바이트 단위로 재생성한다고 보장하지 않는다. 현재 수집기 스냅샷과 과거 `.run.json`을 구분한다.

## 5. 사람이 가이드를 보며 직접 조작하는 모드

```bash
./isaaclab.sh -p "$REPO/tools/collection/record_toy2box_auto.py" \
  --reference_dataset /home/kisuk/toy/toy2box_agibot.hdf5 \
  --dataset_file ./datasets/toy2box_manual_new_20.hdf5 \
  --num_demos 20 --full_sequence --sequence_order rotation_xy_z
```

`--full_sequence`는 사람의 키 입력으로 움직이며 위·왼쪽 화면과 전체 순서를 표시한다. 자동 수집용 `--motion_style varied`를 함께 지정하지 않는다. 표시된 키와 횟수를 보고 사람이 조작한다. 위·왼쪽 카메라는 표시용이며 RGB 저장은 추가하지 않는다.

## 6. BC 학습·평가

실제 학습 config는 `configs/`에 있다. `train.data`와 출력 경로를 환경에 맞게 변경한다. 100개 원본, 최신 20개 원본, 이전 20개에서 만든 Mimic 데이터를 구분한다.

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --algo bc --config /path/to/config.json

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint /path/to/model.pth \
  --num_rollouts 100 --horizon 800 --seed 42 --headless
```

확인한 로컬 기록만으로는 최종 두 학습 seed의 BC 성공률을 확정할 수 없다. 수집 성공률이나 대표 재생의 성공을 BC 평가 결과로 대신하지 않는다.
