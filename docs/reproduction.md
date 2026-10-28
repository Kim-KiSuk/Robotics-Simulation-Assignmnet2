# 설치·학습·평가·데이터 재현

## 환경

실행에는 Agibot Toy2Box task와 자산이 설치된 IsaacLab_RS, Isaac Sim, NVIDIA GPU, `lerobot-arena` Python 환경이 필요하다. 이 저장소는 task·로봇·controller·BC 학습·평가 코드를 `isaaclab_project/`에 포함한다. IsaacLab core와 시뮬레이터 자산은 별도 dependency다.

기반 저장소: `https://github.com/cailab-hy/IsaacLab_RS.git`.
기반 commit과 포함 파일 SHA256: `isaaclab_project/source_manifest.json`.

기존 설치에 포함 코드를 적용하려면 다음을 사용한다. 수정할 파일은 백업한다.

```bash
python tools/install_project_overlay.py --isaaclab_root /home/kisuk/IsaacLab_RS
python tools/install_project_overlay.py --isaaclab_root /home/kisuk/IsaacLab_RS --apply
```

## 선택 모델 평가

```bash
conda activate lerobot-arena
cd /home/kisuk/IsaacLab_RS
REPO=/home/kisuk/IsaacLab_RS/Robotics-Simulation-Assignmnet2

./isaaclab.sh -p "$REPO/isaaclab_project/scripts/imitation_learning/robomimic/play.py" \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint "$REPO/models/final_model.pth" \
  --num_rollouts 100 --horizon 800 --seed 42 --headless
```

LMS용 명령어는 `submission/evaluate.txt`에도 있다. REPO와 IsaacLab_RS 위치를 환경에 맞게 수정한다. 조교 평가에는 지정받은 평가 seed를 사용한다.

Conda C++ 라이브러리 충돌이 발생하면, 추론 확인에 사용한 환경에서는 다음과 같이 환경의 lib 경로를 우선 적용했다.

```bash
LD_LIBRARY_PATH=/home/kisuk/miniforge3/envs/lerobot-arena/lib \
  /home/kisuk/miniforge3/envs/lerobot-arena/bin/python \
  "$REPO/isaaclab_project/scripts/imitation_learning/robomimic/play.py" \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint "$REPO/models/final_model.pth" \
  --num_rollouts 100 --horizon 800 --seed 42 --headless
```

## 학습 코드

선택 모델은 성공 생성 시범 499개로 학습했다. 해당 Mimic HDF5는 이 Release의 공개 원본 두 개에 포함되지 않으므로 재학습에는 별도 로컬 파일이 필요하다. 추론 평가에는 HDF5가 필요하지 않다.

```bash
./isaaclab.sh -p "$REPO/isaaclab_project/scripts/imitation_learning/robomimic/train.py" \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --algo bc \
  --dataset /path/to/toy2box_agibot_mimic_n20_500.hdf5 \
  --epochs 2000 --name final_bc_rnn
```

이 학습 스크립트는 task에 등록된 BC config를 읽는다. `--config` 옵션을 받지 않는다. 과거 실행의 전체 설정은 `configs/final_train_config.json`으로 확인한다. seed나 설정을 바꾸는 실험은 해당 task config와 실제 저장된 학습 config를 함께 관리한다.

## Dataset A / Dataset C 다운로드와 분석

저장소 루트에서 실행한다.

```bash
python tools/download_data.py
python tools/analyze_datasets.py --dataset_dir ./data
```

공개 파일명은 `toy2box_dataset_a.hdf5`, `toy2box_dataset_c.hdf5`다. 원본 파일명·체크섬 대응은 `data/manifest.json`에 있다. 분석 도구는 두 공개 파일명과 수집 당시 원본 이름 모두 읽을 수 있다. 로컬 annotation·Mimic까지 7개 파일을 갖고 있으면 `--include_related`로 전체 통계와 연결 관계를 확인한다.

## 실제 시범 MP4 재생성

```bash
cd /home/kisuk/IsaacLab_RS
./isaaclab.sh -p "$REPO/tools/replay_dataset_video.py" \
  --dataset_files "$REPO/data/toy2box_dataset_a.hdf5" "$REPO/data/toy2box_dataset_c.hdf5" \
  --output_dir /tmp/toy2box_replay_named \
  --samples_per_dataset 3 --headless
```

초기 상태를 복원하고 action 전체를 실행한다. RGB가 없는 HDF5를 카메라 프레임 변환 전용 `hdf5_to_mp4.py`에 바로 넣는 방식이 아니다. 도구는 카메라를 활성화하고 렌더링을 직접 수행한다. 같은 파일이 있으면 덮어쓰지 않는다.

## 수집 코드와 직접 조작 모드

`tools/collection/record_toy2box_auto.py`는 자동 수집과 수동 가이드를 지원한다. Dataset C 설정은 `--motion_style varied --motion_variation_strength 1.0 --target_idle_fraction 0.21`이다. 사람이 직접 움직이는 모드는 `--full_sequence --sequence_order rotation_xy_z`로 실행하며 자동 동작 옵션을 함께 지정하지 않는다. 새 수집에는 새 파일명을 사용한다.

## 검증

```bash
python -m pytest -q tools/collection/test tools/test
python tools/audit_videos.py
```

수집기 오프라인 검증과 HDF5 재생 성공은 BC 정책의 정식 100회 평가 결과와 별개다.
