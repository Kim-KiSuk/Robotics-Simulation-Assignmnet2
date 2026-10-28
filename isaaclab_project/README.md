# Toy2Box IsaacLab 프로젝트 코드

## 포함한 코드

- `source/isaaclab_tasks/.../place/`: Agibot Toy2Box 환경 등록, 관측, 성공·종료 조건, task·BC config.
- `source/isaaclab_assets/.../robots/agibot.py`: 로봇 초기 자세와 actuator 설정.
- `source/isaaclab/.../controllers/config/rmp_flow.py`: RMPFlow controller 설정.
- `scripts/imitation_learning/robomimic/train.py`: 데이터 기반 BC 학습.
- `scripts/imitation_learning/robomimic/play.py`: checkpoint 로드·rollout 평가.

기반 프로젝트는 `https://github.com/cailab-hy/IsaacLab_RS.git`, 정리 시점의 Git commit은 `source_manifest.json`에 고정했다. 필요한 과업 코드를 실제 경로 구조로 보관한 overlay이며, Isaac Sim 엔진이나 IsaacLab 전체 dependency를 복제한 패키지는 아니다. 로봇·트럭·박스 USD와 RMPFlow 자산은 해당 환경의 설치 경로를 사용한다.

## 적용

Isaac Sim·IsaacLab_RS가 설치된 환경에서 저장소 루트 기준으로 실행한다.

```bash
python tools/install_project_overlay.py --isaaclab_root /path/to/IsaacLab_RS
python tools/install_project_overlay.py --isaaclab_root /path/to/IsaacLab_RS --apply
```

첫 명령은 포함 파일의 SHA256과 현재 설치 파일을 확인한다. `--apply`는 변경 파일의 원본을 `toy2box_overlay_backups/`에 보관하고 복사한다. 과업의 초기 물체 분포·성공 판정·종료 조건을 제출용으로 변경하지 않았다.

현재 코드는 추론 확인에 사용한 설치 파일의 사본이다. 파일별 해시는 [source_manifest.json](source_manifest.json)에 있다.
