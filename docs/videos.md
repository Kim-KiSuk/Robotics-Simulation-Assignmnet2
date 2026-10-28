# 재생 영상 목록

## 원본 데이터의 실제 action 재생

2026-10-28 NVIDIA RTX 5080에서 Isaac Sim을 사용해 새로 렌더링했다. 원본 HDF5의 초기 상태를 복원하고 저장된 7차원 action을 순서대로 실행했다. 카메라 해상도는 960×540, 영상은 H.264 MP4, 20 fps이다.

| 원본 | episode | 프레임 | 시간 | 영상 |
|---|---|---:|---:|---|
| 표준 자동 수집 100개 | demo_0 | 240 | 12.00초 | [재생](../videos/toy2box_agibot_auto_100/demo_0.mp4) |
| 표준 자동 수집 100개 | demo_49 | 249 | 12.45초 | [재생](../videos/toy2box_agibot_auto_100/demo_49.mp4) |
| 표준 자동 수집 100개 | demo_99 | 261 | 13.05초 | [재생](../videos/toy2box_agibot_auto_100/demo_99.mp4) |
| 최근 변화형 자동 수집 20개 | demo_0 | 439 | 21.95초 | [재생](../videos/toy2box_agibot_idle21_n20/demo_0.mp4) |
| 최근 변화형 자동 수집 20개 | demo_9 | 424 | 21.20초 | [재생](../videos/toy2box_agibot_idle21_n20/demo_9.mp4) |
| 최근 변화형 자동 수집 20개 | demo_19 | 384 | 19.20초 | [재생](../videos/toy2box_agibot_idle21_n20/demo_19.mp4) |

![각 영상의 처음·중간·마지막 프레임](../videos/previews/contact_sheet.jpg)

확인 사항:

- 디코딩한 프레임 수가 원본 action 수와 6개 모두 일치한다.
- 처음·중간·마지막 프레임이 검은 화면이 아니며, 시작과 끝의 장면 변화가 있다.
- 6개 모두 재생 중 기존 task 성공 조건을 10 step 연속 충족했다.
- 이 선택은 전체 데이터의 처음·중간·마지막 시범이다. 전체 120개를 재생 검증했다고 주장하지 않는다.
- 시범 action 재생이며 학습된 BC 정책의 독립 평가 rollout과 다르다.

[원본별 재생 판정 JSON](../videos/replay_results.json) · [영상 디코딩 검사 JSON](../reports/video_audit.json)

## 기존 BC 정책 참고 영상

기존 로컬 `videos/agibot_bc_mimic/`에서 다음 두 파일을 그대로 보관했다.

- [rollout-episode-0.mp4](../videos/policy_reference/rollout-episode-0.mp4)
- [rollout-episode-1.mp4](../videos/policy_reference/rollout-episode-1.mp4)

정확한 체크포인트, 평가 seed와 episode별 성공 라벨을 연결하는 평가 로그는 확인되지 않았다. 따라서 두 영상은 정책 동작 참고용이며, 최신 20개에서 학습한 정책의 결과나 정량 성공률의 근거로 사용하지 않는다.

## 재생 방법

GitHub 파일 화면에서 MP4를 열거나 다운로드해 재생한다. 재생 영상이 화면에 바로 표시되지 않으면 파일의 Download raw 기능을 사용한다. 시뮬레이터에서 다시 만들려면 [재현 명령](reproduction.md)을 따른다.
