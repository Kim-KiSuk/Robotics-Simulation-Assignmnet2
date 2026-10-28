# 재생 영상

## Dataset A / Dataset C 시범 재생

원본의 초기 상태를 복원하고 저장 action을 실제 Isaac Sim에서 실행했다. NVIDIA RTX 5080, 960×540, H.264 MP4, 20 fps로 렌더링했다.

| 실험 | episode | 프레임 | 시간 | MP4 |
|---|---|---:|---:|---|
| Dataset A | demo_0 | 240 | 12.00초 | [재생](../videos/dataset_a/demo_0.mp4) |
| Dataset A | demo_49 | 249 | 12.45초 | [재생](../videos/dataset_a/demo_49.mp4) |
| Dataset A | demo_99 | 261 | 13.05초 | [재생](../videos/dataset_a/demo_99.mp4) |
| Dataset C | demo_0 | 439 | 21.95초 | [재생](../videos/dataset_c/demo_0.mp4) |
| Dataset C | demo_9 | 424 | 21.20초 | [재생](../videos/dataset_c/demo_9.mp4) |
| Dataset C | demo_19 | 384 | 19.20초 | [재생](../videos/dataset_c/demo_19.mp4) |

![처음·중간·마지막 프레임](../videos/previews/contact_sheet.jpg)

각 데이터의 처음·중간·마지막 시범을 선택했다. 6개 모두 디코딩 프레임 수가 원본 action 수와 일치하며, 검은 화면이 아니고 시작·종료 사이 장면 변화가 있다. 재생 중 기존 성공 조건을 10 step 연속 충족했다. 전체 데이터를 재생 검증한 결과는 아니다.

이 영상은 저장 시범의 action 재생이며 최종 학습 정책의 독립 평가 영상과 구분한다.

[재생 판정 기록](../videos/replay_results.json) · [영상 검사](../reports/video_audit.json)

## 정책 동작 참고

- [rollout-episode-0.mp4](../videos/policy_reference/rollout-episode-0.mp4)
- [rollout-episode-1.mp4](../videos/policy_reference/rollout-episode-1.mp4)

두 파일은 로컬 정책 영상에서 보관한 참고 자료다. 정확한 체크포인트·평가 seed를 연결하는 로그가 확인되지 않아 선택한 최종 모델의 결과라고 표기하지 않는다. 정량 성공률의 근거로도 사용하지 않는다.

GitHub에서 영상 파일을 열거나 Download raw로 내려받아 재생한다. 다시 렌더링하는 명령은 [재현 방법](reproduction.md)에 있다.
