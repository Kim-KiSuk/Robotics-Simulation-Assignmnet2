# 제출 파일 확인

| 항목 | 위치 |
|---|---|
| 환경·로봇·controller·학습·평가 코드 | `isaaclab_project/` |
| 최종 모델 1개 | `models/final_model.pth` |
| 모델 해시·학습 입력 정보 | `models/model_manifest.json` |
| 평가 명령어 | `submission/evaluate.txt` |
| 발표 자료 | `submission/toy2box_presentation_draft.pptx` |
| 설치·재현 방법 | `docs/reproduction.md` |

## 실행 확인

- 모델 경로와 task 이름을 확인한다.
- 평가 설정은 rollout 100회, horizon 800, 검증 seed 42다.
- 모델 가중치의 SHA256은 manifest와 일치해야 한다.
- Isaac Sim과 로봇·물체 자산은 실행 환경에 설치되어 있어야 한다.

## 결과 정리

선택한 모델의 로드·추론 실행을 확인했다. 부분 실행 확인은 정식 100회 평가 결과와 구분한다. 발표 자료에는 정식 평가, 실패 단계 분석, 통제 실험 결과와 조원별 역할을 반영해야 한다.
