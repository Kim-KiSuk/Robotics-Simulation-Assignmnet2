# Toy2Box collection archive — 2026-10-28

HDF5 원본 7개를 원래 파일명과 바이트 그대로 보관합니다. 파일 크기와 SHA256은 저장소의 `data/manifest.json`에서 확인할 수 있습니다.

## 주요 수집 데이터

- `toy2box_agibot_auto_100.hdf5`: 표준 자동 수집 성공 시범 100개, 29,229프레임.
- `toy2box_agibot_idle21_n20.hdf5`: 속도 변화·멈춤·작은 보정을 추가한 최신 자동 수집 성공 시범 20개, 8,406프레임, idle 평균 19.79%.

## 이전 버전과 증강 이력

- `toy2box_agibot_auto_n20.hdf5`: 최신 idle 조절 이전의 자동 시범 20개.
- `toy2box_agibot_annotated_100.hdf5`: 실제 annotation 99개.
- `toy2box_agibot_n20_annotated.hdf5`: 이전 20개의 annotation. 최신 `idle21_n20`의 annotation이 아닙니다.
- `toy2box_agibot_mimic_n100_500.hdf5`: 성공 생성 시범 500개.
- `toy2box_agibot_mimic_n20_500.hdf5`: 성공 생성 시범 499개. 이전 20개 계열입니다.

자동 수집 원본을 사람의 직접 키보드 수집으로 재분류하지 않습니다. 실패 생성 `_failed.hdf5`와 모델 가중치는 포함하지 않았습니다.

## 저장소에 포함된 자료

- 연구 축별 새 주제 3개씩, 총 12개: Markdown·일반 텍스트.
- 원본별 수집 통계, episode CSV, 수집 실행 로그, 버전 연결 검사.
- 원본 두 데이터에서 각 3개씩 새로 렌더링한 실제 시범 재생 MP4 6개.
- 기존 정책 동작 참고 MP4 2개. 정확한 평가 실행 연결은 미확인입니다.
- 다운로드·분석·영상 재생 도구와 수집기 코드 스냅샷.

대표 시범 재생 6개는 모두 기존 성공 조건을 10 step 연속 충족했습니다. 이는 전체 데이터 검증이나 학습된 BC 정책의 100회 평가 결과가 아닙니다.
