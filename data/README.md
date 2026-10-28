# 데이터 다운로드

HDF5 원본은 [collection-2026-10-28 Release](https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2/releases/tag/collection-2026-10-28)의 assets에 있습니다.

```bash
# 주요 원본 두 개: 자동 100개 + 최신 자동 20개
python tools/download_data.py

```

주요 두 원본의 파일명, 역할, 바이트 크기, SHA256, 다운로드 URL은 [manifest.json](manifest.json)에 있습니다. 이전 20개·annotation·Mimic 파일 5개는 원본 공개 없이 통계만 보관했습니다. 다운로드한 HDF5는 `.gitignore`로 Git 이력에서 제외합니다.

카메라 RGB가 없는 상태·action 데이터이며, 기존 파일의 수집 방식이나 내용을 변경하지 않았습니다. 각 데이터의 관계는 [수집 문서](../docs/data_collection.md)를 참고하세요.
