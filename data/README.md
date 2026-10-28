# 공개 데이터

[Release](https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2/releases/tag/collection-2026-10-28)에서 두 원본을 제공한다.

| 실험 | 파일 |
|---|---|
| Dataset A | `toy2box_dataset_a.hdf5` |
| Dataset C | `toy2box_dataset_c.hdf5` |

```bash
python tools/download_data.py
```

[manifest.json](manifest.json)은 실험 이름, 공개 파일명, 수집 원본 파일명, 크기, SHA256과 다운로드 URL을 담는다. 도구는 다운로드 후 원본 해시를 확인한다. 이름을 바꾸어도 데이터 바이트는 변경하지 않았다.

Dataset B·annotation·Mimic은 [통계·연결 관계](../docs/data_collection.md)만 보관한다. 이 파일들은 기본 다운로드에 포함되지 않는다.
