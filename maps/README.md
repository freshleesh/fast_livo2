# Maps

원본 Ubuntu 워크스페이스 (`~/unicorn_ws/ICRA2026_HJ/stack_master/maps/`) 의 맵 폴더들을 여기에 복사.

각 맵은 다음 구조:
```
maps/
  <map_name>/
    global_waypoints.json   # global_republisher 필수
    map.pgm + map.yaml      # (옵션) navigation map
    ...
```

노드들은 다음 우선순위로 맵 경로를 찾음:
1. ROS 파라미터 `map_path` (절대 경로)
2. 환경변수 `IFAC_MAPS_DIR`
3. 이 폴더 (`<src>/IFAC2026_SH/maps/<map>/`)
4. legacy fallback `~/unicorn_ws/ICRA2026_HJ/stack_master/maps/<map>/` (Ubuntu 원본 머신)
