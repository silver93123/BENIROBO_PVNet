"""프로젝트 루트 경로 계산.

이 앱은 프로젝트 폴더 하나 안에 전부 들어있다고 가정한다:

  <프로젝트 루트>/
    main.py
    app/                 <- 이 파일은 app/core/paths.py
    scripts/2_Train_rtmdet_model.py
    configs/rtmdet-ins_bracket.py
    data/dataset/<세션 폴더>/
    models/*.pth
    work_dirs/*/

프로젝트 폴더를 통째로 다른 위치로 옮겨도, 이 파일 기준 상대 위치로
루트를 계산하기 때문에 항상 올바른 경로를 가리킨다.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rtmdet-ins_bracket.py"
DEFAULT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ICP 정합 테스트(탭 4) 관련
DEFAULT_CAD_DIR = PROJECT_ROOT / "data" / "cad"

# 데이터 수집(탭 0) 관련 — 전부 이 프로젝트 안에서 해결한다 (외부 프로젝트 의존 없음).
DEFAULT_COLLECT_SCRIPT = DEFAULT_SCRIPTS_DIR / "collect_dataset.py"
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "configs" / "camera_config.yaml"


def resolve_relative_to_project(path_str: str) -> Path:
    """path_str이 상대경로면 PROJECT_ROOT 기준으로, 절대경로면 그대로 반환."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)