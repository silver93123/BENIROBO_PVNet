"""
mmdet config 파일(.py)을 텍스트 레벨에서 다룬다.

- load_from 한 줄만 override한 복사본 생성 (원본 config는 건드리지 않음)
- config 안의 work_dir 값을 읽어서, 그 폴더의 best_*.pth 중 최신 파일 탐색

주의: mmengine의 Config 파서를 쓰지 않고 정규식으로 텍스트를 다룬다.
rtmdet-ins_bracket.py처럼 `load_from = '...'`, `work_dir = '...'` 가
각각 한 줄로 되어 있는 표준적인 형태를 가정한다. config 구조가 이와 많이
다르면 (예: load_from이 조건문 안에 있는 경우) 정확히 동작하지 않을 수 있다.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.paths import resolve_relative_to_project

LOAD_FROM_RE = re.compile(r"^load_from\s*=.*$", re.MULTILINE)
WORK_DIR_RE = re.compile(r"^work_dir\s*=\s*['\"](.*?)['\"]\s*$", re.MULTILINE)


def make_config_with_override(config_path: str, checkpoint_override: str) -> str:
    """load_from만 override한 config 복사본을 만들고 그 경로를 반환한다."""
    src = Path(config_path)
    text = src.read_text(encoding="utf-8")
    new_line = f"load_from = '{checkpoint_override}'"

    if LOAD_FROM_RE.search(text):
        text = LOAD_FROM_RE.sub(new_line, text, count=1)
    else:
        text = text.rstrip() + f"\n\n# UI에서 override\n{new_line}\n"

    out_path = src.with_name(f"{src.stem}_used.py")
    out_path.write_text(text, encoding="utf-8")
    return str(out_path)


def find_work_dir(config_path: str) -> str | None:
    text = Path(config_path).read_text(encoding="utf-8")
    match = WORK_DIR_RE.search(text)
    return match.group(1) if match else None


def find_latest_best_checkpoint(config_path: str) -> str | None:
    work_dir = find_work_dir(config_path)
    if not work_dir:
        return None
    folder = resolve_relative_to_project(work_dir)
    if not folder.is_dir():
        return None
    candidates = sorted(folder.glob("best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None
