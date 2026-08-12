"""카메라 타입별 노출/촬영 옵션 스키마 + per-camera yaml 읽기/쓰기.

설정 탭(settings_tab.py)이 카메라 타입 콤보박스 선택에 따라 다른 조정 위젯을
동적으로 그리기 위해 이 모듈의 CAMERA_EXPOSURE_SCHEMA를 그대로 사용한다.

이 값들은 app_settings.json(settings_manager)이 아니라 카메라별 개별 yaml
(configs/camera_config_*.yaml)에 저장된다 - src.camera.create_camera()가
그 yaml을 그대로 읽어 드라이버 생성자에 넘기는 기존 구조와 일치시키기 위함.
comments가 많은 yaml이라 일반 PyYAML(safe_dump)로 다시 쓰면 주석이 다
날아가므로, 반드시 ruamel.yaml round-trip 모드로 읽고 써서 주석/포맷을
보존한다.

필요 패키지:
    pip install ruamel.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML

from app.core.paths import CAMERA_CONFIG_PATHS

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


# 필드 하나의 정의:
#   key: yaml/드라이버 kwarg 키
#   label: UI 라벨
#   widget: 'combo' | 'spin_int' | 'spin_double' | 'checkbox' | 'line_edit'
#   choices: widget='combo'일 때 선택지
#   range: widget='spin_*'일 때 (min, max)
#   step / decimals: widget='spin_double'일 때
#   nullable: True면 스핀박스 최소값(보통 0)을 "값 지정 안 함(=null, 카메라
#             현재 설정 유지)"으로 취급
#   tooltip: 도움말
CAMERA_EXPOSURE_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "lucid_helios": [
        {
            "key": "exposure_time_selector",
            "label": "노출 시간",
            "widget": "combo",
            "choices": ["Exp62_5Us", "Exp250Us", "Exp1000Us"],
            "tooltip": "어두우면 길게(Exp1000Us), 모션블러가 우려되면 짧게(Exp62_5Us).",
        },
        {
            "key": "operating_mode",
            "label": "동작 거리 모드",
            "widget": "combo",
            "choices": ["Distance1500mm", "Distance3000mm", "Distance4000mm", "Distance5000mm", "Distance6000mm"],
            "tooltip": "작업 영역에 맞춰 가장 짧은 모드를 선택해야 정밀도가 올라갑니다.",
        },
    ],
    "femto_bolt": [
        {
            "key": "fps",
            "label": "FPS",
            "widget": "spin_int",
            "range": (5, 30),
            "tooltip": "Femto Bolt는 자동 노출만 지원합니다 (직접 노출 시간 조정 불가). "
                       "FPS를 낮추면 프레임당 노출 여유가 늘어나는 간접 효과가 있습니다.",
        },
    ],
    "o3r": [
        {
            "key": "framerate",
            "label": "프레임레이트",
            "widget": "spin_double",
            "range": (0.0, 20.0),
            "step": 0.5,
            "decimals": 1,
            "nullable": True,
            "null_sentinel": 0.0,
            "tooltip": "0 = 값 지정 안 함(카메라 현재 설정 유지). 보통 10.0~20.0, 0.5 단위.",
        },
        {
            "key": "exposure_long_us",
            "label": "노출 시간 Long (µs)",
            "widget": "spin_int",
            "range": (0, 100000),
            "nullable": True,
            "null_sentinel": 0,
            "tooltip": "0 = 값 지정 안 함(카메라 현재 설정 유지). 어두우면 늘리고, "
                       "반사가 강해 포화되면 줄이세요.",
        },
        {
            "key": "exposure_short_us",
            "label": "노출 시간 Short (µs)",
            "widget": "spin_int",
            "range": (0, 100000),
            "nullable": True,
            "null_sentinel": 0,
            "tooltip": "0 = 값 지정 안 함(카메라 현재 설정 유지). HDR 합성용 짧은 노출.",
        },
        {
            "key": "offset",
            "label": "측정 범위 오프셋 (m)",
            "widget": "spin_double",
            "range": (-3.0, 4.0),
            "step": 0.1,
            "decimals": 2,
            "nullable": True,
            "null_sentinel": -3.0,
            "tooltip": "측정 거리 범위를 이동시킵니다. 음수 = 카메라 쪽으로 당김 "
                       "(근거리 작업에 유리, 단 반대쪽 끝은 좁아짐). "
                       "슬라이더 최솟값(-3.00) = 값 지정 안 함(카메라 현재 설정 유지) - "
                       "0을 명시적으로 쓰고 싶으면 -3.00 대신 정확히 0.00으로 맞추세요.",
        },
        {
            "key": "port_3d",
            "label": "3D 포트",
            "widget": "line_edit",
            "tooltip": "3D(ToF) 카메라가 물린 포트명 (예: port0). 실제 배선에 맞게 지정.",
        },
        {
            "key": "port_2d",
            "label": "2D(RGB) 포트 (선택)",
            "widget": "line_edit",
            "tooltip": "RGB 정합 캡처를 쓸 때만 필요. 3D와 같은 헤드의 포트여야 합니다.",
        },
        {
            "key": "capture_rgb",
            "label": "RGB 정합 캡처 사용 (실험적)",
            "widget": "checkbox",
            "tooltip": "depth 격자에 정렬된 RGB도 함께 캡처합니다. 실패해도 3D 캡처에는 영향 없음.",
        },
    ],
}


def load_camera_yaml(camera_type: str):
    """해당 카메라 타입의 yaml 전체를 ruamel CommentedMap으로 로드."""
    path = CAMERA_CONFIG_PATHS.get(camera_type)
    if path is None or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.load(f)


def get_current_exposure_values(camera_type: str) -> dict[str, Any]:
    """스키마에 정의된 필드들의 현재 yaml 값을 dict로 반환.
    yaml 자체가 없거나 필드가 없으면 빈 값(None/기본)으로 채운다."""
    schema = CAMERA_EXPOSURE_SCHEMA.get(camera_type, [])
    doc = load_camera_yaml(camera_type)
    cam_section = (doc or {}).get("camera", {}) if doc else {}

    values = {}
    for field in schema:
        key = field["key"]
        values[key] = cam_section.get(key)
    return values


def save_exposure_values(camera_type: str, values: dict[str, Any]) -> None:
    """UI에서 편집한 값들을 해당 카메라 yaml에 반영 (주석/포맷 보존)."""
    path = CAMERA_CONFIG_PATHS.get(camera_type)
    if path is None:
        raise ValueError(f"알 수 없는 카메라 타입: {camera_type}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"카메라 설정 파일이 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        doc = _yaml.load(f)

    if "camera" not in doc:
        raise ValueError(f"{path}에 'camera' 섹션이 없습니다.")

    for key, value in values.items():
        doc["camera"][key] = value

    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(doc, f)