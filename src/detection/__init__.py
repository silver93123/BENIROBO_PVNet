"""2D 검출 백엔드 추상화 계층.

src/camera(CameraBase/create_camera), app/core/registration
(PoseEstimator/create_registrator)과 동일한 패턴:

    from src.detection import create_detector

    cfg = {
        "type": "rtmdet_ins_rothead",
        "params": {"config": "...", "checkpoint": "...", "device": "cuda:0"},
    }
    detector = create_detector(cfg)
    results = detector.infer(image, depth_mm=depth, camera_intrinsics=K)

새 백엔드를 추가할 때:
    1. DetectorBase를 상속받는 클래스를 이 폴더에 새 파일로 작성
       (rtmdet_inferencer_rothead.py 참고).
    2. 아래 create_detector()에 타입 문자열 분기 한 줄 추가.
    3. AVAILABLE_DETECTOR_TYPES에 문자열 추가 (icp_test_tab.py의
       콤보박스가 이 목록을 그대로 채운다).
그 외에는 아무 것도 손댈 필요 없다 - 호출부는 DetectorBase 인터페이스만
보고 동작한다.
"""
from __future__ import annotations

from .base import DetectionResult, DetectorBase
from .rtmdet_inferencer import RTMDetInferencer  # 하위 호환 (기존 import 유지)

AVAILABLE_DETECTOR_TYPES = ["rtmdet_ins", "rtmdet_ins_rothead", "rtmdet_ins_foundationpose"]

__all__ = [
    "DetectorBase", "DetectionResult", "RTMDetInferencer",
    "create_detector", "AVAILABLE_DETECTOR_TYPES",
]


def create_detector(config: dict) -> DetectorBase:
    """설정 dict로부터 검출 백엔드 인스턴스 생성 (factory).

    Args:
        config: {"type": <백엔드 이름>, "params": {<생성자 kwargs>}} 형태.

    Returns:
        DetectorBase 하위 클래스 인스턴스.

    Raises:
        ValueError: 지원하지 않는 백엔드 타입.
    """
    det_type = config.get("type", "rtmdet_ins").lower()
    params = config.get("params", {}) or {}

    if det_type == "rtmdet_ins":
        from .rtmdet_inferencer import RTMDetInferencer
        return RTMDetInferencer(**params)

    if det_type == "rtmdet_ins_rothead":
        from .rtmdet_inferencer_rothead import RTMDetInferencerRotHead
        return RTMDetInferencerRotHead(**params)

    if det_type == "rtmdet_ins_foundationpose":
        from .rtmdet_inferencer_foundationpose import RTMDetInferencerFoundationPose
        return RTMDetInferencerFoundationPose(**params)

    raise ValueError(
        f"지원하지 않는 검출 백엔드: '{det_type}'. "
        f"지원: {AVAILABLE_DETECTOR_TYPES}"
    )