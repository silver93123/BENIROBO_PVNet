"""Camera 추상화 계층.

사용 예:
    from src.camera import create_camera

    cfg = {"type": "lucid_helios", "serial": None, ...}
    with create_camera(cfg) as cam:
        frame = cam.capture()
        print(frame.points.shape, frame.intensity.shape)

다중 프레임 평균화(2026-07 추가):
    config에 "averaging" 섹션을 넣으면 카메라 종류와 무관하게 N프레임을
    합쳐서 반환하는 AveragingCamera로 자동으로 감싼다. 예:

    cfg = {
        "type": "lucid_helios", ...,
        "averaging": {"num_frames": 8, "method": "median", "min_valid_ratio": 0.6},
    }
    with create_camera(cfg) as cam:
        frame = cam.capture()   # 내부적으로 8프레임 찍어서 합친 결과가 나온다

    averaging 섹션이 없거나 num_frames가 1 이하면 감싸지 않고 원본 카메라를
    그대로 반환한다 (기존 코드/설정 파일은 아무 영향 없음).
"""

from .base import CameraBase, FrameData
from .averaging import AveragingCamera

__all__ = ["CameraBase", "FrameData", "AveragingCamera", "create_camera"]


def create_camera(config: dict) -> CameraBase:
    """설정 dict로부터 카메라 인스턴스 생성 (factory).

    Args:
        config: config.yaml의 'camera' 섹션 dict.

    Returns:
        CameraBase 하위 클래스 인스턴스. "averaging" 섹션이 유효하면
        AveragingCamera로 감싸진 인스턴스.

    Raises:
        ValueError: 지원하지 않는 카메라 타입.
    """
    cam = _create_base_camera(config)

    avg_cfg = config.get("averaging")
    if avg_cfg and int(avg_cfg.get("num_frames", 1)) > 1:
        cam = AveragingCamera(
            cam,
            num_frames=int(avg_cfg.get("num_frames", 5)),
            method=avg_cfg.get("method", "median"),
            min_valid_ratio=float(avg_cfg.get("min_valid_ratio", 0.6)),
        )
    return cam


def _create_base_camera(config: dict) -> CameraBase:
    cam_type = config.get("type", "").lower()

    if cam_type == "lucid_helios":
        from .lucid_helios import LucidHeliosCamera
        return LucidHeliosCamera(
            serial=config.get("serial"),
            pixel_format=config.get("pixel_format", "Coord3D_ABCY16"),
            exposure_time_selector=config.get("exposure_time_selector", "Exp1000Us"),
            operating_mode=config.get("operating_mode", "Distance3000mm"),
            connect_timeout_ms=config.get("connect_timeout_ms", 5000),
            capture_timeout_ms=config.get("capture_timeout_ms", 2000),
            valid_z_range_mm=tuple(config.get("valid_z_range_mm", (100.0, 1500.0))),
        )

    if cam_type == "femto_bolt":
        from .femto_bolt import FemtoBoltCamera
        return FemtoBoltCamera(
            serial=config.get("serial"),
            depth_width=config.get("depth_width", 640),
            depth_height=config.get("depth_height", 576),
            fps=config.get("fps", 15),
            capture_timeout_ms=config.get("capture_timeout_ms", 2000),
            valid_z_range_mm=tuple(config.get("valid_z_range_mm", (100.0, 1500.0))),
            warmup_frames=config.get("warmup_frames", 5),
            capture_rgb=config.get("capture_rgb", False),
        )

    raise ValueError(
        f"지원하지 않는 카메라 타입: '{cam_type}'. "
        f"지원: ['lucid_helios', 'femto_bolt']"
    )