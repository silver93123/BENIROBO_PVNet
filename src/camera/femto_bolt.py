"""Orbbec Femto Bolt ToF 카메라 구현.

대상 하드웨어:
    Orbbec Femto Bolt (Azure Kinect DK 후속, iToF 방식, USB-C 연결)

SDK:
    pyorbbecsdk v2 (Orbbec SDK v2.x Python 바인딩)
    설치: pip install pyorbbecsdk2   ← 패키지명 주의 (v1은 pyorbbecsdk)
    (Windows x64 / Linux x64 / ARM64 프리빌트 휠 제공)

설계 노트 (Helios 드라이버와의 대응 관계):
    - Helios의 'intensity'는 ToF 센서 자체의 반사강도 이미지.
      Femto Bolt에서 이에 해당하는 것은 IR 스트림이다.
      IR과 Depth는 동일 ToF 센서에서 나오므로 픽셀 단위로 정렬되어 있음
      → 별도 align filter 없이 2D 마스크 ↔ organized PCD 인덱싱이 성립.
      (RGB를 detection 입력으로 쓰려면 AlignFilter로 D2C 정렬이 필요해
       파이프라인이 복잡해지므로, 기존 구조 유지에는 IR 사용을 권장)
    - Helios의 Coord3D_ABCY16은 XYZ를 직접 출력하지만, Femto Bolt는
      depth map을 출력하므로 SDK의 PointCloudFilter로 XYZ 변환한다.
      PointCloudFilter는 렌즈 왜곡(Brown-Conrady)을 내부에서 보정하므로
      단순 핀홀 역투영보다 정확하다. 출력은 H*W 행우선(organized) 순서,
      무효 픽셀은 (0,0,0).

검증된 출처:
    - https://github.com/orbbec/pyorbbecsdk (v2-main 브랜치, 35+ 예제)
    - https://orbbec.github.io/pyorbbecsdk/  (공식 문서)

주의:
    pyorbbecsdk2 설치 후 아래 API(특히 PointCloudFilter의
    set_position_data_scaled 등 메서드명)를 examples/ 폴더의
    point cloud 관련 예제와 반드시 대조하세요. SDK 버전별로
    세부 시그니처가 다를 수 있습니다.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .base import CameraBase, FrameData

logger = logging.getLogger(__name__)

try:
    from pyorbbecsdk import (  # type: ignore
        AlignFilter,
        Config,
        OBError,
        OBFormat,
        OBSensorType,
        OBStreamType,
        Pipeline,
        PointCloudFilter,
    )
    _ORBBEC_AVAILABLE = True
    _ORBBEC_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    _ORBBEC_AVAILABLE = False
    _ORBBEC_IMPORT_ERROR = _e


class FemtoBoltCamera(CameraBase):
    """Orbbec Femto Bolt wrapper.

    출력 정규화 (FrameData 규격, Helios 드라이버와 동일):
        - intensity: (H, W) uint8. IR 스트림을 percentile stretch로 8-bit화.
        - points_organized: (H, W, 3) float32, mm 단위. 무효 픽셀 = NaN.
        - points: (N, 3) float32, NaN 제거된 유효 포인트.
        - valid_mask: (H, W) bool.

    Args:
        serial: 카메라 시리얼 번호. None이면 첫 번째 발견 카메라.
        depth_width / depth_height / fps:
            depth·IR 스트림 프로파일. Femto Bolt 기준
            NFOV unbinned = 640x576 (동작거리 약 0.5~3.86 m)
            WFOV binned   = 512x512 (약 0.25~2.5 m, 근거리 유리)
            ※ IR은 depth와 반드시 같은 해상도로 열어야 픽셀 정렬이 유지됨.
        capture_timeout_ms: 한 프레임셋 대기 타임아웃 (밀리초).
        valid_z_range_mm: 이 범위 밖 Z는 무효 처리.
        warmup_frames: 스트림 시작 직후 버릴 프레임 수 (초기 프레임 불안정 대비).
        capture_rgb: True면 depth 격자에 정렬된 RGB도 추가로 캡처한다 (실험적,
            기본값 False). IR/depth 기반 검증된 파이프라인과는 완전히 분리되어
            있어서, 이 옵션이 실패해도(스트림 미지원, AlignFilter 실패 등)
            capture()는 color_rgb=None으로 정상 반환하며 크래시하지 않는다.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        depth_width: int = 640,
        depth_height: int = 576,
        fps: int = 15,
        capture_timeout_ms: int = 2000,
        valid_z_range_mm: tuple = (100.0, 1500.0),
        warmup_frames: int = 5,
        capture_rgb: bool = False,
    ) -> None:
        if not _ORBBEC_AVAILABLE:
            raise ImportError(
                "pyorbbecsdk를 import할 수 없습니다. "
                "'pip install pyorbbecsdk2' 실행 후 다시 시도하세요. "
                "(Linux는 udev rules 설치 필요: "
                "https://github.com/orbbec/pyorbbecsdk 참고) "
                f"원본 에러: {_ORBBEC_IMPORT_ERROR}"
            )

        self.serial = serial
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.fps = int(fps)
        self.capture_timeout_ms = int(capture_timeout_ms)
        self._valid_z_min = float(valid_z_range_mm[0])
        self._valid_z_max = float(valid_z_range_mm[1])
        self.warmup_frames = int(warmup_frames)
        self.capture_rgb = bool(capture_rgb)

        self._pipeline: Optional["Pipeline"] = None
        self._pc_filter: Optional["PointCloudFilter"] = None
        self._align_filter: Optional["AlignFilter"] = None

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        """파이프라인 구성 → depth + IR 스트림 시작 → PointCloudFilter 준비."""
        if self.serial:
            from pyorbbecsdk import Context  # type: ignore
            ctx = Context()
            dev_list = ctx.query_devices()
            device = None
            found = []
            for i in range(dev_list.get_count()):
                d = dev_list.get_device_by_index(i)
                sn = d.get_device_info().get_serial_number()
                found.append(sn)
                if sn == self.serial:
                    device = d
                    break
            if device is None:
                raise RuntimeError(
                    f"시리얼 '{self.serial}'에 해당하는 카메라가 없습니다. "
                    f"발견된 시리얼: {found}"
                )
            self._pipeline = Pipeline(device)
        else:
            self._pipeline = Pipeline()

        config = Config()

        depth_profiles = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        depth_profile = depth_profiles.get_video_stream_profile(
            self.depth_width, self.depth_height, OBFormat.Y16, self.fps
        )
        config.enable_stream(depth_profile)

        ir_profiles = self._pipeline.get_stream_profile_list(
            OBSensorType.IR_SENSOR
        )
        ir_profile = ir_profiles.get_video_stream_profile(
            self.depth_width, self.depth_height, OBFormat.Y16, self.fps
        )
        config.enable_stream(ir_profile)

        # ---- RGB (옵션, 실험적) ----
        # IR/depth 기반 검증된 파이프라인과 완전히 분리: 여기서 뭐가 실패해도
        # capture_rgb를 False로 되돌리고 경고만 남긴 뒤 계속 진행한다.
        if self.capture_rgb:
            try:
                color_profiles = self._pipeline.get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )
                color_profile = None
                for cp in color_profiles:
                    if cp.get_format() == OBFormat.RGB:
                        color_profile = cp
                        break
                if color_profile is None:
                    color_profile = color_profiles.get_default_video_stream_profile()
                    logger.warning(
                        "RGB 포맷 컬러 프로파일을 찾지 못해 기본 프로파일(포맷=%s)을 "
                        "씁니다. AlignFilter가 실패하면 색상 포맷을 확인하세요.",
                        color_profile.get_format(),
                    )
                config.enable_stream(color_profile)
                # color를 depth 격자에 맞춰 정렬 (organized PCD/IR과 같은 격자 유지)
                self._align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
            except Exception as e:
                logger.warning(
                    "RGB 스트림 설정 실패 - RGB 없이 IR/depth만으로 진행합니다: %s", e
                )
                self.capture_rgb = False
                self._align_filter = None

        self._pipeline.start(config)

        self._pc_filter = PointCloudFilter()
        self._pc_filter.set_create_point_format(OBFormat.POINT)

        try:
            info = self._pipeline.get_device().get_device_info()
            logger.info(
                "Femto Bolt opened (S/N=%s, FW=%s) | depth=%dx%d@%dfps",
                info.get_serial_number(), info.get_firmware_version(),
                self.depth_width, self.depth_height, self.fps,
            )
        except Exception:
            pass

        for _ in range(self.warmup_frames):
            try:
                self._pipeline.wait_for_frames(self.capture_timeout_ms)
            except OBError:
                pass
        time.sleep(0.1)

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        except Exception as e:
            logger.warning("pipeline.stop 실패: %s", e)
        self._pipeline = None
        self._pc_filter = None
        self._align_filter = None

    # --------------------------------------------------------------- capture
    def capture(self) -> FrameData:
        if self._pipeline is None:
            raise RuntimeError(
                "카메라가 열려있지 않습니다. open()을 먼저 호출하거나 "
                "with 구문을 사용하세요."
            )

        deadline = time.monotonic() + self.capture_timeout_ms / 1000.0
        frames = depth_frame = ir_frame = None
        while time.monotonic() < deadline:
            frames = self._pipeline.wait_for_frames(self.capture_timeout_ms)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_ir_frame()
            if depth_frame is not None and ir_frame is not None:
                break
        if depth_frame is None or ir_frame is None:
            raise RuntimeError(
                "유효한 depth/IR 프레임셋을 획득하지 못했습니다. "
                "USB 연결(전용 USB3 포트 권장) 및 스트림 프로파일을 확인하세요."
            )

        h = depth_frame.get_height()
        w = depth_frame.get_width()
        depth_scale = float(depth_frame.get_depth_scale())

        ir_u16 = np.frombuffer(
            ir_frame.get_data(), dtype=np.uint16
        ).reshape(ir_frame.get_height(), ir_frame.get_width()).copy()
        if ir_u16.shape != (h, w):
            raise RuntimeError(
                f"IR({ir_u16.shape})과 depth({h},{w}) 해상도 불일치. "
                "두 스트림을 같은 프로파일로 열어야 합니다."
            )
        intensity = self._normalize_intensity(ir_u16)

        self._pc_filter.set_position_data_scaled(depth_scale)
        pc_frame = self._pc_filter.process(depth_frame)
        if pc_frame is None:
            raise RuntimeError("PointCloudFilter 처리 실패 (None 반환).")

        xyz = np.frombuffer(pc_frame.get_data(), dtype=np.float32).copy()
        expected = h * w * 3
        if xyz.size != expected:
            raise RuntimeError(
                f"포인트클라우드 크기 불일치: {xyz.size} floats "
                f"(기대={expected}). SDK 버전별 출력 포맷을 확인하세요."
            )
        points_organized = xyz.reshape(h, w, 3)

        z_mm = points_organized[..., 2]
        valid_mask = (z_mm >= self._valid_z_min) & (z_mm <= self._valid_z_max)
        points_organized = points_organized.astype(np.float32, copy=True)
        points_organized[~valid_mask] = np.nan
        points = points_organized[valid_mask]

        color_rgb = None
        if self.capture_rgb and self._align_filter is not None:
            color_rgb = self._try_extract_aligned_rgb(frames, h, w)

        return FrameData(
            intensity=intensity,
            points=points,
            points_organized=points_organized,
            valid_mask=valid_mask,
            confidence=None,
            color_rgb=color_rgb,
        )

    def _try_extract_aligned_rgb(self, frames, h: int, w: int) -> Optional[np.ndarray]:
        """depth 격자에 정렬된 RGB를 뽑아본다. 실패해도 None만 반환하고 절대
        예외를 위로 던지지 않는다 (IR/depth 기반 캡처 성공에 영향 주지 않기 위함)."""
        try:
            aligned = self._align_filter.process(frames)
            if aligned is None:
                logger.warning("AlignFilter.process()가 None을 반환했습니다.")
                return None
            aligned_fs = aligned.as_frame_set()
            color_frame = aligned_fs.get_color_frame()
            if color_frame is None:
                logger.warning("정렬된 프레임셋에서 color를 가져오지 못했습니다.")
                return None

            c_h, c_w = color_frame.get_height(), color_frame.get_width()
            raw_rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            raw_rgb = raw_rgb[: c_h * c_w * 3].reshape(c_h, c_w, 3).copy()
            if (c_h, c_w) != (h, w):
                raw_rgb = cv2.resize(raw_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            return raw_rgb
        except Exception as e:
            logger.warning("RGB 정렬/추출 실패 (IR/depth 결과에는 영향 없음): %s", e)
            return None

    @staticmethod
    def _normalize_intensity(intensity_u16: np.ndarray) -> np.ndarray:
        if intensity_u16.size == 0:
            return np.zeros_like(intensity_u16, dtype=np.uint8)

        lo, hi = np.percentile(intensity_u16, [1.0, 99.0])
        if hi <= lo:
            return np.zeros_like(intensity_u16, dtype=np.uint8)

        scaled = (intensity_u16.astype(np.float32) - lo) / (hi - lo)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        return (scaled * 255.0).astype(np.uint8)