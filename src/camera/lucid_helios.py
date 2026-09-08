"""LUCID Helios ToF 카메라 구현.

대상 하드웨어:
    LUCID Helios (HLS003S-001), Helios2 (HLT003S-001), Helios2+ (HTP003S-001) 등
    Coord3D_ABCY16 픽셀 포맷을 지원하는 LUCID ToF 카메라 전반.

SDK:
    Arena SDK Python wrapper (arena_api), Python 3.6.8+

Pixel Format - Coord3D_ABCY16:
    픽셀당 4채널, 각 16-bit:
        A = X, B = Y, C = Z, Y = Intensity
    버퍼 레이아웃: [A1, B1, C1, Y1, A2, B2, C2, Y2, ...]

mm 변환:
    LUCID GenICam 노드 사용:
        Scan3dCoordinateScale     - 공통 스케일 (float)
        Scan3dCoordinateSelector  - 'CoordinateA' / 'B' / 'C' 선택
        Scan3dCoordinateOffset    - 선택된 축의 오프셋 (mm)
    공식: physical_mm = raw_uint16 * scale + offset

참고 자료 (검증된 출처):
    - https://thinklucid.com/product/helios-time-of-flight-imx556/
    - https://thinklucid.com/arena-software-development-kit/
    - https://support.thinklucid.com/app-note-helios-3d-point-cloud-with-rgb-color/
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
from .base import CameraBase, FrameData

logger = logging.getLogger(__name__)

# arena_api는 시스템 라이브러리(ArenaC)를 로드하므로,
# 카메라 SDK가 없는 환경에서도 모듈 import 자체는 가능하도록 처리.

try:
    from arena_api.system import system  # type: ignore
    _ARENA_AVAILABLE = True
    _ARENA_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    _ARENA_AVAILABLE = False
    _ARENA_IMPORT_ERROR = _e


class LucidHeliosCamera(CameraBase):
    """LUCID Helios ToF 카메라 wrapper.

    출력 정규화:
        - intensity: (H, W) uint8 mono image. percentile-based contrast stretch 적용.
        - points_organized: (H, W, 3) float32, mm 단위. 무효 픽셀 = NaN.
        - points: (N, 3) float32, NaN 제거된 유효 포인트.
        - valid_mask: (H, W) bool.

    Args:
        serial: 카메라 시리얼 번호. None이면 첫 번째 발견 카메라.
        pixel_format: 픽셀 포맷 (현재 Coord3D_ABCY16만 검증됨).
        exposure_time_selector: 'Exp62_5Us' | 'Exp250Us' | 'Exp1000Us' 중 택1.
        operating_mode: 'Distance1500mm' | 'Distance3000mm' | ... 중 택1.
        connect_timeout_ms: 카메라 발견 타임아웃 (밀리초).
        capture_timeout_ms: 한 프레임 획득 타임아웃 (밀리초).
    """

    SUPPORTED_PIXEL_FORMAT = "Coord3D_ABCY16"

    def __init__(
        self,
        serial: Optional[str] = None,
        pixel_format: str = SUPPORTED_PIXEL_FORMAT,
        exposure_time_selector: str = "Exp250Us",
        operating_mode: str = "Distance1500mm",
        connect_timeout_ms: int = 5000,
        capture_timeout_ms: int = 2000,
        valid_z_range_mm: tuple = (100.0, 1500.0),

    ) -> None:
        if not _ARENA_AVAILABLE:
            raise ImportError(
                "arena_api를 import할 수 없습니다. "
                "LUCID Arena SDK 설치 후 동봉된 wheel을 pip install 하세요. "
                f"원본 에러: {_ARENA_IMPORT_ERROR}"
            )

        if pixel_format != self.SUPPORTED_PIXEL_FORMAT:
            logger.warning(
                "이 클래스는 %s 전용으로 작성되었습니다. "
                "다른 포맷 사용 시 _decode_buffer를 수정해야 합니다.",
                self.SUPPORTED_PIXEL_FORMAT,
            )

        self.serial = serial
        self.pixel_format = pixel_format
        self.exposure_time_selector = exposure_time_selector
        self.operating_mode = operating_mode
        self.connect_timeout_ms = connect_timeout_ms
        self.capture_timeout_ms = capture_timeout_ms

        self._device = None  # arena_api Device

        # Coord3D 변환 파라미터 (open()에서 채워짐)
        self._scale_mm: float = 1.0
        self._offset_x_mm: float = 0.0
        self._offset_y_mm: float = 0.0
        self._offset_z_mm: float = 0.0
        self._valid_z_min = float(valid_z_range_mm[0])
        self._valid_z_max = float(valid_z_range_mm[1])

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        """카메라 발견 → 노드 설정 → 스트리밍 시작."""
        # 1. 디바이스 발견 (1차 즉시, 2차 타임아웃 대기)
        devices = system.create_device()
        if not devices:
            devices = system.create_device(timeout=self.connect_timeout_ms)
        if not devices:
            raise RuntimeError(
                "LUCID 카메라를 발견할 수 없습니다. 다음을 확인하세요:\n"
                "  1) PoE 전원 및 GigE 케이블 연결\n"
                "  2) NIC IP가 카메라와 같은 서브넷인지 (LUCID IpConfig 도구로 확인)\n"
                "  3) 방화벽 / Jumbo Frames(9000) 설정"
            )

        # 2. 시리얼 매칭 (지정된 경우)
        if self.serial:
            target = next(
                (d for d in devices
                 if d.nodemap['DeviceSerialNumber'].value == self.serial),
                None,
            )
            if target is None:
                # 매칭 안 된 디바이스들도 정리
                system.destroy_device()
                raise RuntimeError(
                    f"시리얼 '{self.serial}'에 해당하는 카메라가 없습니다. "
                    f"발견된 시리얼: "
                    f"{[d.nodemap['DeviceSerialNumber'].value for d in devices]}"
                )
            self._device = target
        else:
            self._device = devices[0]

        nodemap = self._device.nodemap
        stream_nodemap = self._device.tl_stream_nodemap

        # 3. 스트림 안정성 옵션
        stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        stream_nodemap['StreamPacketResendEnable'].value = True

        # 4. 픽셀 포맷
        nodemap['PixelFormat'].value = self.pixel_format

        # 5. 운용 거리 모드 (실패해도 진행 - 펌웨어/모델별 차이 있음)
        self._safe_set_node(nodemap, 'Scan3dOperatingMode', self.operating_mode)

        # 6. 노출
        self._safe_set_node(
            nodemap, 'ExposureTimeSelector', self.exposure_time_selector
        )

        # 7. Coord3D 좌표 변환 파라미터 읽기
        #    physical_mm = raw_uint16 * scale + offset
        self._scale_mm = float(nodemap['Scan3dCoordinateScale'].value)
        for axis_name, attr_name in [
            ('CoordinateA', '_offset_x_mm'),
            ('CoordinateB', '_offset_y_mm'),
            ('CoordinateC', '_offset_z_mm'),
        ]:
            nodemap['Scan3dCoordinateSelector'].value = axis_name
            setattr(self, attr_name, float(nodemap['Scan3dCoordinateOffset'].value))

        sn = nodemap['DeviceSerialNumber'].value
        logger.info(
            "LUCID Helios opened (S/N=%s) | scale=%.6f, "
            "offsets=(x=%.3f, y=%.3f, z=%.3f) mm",
            sn, self._scale_mm,
            self._offset_x_mm, self._offset_y_mm, self._offset_z_mm,
        )

        # 8. 스트리밍 시작
        self._device.start_stream()

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        """스트리밍 중지 및 디바이스 해제."""
        if self._device is None:
            return
        try:
            self._device.stop_stream()
        except Exception as e:  # pragma: no cover - 안전망
            logger.warning("stop_stream 실패: %s", e)
        try:
            system.destroy_device()  # 모든 디바이스 정리
        except Exception as e:  # pragma: no cover
            logger.warning("destroy_device 실패: %s", e)
        self._device = None

    # --------------------------------------------------------------- capture
    def capture(self) -> FrameData:
        """한 프레임 획득 후 FrameData로 디코딩."""
        if self._device is None:
            raise RuntimeError(
                "카메라가 열려있지 않습니다. open()을 먼저 호출하거나 "
                "with 구문을 사용하세요."
            )

        buffer = self._device.get_buffer(timeout=self.capture_timeout_ms)
        try:
            return self._decode_buffer(buffer)
        finally:
            # 버퍼는 반드시 반환 (메모리 누수 방지)
            self._device.requeue_buffer(buffer)

    # ----------------------------------------------------- internal helpers
    def _decode_buffer(self, buffer) -> FrameData:
        """Coord3D_ABCY16 버퍼 → FrameData."""
        height = buffer.height
        width = buffer.width

        # 버퍼 데이터 → uint16 배열 (4채널 interleaved)
        # buffer.data는 ctypes 배열일 수 있으므로 numpy로 변환 후 copy()
        # (requeue_buffer 후 원본 메모리가 회수되므로 반드시 copy)
        raw = np.frombuffer(bytes(buffer.data), dtype=np.uint16)

        expected = height * width * 4
        if raw.size != expected:
            raise RuntimeError(
                f"버퍼 크기 불일치: {raw.size} uint16 (기대={expected}). "
                f"PixelFormat이 {self.pixel_format}인지 확인하세요."
            )
        raw = raw.reshape(height, width, 4)

        x_raw = raw[..., 0]
        y_raw = raw[..., 1]
        z_raw = raw[..., 2]
        intensity_u16 = raw[..., 3]

        # mm 변환
        x_mm = x_raw.astype(np.float32) * self._scale_mm + self._offset_x_mm
        y_mm = y_raw.astype(np.float32) * self._scale_mm + self._offset_y_mm
        z_mm = z_raw.astype(np.float32) * self._scale_mm + self._offset_z_mm

        points_organized = np.stack([x_mm, y_mm, z_mm], axis=-1)  # (H, W, 3)

        # 유효 픽셀: Z=0인 픽셀은 측정 실패 (LUCID 규약)
        z_mm = points_organized[..., 2]
        valid_mask = (z_mm >= self._valid_z_min) & (z_mm <= self._valid_z_max)

        # 무효 픽셀은 NaN으로 표시 (Open3D는 NaN을 자동으로 무시함)
        points_organized[~valid_mask] = np.nan

        # 평탄화된 유효 포인트
        points = points_organized[valid_mask]

        # Intensity → uint8
        intensity = self._normalize_intensity(intensity_u16)

        return FrameData(
            intensity=intensity,
            points=points,
            points_organized=points_organized,
            valid_mask=valid_mask,
            confidence=None,  # ABCY16 포맷에는 confidence가 별도로 없음
        )

    @staticmethod
    def _normalize_intensity(intensity_u16: np.ndarray) -> np.ndarray:
        """16-bit intensity → 8-bit (percentile contrast stretch).

        ToF intensity는 표면 반사율과 거리에 따라 동적 범위가 크므로,
        고정 비트 시프트보다 percentile clip이 모델 입력으로 더 안정적입니다.
        """
        if intensity_u16.size == 0:
            return np.zeros_like(intensity_u16, dtype=np.uint8)

        lo, hi = np.percentile(intensity_u16, [1.0, 99.0])
        if hi <= lo:
            return np.zeros_like(intensity_u16, dtype=np.uint8)

        scaled = (intensity_u16.astype(np.float32) - lo) / (hi - lo)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        return (scaled * 255.0).astype(np.uint8)

    @staticmethod
    def _safe_set_node(nodemap, name: str, value) -> None:
        """노드 설정 실패 시 경고만 남기고 계속 진행."""
        try:
            nodemap[name].value = value
        except Exception as e:
            logger.warning(
                "노드 설정 실패 (%s = %r): %s. 기본값으로 진행합니다.",
                name, value, e,
            )