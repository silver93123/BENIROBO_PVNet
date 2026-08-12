"""ifm O3R (OVP8xx VPU + 3D ToF 헤드) 카메라 구현.

대상 하드웨어:
    ifm O3R 플랫폼 (OVP80x VPU + O3R22x/25x 시리즈 ToF 헤드)

SDK:
    ifm3dpy (공식 파이썬 바인딩)
    설치: pip install ifm3dpy
    참고: https://api.ifm3d.com/stable/

설계 노트 (Helios/Femto Bolt 드라이버와의 대응 관계):
    - Helios/Femto의 'intensity'에 대응하는 것은 O3R의 NORM_AMPLITUDE_IMAGE
      (정규화된 반사강도 이미지). AMPLITUDE_IMAGE가 아님에 주의 - O3R
      플랫폼은 raw amplitude 대신 노출 보정이 된 이 버전을 제공한다.
    - XYZ/거리 값은 O3R이 meter 단위로 반환하므로, 프로젝트 공통 규격(mm)에
      맞춰 1000을 곱해 변환한다.
    - confidence 이미지의 최하위 비트(bit 0)가 0이면 유효 픽셀 (ifm 공식
      문서 기준 규칙). z 범위 필터와 AND로 결합해 valid_mask를 만든다.
    - RGB(color_rgb)는 O3R222/225처럼 3D 헤드와 같은 물리 포트 그룹에서
      2D(JPEG) 스트림도 함께 나오는 모델에서만 의미가 있다. 별도의
      2D/3D 포트가 있어야 하며, o3r-algo-utilities 패키지로 2D-3D
      registration(픽셀 좌표 정렬)을 수행해 depth 격자에 맞춰 리샘플링한다.
      IR/depth 기반 검증된 파이프라인과는 완전히 분리되어 있어서, 이 옵션이
      실패해도(패키지 미설치, 다른 헤드의 포트, registration 실패 등)
      capture()는 color_rgb=None으로 정상 반환하며 크래시하지 않는다
      (Femto Bolt의 capture_rgb 옵션과 동일한 안전장치 패턴).

검증된 출처:
    - https://api.ifm3d.com/stable/ (공식 문서)
    - https://github.com/ifm/ifm3d-examples (registration_2d_3d.py 등 공식 예제)

주의:
    O3R은 하나의 VPU에 여러 카메라 헤드(포트)가 물릴 수 있으므로, ip뿐 아니라
    port_3d(필수)와 port_2d(RGB 사용 시)를 반드시 실제 배선에 맞게 지정해야 한다.
    포트 목록/타입은 `ifm3d ovp8xx config get --ip=<IP>` 또는
    O3R(ip=...).get()의 "ports" 키에서 확인 가능하다.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import CameraBase, FrameData

logger = logging.getLogger(__name__)

try:
    from ifm3dpy.device import O3R
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
    _IFM3D_AVAILABLE = True
    _IFM3D_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    _IFM3D_AVAILABLE = False
    _IFM3D_IMPORT_ERROR = _e

try:
    from ifm3dpy.deserialize import RGBInfoV1, TOFInfoV4
    from o3r_algo_utilities.calib.point_correspondences import inverse_intrinsic_projection
    from o3r_algo_utilities.o3r_uncompress_di import evalIntrinsic
    from o3r_algo_utilities.rotmat import rotMat
    _O3R_ALGO_AVAILABLE = True
    _O3R_ALGO_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    _O3R_ALGO_AVAILABLE = False
    _O3R_ALGO_IMPORT_ERROR = _e


class O3RCamera(CameraBase):
    """ifm O3R wrapper.

    출력 정규화 (FrameData 규격, Helios/Femto Bolt 드라이버와 동일):
        - intensity: (H, W) uint8. NORM_AMPLITUDE_IMAGE를 percentile stretch로 8-bit화.
        - points_organized: (H, W, 3) float32, mm 단위. 무효 픽셀 = NaN.
        - points: (N, 3) float32, NaN 제거된 유효 포인트.
        - valid_mask: (H, W) bool. confidence 최하위 비트 + z 범위로 결정.
        - confidence: (H, W) uint16. O3R이 항상 제공.

    Args:
        ip: VPU IP 주소.
        port_3d: 3D(ToF) 카메라가 물린 포트명 (예: 'port0'). 필수.
        port_2d: RGB(2D) 카메라 포트명. capture_rgb=True일 때만 사용.
        connect_timeout_ms / capture_timeout_ms: 연결/캡처 타임아웃.
        valid_z_range_mm: 이 범위 밖 Z는 무효 처리 (mm 단위, 프로젝트 공통 규격).
        capture_rgb: True면 depth 격자에 정렬된 RGB도 추가로 캡처한다 (실험적,
            기본값 False). port_2d가 3D와 같은 헤드가 아니거나
            o3r-algo-utilities가 없으면 자동으로 비활성화되고 경고만 남긴다.
    """

    def __init__(
        self,
        ip: str = "192.168.0.69",
        port_3d: str = "port0",
        port_2d: Optional[str] = None,
        connect_timeout_ms: int = 5000,
        capture_timeout_ms: int = 3000,
        valid_z_range_mm: tuple = (100.0, 1500.0),
        capture_rgb: bool = False,
    ) -> None:
        if not _IFM3D_AVAILABLE:
            raise ImportError(
                "ifm3dpy를 import할 수 없습니다. 'pip install ifm3dpy' 실행 후 "
                f"다시 시도하세요. 원본 에러: {_IFM3D_IMPORT_ERROR}"
            )

        self.ip = ip
        self.port_3d = port_3d
        self.port_2d = port_2d
        self.connect_timeout_ms = int(connect_timeout_ms)
        self.capture_timeout_ms = int(capture_timeout_ms)
        self._valid_z_min = float(valid_z_range_mm[0])
        self._valid_z_max = float(valid_z_range_mm[1])
        self.capture_rgb = bool(capture_rgb)

        if self.capture_rgb and not self.port_2d:
            logger.warning("capture_rgb=True인데 port_2d가 지정되지 않아 RGB 없이 진행합니다.")
            self.capture_rgb = False
        if self.capture_rgb and not _O3R_ALGO_AVAILABLE:
            logger.warning(
                "o3r-algo-utilities를 import할 수 없어 RGB 없이 진행합니다. "
                "'pip install o3r-algo-utilities' 실행 후 다시 시도하세요. "
                "원본 에러: %s", _O3R_ALGO_IMPORT_ERROR,
            )
            self.capture_rgb = False

        self._o3r: Optional["O3R"] = None
        self._fg_3d: Optional["FrameGrabber"] = None
        self._fg_2d: Optional["FrameGrabber"] = None

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        self._o3r = O3R(ip=self.ip)
        config = self._o3r.get()

        if self.port_3d not in config.get("ports", {}):
            raise RuntimeError(
                f"포트 '{self.port_3d}'가 VPU({self.ip})에 없습니다. "
                f"사용 가능한 포트: {list(config.get('ports', {}).keys())}"
            )

        if self.capture_rgb:
            try:
                self._check_same_head(config, self.port_2d, self.port_3d)
            except Exception as e:
                logger.warning("2D/3D 헤드 검증 실패 - RGB 없이 진행합니다: %s", e)
                self.capture_rgb = False

        self._fg_3d = FrameGrabber(self._o3r, self._o3r.port(self.port_3d).pcic_port)
        buffers_3d = [buffer_id.RADIAL_DISTANCE_IMAGE, buffer_id.NORM_AMPLITUDE_IMAGE, buffer_id.CONFIDENCE_IMAGE]
        if self.capture_rgb:
            buffers_3d.append(buffer_id.TOF_INFO)
        self._fg_3d.start(buffers_3d)

        if self.capture_rgb:
            self._fg_2d = FrameGrabber(self._o3r, self._o3r.port(self.port_2d).pcic_port)
            self._fg_2d.start([buffer_id.JPEG_IMAGE, buffer_id.RGB_INFO])

        fw_version = config["device"]["swVersion"]["firmware"]
        logger.info(
            "O3R opened (ip=%s, fw=%s) | port_3d=%s%s",
            self.ip, fw_version, self.port_3d,
            f", port_2d={self.port_2d}(rgb on)" if self.capture_rgb else "",
        )

    @staticmethod
    def _check_same_head(config: dict, port2d: Optional[str], port3d: str) -> None:
        if not port2d:
            raise ValueError("port_2d가 지정되지 않았습니다.")
        sn2d = config["ports"][port2d]["info"]["serialNumber"]
        sn3d = config["ports"][port3d]["info"]["serialNumber"]
        if sn2d != sn3d:
            raise ValueError(
                f"'{port2d}'와 '{port3d}'는 서로 다른 카메라 헤드입니다 "
                f"(2D-3D registration은 같은 헤드에서만 가능)."
            )

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        for fg in (self._fg_3d, self._fg_2d):
            if fg is not None:
                try:
                    fg.stop()
                except Exception as e:
                    logger.warning("FrameGrabber.stop 실패: %s", e)
        self._fg_3d = None
        self._fg_2d = None
        self._o3r = None

    # --------------------------------------------------------------- capture
    def capture(self) -> FrameData:
        if self._fg_3d is None:
            raise RuntimeError(
                "카메라가 열려있지 않습니다. open()을 먼저 호출하거나 with 구문을 사용하세요."
            )

        ok, frame_3d = self._fg_3d.wait_for_frame().wait_for(self.capture_timeout_ms)
        if not ok:
            raise RuntimeError(
                f"3D 포트 '{self.port_3d}' 프레임 획득 실패 (timeout={self.capture_timeout_ms}ms). "
                "케이블/포트명/카메라 상태를 확인하세요."
            )

        distance_m = frame_3d.get_buffer(buffer_id.RADIAL_DISTANCE_IMAGE)
        amplitude = frame_3d.get_buffer(buffer_id.NORM_AMPLITUDE_IMAGE)
        confidence = frame_3d.get_buffer(buffer_id.CONFIDENCE_IMAGE)

        intensity = self._normalize_intensity(amplitude)

        xyz_mm = self._distance_to_xyz_mm(distance_m, frame_3d)
        z_mm = xyz_mm[..., 2]

        conf_valid = (confidence.astype(np.uint16) & 1) == 0
        z_valid = (z_mm >= self._valid_z_min) & (z_mm <= self._valid_z_max)
        valid_mask = conf_valid & z_valid

        points_organized = xyz_mm.astype(np.float32, copy=True)
        points_organized[~valid_mask] = np.nan
        points = points_organized[valid_mask]

        color_rgb = None
        if self.capture_rgb and self._fg_2d is not None:
            color_rgb = self._try_capture_aligned_rgb(distance_m, frame_3d)

        return FrameData(
            intensity=intensity,
            points=points,
            points_organized=points_organized,
            valid_mask=valid_mask,
            confidence=confidence,
            color_rgb=color_rgb,
        )

    @staticmethod
    def _distance_to_xyz_mm(distance_m: np.ndarray, frame_3d) -> np.ndarray:
        """RADIAL_DISTANCE_IMAGE만으로 XYZ가 필요하면 XYZ 버퍼를 직접 받는 편이
        더 간단하지만, TOF_INFO(선택적 RGB 정합)를 함께 쓸 때는 동일 프레임에서
        일관된 값을 보장하기 위해 XYZ 버퍼를 그대로 사용한다."""
        xyz_m = frame_3d.get_buffer(buffer_id.XYZ)
        return (xyz_m * 1000.0).astype(np.float32)

    @staticmethod
    def _normalize_intensity(amplitude: np.ndarray) -> np.ndarray:
        if amplitude.size == 0:
            return np.zeros_like(amplitude, dtype=np.uint8)

        lo, hi = np.percentile(amplitude, [1.0, 99.0])
        if hi <= lo:
            return np.zeros_like(amplitude, dtype=np.uint8)

        scaled = (amplitude.astype(np.float32) - lo) / (hi - lo)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        return (scaled * 255.0).astype(np.uint8)

    def _try_capture_aligned_rgb(self, distance_m: np.ndarray, frame_3d) -> Optional[np.ndarray]:
        """depth 격자(3D 해상도)에 정렬된 RGB를 뽑아본다. 실패해도 None만
        반환하고 절대 예외를 위로 던지지 않는다 (3D 캡처 성공에 영향 주지 않기 위함).
        투영 로직은 ifm 공식 registration_2d_3d.py 기반."""
        try:
            ok_2d, frame_2d = self._fg_2d.wait_for_frame().wait_for(self.capture_timeout_ms)
            if not ok_2d:
                logger.warning("2D 포트 '%s' 프레임 획득 실패.", self.port_2d)
                return None

            import cv2
            jpg = cv2.imdecode(frame_2d.get_buffer(buffer_id.JPEG_IMAGE), cv2.IMREAD_UNCHANGED)
            jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)

            rgb_info = RGBInfoV1().deserialize(frame_2d.get_buffer(buffer_id.RGB_INFO))
            tof_info = TOFInfoV4().deserialize(frame_3d.get_buffer(buffer_id.TOF_INFO))

            h3d, w3d = distance_m.shape
            unit_vectors_3d = evalIntrinsic(
                tof_info.intrinsic_calibration.model_id,
                tof_info.intrinsic_calibration.parameters,
                w3d, h3d,
            )
            pt_cloud_3d_opt = (unit_vectors_3d * distance_m).reshape(3, -1)
            valid = pt_cloud_3d_opt[2] > 0

            ext3d = tof_info.extrinsic_optic_to_user
            rot3d = rotMat(ext3d.rot_x, ext3d.rot_y, ext3d.rot_z)
            trans3d = np.array([ext3d.trans_x, ext3d.trans_y, ext3d.trans_z])
            pt_cloud_user = rot3d.dot(pt_cloud_3d_opt) + trans3d[..., np.newaxis]

            ext2d = rgb_info.extrinsic_optic_to_user
            rot2d = rotMat(ext2d.rot_x, ext2d.rot_y, ext2d.rot_z)
            trans2d = np.array([ext2d.trans_x, ext2d.trans_y, ext2d.trans_z])
            pt_cloud_2d_opt = rot2d.T.dot(pt_cloud_user - trans2d[..., np.newaxis])

            pixels_2d = inverse_intrinsic_projection(
                camXYZ=pt_cloud_2d_opt,
                invIC={
                    "modelID": rgb_info.inverse_intrinsic_calibration.model_id,
                    "modelParameters": rgb_info.inverse_intrinsic_calibration.parameters,
                },
                camRefToOpticalSystem={"rot": (0, 0, 0), "trans": (0, 0, 0)},
                binning=0,
            )
            pixels_2d = np.round(pixels_2d).astype(int)
            row, col = pixels_2d[1], pixels_2d[0]

            in_bounds = valid & (row >= 0) & (row < jpg.shape[0]) & (col >= 0) & (col < jpg.shape[1])

            rgb_on_3d_grid = np.zeros((h3d * w3d, 3), dtype=np.uint8)
            rgb_on_3d_grid[in_bounds] = jpg[row[in_bounds], col[in_bounds]]
            return rgb_on_3d_grid.reshape(h3d, w3d, 3)
        except Exception as e:
            logger.warning("RGB 정합/추출 실패 (3D 결과에는 영향 없음): %s", e)
            return None