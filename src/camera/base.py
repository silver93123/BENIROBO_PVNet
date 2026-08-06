"""Camera 추상 인터페이스.
3D 카메라에 대한 공통 인터페이스를 정의
LUCID Helios 외에 다른 카메라(RealSense, Photoneo 등) 추가 시
CameraBase를 상속받아 open/close/capture 세 메서드만 구현하면 됨
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameData:
    """카메라 한 프레임의 모든 출력.

    빈 피킹 파이프라인에서 핵심은 `points_organized`입니다. 이는
    픽셀 격자 형태의 PCD이므로, 2D 인스턴스 마스크를 그대로 인덱싱하여
    객체별 부분 PCD를 추출할 수 있습니다 (Stage 4-2 단계).

    Attributes:
        intensity:
            (H, W) uint8 mono image. RTMDet-Ins 입력으로 그대로 사용 가능
            (3채널 복제 또는 1채널 입력 모델로 학습).
        points:
            (N, 3) float32, mm 단위. 무효 픽셀이 제거된 평탄화 PCD.
            Open3D PointCloud 생성 시 사용.
        points_organized:
            (H, W, 3) float32, mm 단위. 픽셀 격자 형태의 XYZ.
            무효 픽셀에는 NaN이 들어 있음.
            마스크와 직접 인덱싱하여 객체별 PCD 추출에 사용.
        valid_mask:
            (H, W) bool. 유효한 깊이값을 가진 픽셀 마스크.
        confidence:
            (H, W) uint16 또는 None. ToF 측정 신뢰도 (지원 시).
        color_rgb:
            (H, W, 3) uint8 또는 None. depth 격자에 정렬된 RGB 이미지
            (지원하는 카메라에서, 옵션으로 켰을 때만). 기본 파이프라인
            (intensity/points_organized)과는 독립적인 부가 정보이므로
            None이어도 나머지 필드는 항상 정상 사용 가능해야 한다.
        points_std:
            (H, W, 3) float32 또는 None. 다중 프레임 촬영(AveragingCamera,
            num_frames>1)에서만 채워지는 픽셀별 XYZ 표준편차 - 단일 프레임
            촬영이거나 num_frames=1이면 항상 None이다. 이 값이 있어야
            icp_runner.extract_instance_points_probabilistic()의 Monte Carlo
            확률적 샘플링(픽셀별 Normal(mean, std)에서 재샘플링해 점 개수를
            늘리는 방식)이 가능하다 - None이면 그 함수는 자동으로 일반
            추출로 폴백한다.
    """
    intensity: np.ndarray
    points: np.ndarray
    points_organized: np.ndarray
    valid_mask: np.ndarray
    confidence: Optional[np.ndarray] = None
    color_rgb: Optional[np.ndarray] = None
    points_std: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # 형상 일관성 검증 - 디버깅 시 빠른 실패를 위해
        h, w = self.intensity.shape[:2]
        if self.points_organized.shape != (h, w, 3):
            raise ValueError(
                f"points_organized 형상 불일치: "
                f"intensity=({h},{w}), points_organized={self.points_organized.shape}"
            )
        if self.valid_mask.shape != (h, w):
            raise ValueError(
                f"valid_mask 형상 불일치: ({h},{w}) 기대, "
                f"실제={self.valid_mask.shape}"
            )
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"points 형상 불일치: (N,3) 기대, 실제={self.points.shape}"
            )
        if self.color_rgb is not None and self.color_rgb.shape != (h, w, 3):
            raise ValueError(
                f"color_rgb 형상 불일치: ({h},{w},3) 기대, "
                f"실제={self.color_rgb.shape}"
            )
        if self.points_std is not None and self.points_std.shape != (h, w, 3):
            raise ValueError(
                f"points_std 형상 불일치: ({h},{w},3) 기대, "
                f"실제={self.points_std.shape}"
            )

    @property
    def height(self) -> int:
        return self.intensity.shape[0]

    @property
    def width(self) -> int:
        return self.intensity.shape[1]

    def crop_by_mask(self, mask: np.ndarray) -> np.ndarray:
        """2D 인스턴스 마스크 영역의 유효 PCD만 추출.
        Stage 4-2의 핵심 연산: detection 결과 마스크 → 객체별 PCD.
        Args: mask: (H, W) bool 또는 0/1 uint8 마스크.
        Returns: (M, 3) float32 PCD. 빈 객체일 경우 (0, 3).
        """
        if mask.shape != (self.height, self.width):
            raise ValueError(
                f"mask 형상 불일치: ({self.height},{self.width}) 기대, "
                f"실제={mask.shape}"
            )
        combined = mask.astype(bool) & self.valid_mask
        return self.points_organized[combined]


class CameraBase(ABC):
    """3D 카메라 추상 인터페이스.

    Context manager로 사용을 권장합니다:

        with create_camera(config) as cam:
            frame = cam.capture()
            ...
    """

    @abstractmethod
    def open(self) -> None:
        """카메라 연결 및 스트리밍 시작."""

    @abstractmethod
    def close(self) -> None:
        """스트리밍 중지 및 리소스 해제."""

    @abstractmethod
    def capture(self) -> FrameData:
        """한 프레임 획득.
        Returns: FrameData(intensity + organized PCD)
        """

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # 예외를 흡수하지 않음