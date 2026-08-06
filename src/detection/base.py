"""2D 검출 엔진 공통 인터페이스.

src/camera/base.py의 CameraBase, app/core/registration/base.py의
PoseEstimator와 동일한 패턴을 따른다: 새 검출 백엔드를 추가하고 싶으면
DetectorBase를 상속받아 infer() 하나만 구현하고, src/detection/__init__.py의
create_detector()에 타입 문자열 한 줄만 등록하면 된다. 호출부
(app/core/detector.py, icp_runner.py)는 이 인터페이스만 알면 되고
RTMDet-Ins인지, 회전 헤드가 붙었는지는 몰라도 된다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """한 인스턴스의 검출 결과 (모든 백엔드 공통 포맷).

    Attributes:
        mask: (H, W) bool. 인스턴스 픽셀 마스크.
        bbox: (4,) float32 [x1, y1, x2, y2]. axis-aligned bbox.
        score: float. 검출 신뢰도 [0, 1].
        class_id: int.
        class_name: str.
        initial_pose: source(CAD)->scene 4x4 변환, m 단위. icp_runner.py의
            T_init과 동일한 단위/좌표계 관례를 따른다. 회전 헤드가 없는
            백엔드는 항상 None을 반환하며, 이 경우 icp_runner는 기존
            build_icp_init()(PCA 등 fallback)을 그대로 쓰면 된다.
            즉 이 필드는 "있으면 쓰고 없으면 기존 로직 그대로"인 선택적
            확장이지, 필수 계약이 아니다.
    """
    mask: np.ndarray
    bbox: np.ndarray
    score: float
    class_id: int
    class_name: str
    initial_pose: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.initial_pose is not None:
            assert self.initial_pose.shape == (4, 4), (
                f"initial_pose는 4x4여야 함, 받음: {self.initial_pose.shape}"
            )

    @property
    def n_pixels(self) -> int:
        return int(self.mask.sum())


class DetectorBase(ABC):
    """모든 2D 검출 백엔드가 구현해야 하는 최소 인터페이스."""

    #: 이 백엔드가 initial_pose를 채워서 반환하는지 여부.
    #: UI/로그에서 "이 백엔드가 coarse alignment까지 제공하는지" 표시할 때 사용.
    provides_pose_init: bool = False

    @property
    @abstractmethod
    def class_names(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def infer(self, image: np.ndarray) -> List[DetectionResult]:
        """이미지 한 장 -> DetectionResult 리스트.

        Args:
            image: (H, W, 3) BGR uint8 또는 (H, W) mono uint8.

        Returns:
            score_threshold 이상인 DetectionResult만, score 내림차순.
        """
        raise NotImplementedError