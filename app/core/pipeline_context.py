"""ICP 정합 테스트 섹션의 파이프라인 탭들이 공유하는 입력 계약.

app/tabs/icp_test_tab.py(공유 헤더)가 세션/프레임/CAD 로딩을 끝낸 뒤 이
객체 하나로 묶어서 현재 활성 파이프라인 탭에 넘긴다. 파이프라인 탭은 이
필드들만으로 자기 할 일을 시작할 수 있어야 하고, 필드를 전부 쓸 필요는
없다 (예: detect() 단계에서는 cad_* 필드가 아직 None이어도 됨).

세션당 한 번만 바뀌는 값(checkpoint_path, config_path)과 프레임마다
바뀌는 값(pcd_organized_mm 등)이 섞여 있는데, 의도적으로 그렇게 뒀다 -
파이프라인 탭 입장에서는 "이번 실행에 필요한 것 전부"가 한 객체에
들어있는 게 여러 인자로 흩어져 있는 것보다 다루기 쉽다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameContext:
    session_path: str
    frame_name: str
    image_path: str                              # intensity 이미지 경로 (검출 입력)
    pcd_organized_mm: np.ndarray                  # (H,W,3) mm
    valid_mask: np.ndarray                        # (H,W) bool

    # CAD - detect() 시점에는 None일 수 있음 (CAD는 register() 직전에만 로드됨).
    # register()를 구현/override하는 파이프라인은 이 셋이 채워져 있다고 가정해도 됨.
    cad_pcd: Optional[object] = None               # o3d.geometry.PointCloud
    cad_visible_normal: Optional[object] = None    # o3d.geometry.PointCloud
    cad_visible_flipped: Optional[object] = None   # o3d.geometry.PointCloud

    # 검출 백엔드 공통 설정 - 세션 내내 거의 안 바뀌지만, "이번 실행에
    # 필요한 것 전부"를 한 곳에 모은다는 원칙상 여기 포함.
    checkpoint_path: str = ""
    config_path: str = ""
    score_threshold: float = 0.3

    # 2026-07 추가: Monte Carlo 확률적 샘플링용 픽셀별 표준편차 (H,W,3) mm.
    # 다중 프레임 촬영(AveragingCamera)에서만 채워짐 - 세션에서 로드한
    # 프레임이나 num_frames=1 촬영은 항상 None. None이면
    # icp_runner.extract_instance_points_probabilistic()이 자동으로
    # 일반 추출(그리드 보간/원본)로 폴백한다.
    pcd_std_mm: Optional[np.ndarray] = None