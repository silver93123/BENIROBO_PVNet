"""PVNet 핵심 알고리즘 (네트워크 + RANSAC 투표 + uncertainty-driven PnP).

RTMDet-Ins가 이미 인스턴스 검출/마스크/bbox를 낸 뒤, 그 크롭을 입력받아
DetectionResult.initial_pose(4x4)를 채우는 것이 최종 목표다. 이 모듈은
그 중 핵심 알고리즘만 담고 있다 (이번 구현 범위). 아래는 아직 연결되지
않았다 - 다음 단계에서 채워야 함:
    - RTMDetInferencerRotHead와 동일한 패턴의 추론 wrapper
      (크롭 -> pipeline.estimate_pose_from_crop -> DetectionResult.initial_pose)
    - 학습 스크립트/데이터셋 (키포인트 2D 투영 GT 자동 생성 포함,
      generate_rotation_labels.py와 같은 원리로 CAD pose를 알면
      keypoints_3d를 그 pose로 투영해서 만들면 됨)
    - GUI 탭 (app/tabs/training_pipelines, app/tabs/icp_pipelines)

의존성: 이 패키지는 scipy(least_squares)가 추가로 필요하다.
requirements.txt에 없다면 `pip install scipy`로 설치할 것.

사용 예시 (단일 크롭 end-to-end):
    >>> from src.detection.pvnet import (
    ...     PVNetHead, farthest_point_sampling, estimate_pose_from_crop,
    ... )
    >>> keypoints_3d = farthest_point_sampling(cad_surface_points, num_keypoints=8)
    >>> model = PVNetHead(num_keypoints=len(keypoints_3d))
    >>> result = estimate_pose_from_crop(model, crop_tensor, keypoints_3d, camera_matrix)
    >>> result.pose         # (4, 4)
    >>> result.reprojection_error

단계별로 직접 다루고 싶으면 (voting 결과를 로깅/시각화하고 싶을 때 등):
    >>> from src.detection.pvnet import vote_keypoints, solve_uncertainty_pnp
    >>> seg_logits, vertex = model(crop_tensor.unsqueeze(0))   # PVNetHead.forward
    >>> # ... fg_pixels, vertex_field로 후처리 (pipeline.py 참고) ...
    >>> votes = vote_keypoints(fg_pixels, vertex_field, num_keypoints=9)
"""
from .keypoints import farthest_point_sampling
from .model import PVNetHead, segmentation_loss, vertex_smooth_l1_loss
from .pipeline import estimate_pose_from_crop
from .pnp import PnPResult, solve_uncertainty_pnp
from .symmetry import canonicalize_axial_rotation
from .voting import KeypointVote, vote_keypoints

__all__ = [
    "farthest_point_sampling",
    "PVNetHead",
    "segmentation_loss",
    "vertex_smooth_l1_loss",
    "estimate_pose_from_crop",
    "PnPResult",
    "solve_uncertainty_pnp",
    "canonicalize_axial_rotation",
    "KeypointVote",
    "vote_keypoints",
]