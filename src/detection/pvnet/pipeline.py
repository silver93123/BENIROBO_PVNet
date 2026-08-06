"""핵심 알고리즘 4단계(모델 forward -> 후처리 -> 투표 -> PnP)를 하나로 묶은
편의 함수. 아직 DetectorBase/DetectionResult에는 연결하지 않았다 - 그건 다음
단계(RTMDetInferencerRotHead와 동일한 패턴의 추론 wrapper)에서 할 일이고,
여기서는 크롭 하나에 대한 핵심 알고리즘 동작을 빠르게 확인/테스트하기 위한
용도다.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import PVNetHead
from .pnp import PnPResult, solve_uncertainty_pnp
from .voting import vote_keypoints


def estimate_pose_from_crop(
    model: PVNetHead,
    crop_tensor: torch.Tensor,
    keypoints_3d: np.ndarray,
    camera_matrix: np.ndarray,
    device: str = "cpu",
) -> PnPResult:
    """크롭 이미지 하나 -> (R, t).

    Args:
        model: PVNetHead 인스턴스 (학습된 checkpoint를 로드했거나, 배관
            테스트용이면 미학습 상태여도 코드 자체는 동작함 - 물론 미학습
            상태의 pose는 의미 없음).
        crop_tensor: (3, H, W) float32 [0,1]. rotation_head_model.py의
            crop_and_preprocess() 출력을 그대로 재사용할 수 있다.
        keypoints_3d: (K, 3) keypoints.py의 farthest_point_sampling() 출력과
            동일 순서/좌표계여야 한다.
        camera_matrix: (3, 3) intrinsic. crop_and_preprocess()가 원본 이미지를
            리사이즈/크롭했다면, 그 변환을 반영해 조정된 intrinsic을 넘겨야
            한다 - 이 함수는 좌표계를 자동으로 보정하지 않는다.
        device: model과 동일한 device ("cpu" 또는 "cuda").

    Returns:
        PnPResult(pose, reprojection_error).

    Raises:
        RuntimeError: 세그멘테이션이 전경 픽셀을 2개 미만으로 예측한 경우
            (미학습 모델이거나 크롭이 비정상일 때 흔히 발생).
    """
    model.eval()
    with torch.no_grad():
        batch = crop_tensor.unsqueeze(0).to(device)
        seg_logits, vertex = model(batch)

    seg_pred = seg_logits[0].argmax(dim=0).cpu().numpy()  # (H, W) {0,1}
    fg_yx = np.argwhere(seg_pred == 1)                    # (M, 2) [row, col]
    if fg_yx.shape[0] < 2:
        raise RuntimeError("전경 픽셀이 2개 미만 - 세그멘테이션 실패 또는 미학습 모델")

    fg_pixels = fg_yx[:, ::-1].astype(np.float64)  # (M, 2) -> (x, y)

    k = model.num_keypoints
    vertex_np = vertex[0].cpu().numpy()                          # (K*2, H, W)
    vertex_np = vertex_np.reshape(k, 2, *vertex_np.shape[1:])    # (K, 2, H, W)
    vertex_field = vertex_np[:, :, fg_yx[:, 0], fg_yx[:, 1]]     # (K, 2, M)
    vertex_field = np.transpose(vertex_field, (2, 0, 1))         # (M, K, 2)

    votes = vote_keypoints(fg_pixels, vertex_field, num_keypoints=k)
    means = np.stack([v.mean for v in votes])
    covs = np.stack([v.covariance for v in votes])

    return solve_uncertainty_pnp(keypoints_3d, means, covs, camera_matrix)