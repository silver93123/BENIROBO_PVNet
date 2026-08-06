"""PVNet 키포인트 선택: CAD 표면 포인트 위에서 FPS(farthest point sampling).

Peng et al., "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation",
CVPR 2019, Sec 3.1 "Keypoint selection"의 방식을 그대로 따른다:
    1) 물체 중심(centroid)을 키포인트 집합에 먼저 넣는다.
    2) 현재 집합에서 가장 먼 표면 점을 반복해서 추가한다 (K개가 될 때까지).

bbox 8개 코너 대신 표면 점을 쓰는 이유: 코너는 물체 픽셀에서 멀리 떨어져
있어, 픽셀 -> 키포인트 방향벡터의 각도 오차가 위치 오차로 크게 증폭된다
(논문 Fig.3, ablation에서 FPS 8이 BBox 8보다 일관되게 우수함을 확인).
"""
from __future__ import annotations

import numpy as np

DEFAULT_NUM_SURFACE_KEYPOINTS = 8  # 논문 권장값: 정확도/속도 균형, 12개와 차이 미미


def farthest_point_sampling(
    points: np.ndarray,
    num_keypoints: int = DEFAULT_NUM_SURFACE_KEYPOINTS,
    include_centroid: bool = True,
) -> np.ndarray:
    """CAD 표면 점군에서 K개 키포인트를 FPS로 선택.

    Args:
        points: (N, 3) CAD 모델 표면 점군, 물체 로컬 좌표계(m 단위).
            메시라면 정점(vertex) 또는 표면에서 균일 샘플링한 점을 넣으면 됨.
        num_keypoints: 표면에서 뽑을 키포인트 개수 K (센트로이드는 별도).
        include_centroid: True면 반환값 맨 앞에 물체 중심을 추가해 (K+1, 3)을
            반환한다. PnP에 넘길 3D 대응점 집합은 보통 센트로이드를 포함해
            "표면점들 + 중심" 구성으로 쓰는 게 논문 관례.

    Returns:
        (K, 3) 또는 (K+1, 3) 키포인트 3D 좌표. 순서는 [center?, fps_1, ...,
        fps_K] 이며, 이 순서가 이후 voting.py/pnp.py에서 다루는 키포인트
        인덱스 순서와 항상 일치해야 한다 (network 출력 채널 순서 포함).
    """
    n = points.shape[0]
    if n < num_keypoints:
        raise ValueError(f"점 개수({n})가 키포인트 개수({num_keypoints})보다 적음")

    centroid = points.mean(axis=0)

    # FPS 시작점: 센트로이드에서 가장 먼 점으로 고정하면 결정적(deterministic)이고
    # 물체 형상 중 가장 극단적인 부분부터 커버하게 된다.
    dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
    first_idx = int(np.argmax(dist_to_centroid))

    chosen_idx = [first_idx]
    min_dist = np.linalg.norm(points - points[first_idx], axis=1)
    min_dist[first_idx] = -1.0  # 재선택 방지

    while len(chosen_idx) < num_keypoints:
        next_idx = int(np.argmax(min_dist))
        chosen_idx.append(next_idx)
        d = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[next_idx] = -1.0

    surface_kpts = points[chosen_idx].astype(np.float64)
    if include_centroid:
        return np.concatenate([centroid[None, :].astype(np.float64), surface_kpts], axis=0)
    return surface_kpts