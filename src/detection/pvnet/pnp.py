"""Uncertainty-driven PnP.

Peng et al., CVPR 2019, Sec 3.2, Eq.5. voting.py가 낸 키포인트별
(mu_k, Sigma_k)를 이용해 Mahalanobis 재투영 오차를 최소화하는 (R, t)를 구한다.
일반 EPnP는 모든 키포인트를 동일 신뢰도로 취급하지만, 여기서는 voting
공분산이 작은(확신하는) 키포인트일수록 재투영 오차에 더 큰 가중치가 실린다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class PnPResult:
    pose: np.ndarray            # (4, 4) source(CAD)->camera 변환, DetectionResult.initial_pose와 동일 규약
    reprojection_error: float   # px, 최종 평균 재투영 오차 (참고/로깅용)


def _project(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, distCoeffs=None)
    return proj.reshape(-1, 2)


def solve_uncertainty_pnp(
    points_3d: np.ndarray,
    points_2d_mean: np.ndarray,
    points_2d_cov: np.ndarray,
    camera_matrix: np.ndarray,
    min_points_for_epnp: int = 4,
) -> PnPResult:
    """키포인트 2D 위치의 평균/공분산 + 3D 대응점 -> (R, t).

    Args:
        points_3d: (K, 3) CAD 로컬 좌표계 키포인트. keypoints.py 출력과 순서가
            반드시 일치해야 한다.
        points_2d_mean: (K, 2) voting.py KeypointVote.mean을 쌓은 배열.
        points_2d_cov: (K, 2, 2) voting.py KeypointVote.covariance를 쌓은 배열.
        camera_matrix: (3, 3) intrinsic. 크롭이 원본 이미지에서 잘리거나
            리사이즈됐다면, 호출부에서 크롭 좌표계에 맞게 조정한 intrinsic을
            넘겨야 한다 (이 함수는 좌표계 보정을 하지 않는다).
        min_points_for_epnp: EPnP 초기화에 쓸 최소 포인트 수. 논문은 4개를
            쓰고, 공분산 trace가 가장 작은(가장 확신하는) 키포인트들을 고른다.

    Returns:
        PnPResult(pose, reprojection_error).
    """
    k = points_3d.shape[0]
    if k < min_points_for_epnp:
        raise ValueError(f"PnP에 필요한 최소 키포인트({min_points_for_epnp}개) 미달: {k}개")

    # --- 1) EPnP 초기화: 공분산 trace가 가장 작은 4개만 사용 (논문 Sec 3.2) ---
    trace = np.trace(points_2d_cov, axis1=1, axis2=2)
    best4 = np.argsort(trace)[:min_points_for_epnp]

    ok, rvec0, tvec0 = cv2.solvePnP(
        points_3d[best4].astype(np.float64),
        points_2d_mean[best4].astype(np.float64),
        camera_matrix.astype(np.float64),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        raise RuntimeError("EPnP 초기화 실패 - 신뢰도 상위 4개 키포인트 배치가 퇴화(공면/공선)일 가능성")

    # --- 2) Levenberg-Marquardt로 Mahalanobis 재투영 오차 최소화 (Eq.5) ---
    # Sigma^-1 = L L^T (Cholesky) 이면, err^T Sigma^-1 err = ||L^T err||^2 이므로
    # residual을 L^T err로 whitening해서 넘기면 least_squares(sum of squares)가
    # 자동으로 Mahalanobis distance를 최소화하게 된다.
    cov_inv = np.linalg.inv(points_2d_cov)  # (K, 2, 2)
    chol_inv = np.linalg.cholesky(cov_inv)  # (K, 2, 2), lower-triangular L

    x0 = np.concatenate([rvec0.flatten(), tvec0.flatten()])

    def residual(x: np.ndarray) -> np.ndarray:
        rvec, tvec = x[:3].reshape(3, 1), x[3:].reshape(3, 1)
        proj = _project(points_3d, rvec, tvec, camera_matrix)  # (K, 2)
        err = proj - points_2d_mean
        whitened = np.einsum("kij,kj->ki", chol_inv.transpose(0, 2, 1), err)  # L^T @ err
        return whitened.flatten()

    result = least_squares(residual, x0, method="lm")
    rvec, tvec = result.x[:3].reshape(3, 1), result.x[3:].reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = tvec.flatten()

    proj_final = _project(points_3d, rvec, tvec, camera_matrix)
    reproj_err = float(np.linalg.norm(proj_final - points_2d_mean, axis=1).mean())

    return PnPResult(pose=pose, reprojection_error=reproj_err)