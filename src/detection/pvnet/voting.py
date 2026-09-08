"""RANSAC 기반 키포인트 위치 투표.

Peng et al., CVPR 2019, Sec 3.1 수식을 그대로 구현한다.
    - 가설 생성: 전경 픽셀 쌍을 무작위로 뽑아 두 방향벡터의 교점을 후보
      키포인트 위치로 삼는다.
    - 투표 점수 (Eq.2): 그 후보 위치가 실제로 맞다면, 다른 전경 픽셀들의
      예측 방향과도 일치해야 한다는 논리로 지지표를 센다.
    - 평균/공분산 (Eq.3, Eq.4): 투표 점수를 가중치로 한 가중평균/가중공분산.

논문은 이 과정을 CUDA 커널로 구현해 이미지 전체 멀티인스턴스를 실시간
처리하지만(25fps @ 480x640), 여기서는 크롭 하나(인스턴스 하나) 단위로만
돌리면 되므로 numpy 벡터화로 충분하다. 실시간 멀티인스턴스 처리가 필요해지면
배치 단위 torch 구현으로 옮기면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_NUM_HYPOTHESES = 128  # 논문은 구체값 비공개, 공식 구현 관례상 128 사용
DEFAULT_INLIER_THRESHOLD = 0.99  # 논문 Eq.2의 theta


@dataclass
class KeypointVote:
    """키포인트 하나에 대한 투표 결과."""
    mean: np.ndarray        # (2,) 이미지 좌표 (x, y) - 가중평균 mu_k (Eq.3)
    covariance: np.ndarray  # (2, 2) 가중 공분산 Sigma_k (Eq.4) - pnp.py의 가중치로 쓰임
    inlier_score: float     # 최고 득표 가설의 원점수 (참고/디버깅용, PnP에는 안 씀)


def _line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """두 점 p1,p2에서 각각 방향 d1,d2로 뻗은 직선의 교점.

    h = p1 + t1*d1 = p2 + t2*d2  =>  [d1, -d2][t1; t2] = p2 - p1 (2x2 선형계).

    Args:
        p1, p2, d1, d2: (N, 2). d1, d2는 단위벡터일 필요 없음.

    Returns:
        h: (N, 2) 교점 좌표 (평행이면 값 무의미, valid=False로 표시됨).
        valid: (N,) bool. 두 방향이 충분히 벌어져(평행 아님) 교점이 수치적으로
            안정적인 경우만 True.
    """
    a11, a12 = d1[:, 0], -d2[:, 0]
    a21, a22 = d1[:, 1], -d2[:, 1]
    det = a11 * a22 - a12 * a21
    valid = np.abs(det) > 1e-6

    rhs_x = p2[:, 0] - p1[:, 0]
    rhs_y = p2[:, 1] - p1[:, 1]
    det_safe = np.where(valid, det, 1.0)
    t1 = (rhs_x * a22 - a12 * rhs_y) / det_safe

    h = p1 + t1[:, None] * d1
    return h, valid


def vote_keypoints(
    fg_pixels: np.ndarray,
    vertex_field: np.ndarray,
    num_keypoints: int,
    num_hypotheses: int = DEFAULT_NUM_HYPOTHESES,
    inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
    rng: Optional[np.random.Generator] = None,
) -> List[KeypointVote]:
    """전경 픽셀 + 픽셀별 벡터장 -> 키포인트별 (평균, 공분산).

    Args:
        fg_pixels: (M, 2) 전경(물체) 픽셀 좌표 (x, y).
        vertex_field: (M, K, 2) 각 전경 픽셀에서 각 키포인트로 향하는 예측
            벡터 (fg_pixels와 같은 순서로 정렬돼 있어야 함). model.py 출력을
            이 형태로 재배열해서 넘기면 된다.
        num_keypoints: K.
        num_hypotheses: 가설 개수 N.
        inlier_threshold: 논문 Eq.2의 theta (기본 0.99).
        rng: 재현성이 필요하면 np.random.default_rng(seed)를 넘길 것.

    Returns:
        키포인트 K개 각각에 대한 KeypointVote 리스트 (입력 채널 순서 유지 -
        keypoints.py가 반환한 3D 키포인트 순서와 이 리스트 순서가 일치해야
        pnp.py에서 2D-3D 대응이 올바르게 맞는다).
    """
    if rng is None:
        rng = np.random.default_rng()

    m = fg_pixels.shape[0]
    if m < 2:
        raise ValueError(f"투표에 필요한 최소 전경 픽셀(2개) 미달: {m}개")

    # 모든 키포인트가 같은 픽셀 쌍(N개)을 공유한다 - 공식 구현 관례이자,
    # 여기서 지배적 비용인 (N x M) 코사인 유사도 계산을 키포인트마다
    # 새 쌍으로 다시 뽑지 않아도 되게 해준다.
    idx_i = rng.integers(0, m, size=num_hypotheses)
    idx_j = rng.integers(0, m, size=num_hypotheses)
    same = idx_i == idx_j
    if same.any():
        idx_j[same] = (idx_j[same] + 1) % m  # 같은 픽셀이 뽑히면 옆 인덱스로 대체

    p_i = fg_pixels[idx_i].astype(np.float64)
    p_j = fg_pixels[idx_j].astype(np.float64)
    fg = fg_pixels.astype(np.float64)

    results: List[KeypointVote] = []
    for k in range(num_keypoints):
        v = vertex_field[:, k, :].astype(np.float64)  # (M, 2)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_unit = v / np.clip(norm, 1e-8, None)  # Eq.1 대응 - 단위벡터화

        d_i, d_j = v_unit[idx_i], v_unit[idx_j]
        hyp, valid = _line_intersection(p_i, d_i, p_j, d_j)

        if not valid.any():
            # 극단적으로 운이 나쁜 경우(전부 평행) - 전경 중심으로 폴백하고
            # 공분산을 크게 잡아 pnp.py가 이 키포인트를 낮은 가중치로 취급하게 함.
            results.append(KeypointVote(mean=fg.mean(axis=0), covariance=np.eye(2) * 1e4, inlier_score=0.0))
            continue

        hyp_valid = hyp[valid]  # (N', 2)

        # 투표 점수 (Eq.2): 가설 h에서 모든 전경 픽셀 p로의 방향이 v_unit[p]와
        # threshold 이상 일치하는 픽셀 수.
        diff = hyp_valid[:, None, :] - fg[None, :, :]  # (N', M, 2)
        diff_norm = np.linalg.norm(diff, axis=2, keepdims=True)
        diff_unit = diff / np.clip(diff_norm, 1e-8, None)
        cos_sim = np.sum(diff_unit * v_unit[None, :, :], axis=2)  # (N', M)
        scores = np.sum(cos_sim >= inlier_threshold, axis=1).astype(np.float64)

        weights = scores if scores.sum() > 0 else np.ones_like(scores)

        mean = np.average(hyp_valid, axis=0, weights=weights)  # Eq.3
        centered = hyp_valid - mean
        cov = (centered.T * weights) @ centered / weights.sum()  # Eq.4
        cov = cov + np.eye(2) * 1e-6  # 공분산이 특이(singular)해지는 것 방지

        results.append(KeypointVote(mean=mean, covariance=cov, inlier_score=float(scores.max())))

    return results