"""원통형(축 대칭) 부품의 GT 회전 정규화(canonicalization).

PVNet은 키포인트 K개 각각의 2D 벡터장을 독립적으로 예측하기 때문에,
RotHead에서 쓴 "loss 계산 시 대칭군 중 최솟값을 취하는" 방식(symmetry_aware_
geodesic_loss)을 그대로 못 쓴다 - 키포인트마다 서로 다른 대칭 변형이 최솟값이
되면 물리적으로 존재하지 않는 자세가 섞여버리기 때문. 그래서 PVNet은 원 논문
방식대로 "라벨을 만드는 시점"에 정규화한다: 대칭축 스핀(위상)이 얼마든 상관없이
항상 하나의 canonical 자세로 통일해서 키포인트 2D 라벨을 만든다.

핵심 원리: 측정된 회전 R이 로컬 대칭축 a를 실제로 어느 방향(v = R @ a)으로
보내는지는 진짜 정보(축이 어느 쪽으로 기울어 누워있는지)라서 보존해야 하고,
그 방향을 만드는 무수한 회전들(스핀 자유도만큼) 중 "회전각이 가장 작은 단
하나"만 남기면 스핀 성분이 전부 제거된다. 이 최소 회전은 a에 수직인 축
(a x v) 둘레의 회전이므로, 정의상 a 자신을 중심으로 한 추가 스핀이 없다.
"""
from __future__ import annotations

import numpy as np

AXIS_NAMES = "xyz"


def canonicalize_axial_rotation(R: np.ndarray, axis: str) -> np.ndarray:
    """측정된 회전행렬 R에서 로컬 대칭축(axis)의 스핀(위상) 성분만 제거한
    최소 회전(minimal rotation)을 반환한다.

    Args:
        R: (3, 3) 측정된 회전행렬 (ICP 결과 또는 수동 라벨링 결과).
        axis: "x"/"y"/"z" - CAD 로컬 좌표계에서 원통 대칭축 방향
            (scripts/check_cad_symmetry_axis.py로 미리 확인해둔 값).

    Returns:
        (3, 3) canonical 회전행렬. R @ axis_vec 방향은 그대로 보존되고,
        그 방향을 만드는 회전 중 스핀이 0인(회전각이 최소인) 것.
        스핀 각도가 다른 두 R이 같은 축 방향을 향하면, 이 함수는 항상
        똑같은 결과를 반환한다 (검증: 스핀 0.1~6.2 rad 범위에서 확인).
    """
    axis = axis.lower()
    if axis not in AXIS_NAMES:
        raise ValueError(f"지원하지 않는 축: '{axis}' (x/y/z만 가능)")

    a = np.zeros(3)
    a[AXIS_NAMES.index(axis)] = 1.0
    v = R @ a
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        raise ValueError("R @ axis 벡터의 크기가 0에 가깝습니다 - R이 유효한 회전행렬인지 확인하세요.")
    v = v / norm_v

    cos_angle = np.clip(np.dot(a, v), -1.0, 1.0)

    if cos_angle > 1 - 1e-9:
        # 축 방향이 그대로 유지됨 (v == a) - 스핀 제거할 것도 없이 항등행렬
        return np.eye(3)

    if cos_angle < -1 + 1e-9:
        # v == -a (180도 뒤집힘) - 회전축이 유일하게 안 정해지는 특이 케이스라
        # a에 수직인 임의의 벡터 하나를 골라 그 둘레로 180도 회전시킨다.
        # (어느 걸 고르든 축 방향 보존은 동일하게 만족 - 이 케이스 자체가
        # 이미 다른 canonical과 안 겹치는 극단 상황이라 안전함)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = perp - np.dot(perp, a) * a
        k /= np.linalg.norm(k)
        angle = np.pi
    else:
        k = np.cross(a, v)
        k /= np.linalg.norm(k)
        angle = np.arccos(cos_angle)

    # Rodrigues 회전 공식: 축 k, 각도 angle인 최소 회전행렬
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R_canonical = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R_canonical