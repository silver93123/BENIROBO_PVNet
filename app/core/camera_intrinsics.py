"""카메라 intrinsic 근사 역산 + 3D->2D 투영 공유 유틸리티.

이 레포는 카메라 캘리브레이션 파일을 따로 두지 않는다 - SDK가 왜곡보정까지
끝낸 organized PCD(H,W,3, mm)만 저장하기 때문(src/camera/femto_bolt.py,
lucid_helios.py 참고). 그래서 3D 점을 2D로 투영하려면 핀홀 근사 intrinsic이
필요한데, 이미 가진 (X,Y,Z)<->(u,v) 대응 수만 쌍에서 최소자승으로 fx,fy,cx,cy를
역산하면 별도 캘리브레이션 없이 충분히 정확한 근사치를 얻을 수 있다.

scripts/generate_rotation_labels.py, app/tabs/manual_labeling_tab.py 둘 다
이 모듈을 쓴다 - 같은 계산을 두 번 구현하지 않기 위해 분리해뒀다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_intrinsics_from_organized_pcd(
    points_organized: np.ndarray,
    valid_mask: np.ndarray,
    subsample: int = 20000,
) -> Tuple[float, float, float, float]:
    """organized PCD (H,W,3, mm)와 valid_mask -> 근사 핀홀 (fx, fy, cx, cy).

    u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy 는 각 축이 독립적인 1차
    선형회귀이므로 유효 픽셀 (X,Y,Z,u,v) 표본으로 최소자승 적합하면 된다.

    Args:
        points_organized: (H, W, 3) float, mm.
        valid_mask: (H, W) bool.
        subsample: 적합에 쓸 최대 픽셀 수 (전체 다 쓸 필요 없음, 속도용).

    Returns:
        (fx, fy, cx, cy).
    """
    vs, us = np.where(valid_mask)
    if vs.size == 0:
        raise ValueError("valid_mask에 유효 픽셀이 없음 - intrinsic 추정 불가")

    if vs.size > subsample:
        idx = np.random.default_rng(0).choice(vs.size, size=subsample, replace=False)
        vs, us = vs[idx], us[idx]

    xyz = points_organized[vs, us]  # (S, 3)
    z_ok = xyz[:, 2] > 1e-3  # Z<=0은 카메라 뒤/무효값이므로 제외
    x_over_z = xyz[z_ok, 0] / xyz[z_ok, 2]
    y_over_z = xyz[z_ok, 1] / xyz[z_ok, 2]
    us_f = us[z_ok].astype(np.float64)
    vs_f = vs[z_ok].astype(np.float64)

    fx, cx = np.polyfit(x_over_z, us_f, deg=1)
    fy, cy = np.polyfit(y_over_z, vs_f, deg=1)
    return float(fx), float(fy), float(cx), float(cy)


def project_points(points_cam_mm: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """카메라 좌표계 3D점(mm) -> 픽셀 좌표 (K,2), (x,y)=(col,row) 순서."""
    fx, fy, cx, cy = intrinsics
    z = points_cam_mm[:, 2]
    u = fx * points_cam_mm[:, 0] / z + cx
    v = fy * points_cam_mm[:, 1] / z + cy
    return np.stack([u, v], axis=1)