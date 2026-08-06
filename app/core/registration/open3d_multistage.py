"""Open3D coarse-to-fine point-to-point -> point-to-plane ICP.

icp_runner.py에 있던 run_icp_multistage()의 1:1 이식이다. 로직/파라미터
기본값 모두 동일하게 유지했다 - 이 리팩터링 시점에 결과가 달라지면 안 된다.

  회귀 확인 방법: 리팩터링 전/후 icp_runner.run_icp_for_instance()를 같은
  CAD/씬 입력으로 돌려서 T/fitness/rmse/stage_logs가 동일한지 비교.

voxel이 있는(coarse/mid) stage는 point-to-point를 쓴다 - 초기 정렬이 많이
틀어져 있을 수 있는 초반 단계에서 point-to-plane은 선형화 근사 때문에
발산하기 쉽다. voxel=None인 마지막(fine) stage에서만 point-to-plane으로
정밀화한다 (이미 대략 맞춰진 상태라 안전함).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .base import PoseEstimator, RegistrationResult


@dataclass
class Open3DMultiStageParams:
    """기존 icp_runner.ICPParams 중 '이 알고리즘 고유' 파라미터만 모은 것.

    나머지(outlier 제거, 마스크 침식, CAD 가시면, 회전 구속, xyz 범위 등)는
    알고리즘이 바뀌어도 그대로 쓰이는 공통 파라미터라 여기 넣지 않았다 -
    registration/preprocessing.py 쪽에 남는다 (다음 단계에서 이동 예정).
    """
    icp_stages: list[dict] = field(default_factory=lambda: [
        {"voxel": 0.006, "max_dist": 0.020, "max_iter": 100},
        {"voxel": 0.003, "max_dist": 0.010, "max_iter": 100},
        {"voxel": None,  "max_dist": 0.003, "max_iter": 50},
    ])
    normal_radius_factor: float = 2.5
    normal_radius_final: float = 0.004  # m


def _prep_stage_cloud(base_pcd: o3d.geometry.PointCloud, voxel: float | None,
                       normal_radius: float) -> o3d.geometry.PointCloud:
    cloud = base_pcd.voxel_down_sample(voxel) if voxel is not None else copy.deepcopy(base_pcd)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    cloud.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
    return cloud


class Open3DMultiStageICP(PoseEstimator):
    """기존 icp_runner.run_icp_multistage()와 동일한 다단계 ICP.

    사용 예:
        estimator = Open3DMultiStageICP()                      # 기본 파라미터
        estimator = Open3DMultiStageICP(icp_stages=[...])       # 커스텀 stage
        result = estimator.estimate(cad_visible, scene_pcd, T_init)
    """

    def __init__(self, icp_stages: list[dict] | None = None,
                 normal_radius_factor: float = 2.5,
                 normal_radius_final: float = 0.004):
        defaults = Open3DMultiStageParams()
        self.params = Open3DMultiStageParams(
            icp_stages=icp_stages if icp_stages is not None else defaults.icp_stages,
            normal_radius_factor=normal_radius_factor,
            normal_radius_final=normal_radius_final,
        )

    def estimate(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                 T_init: np.ndarray) -> RegistrationResult:
        p = self.params
        T = T_init.copy()
        stage_logs: list[dict] = []

        for i, stage in enumerate(p.icp_stages):
            voxel = stage.get("voxel")
            n_radius = (voxel * p.normal_radius_factor) if voxel is not None else p.normal_radius_final
            src = _prep_stage_cloud(source, voxel, n_radius)
            tgt = _prep_stage_cloud(target, voxel, n_radius)

            estimation = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
                if voxel is None else
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, stage["max_dist"], T, estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage["max_iter"]),
            )
            T = np.asarray(res.transformation)
            stage_logs.append({
                "stage": i, "voxel": voxel, "max_dist": stage["max_dist"],
                "n_src": len(src.points), "n_tgt": len(tgt.points),
                "fitness": float(res.fitness), "rmse": float(res.inlier_rmse),
                "method": "point-to-plane" if voxel is None else "point-to-point",
            })

        final = o3d.pipelines.registration.evaluate_registration(
            source, target, p.icp_stages[-1]["max_dist"], T)
        return RegistrationResult(T=T, fitness=float(final.fitness),
                                   rmse=float(final.inlier_rmse), stage_logs=stage_logs)