"""FPFH 특징 기반 FGR(Fast Global Registration) - 초기 자세 없이 동작하는 전역 정합.

Open3DMultiStageICP(로컬 정합, T_init 필수)와의 핵심 차이:
    - point-to-point/point-to-plane ICP는 "이미 대충 맞춰진 상태에서 미세
      조정"하는 로컬 최적화라, T_init이 실제 자세와 많이 어긋나 있으면
      전혀 다른 국소 최적점(local minimum)에 수렴해 완전히 틀린 자세가
      나올 수 있다.
    - FGR은 점군 표면의 국소 형상 특징(FPFH: Fast Point Feature Histogram)을
      매칭해서 초기 자세 추정 없이 전역적으로 대응점을 찾는다. 대신 정밀도는
      ICP보다 거칠어서(voxel 단위 정합), 보통 "FGR로 대충 맞추고 → ICP로
      정밀화"하는 2단계 파이프라인으로 쓰인다. 이 클래스는 refine_with_icp
      옵션으로 그 2단계를 하나로 묶어서 제공한다 (기본 켜짐).

언제 쓰나:
    - build_icp_init()이 만드는 "물체 중심만 맞추고 회전은 UI에서 준 고정값"
      초기 자세가 실제 자세와 많이 다를 때(부품이 예상 못한 각도로 놓였을 때)
      open3d_multistage는 엉뚱한 곳에 수렴하기 쉽다. FGR은 이런 케이스에서
      대안이 될 수 있다.
    - 반대로 초기 자세가 이미 잘 맞는 일반적인 빈피킹 상황에서는
      open3d_multistage가 더 빠르고 정밀하다 - FGR을 기본값으로 바꿀
      필요는 없다.

참고 (검증된 출처):
    - Zhou, Park, Koltun, "Fast Global Registration", ECCV 2016.
    - Open3D 공식 문서: https://www.open3d.org/docs/latest/tutorial/Advanced/fast_global_registration.html
    - Open3D 공식 문서: https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .base import PoseEstimator, RegistrationResult
from .open3d_multistage import _prep_stage_cloud


@dataclass
class FGRParams:
    """FGR 자체 파라미터. Open3DMultiStageParams와 마찬가지로 이 알고리즘에서만
    쓰이고, outlier 제거/마스크 침식 등 공통 전처리 파라미터(ICPParams)와는
    독립적이다."""
    voxel_size_m: float = 0.005          # 다운샘플 + FPFH 계산 기준 voxel 크기
    normal_radius_factor: float = 2.0     # normal 추정 반경 = voxel_size_m * 이 값
    fpfh_radius_factor: float = 5.0       # FPFH 특징 계산 반경 = voxel_size_m * 이 값
    distance_threshold_factor: float = 3.0  # FGR 대응점 인정 거리 = voxel_size_m * 이 값
    # centering 이후에도 "CAD 가시면 중심"과 "씬 인스턴스 중심"이 완전히
    # 일치하진 않는다(부분 관측 시야 차이 때문) - 남는 오차를 흡수할 여유를
    # 1.5 -> 3.0으로 넉넉하게 잡았다. 그래도 정합이 안 되면 이 값을 더
    # 키워보는 게 첫 번째로 시도해볼 조정이다.
    refine_with_icp: bool = True          # FGR 결과를 point-to-plane ICP로 한 번 더 정밀화할지
    refine_max_dist_m: float = 0.003      # 정밀화 단계의 max_correspondence_distance


def _compute_fpfh(pcd_down: o3d.geometry.PointCloud, fpfh_radius: float):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100)
    )


class FGRGlobalRegistration(PoseEstimator):
    """FPFH 기반 FGR. T_init을 요구하지 않는다 (인터페이스 시그니처상 받긴
    하지만 내부적으로 사용하지 않는다 - 완전히 무시).

    사용 예:
        estimator = FGRGlobalRegistration()                       # 기본 파라미터
        estimator = FGRGlobalRegistration(voxel_size_m=0.004, refine_with_icp=False)
        result = estimator.estimate(cad_visible, scene_pcd, np.eye(4))  # T_init 무시됨
    """

    def __init__(self, voxel_size_m: float = 0.005, normal_radius_factor: float = 2.0,
                 fpfh_radius_factor: float = 5.0, distance_threshold_factor: float = 1.5,
                 refine_with_icp: bool = True, refine_max_dist_m: float = 0.003):
        self.params = FGRParams(
            voxel_size_m=voxel_size_m,
            normal_radius_factor=normal_radius_factor,
            fpfh_radius_factor=fpfh_radius_factor,
            distance_threshold_factor=distance_threshold_factor,
            refine_with_icp=refine_with_icp,
            refine_max_dist_m=refine_max_dist_m,
        )

    def estimate(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                 T_init: np.ndarray) -> RegistrationResult:
        p = self.params
        stage_logs: list[dict] = []

        normal_radius = p.voxel_size_m * p.normal_radius_factor
        fpfh_radius = p.voxel_size_m * p.fpfh_radius_factor
        distance_threshold = p.voxel_size_m * p.distance_threshold_factor

        # --- 버그 수정 (2026-07): FGR 실행 전 centering ---------------------
        # source(CAD 가시면)는 CAD 로컬 좌표계 원본(icp_runner.build_visible_cad
        # 참고), target(씬 인스턴스)은 카메라 좌표계 - 둘 사이에 보통 수백mm
        # 오프셋이 있다. FGR의 maximum_correspondence_distance는 GNC 최적화의
        # "시작 스케일"로 쓰이는데, 이 값(voxel_size_m 몇 배 = mm 단위)이
        # 실제 오프셋보다 훨씬 작으면 최적화가 시작부터 모든 대응점을
        # outlier로 취급해서 사실상 아무것도 못 맞춘다 (정합 실패의 원인).
        # -> 두 점군을 각자 중심으로 원점 이동시켜서 FGR이 풀어야 하는 남은
        #    오프셋을 "물체 크기 수준(몇 mm~몇 cm)"으로 줄여준다. 이후 실제
        #    좌표계로는 아래 T_final 조합식으로 되돌린다.
        t_src = np.asarray(source.get_center())
        t_tgt = np.asarray(target.get_center())
        source_centered = copy.deepcopy(source).translate(-t_src)
        target_centered = copy.deepcopy(target).translate(-t_tgt)

        src_down = _prep_stage_cloud(source_centered, p.voxel_size_m, normal_radius)
        tgt_down = _prep_stage_cloud(target_centered, p.voxel_size_m, normal_radius)
        src_fpfh = _compute_fpfh(src_down, fpfh_radius)
        tgt_fpfh = _compute_fpfh(tgt_down, fpfh_radius)

        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
        fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down, tgt_down, src_fpfh, tgt_fpfh, option
        )
        T_rel = np.asarray(fgr_result.transformation)

        # centered 좌표계에서 구한 T_rel을 원래 좌표계로 되돌린다:
        #   target_point = T_to_tgt @ T_rel @ T_to_origin_src @ source_point
        T_to_origin_src = np.eye(4); T_to_origin_src[:3, 3] = -t_src
        T_to_tgt = np.eye(4); T_to_tgt[:3, 3] = t_tgt
        T = T_to_tgt @ T_rel @ T_to_origin_src

        stage_logs.append({
            "stage": "fgr", "voxel": p.voxel_size_m, "max_dist": distance_threshold,
            "n_src": len(src_down.points), "n_tgt": len(tgt_down.points),
            "fitness": float(fgr_result.fitness), "rmse": float(fgr_result.inlier_rmse),
            "method": "fgr (fpfh feature matching, centered, no T_init)",
        })

        # FGR 단독 결과는 voxel 단위 정밀도라 거칠다. 실사용 가능한 정밀도가
        # 필요하면 대부분 이 정밀화 단계가 필요하다 (기본 켜짐).
        if p.refine_with_icp:
            n_radius_fine = p.refine_max_dist_m * 2.0
            src_fine = _prep_stage_cloud(source, None, n_radius_fine)
            tgt_fine = _prep_stage_cloud(target, None, n_radius_fine)
            icp_result = o3d.pipelines.registration.registration_icp(
                src_fine, tgt_fine, p.refine_max_dist_m, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            T = np.asarray(icp_result.transformation)
            stage_logs.append({
                "stage": "fgr_icp_refine", "voxel": None, "max_dist": p.refine_max_dist_m,
                "n_src": len(src_fine.points), "n_tgt": len(tgt_fine.points),
                "fitness": float(icp_result.fitness), "rmse": float(icp_result.inlier_rmse),
                "method": "point-to-plane",
            })

        final = o3d.pipelines.registration.evaluate_registration(
            source, target, p.refine_max_dist_m if p.refine_with_icp else distance_threshold, T
        )
        return RegistrationResult(T=T, fitness=float(final.fitness),
                                   rmse=float(final.inlier_rmse), stage_logs=stage_logs)