"""정합(registration) 알고리즘 공통 인터페이스.

src/camera/base.py의 CameraBase 패턴을 그대로 따른다: 새 정합 알고리즘을
추가하고 싶으면 PoseEstimator를 상속받아 estimate() 하나만 구현하고,
registration/__init__.py의 create_registrator()에 타입 문자열 한 줄만
등록하면 된다. icp_test_tab.py 등 호출부는 이 인터페이스만 알면 되고
Open3D인지, FGR인지, TEASER++인지는 몰라도 된다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


@dataclass
class RegistrationResult:
    """정합 알고리즘 '한 번 호출'의 원시 결과 (알고리즘 무관 공통 포맷).

    icp_runner.ICPResult와는 레벨이 다르다 - ICPResult는 인스턴스 하나에
    대한 최종 결과(뒤집힘 보정 + 회전 구속 검사까지 끝난 후)를 담는
    상위 레벨 객체이고, RegistrationResult는 PoseEstimator.estimate()
    단일 호출의 결과만 담는 하위 레벨 객체다. correct_flipped_pose()가
    뒤집힘 감지 시 estimate()를 한 번 더 호출할 수 있는 이유가 이 분리
    때문이다.

    Attributes:
        T: source -> target 4x4 변환 행렬 (m 단위).
        fitness: Open3D 관례를 따름 - 높을수록 좋음 (0~1, inlier 비율).
        rmse: inlier 점들의 평균 제곱근 오차 (m 단위).
        stage_logs: 알고리즘 내부 단계별 진단 정보. 알고리즘마다 내용은
            달라도 되지만, UI 로그 테이블 호환을 위해 각 dict에 최소
            "fitness"/"rmse" 키는 채워두는 걸 권장.
    """
    T: np.ndarray
    fitness: float
    rmse: float
    stage_logs: list[dict] = field(default_factory=list)


class PoseEstimator(ABC):
    """CAD 점군을 씬(카메라) 점군에 정합하는 알고리즘의 공통 인터페이스.

    구현체 예시: open3d_multistage.Open3DMultiStageICP (기본, 현재 로직 이식).
    추가 예정: FPFH+RANSAC 기반 전역 정합 등 (T_init 없이도 동작 가능).
    """

    @abstractmethod
    def estimate(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                 T_init: np.ndarray) -> RegistrationResult:
        """source(CAD 가시면)를 target(씬 인스턴스 점군)에 정합한다.

        Args:
            source: CAD 가시면 점군. 원본(미변환) 좌표계, m 단위.
                voxel 다운샘플이나 normal 추정은 구현체 내부에서 알아서
                한다 - 호출부는 원본 밀도 그대로 넘기면 된다.
            target: 씬 인스턴스 점군. 카메라 좌표계, m 단위, outlier
                제거는 호출부(preprocessing)에서 이미 끝난 상태로 들어옴.
            T_init: source -> target 초기 변환 행렬 (4x4, m 단위).
                전역 정합(초기 자세가 필요 없는 알고리즘)을 구현할 경우
                이 인자를 무시해도 되지만, 인터페이스 시그니처는 그대로
                유지해서 호출부가 알고리즘 종류를 몰라도 되게 한다.

        Returns:
            RegistrationResult. T는 source -> target 방향으로 정의된다
            (icp_runner의 기존 관례와 동일).
        """
        raise NotImplementedError