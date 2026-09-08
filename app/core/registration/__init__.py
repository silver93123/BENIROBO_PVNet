"""정합(registration) 알고리즘 추상화 계층.

src/camera의 CameraBase/create_camera 패턴과 동일하게 맞췄다:

    from app.core.registration import create_registrator

    cfg = {"type": "open3d_multistage", "params": {"icp_stages": [...]}}
    estimator = create_registrator(cfg)
    result = estimator.estimate(source_pcd, target_pcd, T_init)

새 알고리즘을 추가할 때:
    1. PoseEstimator를 상속받는 클래스를 이 폴더에 새 파일로 작성
       (open3d_multistage.py 참고).
    2. 아래 create_registrator()에 타입 문자열 분기 한 줄 추가.
    3. (선택) icp_test_tab.py의 알고리즘 콤보박스에 타입 문자열 추가.
그 외에는 아무 것도 손댈 필요 없다 - 호출부는 PoseEstimator 인터페이스만
보고 동작한다.
"""

from .base import PoseEstimator, RegistrationResult

# UI(icp_test_tab.py)의 알고리즘 콤보박스가 이 목록을 그대로 채운다.
# 새 알고리즘을 추가하면 create_registrator()의 분기뿐 아니라 여기에도
# 문자열을 추가해야 콤보박스에 나타난다 (일부러 자동 탐색 안 하고 명시
# 목록으로 관리 - 어떤 알고리즘이 "정식으로 노출"됐는지 한눈에 보이게).
AVAILABLE_REGISTRATION_TYPES = ["open3d_multistage", "fgr_global"]

__all__ = [
    "PoseEstimator", "RegistrationResult",
    "create_registrator", "AVAILABLE_REGISTRATION_TYPES",
]


def create_registrator(config: dict) -> PoseEstimator:
    """설정 dict로부터 정합 알고리즘 인스턴스 생성 (factory).

    Args:
        config: {"type": <알고리즘 이름>, "params": {<생성자 kwargs>}} 형태.
            "params"는 생략 가능 (알고리즘 기본값 사용).

    Returns:
        PoseEstimator 하위 클래스 인스턴스.

    Raises:
        ValueError: 지원하지 않는 알고리즘 타입.
    """
    algo_type = config.get("type", "open3d_multistage").lower()
    params = config.get("params", {}) or {}

    if algo_type == "open3d_multistage":
        from .open3d_multistage import Open3DMultiStageICP
        return Open3DMultiStageICP(**params)

    if algo_type == "fgr_global":
        from .fgr_global import FGRGlobalRegistration
        return FGRGlobalRegistration(**params)

    raise ValueError(
        f"지원하지 않는 정합 알고리즘: '{algo_type}'. "
        f"지원: {AVAILABLE_REGISTRATION_TYPES}"
    )