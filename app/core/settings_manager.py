"""애플리케이션 전역 설정 저장/로드.

ICP 파라미터, RTMDet-Ins 체크포인트/config 경로처럼 "부품이 바뀌어도 세션마다
새로 입력하기 귀찮은" 값들을 하나의 JSON 파일로 모아서 관리한다. 이전에는 이
값들이 ICPWorkbenchTab 좌측 패널에 탭마다(수동 라벨링, PVNet 라벨 생성 등)
각각 편집 가능한 위젯으로 떠 있었고, 앱을 재시작하면 전부 ICPParams()의
하드코딩된 기본값으로 리셋됐다 - 매번 다시 입력해야 했다.

이 모듈 도입 이후 흐름:
    1. "설정" 탭(settings_tab.py)에서만 값을 편집하고 [저장] 버튼으로 커밋한다.
    2. 다른 모든 탭(icp_workbench_base.py 및 그 서브클래스)은 검출/ICP를
       실행하는 매 순간 load_settings()를 다시 읽어 최신 값을 쓴다 - 탭을
       미리 열어두지 않았어도, 다른 탭에서 방금 저장을 눌렀어도 항상 최신
       값이 반영된다.
    3. 프로그램을 재시작해도 data/app_settings.json에 남아있는 값을 그대로
       불러온다.

data/ 폴더는 .gitignore에 이미 포함돼 있으므로(머신마다 다른 체크포인트
경로 등이 git에 안 들어가는 게 맞다), 이 설정 파일도 자연스럽게 커밋 대상에서
빠진다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"

# ICPParams()(app/core/icp_runner.py)의 기본값과 반드시 동기화할 것 - 그쪽이
# "진짜" 정의고, 여기 DEFAULT_SETTINGS는 설정 탭 UI가 초기값으로 쓰기 위한
# 미러다. 필드를 추가/변경하면 icp_runner.ICPParams도 같이 확인할 것.
DEFAULT_SETTINGS: dict[str, Any] = {
    # --- RTMDet-Ins 마스킹 검출 ---
    "checkpoint_path": "",
    "config_path": str(DEFAULT_CONFIG_PATH),
    "score_threshold": 0.3,

    # --- 카메라 (LiveCaptureICPTab의 '촬영' 모드가 사용) ---
    "camera_type": "lucid_helios",
    "averaging_num_frames": 8,       # 1이면 평균화 없음(기존과 동일)
    "averaging_min_valid_ratio": 0.6,

    # --- CAD 모델 ('CAD 모델 설정' 탭에서 관리) ---
    "cad_path": "",

    # --- ICP 공통 (등록 알고리즘 무관) ---
    "use_visible_face_filtering": True,
    "mask_erode_px": 1,
    "cad_hpr_ref_distance_m": 0.6,
    "pc_upsample_factor": 1,
    "pc_upsample_method": "linear",
    "outlier_nb_neighbors": 20,
    "outlier_std_ratio": 1.5,
    "fitness_threshold": 0.7,
    "xyz_max_m": 2.0,
    "roll_limit_deg": 180.0,
    "pitch_limit_deg": 180.0,
    "yaw_limit_deg": 180.0,
    "init_roll_deg": 180.0,
    "init_pitch_deg": 180.0,
    "init_yaw_deg": 90.0,
    "cad_axis_roll_deg": -90.0,
    "cad_axis_pitch_deg": 90.0,
    "cad_axis_yaw_deg": 90.0,
    "registration_type": "open3d_multistage",
    "use_pca_init": False,

    # --- FGR (registration_type == "fgr_global"일 때만 쓰임) ---
    "fgr_voxel_size_m": 0.005,
    "fgr_normal_radius_factor": 2.0,
    "fgr_fpfh_radius_factor": 5.0,
    "fgr_distance_threshold_factor": 3.0,
    "fgr_refine_with_icp": True,
    "fgr_refine_max_dist_m": 0.003,
    "fgr_use_rotation_prior": True,
    "fgr_max_rotation_deviation_deg": 60.0,
}

# settings dict 중 ICPParams(**kwargs)로 그대로 넘길 수 있는 키만 추림.
# checkpoint_path/config_path/score_threshold는 ICPParams 필드가 아니라
# Detector/FrameContext 쪽에서 쓰는 값이라 제외한다.
_ICP_PARAM_KEYS = [
    "use_visible_face_filtering",
    "mask_erode_px", "cad_hpr_ref_distance_m", "pc_upsample_factor",
    "pc_upsample_method", "outlier_nb_neighbors", "outlier_std_ratio",
    "fitness_threshold", "xyz_max_m", "roll_limit_deg", "pitch_limit_deg",
    "yaw_limit_deg", "init_roll_deg", "init_pitch_deg", "init_yaw_deg",
    "cad_axis_roll_deg", "cad_axis_pitch_deg", "cad_axis_yaw_deg",
    "registration_type", "use_pca_init", "fgr_voxel_size_m", "fgr_normal_radius_factor",
    "fgr_fpfh_radius_factor", "fgr_distance_threshold_factor",
    "fgr_refine_with_icp", "fgr_refine_max_dist_m", "fgr_use_rotation_prior",
    "fgr_max_rotation_deviation_deg",
]


def load_settings() -> dict[str, Any]:
    """저장된 설정을 읽어 DEFAULT_SETTINGS 위에 덮어씌운 dict를 반환.

    파일이 없거나(첫 실행) 손상됐으면(수동 편집 실수 등) 조용히 기본값으로
    시작한다 - 설정 파일 문제로 앱이 아예 못 뜨는 것보다는 기본값으로라도
    계속 쓸 수 있는 편이 낫다. 저장된 파일에 없는 키는 기본값을 쓰고,
    기본값에 없는(예전 버전이 남긴) 키는 무시한다 - 필드가 추가/삭제돼도
    항상 안전하게 로드된다.
    """
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.is_file():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key in DEFAULT_SETTINGS:
                if key in saved:
                    settings[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """settings(DEFAULT_SETTINGS와 같은 키 집합, 일부만 있어도 됨)를 JSON으로 저장.

    settings에 없는 키는 현재 저장돼 있던 값(없으면 기본값)을 유지한다 -
    호출부가 전체 키를 매번 다 채워 넘길 필요가 없다.
    """
    current = load_settings()
    current.update({k: v for k, v in settings.items() if k in DEFAULT_SETTINGS})
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)


def icp_params_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """settings dict -> ICPParams(**kwargs)에 바로 넣을 수 있는 부분집합."""
    return {key: settings[key] for key in _ICP_PARAM_KEYS if key in settings}