"""ICP 워크벤치 공유 베이스.

"4. ICP 정합 테스트"(세션 폴더에서 저장된 프레임 로드)와
"5. ICP 정합테스트(TCP)"(카메라로 즉시 촬영)의 공통 부분을 여기 담는다.
두 탭의 차이는 오직 "FrameContext를 어떻게 채우는가" 하나뿐이다:

    [세션 탭]  세션 폴더 선택 -> 저장된 .npy 로드          -> FrameContext
    [실시간 탭] create_camera()로 즉시 촬영(평균화 적용)   -> FrameContext
                                    │
                                    ▼
                (공통) pipeline_tabs.detect(ctx) / register(...)
                (공통) ICP 파라미터 / FGR 파라미터 / 결과 패널 / 3D 뷰어

서브클래스가 구현해야 하는 것은 딱 하나, `_build_acquisition_panel()`
(좌측 상단의 "프레임을 어떻게 얻을지" UI)이다. 새 프레임 획득 방식이
추가되면(예: 다중 뷰 스캔) 이 메서드 하나만 구현한 새 서브클래스를 만들면
되고, 이 베이스 파일은 손댈 필요가 없다.

서브클래스는 프레임이 준비될 때마다 아래 3개 속성을 채우고
`_on_new_frame_acquired(label)`를 호출해야 한다:
    self._current_image_path : str   (intensity 이미지 파일 경로)
    self._pcd_organized      : np.ndarray (H,W,3) mm
    self._valid_mask         : np.ndarray (H,W) bool
"""
from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt, QProcess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QLineEdit, QComboBox, QScrollArea, QFrame,
    QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
)

from app.core.detector import Detection
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.paths import DEFAULT_CONFIG_PATH, DEFAULT_CAD_DIR
from app.core.pipeline_context import FrameContext
from app.widgets.image_viewer import ImageViewer
from app.core import icp_runner
from app.core.icp_runner import ICPResult, ICPParams
from app.core.registration import AVAILABLE_REGISTRATION_TYPES
from app.tabs.icp_pipelines import AVAILABLE_ICP_PIPELINES

DEFAULT_SCORE_THRESHOLD = 0.3
CAD_EXTS = {".stl", ".ply", ".obj"}


class ICPWorkbenchTab(QWidget):
    log_message = pyqtSignal(str)

    #: 로그 메시지 접두어. 서브클래스가 오버라이드해서 탭을 구분한다
    #: (예: "ICP 탭" vs "ICP(TCP) 탭").
    LOG_PREFIX = "ICP 워크벤치"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_path: str | None = None       # 세션 기반 탭만 의미 있음 (로깅/로 참고용)
        self._current_frame: str | None = None       # 프레임 라벨 (로깅/파일명용)
        self._current_image_path: str | None = None  # intensity 이미지 경로 - 모든 서브클래스 공통
        self._pcd_organized: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        # 2026-07 추가: Monte Carlo 확률적 샘플링용 픽셀별 표준편차.
        # 세션 탭(SessionICPTab)은 항상 None으로 남겨둔다 (저장된 세션엔
        # 표준편차 정보가 없음). 다중 프레임 촬영 탭(LiveCaptureICPTab)만
        # 채운다.
        self._pcd_std: np.ndarray | None = None
        self._last_detections: list[Detection] = []
        self._last_icp_results: list[ICPResult] = []
        self._cad_pcd = None
        self._cad_visible_normal = None
        self._cad_visible_flipped = None
        self._cad_path_loaded: str | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None
        self._cad_init_rot_loaded: tuple[float, float, float] | None = None
        self._cad_ref_dist_loaded: float | None = None
        self._viewer_process: QProcess | None = None
        self._build_ui()
        self._prefill_latest_checkpoint()
        self._refresh_cad_list()

    # =============================================================
    # 서브클래스 필수 구현
    # =============================================================
    def _build_acquisition_panel(self) -> QWidget:
        """좌측 상단: "프레임을 어떻게 얻을지" UI. 서브클래스 필수 구현.

        세션 탭 -> 세션 폴더 선택 + 프레임 목록
        실시간 탭 -> 카메라 타입 선택 + 촬영 버튼 + 촬영 이력
        """
        raise NotImplementedError

    def _on_new_frame_acquired(self, frame_label: str) -> None:
        """서브클래스가 self._current_image_path/_pcd_organized/_valid_mask를
        채운 뒤 호출한다. 이전 검출/ICP 결과를 리셋하고 뷰어를 갱신한다."""
        self._current_frame = frame_label
        if self._current_image_path:
            self.image_viewer.load_image(self._current_image_path)
        self._reset_frame_state(keep_frame=True)
        self.log_message.emit(f"[{self.LOG_PREFIX}] 프레임 준비: {frame_label}")

    # =============================================================
    # UI 조립
    # =============================================================
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ------------------------------------------------------- 좌측
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(340)

        left.addWidget(self._build_acquisition_panel())

        left.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        left.addWidget(self.checkpoint_edit)
        ckpt_btn_row = QHBoxLayout()
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        ckpt_btn_row.addWidget(btn_browse_ckpt)
        ckpt_btn_row.addStretch(1)
        left.addLayout(ckpt_btn_row)

        left.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        left.addWidget(self.config_edit)
        cfg_btn_row = QHBoxLayout()
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_btn_row.addWidget(btn_browse_cfg)
        cfg_btn_row.addStretch(1)
        left.addLayout(cfg_btn_row)

        left.addWidget(QLabel("CAD 모델"))
        cad_row = QHBoxLayout()
        self.cad_combo = QComboBox()
        cad_row.addWidget(self.cad_combo, stretch=1)
        btn_refresh_cad = QPushButton("↻")
        btn_refresh_cad.setFixedWidth(28)
        btn_refresh_cad.setToolTip("data/cad/ 폴더 다시 스캔")
        btn_refresh_cad.clicked.connect(self._refresh_cad_list)
        cad_row.addWidget(btn_refresh_cad)
        left.addLayout(cad_row)
        cad_hint = QLabel(f"{DEFAULT_CAD_DIR} 폴더 스캔")
        cad_hint.setStyleSheet("color: #888; font-size: 10px;")
        cad_hint.setWordWrap(True)
        left.addWidget(cad_hint)
        left.addStretch(1)

        root.addWidget(left_widget)

        # ------------------------------------------------------- 중앙
        center = QVBoxLayout()

        self.pipeline_tabs = QTabWidget()
        for name, cls in AVAILABLE_ICP_PIPELINES:
            self.pipeline_tabs.addTab(cls(), name)
        center.addWidget(self.pipeline_tabs)

        run_row = QHBoxLayout()
        self.btn_run_detect = QPushButton("2D 검출 실행")
        self.btn_run_detect.clicked.connect(self._on_run_detection)
        run_row.addWidget(self.btn_run_detect)

        self.btn_run_icp = QPushButton("ICP 정합 실행")
        self.btn_run_icp.clicked.connect(self._on_run_icp)
        run_row.addWidget(self.btn_run_icp)

        run_row.addWidget(QLabel("conf"))
        from PyQt6.QtWidgets import QSlider
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_SCORE_THRESHOLD * 100))
        self.thresh_slider.setFixedWidth(90)
        self.thresh_label = QLabel(f"{DEFAULT_SCORE_THRESHOLD:.2f}")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_label.setText(f"{v / 100:.2f}")
        )
        run_row.addWidget(self.thresh_slider)
        run_row.addWidget(self.thresh_label)
        run_row.addStretch(1)
        center.addLayout(run_row)

        params_container = QWidget()
        params_layout = QVBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.fgr_box = self._build_fgr_params_box()
        params_layout.addWidget(self._build_icp_params_box())
        params_layout.addWidget(self.fgr_box)
        params_layout.addStretch(1)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_container)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        params_scroll.setMaximumHeight(320)
        self.params_scroll = params_scroll
        params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        center.addWidget(params_scroll)

        self.image_viewer = ImageViewer()
        center.addWidget(self.image_viewer, stretch=1)
        root.addLayout(center, stretch=2)

        # ------------------------------------------------------- 우측
        right = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(280)

        self.result_title_label = QLabel("ICP 결과")
        right.addWidget(self.result_title_label)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.addStretch(1)
        self.result_scroll.setWidget(self.result_container)
        right.addWidget(self.result_scroll, stretch=1)

        self.btn_open_viewer = QPushButton("3D 뷰어 열기")
        self.btn_open_viewer.clicked.connect(self._on_open_viewer)
        self.btn_open_viewer.setEnabled(False)
        right.addWidget(self.btn_open_viewer)

        root.addWidget(right_widget)

    def _build_icp_params_box(self) -> QGroupBox:
        defaults = ICPParams()
        box = QGroupBox("ICP 파라미터 (모든 파이프라인 공유)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        def add_double(row, col, label, value, minimum, maximum, step, decimals=3):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setFixedWidth(80)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        def add_int(row, col, label, value, minimum, maximum, step=1):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setFixedWidth(80)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.spin_mask_erode = add_int(0, 0, "마스크 침식 px", defaults.mask_erode_px, 0, 10)
        self.spin_cad_ref_dist = add_double(0, 1, "카메라~부품 거리(m)", defaults.cad_hpr_ref_distance_m, 0.05, 5.0, 0.01, 3)
        erode_hint = QLabel("마스크 침식: depth 경계 노이즈 완충용 (0=끔).\n"
                             "카메라~부품 거리: CAD 가시면(보이는 면만 정합) 계산 기준값.")
        erode_hint.setStyleSheet("color: #888; font-size: 10px;")
        erode_hint.setWordWrap(True)
        grid.addWidget(erode_hint, 10, 0, 1, 4)

        # 2026-07 추가: 포인트클라우드 업샘플링. 카메라 해상도 자체가 낮아
        # 인스턴스당 포인트가 부족할 때, 마스크 bbox 영역만 격자 보간으로
        # 밀도를 높여서 ICP 대응점을 늘린다 (extract_instance_points_mm 참고).
        self.spin_pc_upsample = add_int(13, 0, "PC 업샘플링 배수", defaults.pc_upsample_factor, 1, 8)
        grid.addWidget(QLabel("업샘플 방식"), 13, 2)
        self.combo_pc_upsample_method = QComboBox()
        self.combo_pc_upsample_method.addItems(["linear", "cubic", "probabilistic"])
        idx = self.combo_pc_upsample_method.findText(defaults.pc_upsample_method)
        self.combo_pc_upsample_method.setCurrentIndex(max(0, idx))
        self.combo_pc_upsample_method.setFixedWidth(90)
        grid.addWidget(self.combo_pc_upsample_method, 13, 3)
        upsample_hint = QLabel("1=끔(기존과 동일). linear/cubic은 격자 보간(매끈한 중간값 생성).\n"
                                "probabilistic은 다중 프레임 촬영(탭5, averaging)의 픽셀별 표준편차에서\n"
                                "Monte Carlo 재샘플링 - 표준편차 정보 없는 프레임(세션 로드 등)에선\n"
                                "자동으로 원본 추출로 폴백됨(개수 안 늘어남). cubic은 경계에서\n"
                                "오버슈트 위험 있어 마스크 침식 px를 같이 늘리는 걸 권장.")
        upsample_hint.setStyleSheet("color: #888; font-size: 10px;")
        upsample_hint.setWordWrap(True)
        grid.addWidget(upsample_hint, 14, 0, 1, 4)

        self.spin_outlier_n = add_int(1, 0, "outlier n", defaults.outlier_nb_neighbors, 1, 200)
        self.spin_outlier_std = add_double(1, 1, "outlier σ", defaults.outlier_std_ratio, 0.1, 10.0, 0.1, 2)

        self.spin_fitness = add_double(2, 0, "fitness ≥", defaults.fitness_threshold, 0.0, 1.0, 0.01, 2)
        self.spin_xyz_max = add_double(2, 1, "XYZ max (m)", defaults.xyz_max_m, 0.1, 10.0, 0.1, 2)

        self.spin_roll_limit = add_double(3, 0, "roll ± deg", defaults.roll_limit_deg, 0.0, 180.0, 1.0, 1)
        self.spin_pitch_limit = add_double(3, 1, "pitch ± deg", defaults.pitch_limit_deg, 0.0, 180.0, 1.0, 1)
        self.spin_yaw_limit = add_double(4, 0, "yaw ± deg", defaults.yaw_limit_deg, 0.0, 180.0, 1.0, 1)

        self.spin_init_roll = add_double(5, 0, "초기 roll deg", defaults.init_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_pitch = add_double(5, 1, "초기 pitch deg", defaults.init_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_yaw = add_double(6, 0, "초기 yaw deg", defaults.init_yaw_deg, -180.0, 180.0, 1.0, 1)

        self.spin_axis_roll = add_double(7, 0, "CAD 축보정 roll", defaults.cad_axis_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_pitch = add_double(7, 1, "CAD 축보정 pitch", defaults.cad_axis_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_yaw = add_double(8, 0, "CAD 축보정 yaw", defaults.cad_axis_yaw_deg, -180.0, 180.0, 1.0, 1)
        axis_hint = QLabel("ICP는 회전 없이 중심만 맞추고 시작합니다 - CAD가 실제 물체 방향과\n안 맞으면 여기부터 조정하세요 (CAD 바뀔 때마다 다시 맞춰야 함).")
        axis_hint.setStyleSheet("color: #888; font-size: 10px;")
        axis_hint.setWordWrap(True)
        grid.addWidget(axis_hint, 9, 0, 1, 4)

        btn_reset = QPushButton("기본값으로")
        btn_reset.clicked.connect(lambda: self._reset_icp_params(defaults))
        grid.addWidget(btn_reset, 8, 2, 1, 2)

        grid.addWidget(QLabel("정합 알고리즘 (fallback)"), 11, 0)
        self.combo_registration_type = QComboBox()
        self.combo_registration_type.addItems(AVAILABLE_REGISTRATION_TYPES)
        default_idx = self.combo_registration_type.findText(defaults.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, default_idx))
        grid.addWidget(self.combo_registration_type, 11, 1)
        self.combo_registration_type.currentTextChanged.connect(self._on_registration_type_changed)
        algo_hint = QLabel("파이프라인 탭이 initial pose를 직접 못 내는 경우 여기로 fallback합니다.\n"
                            "알고리즘별 세부 파라미터는 아래(open3d_multistage는 이 박스,\n"
                            "fgr_global은 바로 아래 'FGR 파라미터' 박스)에서 조정합니다.")
        algo_hint.setStyleSheet("color: #888; font-size: 10px;")
        algo_hint.setWordWrap(True)
        grid.addWidget(algo_hint, 12, 0, 1, 4)

        return box

    def _on_registration_type_changed(self, algo_type: str) -> None:
        self.fgr_box.setVisible(algo_type == "fgr_global")

    def _build_fgr_params_box(self) -> QGroupBox:
        defaults = ICPParams()
        box = QGroupBox("FGR 파라미터 (registration_type=fgr_global)")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        def add_double(row, col, label, value, minimum, maximum, step, decimals=4):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setFixedWidth(90)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.spin_fgr_voxel = add_double(0, 0, "voxel size (m)", defaults.fgr_voxel_size_m, 0.0005, 0.05, 0.0005, 4)
        self.spin_fgr_normal_factor = add_double(0, 1, "normal 반경 배수", defaults.fgr_normal_radius_factor, 0.5, 10.0, 0.5, 2)
        self.spin_fgr_fpfh_factor = add_double(1, 0, "FPFH 반경 배수", defaults.fgr_fpfh_radius_factor, 1.0, 20.0, 0.5, 2)
        self.spin_fgr_dist_factor = add_double(1, 1, "대응거리 배수", defaults.fgr_distance_threshold_factor, 0.5, 10.0, 0.5, 2)

        self.check_fgr_refine = QCheckBox("ICP로 정밀화 (refine_with_icp)")
        self.check_fgr_refine.setChecked(defaults.fgr_refine_with_icp)
        grid.addWidget(self.check_fgr_refine, 2, 0, 1, 2)
        self.spin_fgr_refine_dist = add_double(3, 0, "정밀화 max_dist (m)", defaults.fgr_refine_max_dist_m, 0.0005, 0.02, 0.0005, 4)

        self.check_fgr_rotation_prior = QCheckBox("회전 prior 검증 (use_rotation_prior)")
        self.check_fgr_rotation_prior.setChecked(defaults.fgr_use_rotation_prior)
        grid.addWidget(self.check_fgr_rotation_prior, 4, 0, 1, 2)
        self.spin_fgr_max_dev = add_double(5, 0, "최대 허용 편차 (deg)", defaults.fgr_max_rotation_deviation_deg, 0.0, 180.0, 5.0, 1)

        hint = QLabel("voxel size: 부품 크기에 맞춰 조정 (작은 부품은 3mm 이하 권장).\n"
                      "대응거리 배수: 이 값 * voxel size가 FGR이 대응점으로 인정하는 최대 거리.\n"
                      "회전 prior 검증: 결과 회전이 '초기 roll/pitch/yaw'와 너무 다르면\n"
                      "대칭/반복 형상 오탐으로 보고 초기값 기반 ICP로 대체합니다.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        grid.addWidget(hint, 6, 0, 1, 4)

        box.setVisible(defaults.registration_type == "fgr_global")
        return box

    def _reset_icp_params(self, defaults: ICPParams) -> None:
        self.spin_mask_erode.setValue(defaults.mask_erode_px)
        self.spin_cad_ref_dist.setValue(defaults.cad_hpr_ref_distance_m)
        self.spin_pc_upsample.setValue(defaults.pc_upsample_factor)
        idx = self.combo_pc_upsample_method.findText(defaults.pc_upsample_method)
        self.combo_pc_upsample_method.setCurrentIndex(max(0, idx))
        self.spin_outlier_n.setValue(defaults.outlier_nb_neighbors)
        self.spin_outlier_std.setValue(defaults.outlier_std_ratio)
        self.spin_fitness.setValue(defaults.fitness_threshold)
        self.spin_xyz_max.setValue(defaults.xyz_max_m)
        self.spin_roll_limit.setValue(defaults.roll_limit_deg)
        self.spin_pitch_limit.setValue(defaults.pitch_limit_deg)
        self.spin_yaw_limit.setValue(defaults.yaw_limit_deg)
        self.spin_init_roll.setValue(defaults.init_roll_deg)
        self.spin_init_pitch.setValue(defaults.init_pitch_deg)
        self.spin_init_yaw.setValue(defaults.init_yaw_deg)
        self.spin_axis_roll.setValue(defaults.cad_axis_roll_deg)
        self.spin_axis_pitch.setValue(defaults.cad_axis_pitch_deg)
        self.spin_axis_yaw.setValue(defaults.cad_axis_yaw_deg)
        idx = self.combo_registration_type.findText(defaults.registration_type)
        self.combo_registration_type.setCurrentIndex(max(0, idx))
        self.spin_fgr_voxel.setValue(defaults.fgr_voxel_size_m)
        self.spin_fgr_normal_factor.setValue(defaults.fgr_normal_radius_factor)
        self.spin_fgr_fpfh_factor.setValue(defaults.fgr_fpfh_radius_factor)
        self.spin_fgr_dist_factor.setValue(defaults.fgr_distance_threshold_factor)
        self.check_fgr_refine.setChecked(defaults.fgr_refine_with_icp)
        self.spin_fgr_refine_dist.setValue(defaults.fgr_refine_max_dist_m)
        self.check_fgr_rotation_prior.setChecked(defaults.fgr_use_rotation_prior)
        self.spin_fgr_max_dev.setValue(defaults.fgr_max_rotation_deviation_deg)

    def _build_icp_params(self) -> ICPParams:
        return ICPParams(
            mask_erode_px=self.spin_mask_erode.value(),
            cad_hpr_ref_distance_m=self.spin_cad_ref_dist.value(),
            pc_upsample_factor=self.spin_pc_upsample.value(),
            pc_upsample_method=self.combo_pc_upsample_method.currentText(),
            outlier_nb_neighbors=self.spin_outlier_n.value(),
            outlier_std_ratio=self.spin_outlier_std.value(),
            fitness_threshold=self.spin_fitness.value(),
            xyz_max_m=self.spin_xyz_max.value(),
            cad_axis_roll_deg=self.spin_axis_roll.value(),
            cad_axis_pitch_deg=self.spin_axis_pitch.value(),
            cad_axis_yaw_deg=self.spin_axis_yaw.value(),
            init_roll_deg=self.spin_init_roll.value(),
            init_pitch_deg=self.spin_init_pitch.value(),
            init_yaw_deg=self.spin_init_yaw.value(),
            roll_limit_deg=self.spin_roll_limit.value(),
            pitch_limit_deg=self.spin_pitch_limit.value(),
            yaw_limit_deg=self.spin_yaw_limit.value(),
            registration_type=self.combo_registration_type.currentText(),
            fgr_voxel_size_m=self.spin_fgr_voxel.value(),
            fgr_normal_radius_factor=self.spin_fgr_normal_factor.value(),
            fgr_fpfh_radius_factor=self.spin_fgr_fpfh_factor.value(),
            fgr_distance_threshold_factor=self.spin_fgr_dist_factor.value(),
            fgr_refine_with_icp=self.check_fgr_refine.isChecked(),
            fgr_refine_max_dist_m=self.spin_fgr_refine_dist.value(),
            fgr_use_rotation_prior=self.check_fgr_rotation_prior.isChecked(),
            fgr_max_rotation_deviation_deg=self.spin_fgr_max_dev.value(),
        )

    # ------------------------------------------------------ 체크포인트
    def _prefill_latest_checkpoint(self) -> None:
        if not DEFAULT_CONFIG_PATH.is_file():
            return
        cfg_path = str(DEFAULT_CONFIG_PATH)
        self.config_edit.setText(cfg_path)
        self.config_edit.setToolTip(cfg_path)
        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self.checkpoint_edit.setToolTip(best)
            self.log_message.emit(f"[{self.LOG_PREFIX}] 최신 best 체크포인트 자동 설정: {best}")

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)
            self.checkpoint_edit.setToolTip(path)

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", "", "Python (*.py)")
        if path:
            self.config_edit.setText(path)
            self.config_edit.setToolTip(path)

    # ------------------------------------------------------------ CAD
    def _refresh_cad_list(self) -> None:
        self.cad_combo.clear()
        if not DEFAULT_CAD_DIR.is_dir():
            self.log_message.emit(f"[{self.LOG_PREFIX}] CAD 폴더 없음: {DEFAULT_CAD_DIR}")
            return
        files = sorted(
            f for f in DEFAULT_CAD_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in CAD_EXTS
        )
        for f in files:
            self.cad_combo.addItem(f.name, str(f))
        self.log_message.emit(f"[{self.LOG_PREFIX}] CAD 폴더 스캔: {len(files)}개")

    # ------------------------------------------------------ 상태 리셋
    def _reset_frame_state(self, keep_frame: bool = False) -> None:
        if not keep_frame:
            self._current_frame = None
            self._current_image_path = None
            self._pcd_organized = None
            self._valid_mask = None
            self._pcd_std = None
        self._last_detections = []
        self._last_icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

    # ------------------------------------------------------ FrameContext
    def _build_context(self, cad_loaded: bool) -> FrameContext:
        return FrameContext(
            session_path=self._session_path or "",
            frame_name=self._current_frame or "",
            image_path=self._current_image_path,
            pcd_organized_mm=self._pcd_organized,
            valid_mask=self._valid_mask,
            pcd_std_mm=self._pcd_std,
            cad_pcd=self._cad_pcd if cad_loaded else None,
            cad_visible_normal=self._cad_visible_normal if cad_loaded else None,
            cad_visible_flipped=self._cad_visible_flipped if cad_loaded else None,
            checkpoint_path=self.checkpoint_edit.text().strip(),
            config_path=self.config_edit.text().strip(),
            score_threshold=self.thresh_slider.value() / 100.0,
        )

    # -------------------------------------------------------------- 2D 검출
    def _on_run_detection(self) -> None:
        if not self._current_image_path:
            QMessageBox.warning(self, "알림", "먼저 프레임을 준비하세요.")
            return
        if not self.checkpoint_edit.text().strip() or not self.config_edit.text().strip():
            QMessageBox.warning(self, "알림", "체크포인트와 config를 모두 지정하세요.")
            return

        active_tab = self.pipeline_tabs.currentWidget()
        ctx = self._build_context(cad_loaded=False)

        try:
            detections = active_tab.detect(ctx)
        except ImportError as exc:
            QMessageBox.critical(self, "추론 엔진 없음", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "추론 오류", str(exc))
            return

        n_with_pose = sum(1 for d in detections if getattr(d, "initial_pose", None) is not None)
        if n_with_pose:
            self.log_message.emit(
                f"[{self.LOG_PREFIX}] {active_tab.pipeline_name}: initial_pose 제공 "
                f"{n_with_pose}/{len(detections)}건 (나머지는 fallback 정합 알고리즘 사용)"
            )

        self._last_detections = detections
        self.image_viewer.set_detections(self._last_detections)
        self._last_icp_results = []
        self._clear_result_panel()
        self.btn_open_viewer.setEnabled(False)

        self.log_message.emit(f"[{self.LOG_PREFIX}] {active_tab.pipeline_name} 검출 완료: {len(detections)}건")
        if not detections:
            QMessageBox.information(self, "알림", "마스크가 있는 검출 결과가 없습니다.")

    # -------------------------------------------------------------- ICP 실행
    def _on_run_icp(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 2D 검출을 실행하세요.")
            return
        cad_index = self.cad_combo.currentIndex()
        if cad_index < 0:
            QMessageBox.warning(self, "알림", "CAD 모델을 선택하세요 (data/cad/ 폴더가 비어있지 않은지 확인).")
            return

        cad_path = self.cad_combo.itemData(cad_index)
        params = self._build_icp_params()
        try:
            self._ensure_cad_loaded(cad_path, params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "CAD 로드 오류", str(exc))
            return

        active_tab = self.pipeline_tabs.currentWidget()
        ctx = self._build_context(cad_loaded=True)

        self.log_message.emit(
            f"[{self.LOG_PREFIX}] {active_tab.pipeline_name} 파이프라인 ICP 정합 시작: "
            f"인스턴스 {len(self._last_detections)}개 "
            f"(fitness≥{params.fitness_threshold:.2f}, 회전구속 R±{params.roll_limit_deg:.0f} "
            f"P±{params.pitch_limit_deg:.0f} Y±{params.yaw_limit_deg:.0f}deg)"
        )

        results = active_tab.register(self._last_detections, ctx, params)

        for i, (det, result) in enumerate(zip(self._last_detections, results)):
            if result.ok:
                init_src = "파이프라인 제공" if getattr(det, "initial_pose", None) is not None else "fallback"
                self.log_message.emit(
                    f"[{self.LOG_PREFIX}]  obj{i} ✓ fitness={result.fitness:.3f} (init={init_src}) "
                    f"pick={tuple(round(v, 1) for v in result.pick_point_mm)} mm"
                )
            else:
                self.log_message.emit(f"[{self.LOG_PREFIX}]  obj{i} ✗ {result.error}")
                if result.stage_logs:
                    for sl in result.stage_logs:
                        voxel_str = f"{sl['voxel']}" if sl["voxel"] is not None else "원본밀도"
                        self.log_message.emit(
                            f"[{self.LOG_PREFIX}]     stage{sl['stage']}({sl['method']}, voxel={voxel_str}): "
                            f"src={sl['n_src']}pt tgt={sl['n_tgt']}pt "
                            f"fitness={sl['fitness']:.3f} rmse={sl['rmse']*1000:.2f}mm"
                        )

        self._last_icp_results = results
        self._render_result_panel(results)
        # 2026-07 변경: 실패한 인스턴스도 이제 뷰어에 별도 색으로 표시되므로
        # (build_scene_components의 Failed ICP Instances 레이어), 성공 여부와
        # 무관하게 결과가 하나라도 있으면 뷰어를 열 수 있게 한다.
        self.btn_open_viewer.setEnabled(len(results) > 0)

        n_ok = sum(r.ok for r in results)
        self.log_message.emit(f"[{self.LOG_PREFIX}] {active_tab.pipeline_name} ICP 완료: 성공 {n_ok}/{len(results)}")

    def _ensure_cad_loaded(self, cad_path: str, params: ICPParams) -> None:
        axis = params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis:
            self.log_message.emit(
                f"[{self.LOG_PREFIX}] CAD 로드 중: {cad_path} (축보정 R{axis[0]:.0f} P{axis[1]:.0f} Y{axis[2]:.0f}deg)"
            )
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis
            self._cad_visible_normal = None

        init_rot = params.init_rotation_deg
        ref_dist = params.cad_hpr_ref_distance_m
        if (self._cad_visible_normal is None
                or self._cad_init_rot_loaded != init_rot
                or self._cad_ref_dist_loaded != ref_dist):
            visible_normal, visible_flipped = icp_runner.build_visible_cad_pair(self._cad_pcd, params)
            total = len(self._cad_pcd.points)
            vis = len(visible_normal.points)
            MIN_VISIBLE_RATIO = 0.05
            if total == 0 or vis / total < MIN_VISIBLE_RATIO:
                self.log_message.emit(
                    f"[{self.LOG_PREFIX}] ⚠ CAD 가시면이 비정상적으로 적음({vis}/{total}점) - "
                    f"'카메라~부품 거리(m)' 값을 확인하세요. 일단 CAD 전체로 폴백합니다."
                )
                visible_normal = self._cad_pcd
                visible_flipped = self._cad_pcd
            self._cad_visible_normal = visible_normal
            self._cad_visible_flipped = visible_flipped
            self._cad_init_rot_loaded = init_rot
            self._cad_ref_dist_loaded = ref_dist
            self.log_message.emit(
                f"[{self.LOG_PREFIX}] CAD 가시면 준비 완료: 전체 {total}점 -> 가시 {vis}점 "
                f"({100*vis/total:.1f}%, 기준거리={ref_dist:.2f}m)"
            )

    # -------------------------------------------------------------- 결과 패널
    def _clear_result_panel(self) -> None:
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_result_panel(self, results: list[ICPResult]) -> None:
        self._clear_result_panel()
        for r in results:
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet(
                "QFrame { border: 1px solid #ddd; border-radius: 6px; padding: 4px; margin-bottom: 4px; }"
            )
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 6, 8, 6)

            title = QLabel(f"obj{r.instance_id}")
            title.setStyleSheet("font-weight: 600;")
            layout.addWidget(title)

            if r.ok:
                pose = r.pose["euler_deg"]
                pos = r.pick_point_mm
                status = QLabel(f"fitness {r.fitness:.3f}")
                status.setStyleSheet("color: #2a8a2a;")
                layout.addWidget(status)
                layout.addWidget(QLabel(f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm"))
                layout.addWidget(QLabel(
                    f"R{pose['roll_deg']:+.1f} P{pose['pitch_deg']:+.1f} Y{pose['yaw_deg']:+.1f} deg"
                ))
                if r.was_flipped:
                    flip_label = QLabel("뒤집힘 보정됨")
                    flip_label.setStyleSheet("color: #888; font-size: 10px;")
                    layout.addWidget(flip_label)
            else:
                status = QLabel(r.error or "실패")
                status.setStyleSheet("color: #c0392b;")
                status.setWordWrap(True)
                layout.addWidget(status)
                if r.fitness is not None:
                    layout.addWidget(QLabel(f"fitness {r.fitness:.3f}"))

            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    # -------------------------------------------------------------- 3D 뷰어
    def _on_open_viewer(self) -> None:
        if not self._last_icp_results or self._cad_pcd is None:
            return

        exclude_mask = None
        if self._last_detections and self._valid_mask is not None:
            exclude_mask = np.zeros_like(self._valid_mask, dtype=bool)
            for det in self._last_detections:
                if det.mask is not None:
                    exclude_mask |= det.mask.astype(bool)
        background_pcd = None
        if self._pcd_organized is not None and self._valid_mask is not None:
            background_pcd = icp_runner.build_background_pcd(
                self._pcd_organized, self._valid_mask, exclude_mask=exclude_mask,
                color_mode="height",
            )

        # 2026-07 개편: 레이어를 하나로 합치지 않고 이름별로 분리해서 각각
        # PLY로 저장 - 뷰어에서 레이어별 체크박스로 켜고 끌 수 있게 하기 위함
        # (app/core/icp_viewer.py의 매니페스트 방식 참고).
        components = icp_runner.build_scene_components(self._last_icp_results, self._cad_pcd, background_pcd)
        if not components:
            QMessageBox.information(self, "알림", "표시할 포인트클라우드가 없습니다 (배경/검출 결과 모두 비어있음).")
            return

        import open3d as o3d

        view_dir = Path(tempfile.gettempdir()) / f"icp_view_{self._current_frame or 'frame'}"
        view_dir.mkdir(parents=True, exist_ok=True)

        layers = []
        for i, (name, pcd) in enumerate(components.items()):
            filename = f"layer_{i}.ply"
            o3d.io.write_point_cloud(str(view_dir / filename), pcd, write_ascii=False)
            layers.append({"name": name, "file": filename, "visible": True})

        manifest_path = view_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"layers": layers}, f, ensure_ascii=False, indent=2)

        if self._viewer_process is not None and self._viewer_process.state() != QProcess.ProcessState.NotRunning:
            self._viewer_process.kill()

        self._viewer_process = QProcess(self)
        self._viewer_process.start(
            sys.executable,
            ["-m", "app.core.icp_viewer", str(manifest_path), "--title", f"ICP 결과 - {self._current_frame}"],
        )
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 3D 뷰어 실행: {manifest_path} ({len(layers)}개 레이어)"
        )