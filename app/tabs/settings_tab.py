"""탭 0: 설정 - ICP 파라미터 + RTMDet-Ins 체크포인트/config 경로.

이전에는 이 값들이 ICP 관련 탭(수동 라벨링, PVNet 라벨 생성 등)마다 각각
편집 가능한 위젯으로 중복돼 있었고, 앱을 재시작하면 하드코딩된 기본값으로
리셋됐다. 이 탭이 유일한 편집 지점이 되고, [저장] 버튼을 눌러야
data/app_settings.json에 커밋된다 - 다른 탭들은 검출/ICP를 실행하는 매 순간
이 파일을 다시 읽으므로(app/core/settings_manager.py의 load_settings()),
이 탭을 열어두지 않았어도 항상 최신 저장값이 적용된다.

UI 구성(그리드 레이아웃 포함)은 원래 icp_workbench_base.py의
_build_icp_params_box()/_build_fgr_params_box()를 그대로 옮겨온 것이다 -
값의 의미/기본값/범위는 전혀 바뀌지 않았고, "매 프레임마다 즉석에서 조정하는
값"에서 "한 번 세팅해두고 계속 쓰는 값"으로 위치만 옮겼다.
"""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from app.core import settings_manager
from app.core.config_patcher import find_latest_best_checkpoint
from app.core.icp_runner import ICPParams
from app.core.paths import CAMERA_CONFIG_PATHS
from app.core.registration import AVAILABLE_REGISTRATION_TYPES


class SettingsTab(QWidget):
    log_message = pyqtSignal(str)
    LOG_PREFIX = "설정 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_from_settings()

    # ----------------------------------------------------------- UI 조립
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_detection_group())
        layout.addWidget(self._build_camera_group())
        layout.addWidget(self._build_icp_params_box())
        self.fgr_box = self._build_fgr_params_box()
        layout.addWidget(self.fgr_box)
        layout.addStretch(1)

        root.addWidget(scroll, stretch=1)

        bottom = QHBoxLayout()
        self.btn_save = QPushButton("저장")
        self.btn_save.setToolTip("여기서 바꾼 값은 이 버튼을 눌러야 다른 탭에 반영/영구 저장됩니다.")
        self.btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self.btn_save)

        self.btn_reset_defaults = QPushButton("전체 기본값으로")
        self.btn_reset_defaults.clicked.connect(self._on_reset_defaults)
        bottom.addWidget(self.btn_reset_defaults)

        bottom.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        bottom.addWidget(self.status_label)
        root.addLayout(bottom)

    # ----------------------------------------------------- RTMDet-Ins 검출
    def _build_detection_group(self) -> QGroupBox:
        group = QGroupBox("RTMDet-Ins 마스킹 검출")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("체크포인트 (.pth)"))
        ckpt_row = QHBoxLayout()
        self.checkpoint_edit = QLineEdit()
        ckpt_row.addWidget(self.checkpoint_edit, stretch=1)
        btn_browse_ckpt = QPushButton("선택")
        btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(btn_browse_ckpt)
        layout.addLayout(ckpt_row)

        layout.addWidget(QLabel("config (.py)"))
        cfg_row = QHBoxLayout()
        self.config_edit = QLineEdit()
        cfg_row.addWidget(self.config_edit, stretch=1)
        btn_browse_cfg = QPushButton("선택")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_row.addWidget(btn_browse_cfg)
        layout.addLayout(cfg_row)

        btn_auto = QPushButton("config 기준 최신 best 체크포인트 자동 감지")
        btn_auto.clicked.connect(self._on_autodetect_checkpoint)
        layout.addWidget(btn_auto)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("score threshold"))
        self.spin_score_threshold = QDoubleSpinBox()
        self.spin_score_threshold.setRange(0.0, 1.0)
        self.spin_score_threshold.setSingleStep(0.01)
        self.spin_score_threshold.setDecimals(2)
        thresh_row.addWidget(self.spin_score_threshold)
        thresh_hint = QLabel("각 탭의 'conf' 슬라이더 초기값으로 쓰입니다 (탭에서 세션 중 임시로 더 조정 가능).")
        thresh_hint.setStyleSheet("color: #888; font-size: 10px;")
        thresh_hint.setWordWrap(True)
        thresh_row.addWidget(thresh_hint, stretch=1)
        layout.addLayout(thresh_row)

        return group

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "체크포인트 선택", self.checkpoint_edit.text(), "PyTorch (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "config 파일 선택", self.config_edit.text(), "Python (*.py)")
        if path:
            self.config_edit.setText(path)

    def _on_autodetect_checkpoint(self) -> None:
        cfg_path = self.config_edit.text().strip()
        if not cfg_path:
            QMessageBox.warning(self, "알림", "먼저 config 경로를 지정하세요.")
            return
        best = find_latest_best_checkpoint(cfg_path)
        if best:
            self.checkpoint_edit.setText(best)
            self.status_label.setText(f"자동 감지됨: {best}")
        else:
            QMessageBox.information(self, "알림", f"config 기준으로 체크포인트를 찾지 못했습니다: {cfg_path}")

    # ----------------------------------------------------------------- 카메라
    def _build_camera_group(self) -> QGroupBox:
        """LiveCaptureICPTab(수동 라벨링/PVNet 라벨 생성 탭의 '촬영' 모드)이
        실제 촬영을 수행할 때마다 이 값을 다시 읽는다 - '설정' 탭이 유일한
        편집 지점이라는 원칙은 ICP 파라미터와 동일하다."""
        group = QGroupBox("카메라 (촬영 모드)")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("카메라 타입"))
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(list(CAMERA_CONFIG_PATHS.keys()))
        layout.addWidget(self.camera_type_combo)

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("평균화 프레임 수"))
        self.spin_avg_frames = QSpinBox()
        self.spin_avg_frames.setRange(1, 30)
        avg_row.addWidget(self.spin_avg_frames)
        layout.addLayout(avg_row)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("min_valid_ratio"))
        self.spin_min_valid_ratio = QDoubleSpinBox()
        self.spin_min_valid_ratio.setRange(0.05, 1.0)
        self.spin_min_valid_ratio.setSingleStep(0.05)
        self.spin_min_valid_ratio.setDecimals(2)
        ratio_row.addWidget(self.spin_min_valid_ratio)
        layout.addLayout(ratio_row)

        hint = QLabel(
            "평균화 프레임 수: 1=끔(기존과 동일). 5~10부터 depth 노이즈 감소가 체감됨 -\n"
            "커질수록 촬영 시간도 비례해서 늘어남.\n"
            "min_valid_ratio: 픽셀이 최종 유효로 인정되려면 N프레임 중 최소 이 비율\n"
            "이상에서 유효해야 함. 낮출수록 커버리지는 늘지만 노이즈 유입 위험도 증가."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return group

    # ----------------------------------------------------------- ICP 파라미터
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
            spin.setFixedWidth(90)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        def add_int(row, col, label, value, minimum, maximum, step=1):
            grid.addWidget(QLabel(label), row, col * 2)
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setFixedWidth(90)
            grid.addWidget(spin, row, col * 2 + 1)
            return spin

        self.spin_mask_erode = add_int(0, 0, "마스크 침식 px", defaults.mask_erode_px, 0, 10)
        self.spin_cad_ref_dist = add_double(0, 1, "카메라~부품 거리(m)", defaults.cad_hpr_ref_distance_m, 0.05, 5.0, 0.01, 3)
        erode_hint = QLabel("마스크 침식: depth 경계 노이즈 완충용 (0=끔).\n"
                             "카메라~부품 거리: CAD 가시면(보이는 면만 정합) 계산 기준값.")
        erode_hint.setStyleSheet("color: #888; font-size: 10px;")
        erode_hint.setWordWrap(True)
        grid.addWidget(erode_hint, 10, 0, 1, 4)

        self.spin_pc_upsample = add_int(13, 0, "PC 업샘플링 배수", defaults.pc_upsample_factor, 1, 8)
        grid.addWidget(QLabel("업샘플 방식"), 13, 2)
        self.combo_pc_upsample_method = QComboBox()
        self.combo_pc_upsample_method.addItems(["linear", "cubic", "probabilistic"])
        idx = self.combo_pc_upsample_method.findText(defaults.pc_upsample_method)
        self.combo_pc_upsample_method.setCurrentIndex(max(0, idx))
        self.combo_pc_upsample_method.setFixedWidth(100)
        grid.addWidget(self.combo_pc_upsample_method, 13, 3)
        upsample_hint = QLabel("1=끔(기존과 동일). linear/cubic은 격자 보간(매끈한 중간값 생성).\n"
                                "probabilistic은 다중 프레임 촬영(카메라 averaging)의 픽셀별 표준편차에서\n"
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

    # ----------------------------------------------------- 값 <-> 위젯 변환
    def _widgets_to_dict(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_edit.text().strip(),
            "config_path": self.config_edit.text().strip(),
            "score_threshold": self.spin_score_threshold.value(),
            "camera_type": self.camera_type_combo.currentText(),
            "averaging_num_frames": self.spin_avg_frames.value(),
            "averaging_min_valid_ratio": self.spin_min_valid_ratio.value(),
            "mask_erode_px": self.spin_mask_erode.value(),
            "cad_hpr_ref_distance_m": self.spin_cad_ref_dist.value(),
            "pc_upsample_factor": self.spin_pc_upsample.value(),
            "pc_upsample_method": self.combo_pc_upsample_method.currentText(),
            "outlier_nb_neighbors": self.spin_outlier_n.value(),
            "outlier_std_ratio": self.spin_outlier_std.value(),
            "fitness_threshold": self.spin_fitness.value(),
            "xyz_max_m": self.spin_xyz_max.value(),
            "roll_limit_deg": self.spin_roll_limit.value(),
            "pitch_limit_deg": self.spin_pitch_limit.value(),
            "yaw_limit_deg": self.spin_yaw_limit.value(),
            "init_roll_deg": self.spin_init_roll.value(),
            "init_pitch_deg": self.spin_init_pitch.value(),
            "init_yaw_deg": self.spin_init_yaw.value(),
            "cad_axis_roll_deg": self.spin_axis_roll.value(),
            "cad_axis_pitch_deg": self.spin_axis_pitch.value(),
            "cad_axis_yaw_deg": self.spin_axis_yaw.value(),
            "registration_type": self.combo_registration_type.currentText(),
            "fgr_voxel_size_m": self.spin_fgr_voxel.value(),
            "fgr_normal_radius_factor": self.spin_fgr_normal_factor.value(),
            "fgr_fpfh_radius_factor": self.spin_fgr_fpfh_factor.value(),
            "fgr_distance_threshold_factor": self.spin_fgr_dist_factor.value(),
            "fgr_refine_with_icp": self.check_fgr_refine.isChecked(),
            "fgr_refine_max_dist_m": self.spin_fgr_refine_dist.value(),
            "fgr_use_rotation_prior": self.check_fgr_rotation_prior.isChecked(),
            "fgr_max_rotation_deviation_deg": self.spin_fgr_max_dev.value(),
        }

    def _apply_dict_to_widgets(self, settings: dict) -> None:
        self.checkpoint_edit.setText(settings["checkpoint_path"])
        self.config_edit.setText(settings["config_path"])
        self.spin_score_threshold.setValue(settings["score_threshold"])
        idx = self.camera_type_combo.findText(settings["camera_type"])
        self.camera_type_combo.setCurrentIndex(max(0, idx))
        self.spin_avg_frames.setValue(settings["averaging_num_frames"])
        self.spin_min_valid_ratio.setValue(settings["averaging_min_valid_ratio"])
        self.spin_mask_erode.setValue(settings["mask_erode_px"])
        self.spin_cad_ref_dist.setValue(settings["cad_hpr_ref_distance_m"])
        self.spin_pc_upsample.setValue(settings["pc_upsample_factor"])
        idx = self.combo_pc_upsample_method.findText(settings["pc_upsample_method"])
        self.combo_pc_upsample_method.setCurrentIndex(max(0, idx))
        self.spin_outlier_n.setValue(settings["outlier_nb_neighbors"])
        self.spin_outlier_std.setValue(settings["outlier_std_ratio"])
        self.spin_fitness.setValue(settings["fitness_threshold"])
        self.spin_xyz_max.setValue(settings["xyz_max_m"])
        self.spin_roll_limit.setValue(settings["roll_limit_deg"])
        self.spin_pitch_limit.setValue(settings["pitch_limit_deg"])
        self.spin_yaw_limit.setValue(settings["yaw_limit_deg"])
        self.spin_init_roll.setValue(settings["init_roll_deg"])
        self.spin_init_pitch.setValue(settings["init_pitch_deg"])
        self.spin_init_yaw.setValue(settings["init_yaw_deg"])
        self.spin_axis_roll.setValue(settings["cad_axis_roll_deg"])
        self.spin_axis_pitch.setValue(settings["cad_axis_pitch_deg"])
        self.spin_axis_yaw.setValue(settings["cad_axis_yaw_deg"])
        idx = self.combo_registration_type.findText(settings["registration_type"])
        self.combo_registration_type.setCurrentIndex(max(0, idx))
        self.spin_fgr_voxel.setValue(settings["fgr_voxel_size_m"])
        self.spin_fgr_normal_factor.setValue(settings["fgr_normal_radius_factor"])
        self.spin_fgr_fpfh_factor.setValue(settings["fgr_fpfh_radius_factor"])
        self.spin_fgr_dist_factor.setValue(settings["fgr_distance_threshold_factor"])
        self.check_fgr_refine.setChecked(settings["fgr_refine_with_icp"])
        self.spin_fgr_refine_dist.setValue(settings["fgr_refine_max_dist_m"])
        self.check_fgr_rotation_prior.setChecked(settings["fgr_use_rotation_prior"])
        self.spin_fgr_max_dev.setValue(settings["fgr_max_rotation_deviation_deg"])
        self.fgr_box.setVisible(settings["registration_type"] == "fgr_global")

    # ----------------------------------------------------------- 로드/저장
    def _load_from_settings(self) -> None:
        settings = settings_manager.load_settings()
        self._apply_dict_to_widgets(settings)
        if settings["checkpoint_path"]:
            self.status_label.setText("저장된 설정을 불러왔습니다.")
        else:
            self.status_label.setText(
                "체크포인트가 아직 지정되지 않았습니다 - 위에서 선택하거나 자동 감지를 눌러주세요."
            )

    def _on_save(self) -> None:
        values = self._widgets_to_dict()
        settings_manager.save_settings(values)
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"저장됨 ({now}) - 다른 탭에서 다음 실행부터 이 값을 사용합니다.")
        self.log_message.emit(f"[{self.LOG_PREFIX}] 설정 저장됨 (체크포인트={values['checkpoint_path'] or '(없음)'})")

    def _on_reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self, "기본값으로 초기화",
            "체크포인트/config 경로를 포함한 모든 설정을 기본값으로 되돌립니다.\n"
            "(아직 저장은 안 됨 - 마음에 들면 [저장]을 눌러야 반영됩니다.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._apply_dict_to_widgets(settings_manager.DEFAULT_SETTINGS)
        self.status_label.setText("기본값으로 초기화됨 (아직 저장 안 됨)")