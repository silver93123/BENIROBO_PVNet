"""탭 "CAD 모델 설정" - CAD 선택, ICP 초기 자세/가시면 파라미터, 그리고
"초기 자세가 실제로 맞는지" 카메라 포인트클라우드에 직접 대조해보는 기능.

이전에는 CAD 경로가 각 ICP 탭(수동 라벨링/PVNet 라벨 생성)의 좌측 패널에,
초기 roll/pitch/yaw·CAD 축보정·카메라~부품 거리·HPR on/off는 "설정" 탭에
흩어져 있었다. CAD 관련 값들은 서로 강하게 얽혀 있고("초기 자세"가 HPR
가시면 판정 기준이기도 함) CAD가 바뀔 때마다 세트로 다시 맞춰야 하는
값들이라, 한 탭으로 모았다.

핵심 기능(신규): "초기 자세 vs 카메라 포인트클라우드 비교" - 지금까지는
초기 roll/pitch/yaw 값이 실제 부품이 카메라에 놓인 방향과 맞는지 확인할
방법이 "CAD 가시면 미리보기"(그것도 간접적)뿐이었다. 이제 카메라로 직접
찍거나(또는 저장된 세션에서 불러와서) 그 포인트클라우드 위에 "지금 입력된
초기 roll/pitch/yaw로 CAD를 놓으면 이렇게 보인다"를 겹쳐서 3D로 바로 볼 수
있다 - 값을 저장하기 전, 화면에 입력만 한 상태에서도 즉시 비교 가능하다
(_params_from_widgets가 위젯의 현재 값을 그대로 씀).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from app.core import icp_runner, settings_manager
from app.core.icp_runner import ICPParams
from app.core.paths import CAD_EXTS, CAMERA_CONFIG_PATHS, DEFAULT_CAD_DIR, DEFAULT_DATASET_ROOT
from app.tabs.live_capture_icp_tab import _usable_session_frames
from app.tabs.viewer_launcher import Viewer3DMixin


class CADModelSettingsTab(Viewer3DMixin, QWidget):
    log_message = pyqtSignal(str)
    #: '카메라 설정 열기' 버튼 -> main_window.py가 받아서 '설정' 탭으로 전환.
    open_settings_requested = pyqtSignal()
    LOG_PREFIX = "CAD 모델 설정 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewer_process = None

        # CAD 전체(축보정만 적용) 캐시 - 이 탭은 HPR 가시면 서브셋까지는
        # 캐싱하지 않는다(미리보기/비교 버튼 누를 때 그때그때 계산 - 사용
        # 빈도가 낮아 캐싱 이득이 적음).
        self._cad_pcd = None
        self._cad_path_loaded: str | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None

        # 참조 포인트클라우드 (촬영 또는 세션에서 불러온 것 하나만 보관 -
        # 이 탭은 라벨링 이력이 아니라 "지금 이 순간의 비교"가 목적).
        self._ref_pcd_organized: np.ndarray | None = None
        self._ref_valid_mask: np.ndarray | None = None

        self._build_ui()
        self._load_from_settings()
        self._refresh_cad_list()

    # ----------------------------------------------------------- UI 조립
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_cad_select_group())
        layout.addWidget(self._build_pose_params_group())
        layout.addWidget(self._build_visibility_preview_group())
        layout.addWidget(self._build_reference_compare_group())
        layout.addStretch(1)

        root.addWidget(scroll, stretch=1)

        bottom = QHBoxLayout()
        self.btn_save = QPushButton("저장")
        self.btn_save.setToolTip("CAD 선택 + 초기 자세/HPR 값을 저장합니다. 다른 탭은 다음 실행부터 반영됩니다.")
        self.btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self.btn_save)

        self.btn_reset_defaults = QPushButton("기본값으로")
        self.btn_reset_defaults.clicked.connect(self._on_reset_defaults)
        bottom.addWidget(self.btn_reset_defaults)

        bottom.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        bottom.addWidget(self.status_label)
        root.addLayout(bottom)

    # --------------------------------------------------------- CAD 선택
    def _build_cad_select_group(self) -> QGroupBox:
        group = QGroupBox("CAD 모델")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.cad_combo = QComboBox()
        row.addWidget(self.cad_combo, stretch=1)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip(f"{DEFAULT_CAD_DIR} 폴더 다시 스캔")
        btn_refresh.clicked.connect(self._refresh_cad_list)
        row.addWidget(btn_refresh)
        layout.addLayout(row)

        hint = QLabel(f"{DEFAULT_CAD_DIR} 폴더 스캔. 여기서 고른 CAD를 모든 ICP 탭이 공유합니다.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return group

    def _refresh_cad_list(self) -> None:
        current = self.cad_combo.currentData()
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
        if current:
            idx = self.cad_combo.findData(current)
            if idx >= 0:
                self.cad_combo.setCurrentIndex(idx)
        self.log_message.emit(f"[{self.LOG_PREFIX}] CAD 폴더 스캔: {len(files)}개")

    # --------------------------------------------------- 초기 자세/HPR 파라미터
    def _build_pose_params_group(self) -> QGroupBox:
        defaults = ICPParams()
        group = QGroupBox("초기 자세 / 가시면(HPR) 파라미터")
        grid = QGridLayout(group)
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

        self.spin_init_roll = add_double(0, 0, "초기 roll deg", defaults.init_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_pitch = add_double(0, 1, "초기 pitch deg", defaults.init_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_init_yaw = add_double(1, 0, "초기 yaw deg", defaults.init_yaw_deg, -180.0, 180.0, 1.0, 1)
        init_hint = QLabel(
            "ICP는 회전 없이 위치(중심)만 맞추고 시작합니다 - 회전은 이 값을\n"
            "그대로 씁니다. 아래 '참조 포인트클라우드와 비교'로 실제 놓인\n"
            "방향과 맞는지 확인하세요."
        )
        init_hint.setStyleSheet("color: #888; font-size: 10px;")
        init_hint.setWordWrap(True)
        grid.addWidget(init_hint, 2, 0, 1, 4)

        self.spin_axis_roll = add_double(3, 0, "CAD 축보정 roll", defaults.cad_axis_roll_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_pitch = add_double(3, 1, "CAD 축보정 pitch", defaults.cad_axis_pitch_deg, -180.0, 180.0, 1.0, 1)
        self.spin_axis_yaw = add_double(4, 0, "CAD 축보정 yaw", defaults.cad_axis_yaw_deg, -180.0, 180.0, 1.0, 1)
        axis_hint = QLabel(
            "CAD 파일 자체의 로컬 좌표축이 이상한 방향을 향하고 있을 때 여기서\n"
            "한 번 바로잡습니다(CAD 로드 시 1회 적용, 위 '초기 자세'와는 별개)."
        )
        axis_hint.setStyleSheet("color: #888; font-size: 10px;")
        axis_hint.setWordWrap(True)
        grid.addWidget(axis_hint, 5, 0, 1, 4)

        self.spin_cad_ref_dist = add_double(6, 0, "카메라~부품 거리(m)", defaults.cad_hpr_ref_distance_m, 0.05, 5.0, 0.01, 3)
        self.check_use_hpr = QCheckBox("CAD 가시면 필터링 사용 (HPR)")
        self.check_use_hpr.setChecked(defaults.use_visible_face_filtering)
        self.check_use_hpr.setToolTip(
            "꺼지면 CAD 전체를 그대로 정합에 씁니다(카메라에 안 보이는 뒷면 포함)."
        )
        grid.addWidget(self.check_use_hpr, 6, 2, 1, 2)
        hpr_hint = QLabel(
            "카메라~부품 거리: 이 거리에 CAD를 놓고 봤을 때 보이는 면만 정합에\n"
            "사용합니다(HPR 꺼지면 미리보기 외에는 안 쓰임)."
        )
        hpr_hint.setStyleSheet("color: #888; font-size: 10px;")
        hpr_hint.setWordWrap(True)
        grid.addWidget(hpr_hint, 7, 0, 1, 4)

        return group

    # ------------------------------------------------------ 가시면 미리보기
    def _build_visibility_preview_group(self) -> QGroupBox:
        group = QGroupBox("CAD 가시면 미리보기")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "정합에 실제로 쓰이는 '가시면'(초록)과 필터링돼서 제외된 면(빨강)을\n"
            "3D로 직접 확인합니다. 카메라 없이 CAD와 위 설정값만으로 바로 볼 수 있습니다."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        btn = QPushButton("CAD 가시면 미리보기")
        btn.clicked.connect(self._on_preview_cad_visibility)
        layout.addWidget(btn)
        return group

    # ------------------------------------------------ 참조 포인트클라우드 비교
    def _build_reference_compare_group(self) -> QGroupBox:
        group = QGroupBox("초기 자세 vs 카메라 포인트클라우드 비교")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "위 '초기 roll/pitch/yaw' 값으로 CAD를 놓으면 실제로 어떻게 보이는지,\n"
            "카메라로 찍은(또는 저장된 세션의) 포인트클라우드 위에 겹쳐서 확인합니다."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addWidget(QLabel("참조 포인트클라우드 획득 방식"))
        self.ref_mode_combo = QComboBox()
        self.ref_mode_combo.addItems(["촬영 (카메라)", "세션 폴더에서 불러오기"])
        self.ref_mode_combo.currentIndexChanged.connect(self._on_ref_mode_changed)
        layout.addWidget(self.ref_mode_combo)

        # --- 촬영 모드 ---
        self.ref_capture_widget = QWidget()
        cap_layout = QVBoxLayout(self.ref_capture_widget)
        cap_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_camera_label = QLabel()
        self.ref_camera_label.setStyleSheet("color: #666; font-size: 11px;")
        self.ref_camera_label.setWordWrap(True)
        cap_layout.addWidget(self.ref_camera_label)
        btn_open_cam_settings = QPushButton("카메라 설정 열기")
        btn_open_cam_settings.clicked.connect(self.open_settings_requested.emit)
        cap_layout.addWidget(btn_open_cam_settings)
        self.btn_ref_capture = QPushButton("촬영")
        self.btn_ref_capture.clicked.connect(self._on_ref_capture)
        cap_layout.addWidget(self.btn_ref_capture)
        layout.addWidget(self.ref_capture_widget)

        # --- 세션에서 불러오기 모드 ---
        self.ref_load_widget = QWidget()
        load_layout = QVBoxLayout(self.ref_load_widget)
        load_layout.setContentsMargins(0, 0, 0, 0)
        load_layout.addWidget(QLabel("세션 폴더"))
        session_row = QHBoxLayout()
        self.ref_session_combo = QComboBox()
        self.ref_session_combo.setEditable(True)
        session_row.addWidget(self.ref_session_combo, stretch=1)
        btn_refresh_sessions = QPushButton("↻")
        btn_refresh_sessions.setFixedWidth(28)
        btn_refresh_sessions.clicked.connect(self._refresh_ref_session_list)
        session_row.addWidget(btn_refresh_sessions)
        load_layout.addLayout(session_row)
        btn_browse_session = QPushButton("다른 폴더 찾아보기...")
        btn_browse_session.clicked.connect(self._on_browse_ref_session_dir)
        load_layout.addWidget(btn_browse_session)

        load_layout.addWidget(QLabel("프레임"))
        frame_row = QHBoxLayout()
        self.ref_frame_combo = QComboBox()
        frame_row.addWidget(self.ref_frame_combo, stretch=1)
        btn_scan_frames = QPushButton("목록 갱신")
        btn_scan_frames.clicked.connect(self._on_scan_ref_frames)
        frame_row.addWidget(btn_scan_frames)
        load_layout.addLayout(frame_row)
        btn_use_frame = QPushButton("이 프레임 사용")
        btn_use_frame.clicked.connect(self._on_use_ref_frame)
        load_layout.addWidget(btn_use_frame)
        layout.addWidget(self.ref_load_widget)

        self.ref_status_label = QLabel("참조 포인트클라우드: 없음")
        self.ref_status_label.setStyleSheet("color: #666;")
        self.ref_status_label.setWordWrap(True)
        layout.addWidget(self.ref_status_label)

        btn_compare = QPushButton("초기 자세 vs 카메라 포인트클라우드 비교")
        btn_compare.setToolTip("지금 화면에 입력된(아직 저장 안 했어도 됨) 초기 roll/pitch/yaw로 CAD를 놓아 비교합니다.")
        btn_compare.clicked.connect(self._on_compare_initial_pose)
        layout.addWidget(btn_compare)

        self._on_ref_mode_changed(0)
        return group

    def _on_ref_mode_changed(self, index: int) -> None:
        is_capture = index == 0
        self.ref_capture_widget.setVisible(is_capture)
        self.ref_load_widget.setVisible(not is_capture)
        if is_capture:
            self._refresh_ref_camera_label()
        elif self.ref_session_combo.count() == 0:
            self._refresh_ref_session_list()

    def _refresh_ref_camera_label(self) -> None:
        settings = settings_manager.load_settings()
        self.ref_camera_label.setText(
            f"카메라: {settings['camera_type']} · 평균화 {settings['averaging_num_frames']}프레임"
        )

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override 관례
        super().showEvent(event)
        if self.ref_mode_combo.currentIndex() == 0:
            self._refresh_ref_camera_label()

    # ----- 촬영 -----
    def _on_ref_capture(self) -> None:
        settings = settings_manager.load_settings()
        camera_type = settings["camera_type"]
        config_path = CAMERA_CONFIG_PATHS.get(camera_type)
        if config_path is None or not config_path.is_file():
            QMessageBox.critical(
                self, "설정 파일 없음",
                f"{config_path} 파일을 찾을 수 없습니다. '설정' 탭에서 카메라 타입을 확인하세요.",
            )
            return

        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        cam_cfg = dict(full_cfg["camera"])
        avg_cfg = dict(cam_cfg.get("averaging") or {})
        avg_cfg["num_frames"] = settings["averaging_num_frames"]
        avg_cfg["min_valid_ratio"] = settings["averaging_min_valid_ratio"]
        cam_cfg["averaging"] = avg_cfg

        self.btn_ref_capture.setEnabled(False)
        self.log_message.emit(f"[{self.LOG_PREFIX}] 참조용 촬영 시작: {camera_type}")
        try:
            from src.camera import create_camera
            with create_camera(cam_cfg) as cam:
                frame = cam.capture()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "촬영 오류", str(exc))
            return
        finally:
            self.btn_ref_capture.setEnabled(True)

        self._ref_pcd_organized = frame.points_organized
        self._ref_valid_mask = frame.valid_mask
        valid_ratio = 100.0 * frame.valid_mask.sum() / frame.valid_mask.size
        self.ref_status_label.setText(f"참조 포인트클라우드: 방금 촬영함 (유효 픽셀 {valid_ratio:.1f}%)")
        self.log_message.emit(f"[{self.LOG_PREFIX}] 참조용 촬영 완료 (유효 픽셀 {valid_ratio:.1f}%)")

    # ----- 세션에서 불러오기 -----
    def _refresh_ref_session_list(self) -> None:
        current_text = self.ref_session_combo.currentText()
        self.ref_session_combo.clear()
        if not DEFAULT_DATASET_ROOT.is_dir():
            self.log_message.emit(f"[{self.LOG_PREFIX}] 데이터셋 폴더 없음: {DEFAULT_DATASET_ROOT}")
            return
        sessions = sorted(
            d.name for d in DEFAULT_DATASET_ROOT.iterdir()
            if d.is_dir() and _usable_session_frames(d)
        )
        for name in sessions:
            self.ref_session_combo.addItem(name, str(DEFAULT_DATASET_ROOT / name))
        if current_text:
            self.ref_session_combo.setEditText(current_text)

    def _on_browse_ref_session_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", str(DEFAULT_DATASET_ROOT))
        if path:
            self.ref_session_combo.setEditText(path)
            self._on_scan_ref_frames()

    def _resolve_ref_session_dir(self) -> Path | None:
        text = self.ref_session_combo.currentText().strip()
        if not text:
            return None
        idx = self.ref_session_combo.findText(text)
        if idx >= 0:
            data = self.ref_session_combo.itemData(idx)
            if data:
                return Path(data)
        return Path(text)

    def _on_scan_ref_frames(self) -> None:
        session_dir = self._resolve_ref_session_dir()
        if session_dir is None or not session_dir.is_dir():
            QMessageBox.warning(self, "알림", "세션 폴더를 먼저 지정하세요.")
            return
        frames = _usable_session_frames(session_dir)
        self.ref_frame_combo.clear()
        for frame_name in frames:
            self.ref_frame_combo.addItem(frame_name, str(session_dir))
        self.log_message.emit(f"[{self.LOG_PREFIX}] {session_dir.name}: 프레임 {len(frames)}개")

    def _on_use_ref_frame(self) -> None:
        idx = self.ref_frame_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "알림", "먼저 프레임 목록을 갱신하고 선택하세요.")
            return
        frame_name = self.ref_frame_combo.currentText()
        session_dir = Path(self.ref_frame_combo.currentData())
        try:
            pcd_organized = np.load(session_dir / "pointcloud_organized" / f"{frame_name}.npy")
            valid_mask = np.load(session_dir / "valid_mask" / f"{frame_name}.npy")
        except OSError as exc:
            QMessageBox.critical(self, "불러오기 오류", f"프레임을 읽을 수 없습니다: {exc}")
            return
        self._ref_pcd_organized = pcd_organized
        self._ref_valid_mask = valid_mask
        self.ref_status_label.setText(f"참조 포인트클라우드: {session_dir.name}/{frame_name}")
        self.log_message.emit(f"[{self.LOG_PREFIX}] 참조 프레임 로드: {session_dir.name}/{frame_name}")

    # ----------------------------------------------------------- CAD 로드
    def _ensure_cad_loaded(self, cad_path: str, params: ICPParams) -> None:
        axis = params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis:
            self.log_message.emit(f"[{self.LOG_PREFIX}] CAD 로드 중: {cad_path}")
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis

    def _params_from_widgets(self) -> ICPParams:
        """저장 여부와 무관하게 지금 화면에 입력된 값으로 미리보기하기 위해,
        이 탭이 편집하지 않는 나머지 설정은 저장된 값을 쓰고, 이 탭이
        편집하는 필드만 현재 위젯 값으로 덮어쓴다."""
        settings = dict(settings_manager.load_settings())
        settings.update({
            "init_roll_deg": self.spin_init_roll.value(),
            "init_pitch_deg": self.spin_init_pitch.value(),
            "init_yaw_deg": self.spin_init_yaw.value(),
            "cad_axis_roll_deg": self.spin_axis_roll.value(),
            "cad_axis_pitch_deg": self.spin_axis_pitch.value(),
            "cad_axis_yaw_deg": self.spin_axis_yaw.value(),
            "cad_hpr_ref_distance_m": self.spin_cad_ref_dist.value(),
            "use_visible_face_filtering": self.check_use_hpr.isChecked(),
        })
        return ICPParams(**settings_manager.icp_params_kwargs(settings))

    def _current_cad_path(self) -> str | None:
        return self.cad_combo.currentData()

    # ----------------------------------------------------- 가시면 미리보기
    def _on_preview_cad_visibility(self) -> None:
        cad_path = self._current_cad_path()
        if not cad_path:
            QMessageBox.warning(self, "알림", "먼저 CAD 모델을 선택하세요.")
            return

        params = self._params_from_widgets()
        try:
            self._ensure_cad_loaded(cad_path, params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "CAD 로드 오류", str(exc))
            return

        components, default_hidden = icp_runner.build_cad_visibility_components(self._cad_pcd, params)
        if not components:
            QMessageBox.information(self, "알림", "가시면 계산 결과가 비어있습니다 (카메라~부품 거리 설정을 확인하세요).")
            return

        self._launch_viewer(
            components, title=f"CAD Visibility Preview - {Path(cad_path).name}",
            dir_tag="cad_visibility", default_hidden=default_hidden,
        )

    # ------------------------------------------------- 초기 자세 vs 카메라 비교
    def _on_compare_initial_pose(self) -> None:
        if self._ref_pcd_organized is None or self._ref_valid_mask is None:
            QMessageBox.warning(self, "알림", "먼저 위에서 참조 포인트클라우드를 촬영하거나 불러오세요.")
            return
        cad_path = self._current_cad_path()
        if not cad_path:
            QMessageBox.warning(self, "알림", "먼저 CAD 모델을 선택하세요.")
            return

        params = self._params_from_widgets()
        try:
            self._ensure_cad_loaded(cad_path, params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "CAD 로드 오류", str(exc))
            return

        scene_pcd = icp_runner.build_background_pcd(
            self._ref_pcd_organized, self._ref_valid_mask, color_mode="height",
        )
        if len(scene_pcd.points) == 0:
            QMessageBox.warning(self, "알림", "참조 포인트클라우드에 유효한 점이 없습니다.")
            return

        # build_icp_init()과 완전히 동일한 계산 - 실제 ICP가 시작하는 지점을
        # 그대로 보여준다(위치는 두 중심을 맞추고, 회전은 '초기 roll/pitch/yaw'
        # 그대로).
        T_init = icp_runner.build_icp_init(scene_pcd, self._cad_pcd, params)
        cad_preview = copy.deepcopy(self._cad_pcd)
        cad_preview.transform(T_init)
        cad_preview.paint_uniform_color([0.05, 0.85, 0.95])

        components = {
            "Camera Point Cloud": scene_pcd,
            "CAD (Initial Pose)": cad_preview,
        }
        self._launch_viewer(
            components, title=f"Initial Pose Check - {Path(cad_path).name}",
            dir_tag="cad_init_pose_check",
        )
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 초기 자세 비교 뷰어 실행 "
            f"(roll={params.init_roll_deg:.1f} pitch={params.init_pitch_deg:.1f} yaw={params.init_yaw_deg:.1f})"
        )

    # ----------------------------------------------------------- 값 <-> 위젯
    def _widgets_to_dict(self) -> dict:
        return {
            "cad_path": self._current_cad_path() or "",
            "init_roll_deg": self.spin_init_roll.value(),
            "init_pitch_deg": self.spin_init_pitch.value(),
            "init_yaw_deg": self.spin_init_yaw.value(),
            "cad_axis_roll_deg": self.spin_axis_roll.value(),
            "cad_axis_pitch_deg": self.spin_axis_pitch.value(),
            "cad_axis_yaw_deg": self.spin_axis_yaw.value(),
            "cad_hpr_ref_distance_m": self.spin_cad_ref_dist.value(),
            "use_visible_face_filtering": self.check_use_hpr.isChecked(),
        }

    def _apply_dict_to_widgets(self, settings: dict) -> None:
        cad_path = settings.get("cad_path", "")
        if cad_path:
            idx = self.cad_combo.findData(cad_path)
            if idx >= 0:
                self.cad_combo.setCurrentIndex(idx)
        self.spin_init_roll.setValue(settings["init_roll_deg"])
        self.spin_init_pitch.setValue(settings["init_pitch_deg"])
        self.spin_init_yaw.setValue(settings["init_yaw_deg"])
        self.spin_axis_roll.setValue(settings["cad_axis_roll_deg"])
        self.spin_axis_pitch.setValue(settings["cad_axis_pitch_deg"])
        self.spin_axis_yaw.setValue(settings["cad_axis_yaw_deg"])
        self.spin_cad_ref_dist.setValue(settings["cad_hpr_ref_distance_m"])
        self.check_use_hpr.setChecked(settings["use_visible_face_filtering"])

    # ----------------------------------------------------------- 로드/저장
    def _load_from_settings(self) -> None:
        settings = settings_manager.load_settings()
        self._apply_dict_to_widgets(settings)
        if settings.get("cad_path"):
            self.status_label.setText("저장된 설정을 불러왔습니다.")
        else:
            self.status_label.setText("CAD가 아직 지정되지 않았습니다 - 위에서 선택하세요.")

    def _on_save(self) -> None:
        if not self._current_cad_path():
            QMessageBox.warning(self, "알림", "CAD 모델을 선택하세요.")
            return
        values = self._widgets_to_dict()
        settings_manager.save_settings(values)
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"저장됨 ({now}) - 다른 탭에서 다음 실행부터 이 값을 사용합니다.")
        self.log_message.emit(f"[{self.LOG_PREFIX}] 설정 저장됨 (CAD={Path(values['cad_path']).name})")

    def _on_reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self, "기본값으로 초기화",
            "CAD 선택을 제외한 초기 자세/축보정/HPR 값을 기본값으로 되돌립니다.\n"
            "(아직 저장은 안 됨 - 마음에 들면 [저장]을 눌러야 반영됩니다.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        defaults = ICPParams()
        self.spin_init_roll.setValue(defaults.init_roll_deg)
        self.spin_init_pitch.setValue(defaults.init_pitch_deg)
        self.spin_init_yaw.setValue(defaults.init_yaw_deg)
        self.spin_axis_roll.setValue(defaults.cad_axis_roll_deg)
        self.spin_axis_pitch.setValue(defaults.cad_axis_pitch_deg)
        self.spin_axis_yaw.setValue(defaults.cad_axis_yaw_deg)
        self.spin_cad_ref_dist.setValue(defaults.cad_hpr_ref_distance_m)
        self.check_use_hpr.setChecked(defaults.use_visible_face_filtering)
        self.status_label.setText("기본값으로 초기화됨 (아직 저장 안 됨, CAD 선택은 유지)")