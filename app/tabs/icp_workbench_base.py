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
    QGroupBox, QTabWidget, QDoubleSpinBox, QCheckBox,
)

from app.core.detector import Detection
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points
from app.core.paths import DEFAULT_CAD_DIR
from app.core.pipeline_context import FrameContext
from app.widgets.image_viewer import ImageViewer
from app.core import icp_runner, settings_manager
from app.core.icp_runner import ICPResult, ICPParams
from app.tabs.icp_pipelines import AVAILABLE_ICP_PIPELINES

CAD_EXTS = {".stl", ".ply", ".obj"}
CAD_OVERLAY_MAX_POINTS = 300  # ICP 결과 오버레이용 CAD 서브샘플 점 개수 (속도/시인성용)


class ICPWorkbenchTab(QWidget):
    log_message = pyqtSignal(str)
    #: '설정 열기' 버튼이 눌리면 발생 - main_window.py가 이 시그널을 받아
    #: nav_tree에서 설정 탭을 선택하도록 연결한다.
    open_settings_requested = pyqtSignal()

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
        self._icp_original_T: dict[int, np.ndarray] = {}
        self._result_card_widgets: dict[int, dict] = {}
        self._cad_pcd = None
        self._cad_visible_normal = None
        self._cad_visible_flipped = None
        self._cad_path_loaded: str | None = None
        self._cad_axis_loaded: tuple[float, float, float] | None = None
        self._cad_init_rot_loaded: tuple[float, float, float] | None = None
        self._cad_ref_dist_loaded: float | None = None
        self._viewer_process: QProcess | None = None
        self._build_ui()
        self._refresh_checkpoint_display()
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

        ckpt_group = QGroupBox("RTMDet-Ins (설정 탭에서 관리)")
        ckpt_layout = QVBoxLayout(ckpt_group)
        ckpt_layout.addWidget(QLabel("체크포인트"))
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setReadOnly(True)
        ckpt_layout.addWidget(self.checkpoint_edit)
        ckpt_layout.addWidget(QLabel("config"))
        self.config_edit = QLineEdit()
        self.config_edit.setReadOnly(True)
        ckpt_layout.addWidget(self.config_edit)
        btn_open_settings_ckpt = QPushButton("설정 열기")
        btn_open_settings_ckpt.clicked.connect(self.open_settings_requested.emit)
        ckpt_layout.addWidget(btn_open_settings_ckpt)
        left.addWidget(ckpt_group)

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
        self.thresh_slider.setValue(int(settings_manager.load_settings()["score_threshold"] * 100))
        self.thresh_slider.setFixedWidth(90)
        self.thresh_label = QLabel(f"{self.thresh_slider.value() / 100:.2f}")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_label.setText(f"{v / 100:.2f}")
        )
        run_row.addWidget(self.thresh_slider)
        run_row.addWidget(self.thresh_label)
        run_row.addStretch(1)
        center.addLayout(run_row)

        # ICP/FGR 파라미터는 더 이상 여기서 편집하지 않는다 - "설정" 탭
        # (settings_tab.py)이 유일한 편집 지점이고, _build_icp_params()가
        # 실행 시점에 항상 최신 저장값을 다시 읽는다. params_scroll 위젯
        # 자체는 이름을 유지한다 - manual_labeling_tab.py 등 서브클래스가
        # self.params_scroll.hide()로 참조하기 때문.
        params_placeholder = QWidget()
        params_layout = QVBoxLayout(params_placeholder)
        params_layout.setContentsMargins(4, 4, 4, 4)
        params_layout.addWidget(QLabel("ICP/FGR 파라미터는 '설정' 탭에서 관리됩니다."))
        btn_open_settings_params = QPushButton("설정 열기")
        btn_open_settings_params.clicked.connect(self.open_settings_requested.emit)
        params_layout.addWidget(btn_open_settings_params)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_placeholder)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        params_scroll.setMaximumHeight(80)
        self.params_scroll = params_scroll
        center.addWidget(params_scroll)

        self.image_viewer = ImageViewer()
        gizmo_toggle_row = QHBoxLayout()
        gizmo_toggle_row.addStretch(1)
        self.check_show_axis_gizmo = QCheckBox("좌표축 표시")
        self.check_show_axis_gizmo.setChecked(True)
        self.check_show_axis_gizmo.setToolTip(
            "이미지 우측 상단에 Roll/Pitch/Yaw이 어느 축인지 보여주는 작은 범례.\n"
            "검출 박스와 겹치면 여기서 끌 수 있습니다."
        )
        self.check_show_axis_gizmo.toggled.connect(self.image_viewer.set_axis_gizmo_visible)
        gizmo_toggle_row.addWidget(self.check_show_axis_gizmo)
        center.addLayout(gizmo_toggle_row)
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

    # ------------------------------------------------------ ICP 파라미터
    def _build_icp_params(self) -> ICPParams:
        """실행 시점마다 항상 최신 저장 설정을 다시 읽는다 - 이 탭 인스턴스가
        먼저 만들어진 뒤 사용자가 '설정' 탭에서 값을 바꿨어도(혹은 다른
        프로세스가 재시작 사이 파일을 바꿨어도) 항상 최신값을 쓴다."""
        settings = settings_manager.load_settings()
        return ICPParams(**settings_manager.icp_params_kwargs(settings))

    # ------------------------------------------------------ 체크포인트
    def _refresh_checkpoint_display(self) -> None:
        """'설정' 탭에 저장된 체크포인트/config 경로를 읽어 읽기전용 필드에 반영."""
        settings = settings_manager.load_settings()
        self.checkpoint_edit.setText(settings["checkpoint_path"])
        self.checkpoint_edit.setToolTip(settings["checkpoint_path"])
        self.config_edit.setText(settings["config_path"])
        self.config_edit.setToolTip(settings["config_path"])

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override 관례
        super().showEvent(event)
        # 탭이 화면에 보일 때마다 갱신 - '설정' 탭에서 방금 저장하고 이
        # 탭으로 돌아왔을 때 읽기전용 필드가 바로 최신값을 보여주게 한다.
        self._refresh_checkpoint_display()

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
        # 표시용 self.checkpoint_edit/config_edit(읽기전용)가 아니라 설정
        # 파일을 직접 다시 읽는다 - showEvent가 아직 안 붙었을 수도 있는
        # 상황(예: 탭 전환 없이 바로 실행)에서도 항상 최신값을 보장하기 위함.
        settings = settings_manager.load_settings()
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
            checkpoint_path=settings["checkpoint_path"],
            config_path=settings["config_path"],
            score_threshold=self.thresh_slider.value() / 100.0,
        )

    # -------------------------------------------------------------- 2D 검출
    def _on_run_detection(self) -> None:
        if not self._current_image_path:
            QMessageBox.warning(self, "알림", "먼저 프레임을 준비하세요.")
            return
        settings = settings_manager.load_settings()
        if not settings["checkpoint_path"] or not settings["config_path"]:
            QMessageBox.warning(
                self, "알림",
                "체크포인트와 config가 아직 지정되지 않았습니다. '설정' 탭에서 지정하고 저장하세요.",
            )
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
        self.image_viewer.clear_pose_overlays()
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
        self._update_cad_overlay(results)
        # 2026-07 변경: 실패한 인스턴스도 이제 뷰어에 별도 색으로 표시되므로
        # (build_scene_components의 Failed ICP Instances 레이어), 성공 여부와
        # 무관하게 결과가 하나라도 있으면 뷰어를 열 수 있게 한다.
        self.btn_open_viewer.setEnabled(len(results) > 0)

        n_ok = sum(r.ok for r in results)
        self.log_message.emit(f"[{self.LOG_PREFIX}] {active_tab.pipeline_name} ICP 완료: 성공 {n_ok}/{len(results)}")

    def _update_cad_overlay(self, results: list[ICPResult]) -> None:
        """ICP 정합이 낸 pose(4x4)로 CAD 점군을 이미지에 투영해서 반투명
        오버레이로 보여준다. manual_labeling_tab.py가 수동 각도 입력 시
        실시간 미리보기에 쓰는 것과 동일한 ImageViewer.set_pose_overlay()
        메커니즘을 재사용한다 - "정합이 실제로 잘 맞았는지"를 눈으로 바로
        확인할 수 있다.

        정합에 성공한 인스턴스만 그린다(실패는 pose 자체가 신뢰할 수 없으므로
        오버레이도 안 그리는 게 맞음). 카메라 intrinsic을 추정할 수 없거나
        (유효 픽셀 없음) CAD가 아직 로드되지 않았으면 조용히 건너뛴다.
        """
        self.image_viewer.clear_pose_overlays()
        if self._cad_pcd is None or self._pcd_organized is None or self._valid_mask is None:
            return
        try:
            intrinsics = estimate_intrinsics_from_organized_pcd(self._pcd_organized, self._valid_mask)
        except ValueError:
            return

        cad_points_m = np.asarray(self._cad_pcd.points)
        if cad_points_m.shape[0] > CAD_OVERLAY_MAX_POINTS:
            idx = np.random.default_rng(0).choice(cad_points_m.shape[0], size=CAD_OVERLAY_MAX_POINTS, replace=False)
            cad_points_m = cad_points_m[idx]

        for i, result in enumerate(results):
            if not result.ok or result.T is None:
                continue
            R, t_m = result.T[:3, :3], result.T[:3, 3]
            points_cam_mm = (R @ cad_points_m.T).T * 1000.0 + t_m * 1000.0
            points_2d = project_points(points_cam_mm, intrinsics)
            self.image_viewer.set_pose_overlay(i, points_2d)

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
        self._icp_original_T = {}
        self._result_card_widgets = {}

    def _render_result_panel(self, results: list[ICPResult]) -> None:
        self._clear_result_panel()
        # 미세조정 델타는 항상 "이번 ICP 실행 직후"의 원본 pose 기준으로
        # 다시 계산한다 (스핀박스 값을 누적 곱하지 않음) - 그래야 여러 번
        # 만지작거려도 부동소수점 오차가 안 쌓이고, [리셋]도 스핀박스를
        # 0으로 되돌리기만 하면 정확히 원래 값으로 복귀한다.
        self._icp_original_T = {r.instance_id: r.T.copy() for r in results if r.ok and r.T is not None}
        self._result_card_widgets = {}

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
                pos_label = QLabel(f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm")
                layout.addWidget(pos_label)
                rot_label = QLabel(
                    f"R{pose['roll_deg']:+.1f} P{pose['pitch_deg']:+.1f} Y{pose['yaw_deg']:+.1f} deg"
                )
                layout.addWidget(rot_label)
                if r.was_flipped:
                    flip_label = QLabel("뒤집힘 보정됨")
                    flip_label.setStyleSheet("color: #888; font-size: 10px;")
                    layout.addWidget(flip_label)

                layout.addWidget(self._build_rotation_tune_row(r.instance_id, pos_label, rot_label))
            else:
                status = QLabel(r.error or "실패")
                status.setStyleSheet("color: #c0392b;")
                status.setWordWrap(True)
                layout.addWidget(status)
                if r.fitness is not None:
                    layout.addWidget(QLabel(f"fitness {r.fitness:.3f}"))

            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    def _build_rotation_tune_row(self, instance_id: int, pos_label: QLabel, rot_label: QLabel) -> QWidget:
        """ICP 결과가 살짝 틀어졌을 때 눈으로 보면서 미세조정하는 컨트롤.

        Δroll/Δpitch/Δyaw(deg)는 항상 "이 ICP 결과 원본" 기준 델타이고,
        카메라(씬) 좌표계에서 회전을 덧씌운다(R_new = R_delta @ R_icp) -
        물체 자신의 로컬 축이 아니라 화면에서 보이는 대로 X/Y/Z 축을
        돌리는 감각에 가깝다. 위치(pick point)는 건드리지 않는다 -
        회전만 미세조정한다는 요청 그대로.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(QLabel("회전 미세조정 (Δdeg)"))

        row = QHBoxLayout()
        spins = {}
        for axis_label, key in (("R", "droll"), ("P", "dpitch"), ("Y", "dyaw")):
            row.addWidget(QLabel(axis_label))
            spin = QDoubleSpinBox()
            spin.setRange(-90.0, 90.0)
            spin.setSingleStep(0.5)
            spin.setDecimals(1)
            spin.setFixedWidth(60)
            row.addWidget(spin)
            spins[key] = spin
        layout.addLayout(row)

        btn_reset = QPushButton("리셋")
        btn_reset.setToolTip("이 인스턴스의 ICP 원본 pose로 되돌립니다.")
        layout.addWidget(btn_reset)

        def on_delta_changed(_value=None):
            self._apply_rotation_delta(
                instance_id, spins["droll"].value(), spins["dpitch"].value(), spins["dyaw"].value(),
                pos_label, rot_label,
            )

        for spin in spins.values():
            spin.valueChanged.connect(on_delta_changed)

        def on_reset():
            for spin in spins.values():
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)
            on_delta_changed()

        btn_reset.clicked.connect(on_reset)

        self._result_card_widgets[instance_id] = {
            "pos_label": pos_label, "rot_label": rot_label, "spins": spins,
        }
        return widget

    def _apply_rotation_delta(
        self, instance_id: int, droll: float, dpitch: float, dyaw: float,
        pos_label: QLabel, rot_label: QLabel,
    ) -> None:
        """Δroll/Δpitch/Δyaw를 화면에 찍힌 R/P/Y 숫자에 직접 더한다.

        이전 버전은 카메라 좌표계에서 R_delta @ R_원본으로 회전을 "합성"했는데,
        3D 회전 합성은 교환법칙이 성립하지 않아서(비가환) - 특히 원본 회전이
        이미 항등행렬에서 많이 벗어나 있으면(예: roll이 -75도처럼 큰 값) -
        R만 건드려도 재분해된 P/Y 표시값까지 같이 바뀌어버린다. 이건 수학적으로는
        틀린 게 아니지만, "R 슬라이더는 R만 바꾼다"는 직관과 안 맞아 혼란스럽다.

        그래서 여기서는 manual_labeling_tab.py가 각도 입력에서 pose를 만드는
        방식(R = Rz(yaw) @ Ry(pitch) @ Rx(roll))과 동일하게, 원본을
        roll/pitch/yaw로 분해한 뒤 델타를 각 성분에 독립적으로 더하고 그
        값으로 새 회전행렬을 처음부터 다시 만든다 - 이러면 Δroll은 R
        표시값에만, Δpitch는 P에만, Δyaw는 Y에만 정확히 반영된다.
        """
        from app.core.icp_runner import _Rx, _Ry, _Rz  # 언더스코어 접두 - 이 파일 밖 재사용 관례(manual_labeling_tab.py)와 동일

        T_original = self._icp_original_T.get(instance_id)
        if T_original is None:
            return

        orig_euler = icp_runner.transform_to_pose(T_original)["euler_deg"]
        new_roll = orig_euler["roll_deg"] + droll
        new_pitch = orig_euler["pitch_deg"] + dpitch
        new_yaw = orig_euler["yaw_deg"] + dyaw

        R_new = _Rz(new_yaw) @ _Ry(new_pitch) @ _Rx(new_roll)
        T_new = T_original.copy()
        T_new[:3, :3] = R_new
        # 위치(translation)는 원본 그대로 유지 - 회전만 미세조정.

        target = next((r for r in self._last_icp_results if r.instance_id == instance_id), None)
        if target is None:
            return
        target.T = T_new
        target.pose = icp_runner.transform_to_pose(T_new)

        if self._cad_pcd is not None:
            cad_center_m = np.asarray(self._cad_pcd.get_center())
            target.pick_point_mm = ((T_new[:3, :3] @ cad_center_m + T_new[:3, 3]) * 1000.0).tolist()

        pose = target.pose["euler_deg"]
        pos = target.pick_point_mm
        pos_label.setText(f"pick X{pos[0]:+.1f} Y{pos[1]:+.1f} Z{pos[2]:+.1f} mm")
        rot_label.setText(f"R{pose['roll_deg']:+.1f} P{pose['pitch_deg']:+.1f} Y{pose['yaw_deg']:+.1f} deg")

        self._update_cad_overlay(self._last_icp_results)

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