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
from PyQt6.QtCore import pyqtSignal, Qt, QProcess, QEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QLineEdit, QScrollArea, QFrame,
    QGroupBox, QTabWidget, QDoubleSpinBox, QCheckBox, QSplitter,
)

from app.core.detector import Detection
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points
from app.core.pipeline_context import FrameContext
from app.widgets.image_viewer import (
    ImageViewer, DEFAULT_MASK_ALPHA, DEFAULT_LABEL_ALPHA, DEFAULT_POSE_OVERLAY_ALPHA,
)
from app.core import icp_runner, settings_manager
from app.core.icp_runner import ICPResult, ICPParams
from app.tabs.icp_pipelines import AVAILABLE_ICP_PIPELINES
from app.tabs.viewer_launcher import Viewer3DMixin

CAD_OVERLAY_MAX_POINTS = 300  # ICP 결과 오버레이용 CAD 서브샘플 점 개수 (속도/시인성용)


class ICPWorkbenchTab(Viewer3DMixin, QWidget):
    log_message = pyqtSignal(str)
    #: '설정 열기' 버튼이 눌리면 발생 - main_window.py가 이 시그널을 받아
    #: nav_tree에서 설정 탭을 선택하도록 연결한다.
    open_settings_requested = pyqtSignal()
    #: 'CAD 모델 설정 열기' 버튼이 눌리면 발생 - main_window.py가 받아서
    #: nav_tree에서 'CAD 모델 설정' 탭을 선택하도록 연결한다.
    open_cad_settings_requested = pyqtSignal()

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
        self._cad_use_hpr_loaded: bool | None = None
        self._viewer_process: QProcess | None = None
        self._build_ui()
        self._refresh_checkpoint_display()
        self._refresh_cad_display()

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
        root.setContentsMargins(0, 0, 0, 0)

        # 좌/중앙/우 세 패널을 QSplitter에 담아서 경계선을 마우스로 드래그해
        # 폭을 조절할 수 있게 한다 - 예전엔 좌(340px)/우(280px) 폭이
        # 고정이라 결과 패널이 좁아도 늘릴 방법이 없었다.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ------------------------------------------------------- 좌측
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(220)

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

        cad_group = QGroupBox("CAD 모델 (설정은 'CAD 모델 설정' 탭에서 관리)")
        cad_layout = QVBoxLayout(cad_group)
        self.cad_display_label = QLabel()
        self.cad_display_label.setWordWrap(True)
        cad_layout.addWidget(self.cad_display_label)
        btn_open_cad_settings = QPushButton("CAD 모델 설정 열기")
        btn_open_cad_settings.clicked.connect(self.open_cad_settings_requested.emit)
        cad_layout.addWidget(btn_open_cad_settings)
        left.addWidget(cad_group)

        left.addStretch(1)

        splitter.addWidget(left_widget)

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

        run_row.addWidget(QLabel("fitness"))
        self.fitness_slider = QSlider(Qt.Orientation.Horizontal)
        self.fitness_slider.setRange(0, 100)
        self.fitness_slider.setValue(int(settings_manager.load_settings()["fitness_threshold"] * 100))
        self.fitness_slider.setFixedWidth(90)
        self.fitness_slider.setToolTip(
            "ICP 정합 성공 기준(fitness ≥ 이 값). 예전엔 '설정' 탭에서만 바꿀 수 있었는데,\n"
            "결과를 보면서 바로바로 조정할 수 있게 여기로 옮겼습니다."
        )
        self.fitness_label = QLabel(f"{self.fitness_slider.value() / 100:.2f}")
        self.fitness_slider.valueChanged.connect(
            lambda v: self.fitness_label.setText(f"{v / 100:.2f}")
        )
        run_row.addWidget(self.fitness_slider)
        run_row.addWidget(self.fitness_label)
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

        opacity_row = QHBoxLayout()
        self.mask_alpha_slider = self._add_opacity_control(
            opacity_row, "마스크", DEFAULT_MASK_ALPHA, self.image_viewer.set_mask_alpha,
        )
        self.label_alpha_slider = self._add_opacity_control(
            opacity_row, "라벨/박스", DEFAULT_LABEL_ALPHA, self.image_viewer.set_label_alpha,
        )
        self.pose_alpha_slider = self._add_opacity_control(
            opacity_row, "CAD 오버레이", DEFAULT_POSE_OVERLAY_ALPHA, self.image_viewer.set_pose_overlay_alpha,
        )
        opacity_row.addStretch(1)
        center.addLayout(opacity_row)

        roi_zoom_row = QHBoxLayout()
        self.btn_roi_draw = QPushButton("ROI 지정")
        self.btn_roi_draw.setCheckable(True)
        self.btn_roi_draw.setToolTip("누른 뒤 이미지 위를 드래그해서 검출 대상 영역을 지정합니다.")
        self.btn_roi_draw.toggled.connect(self.image_viewer.set_roi_draw_mode)
        self.image_viewer.roi_draw_mode_changed.connect(self.btn_roi_draw.setChecked)
        roi_zoom_row.addWidget(self.btn_roi_draw)

        btn_roi_clear = QPushButton("ROI 해제")
        btn_roi_clear.clicked.connect(self.image_viewer.clear_roi)
        roi_zoom_row.addWidget(btn_roi_clear)

        self.roi_status_label = QLabel("ROI: 지정 안 됨 (전체 영역)")
        self.roi_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.image_viewer.roi_changed.connect(self._on_roi_changed)
        roi_zoom_row.addWidget(self.roi_status_label)

        roi_zoom_row.addStretch(1)

        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setFixedWidth(28)
        btn_zoom_out.clicked.connect(self.image_viewer.zoom_out)
        roi_zoom_row.addWidget(btn_zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(44)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roi_zoom_row.addWidget(self.zoom_label)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedWidth(28)
        btn_zoom_in.clicked.connect(self.image_viewer.zoom_in)
        roi_zoom_row.addWidget(btn_zoom_in)

        btn_zoom_fit = QPushButton("맞춤")
        btn_zoom_fit.setToolTip("확대/축소를 화면에 꽉 맞춤(100%)으로 리셋합니다. 마우스 휠로도 확대/축소할 수 있습니다.")
        btn_zoom_fit.clicked.connect(self.image_viewer.zoom_fit)
        roi_zoom_row.addWidget(btn_zoom_fit)

        center.addLayout(roi_zoom_row)

        # ImageViewer를 QScrollArea에 담아서 확대했을 때 스크롤로 볼 수 있게 한다.
        # zoom=1.0(기본)일 때는 뷰포트에 꽉 맞춰 그려지므로 기존과 동일하게 보인다.
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(False)
        image_scroll.setWidget(self.image_viewer)
        image_scroll.setStyleSheet("QScrollArea { border: none; }")
        center.addWidget(image_scroll, stretch=1)

        # 줌 배율 라벨은 버튼 클릭이든 마우스 휠이든 어느 경로로 바뀌든
        # image_viewer.zoom_changed 시그널 하나로 갱신한다 (실제 값이 바뀐
        # "이후"에 emit되므로 항상 최신값을 보여준다).
        self.image_viewer.zoom_changed.connect(lambda pct: self.zoom_label.setText(f"{pct}%"))

        center_widget = QWidget()
        center_widget.setLayout(center)
        center_widget.setMinimumWidth(320)
        splitter.addWidget(center_widget)

        # ------------------------------------------------------- 우측
        right = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(220)

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

        splitter.addWidget(right_widget)

        # 초기 폭 비율 (이후엔 사용자가 경계선을 드래그해서 자유롭게 조절).
        # 사용자가 특히 우측 결과 패널을 넓혀 쓰고 싶어했으므로 조금 더 준다.
        splitter.setSizes([340, 700, 340])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    # ------------------------------------------------------ ICP 파라미터
    def _build_icp_params(self) -> ICPParams:
        """실행 시점마다 항상 최신 저장 설정을 다시 읽는다 - 이 탭 인스턴스가
        먼저 만들어진 뒤 사용자가 '설정' 탭에서 값을 바꿨어도(혹은 다른
        프로세스가 재시작 사이 파일을 바꿨어도) 항상 최신값을 쓴다.

        fitness_threshold만은 예외 - '설정' 탭이 아니라 이 탭의 fitness
        슬라이더(conf 슬라이더와 동일한 패턴) 값을 그대로 쓴다. 결과를
        보면서 바로바로 조정하고 싶다는 요청 반영."""
        settings = settings_manager.load_settings()
        kwargs = settings_manager.icp_params_kwargs(settings)
        kwargs["fitness_threshold"] = self.fitness_slider.value() / 100.0
        return ICPParams(**kwargs)

    # ------------------------------------------------------ 체크포인트
    @staticmethod
    def _add_opacity_control(layout, label: str, default_alpha: int, setter) -> "QSlider":
        """오버레이 요소 하나(마스크/라벨박스/CAD 오버레이)의 투명도 슬라이더를
        만들어 layout에 붙이고 setter(0~255)에 연결한다. 0=완전 투명,
        255=완전 불투명. 기본값은 기존 시각적 결과와 동일하게 맞춰뒀다."""
        from PyQt6.QtWidgets import QSlider
        layout.addWidget(QLabel(label))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(default_alpha)
        slider.setFixedWidth(70)
        slider.valueChanged.connect(setter)
        layout.addWidget(slider)
        return slider

    @staticmethod
    def _bbox_center_in_roi(bbox, roi: tuple[int, int, int, int]) -> bool:
        """검출 bbox의 중심점이 ROI 사각형 안에 있는지. 중심점 기준으로 판정하는
        이유: bbox 전체 포함/겹침 기준이면 ROI 경계에 걸친 물체가 애매하게
        판정되는데(반쯤 걸친 물체를 포함시킬지 뺄지), 중심점 기준이 "이 물체가
        ROI 안에 있다"는 직관과 가장 가깝다."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx1, ry1, rx2, ry2 = roi
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _on_roi_changed(self, roi) -> None:
        if roi is None:
            self.roi_status_label.setText("ROI: 지정 안 됨 (전체 영역)")
        else:
            x1, y1, x2, y2 = roi
            self.roi_status_label.setText(f"ROI: ({x1},{y1})–({x2},{y2})")

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
        self._refresh_cad_display()

    # ------------------------------------------------------------ CAD
    def _refresh_cad_display(self) -> None:
        """'CAD 모델 설정' 탭에 저장된 CAD 경로를 읽어 읽기전용 표시에 반영."""
        cad_path = settings_manager.load_settings().get("cad_path", "")
        if cad_path:
            self.cad_display_label.setText(f"CAD: {Path(cad_path).name}")
            self.cad_display_label.setToolTip(cad_path)
        else:
            self.cad_display_label.setText("CAD가 아직 지정되지 않았습니다.")

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

        roi = self.image_viewer.get_roi()
        if roi is not None:
            n_before = len(detections)
            detections = [d for d in detections if self._bbox_center_in_roi(d.bbox, roi)]
            self.log_message.emit(
                f"[{self.LOG_PREFIX}] ROI 필터: {n_before}건 중 {len(detections)}건이 ROI 안쪽 "
                f"(bbox 중심 기준, ROI={roi})"
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
        cad_path = settings_manager.load_settings().get("cad_path", "")
        if not cad_path:
            QMessageBox.warning(
                self, "알림",
                "CAD 모델이 아직 지정되지 않았습니다. 'CAD 모델 설정' 탭에서 지정하고 저장하세요.",
            )
            return

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

        try:
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
                    for sl in (result.stage_logs or []):
                        self.log_message.emit(f"[{self.LOG_PREFIX}]     {self._format_stage_log(sl)}")

            self._last_icp_results = results
            self._render_result_panel(results)
            self._update_cad_overlay(results)
            # 2026-07 변경: 실패한 인스턴스도 이제 뷰어에 별도 색으로 표시되므로
            # (build_scene_components의 Failed ICP Instances 레이어), 성공 여부와
            # 무관하게 결과가 하나라도 있으면 뷰어를 열 수 있게 한다.
            self.btn_open_viewer.setEnabled(len(results) > 0)

            n_ok = sum(r.ok for r in results)
            self.log_message.emit(f"[{self.LOG_PREFIX}] {active_tab.pipeline_name} ICP 완료: 성공 {n_ok}/{len(results)}")
        except Exception as exc:  # noqa: BLE001
            # 2026-08 추가: 이전엔 이 블록 안에서 예외가 나면(예: stage_logs
            # 항목마다 필드 구성이 달라서 dict 키 접근이 깨지는 경우) 조용히
            # 함수가 죽어서 남은 인스턴스는 처리도 안 되고 결과 패널도 텅 빈
            # 채로 남았다 - 사용자 입장에선 "아무 이유 없이 아무것도 안 됨"
            # 으로 보였다. 이제 무슨 일이 있었는지 최소한 로그+팝업으로는 보인다.
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[{self.LOG_PREFIX}] ⚠ ICP 처리 중 예상치 못한 오류: {exc}")
            QMessageBox.critical(self, "ICP 처리 오류", f"ICP 정합 처리 중 오류가 발생했습니다:\n\n{exc}")

    @staticmethod
    def _format_stage_log(sl: dict) -> str:
        """stage_logs 항목 하나를 로그 한 줄로 포맷.

        표준 다단계 ICP stage(voxel/method/n_src/n_tgt/fitness/rmse 전부 있음)
        뿐 아니라, PCA 프리스크린/포인트부족 폴백(stage/fitness/rmse만 있거나
        stage/reason만 있음), FGR 회전-prior reject(stage/deviation_deg/
        max_allowed_deg/method) 등 스키마가 다른 항목도 안전하게(키 없다고
        크래시 없이) 표시한다 - 전부 .get()으로 접근하고 있는 정보만 보여준다.
        예전엔 sl['voxel'] 같은 직접 인덱싱이라, 표준 스키마가 아닌 항목이
        하나라도 섞이면 그 즉시 KeyError로 전체 함수가 죽었었다(그리고
        PyQt 슬롯 안 예외는 콘솔에만 찍히고 조용히 삼켜져서, 사용자 입장에선
        "이유 없이 나머지 인스턴스 처리가 멈춤"으로 보였다).
        """
        stage = sl.get("stage", "?")
        header = f"stage{stage}"
        if "voxel" in sl and "method" in sl:
            voxel = sl["voxel"]
            voxel_str = f"{voxel}" if voxel is not None else "원본밀도"
            header = f"stage{stage}({sl['method']}, voxel={voxel_str})"

        bits = []
        if "n_src" in sl and "n_tgt" in sl:
            bits.append(f"src={sl['n_src']}pt tgt={sl['n_tgt']}pt")
        if "fitness" in sl:
            bits.append(f"fitness={sl['fitness']:.3f}")
        if sl.get("rmse") is not None:
            bits.append(f"rmse={sl['rmse']*1000:.2f}mm")
        if "deviation_deg" in sl:
            bits.append(f"편차={sl['deviation_deg']:.1f}deg(허용 {sl.get('max_allowed_deg', '?')}deg)")
        if "reason" in sl:
            bits.append(sl["reason"])
        if "method" in sl and "voxel" not in sl:
            # voxel 없이 method만 있으면 그 자체가 설명 문장인 경우(FGR reject 등)
            bits.append(str(sl["method"]))

        return header + (": " + " ".join(bits) if bits else "")

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
        use_hpr = params.use_visible_face_filtering
        if (self._cad_visible_normal is None
                or self._cad_init_rot_loaded != init_rot
                or self._cad_ref_dist_loaded != ref_dist
                or self._cad_use_hpr_loaded != use_hpr):
            if not use_hpr:
                # HPR 꺼짐 - CAD 전체를 그대로 정합 소스로 쓴다. 뒤집힘
                # 재정합용 서브셋도 동일하게 CAD 전체가 된다(그래도
                # correct_flipped_pose()는 "뒤집힌 초기 자세로 다시 정합
                # 시도"라는 의미 자체는 그대로 유효함 - 소스 점군만 필터링
                # 안 된 것뿐).
                visible_normal = self._cad_pcd
                visible_flipped = self._cad_pcd
                total = vis = len(self._cad_pcd.points)
                self.log_message.emit(
                    f"[{self.LOG_PREFIX}] CAD 가시면 필터링 꺼짐 - CAD 전체({total}점)를 그대로 사용"
                )
            else:
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
            self._cad_use_hpr_loaded = use_hpr
            if use_hpr:
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

            header_row = QHBoxLayout()
            title = QLabel(f"obj{r.instance_id}")
            title.setStyleSheet("font-weight: 600;")
            header_row.addWidget(title)

            if r.ok:
                status = QLabel(f"fitness {r.fitness:.3f}")
                status.setStyleSheet("color: #2a8a2a;")
                header_row.addWidget(status)
                header_row.addStretch(1)
                layout.addLayout(header_row)
                if r.was_flipped:
                    flip_label = QLabel("뒤집힘 보정됨")
                    flip_label.setStyleSheet("color: #888; font-size: 10px;")
                    layout.addWidget(flip_label)

                layout.addWidget(self._build_pose_edit_row(r.instance_id))
            else:
                header_row.addStretch(1)
                layout.addLayout(header_row)
                status = QLabel(r.error or "실패")
                status.setStyleSheet("color: #c0392b;")
                status.setWordWrap(True)
                layout.addWidget(status)
                if r.fitness is not None:
                    layout.addWidget(QLabel(f"fitness {r.fitness:.3f}"))

            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    def _build_pose_edit_row(self, instance_id: int) -> QWidget:
        """ICP 결과의 위치(mm)/회전(deg)을 "델타 보정량"이 아니라 현재 값을
        직접 편집하는 방식으로 바꾼다.

        이전 버전은 Δroll/Δpitch/Δyaw 입력칸이 항상 0에서 시작해서, "지금
        실제 각도가 몇 도인지"는 옆 라벨을 따로 봐야 했다. 이제는 스핀박스
        자체가 현재 값을 그대로 보여주고, 편집하면 그 값 자체가 곧 새
        pose가 된다 - 두 세트의 숫자를 오가며 볼 필요가 없다.

        위치는 "pick point"(CAD 중심을 이 pose로 옮긴 좌표, 카드에 항상
        표시되던 값) 기준으로 편집한다. 회전 칸만 바꾸면 위치 칸은 그대로
        유지되므로, 회전이 pick point를 중심으로 일어나 물체가 제자리에서
        도는 것처럼 보인다(구현: t = pick_point - R_new @ cad_center로
        역산). 회전 범위 제한(예전 ±90도)도 없앴다 - _Rx/_Ry/_Rz는 임의의
        각도에 대해 항상 well-defined라 굳이 좁게 막을 이유가 없다.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        target = next((r for r in self._last_icp_results if r.instance_id == instance_id), None)
        if target is None or target.T is None:
            return widget
        pos0 = target.pick_point_mm or [0.0, 0.0, 0.0]
        euler0 = (target.pose or {}).get("euler_deg", {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0})

        pos_row = QHBoxLayout()
        pos_spins = {}
        for axis_label, key, val in (("X", "x", pos0[0]), ("Y", "y", pos0[1]), ("Z", "z", pos0[2])):
            pos_row.addWidget(QLabel(axis_label))
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)  # mm - 실질적으로 무제한 (신뢰 범위는 '설정' 탭의 xyz_max_m으로 이미 걸러짐)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setFixedWidth(62)
            spin.setValue(val)
            pos_row.addWidget(spin)
            pos_spins[key] = spin
        layout.addLayout(pos_row)

        rot_row = QHBoxLayout()
        rot_spins = {}
        for axis_label, key, val in (
            ("R", "roll", euler0["roll_deg"]), ("P", "pitch", euler0["pitch_deg"]), ("Y", "yaw", euler0["yaw_deg"]),
        ):
            rot_row.addWidget(QLabel(axis_label))
            spin = QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)  # 예전 ±90도 제한 제거
            spin.setSingleStep(0.5)
            spin.setDecimals(1)
            spin.setFixedWidth(62)
            spin.setValue(val)
            rot_row.addWidget(spin)
            rot_spins[key] = spin
        layout.addLayout(rot_row)

        btn_reset = QPushButton("ICP 원본으로 리셋")
        btn_reset.setToolTip("이 인스턴스의 위치/회전을 ICP가 처음 낸 값으로 되돌립니다.")
        layout.addWidget(btn_reset)

        all_spins = list(pos_spins.values()) + list(rot_spins.values())

        def on_changed(_value=None):
            self._apply_pose_edit(
                instance_id,
                pos_spins["x"].value(), pos_spins["y"].value(), pos_spins["z"].value(),
                rot_spins["roll"].value(), rot_spins["pitch"].value(), rot_spins["yaw"].value(),
            )

        for spin in all_spins:
            spin.valueChanged.connect(on_changed)

        def on_reset():
            T_original = self._icp_original_T.get(instance_id)
            if T_original is None:
                return
            orig_euler = icp_runner.transform_to_pose(T_original)["euler_deg"]
            if self._cad_pcd is not None:
                cad_center_m = np.asarray(self._cad_pcd.get_center())
                orig_pick_mm = ((T_original[:3, :3] @ cad_center_m + T_original[:3, 3]) * 1000.0).tolist()
            else:
                orig_pick_mm = (T_original[:3, 3] * 1000.0).tolist()

            for spin in all_spins:
                spin.blockSignals(True)
            pos_spins["x"].setValue(orig_pick_mm[0])
            pos_spins["y"].setValue(orig_pick_mm[1])
            pos_spins["z"].setValue(orig_pick_mm[2])
            rot_spins["roll"].setValue(orig_euler["roll_deg"])
            rot_spins["pitch"].setValue(orig_euler["pitch_deg"])
            rot_spins["yaw"].setValue(orig_euler["yaw_deg"])
            for spin in all_spins:
                spin.blockSignals(False)
            on_changed()

        btn_reset.clicked.connect(on_reset)

        self._result_card_widgets[instance_id] = {"pos_spins": pos_spins, "rot_spins": rot_spins}
        return widget

    def _apply_pose_edit(
        self, instance_id: int,
        x_mm: float, y_mm: float, z_mm: float,
        roll_deg: float, pitch_deg: float, yaw_deg: float,
    ) -> None:
        """스핀박스에 입력된 절대 위치/회전값으로 pose(T)를 직접 재구성한다
        (Δ보정량을 더하는 방식이 아님 - 스핀박스 값 자체가 곧 새 pose)."""
        from app.core.icp_runner import _Rx, _Ry, _Rz

        target = next((r for r in self._last_icp_results if r.instance_id == instance_id), None)
        if target is None:
            return

        R_new = _Rz(yaw_deg) @ _Ry(pitch_deg) @ _Rx(roll_deg)
        pick_point_m = np.array([x_mm, y_mm, z_mm]) / 1000.0

        if self._cad_pcd is not None:
            cad_center_m = np.asarray(self._cad_pcd.get_center())
            t_new = pick_point_m - R_new @ cad_center_m
        else:
            t_new = pick_point_m

        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new

        target.T = T_new
        target.pose = icp_runner.transform_to_pose(T_new)
        target.pick_point_mm = [x_mm, y_mm, z_mm]

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

        self._launch_viewer(components, title=f"ICP 결과 - {self._current_frame}", dir_tag=self._current_frame or "frame")