"""탭 5: ICP 정합테스트(TCP) - 촬영 또는 세션 폴더에서 이미지 불러오기.

공통 로직(파이프라인 실행, ICP 파라미터, 결과 패널, 3D 뷰어)은
ICPWorkbenchTab에 있고, 이 파일은 "프레임을 어떻게 얻는지"를 구현한다.
2026-08 개편: 이전에는 카메라 촬영만 지원했는데, 이제 탭 안에서 방식을
고를 수 있다:

    [촬영]          create_camera()로 즉시 촬영 (카메라 타입/평균화 설정은
                     '설정' 탭에서 관리 - _on_capture()가 실행 시점마다
                     settings_manager.load_settings()로 최신값을 읽는다)
    [이미지 불러오기] data/dataset/<session>/{intensity,pointcloud_organized,
                     valid_mask}/ 구조로 이미 저장된 세션 폴더에서 프레임을
                     골라 불러온다 (촬영 후 "세션으로 저장"을 누르면 만들어지는
                     바로 그 구조 - generate_pvnet_labels.py가 읽는 것과 동일한
                     스키마).

두 방식으로 얻은 프레임은 모두 같은 이력 목록(capture_list)에 쌓이고,
목록에서 고르면 동일하게 self._current_image_path/_pcd_organized/_valid_mask
가 채워진 뒤 _on_new_frame_acquired()가 호출된다 - 이후 파이프라인(검출/ICP)
입장에서는 프레임이 어디서 왔는지 구분할 필요가 없다.

주의 (알려진 제약):
    촬영(cam.capture())은 이 탭 안에서 동기적으로 실행된다 - 즉 촬영 중에는
    UI가 잠깐 멈춘다. 평균화 프레임 수를 크게 잡을수록 촬영 시간이 비례해서
    늘어나므로 몇 초간 응답이 없을 수 있다. 테스트용 도구라서 우선 단순하게
    만들었고, 실제로 불편하면 QThread로 옮기면 된다.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QListWidget, QListWidgetItem, QMessageBox,
)

from app.core import settings_manager
from app.core.paths import CAMERA_CONFIG_PATHS, DEFAULT_DATASET_ROOT
from app.tabs.icp_workbench_base import ICPWorkbenchTab

MODE_CAPTURE = 0
MODE_LOAD = 1


@dataclass
class _FrameEntry:
    """이력 목록 한 줄이 가리키는 프레임. kind로 로딩 방식을 구분한다.

    kind="capture": image_path/pcd_organized/valid_mask/pcd_std가 이미 메모리에
        있음(촬영 직후). kind="session": session_dir/frame_name만 들고 있다가
        실제로 선택됐을 때(_on_capture_row_changed) 디스크에서 읽는다 - 목록에
        수십 개 프레임이 스캔돼도 선택 전까지는 메모리를 안 쓴다.
    """
    kind: str
    image_path: str | None = None
    pcd_organized: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    pcd_std: np.ndarray | None = None
    session_dir: Path | None = None
    frame_name: str | None = None


def _usable_session_frames(session_dir: Path) -> list[str]:
    """generate_pvnet_labels.py의 _usable_frames()와 동일한 기준 - intensity/
    pointcloud_organized/valid_mask 세 폴더에 전부 존재하는 프레임 이름만."""
    intensity_dir = session_dir / "intensity"
    organized_dir = session_dir / "pointcloud_organized"
    mask_dir = session_dir / "valid_mask"
    if not intensity_dir.is_dir():
        return []
    stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
    return [
        s for s in stems
        if (organized_dir / f"{s}.npy").is_file() and (mask_dir / f"{s}.npy").is_file()
    ]


class LiveCaptureICPTab(ICPWorkbenchTab):
    LOG_PREFIX = "ICP(TCP) 탭"

    def __init__(self, parent=None):
        self._frame_entries: dict[str, _FrameEntry] = {}
        super().__init__(parent)

    # ----------------------------------------------------------- UI 조립
    def _build_acquisition_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("이미지 획득 방식"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["촬영 (카메라)", "이미지 불러오기 (세션 폴더)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        layout.addWidget(self._build_capture_mode_widget())
        layout.addWidget(self._build_load_mode_widget())

        layout.addWidget(QLabel("이력 (촬영/불러온 프레임, 이번 실행 중 메모리에만 보관)"))
        self.capture_list = QListWidget()
        self.capture_list.currentRowChanged.connect(self._on_capture_row_changed)
        layout.addWidget(self.capture_list, stretch=1)

        self.btn_save_session = QPushButton("세션으로 저장")
        self.btn_save_session.setToolTip(
            "현재 촬영본을 data/dataset/ 밑에 표준 세션 폴더로 저장 - "
            "이후 '이미지 불러오기' 모드나 학습 파이프라인에서 재사용 가능\n"
            "(이미 세션 폴더에서 불러온 프레임은 다시 저장할 필요가 없어 비활성화됩니다.)"
        )
        self.btn_save_session.clicked.connect(self._on_save_as_session)
        self.btn_save_session.setEnabled(False)
        layout.addWidget(self.btn_save_session)

        self._on_mode_changed(MODE_CAPTURE)
        return panel

    def _build_capture_mode_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.camera_settings_label = QLabel()
        self.camera_settings_label.setStyleSheet("color: #666; font-size: 11px;")
        self.camera_settings_label.setWordWrap(True)
        layout.addWidget(self.camera_settings_label)

        btn_open_settings_cam = QPushButton("카메라 설정 열기")
        btn_open_settings_cam.clicked.connect(self.open_settings_requested.emit)
        layout.addWidget(btn_open_settings_cam)

        self.btn_capture = QPushButton("촬영")
        self.btn_capture.clicked.connect(self._on_capture)
        layout.addWidget(self.btn_capture)

        self.capture_mode_widget = widget
        return widget

    def _build_load_mode_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("세션 폴더"))
        session_row = QHBoxLayout()
        self.session_dir_combo = QComboBox()
        self.session_dir_combo.setEditable(True)
        self.session_dir_combo.setToolTip(
            f"{DEFAULT_DATASET_ROOT} 밑의 세션 폴더 목록입니다.\n"
            "직접 경로를 입력하거나 '다른 폴더 찾아보기'로 바깥 폴더도 지정할 수 있습니다."
        )
        session_row.addWidget(self.session_dir_combo, stretch=1)
        btn_refresh_sessions = QPushButton("↻")
        btn_refresh_sessions.setFixedWidth(28)
        btn_refresh_sessions.setToolTip(f"{DEFAULT_DATASET_ROOT} 다시 스캔")
        btn_refresh_sessions.clicked.connect(self._refresh_session_list)
        session_row.addWidget(btn_refresh_sessions)
        layout.addLayout(session_row)

        btn_browse_session = QPushButton("다른 폴더 찾아보기...")
        btn_browse_session.clicked.connect(self._on_browse_session_dir)
        layout.addWidget(btn_browse_session)

        btn_scan_frames = QPushButton("프레임 목록 불러오기")
        btn_scan_frames.clicked.connect(self._on_scan_frames)
        layout.addWidget(btn_scan_frames)

        self.load_mode_widget = widget
        return widget

    def _on_mode_changed(self, index: int) -> None:
        is_capture = index == MODE_CAPTURE
        self.capture_mode_widget.setVisible(is_capture)
        self.load_mode_widget.setVisible(not is_capture)
        if is_capture:
            self._refresh_camera_settings_label()
        elif self.session_dir_combo.count() == 0:
            self._refresh_session_list()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override 관례
        super().showEvent(event)
        if self.mode_combo.currentIndex() == MODE_CAPTURE:
            self._refresh_camera_settings_label()

    def _refresh_camera_settings_label(self) -> None:
        settings = settings_manager.load_settings()
        self.camera_settings_label.setText(
            f"카메라: {settings['camera_type']} · 평균화 {settings['averaging_num_frames']}프레임 "
            f"· min_valid_ratio={settings['averaging_min_valid_ratio']:.2f}"
        )

    # ----------------------------------------------------------------- 촬영
    def _on_capture(self) -> None:
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
        # '설정' 탭 값으로 averaging.num_frames/min_valid_ratio만 override -
        # 나머지(노출시간, 동작거리 모드 등)는 config 파일 값 그대로 사용.
        avg_cfg = dict(cam_cfg.get("averaging") or {})
        avg_cfg["num_frames"] = settings["averaging_num_frames"]
        avg_cfg["min_valid_ratio"] = settings["averaging_min_valid_ratio"]
        cam_cfg["averaging"] = avg_cfg

        self.btn_capture.setEnabled(False)
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 촬영 시작: {camera_type} "
            f"(averaging={avg_cfg['num_frames']}프레임, method={avg_cfg.get('method', 'median')}, "
            f"min_valid_ratio={avg_cfg['min_valid_ratio']:.2f})"
        )
        try:
            from src.camera import create_camera
            with create_camera(cam_cfg) as cam:
                frame = cam.capture()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "촬영 오류", str(exc))
            return
        finally:
            self.btn_capture.setEnabled(True)

        label = datetime.now().strftime("capture_%H%M%S")
        tmp_path = Path(tempfile.gettempdir()) / f"tcp_{label}.png"
        cv2.imwrite(str(tmp_path), frame.intensity)

        self._frame_entries[label] = _FrameEntry(
            kind="capture",
            image_path=str(tmp_path),
            pcd_organized=frame.points_organized,
            valid_mask=frame.valid_mask,
            pcd_std=frame.points_std,  # num_frames=1이면 None (자동 폴백 대상)
        )
        self.capture_list.addItem(QListWidgetItem(label))
        self.capture_list.setCurrentRow(self.capture_list.count() - 1)

        valid_ratio = 100.0 * frame.valid_mask.sum() / frame.valid_mask.size
        self.log_message.emit(f"[{self.LOG_PREFIX}] 촬영 완료: {label} (유효 픽셀 {valid_ratio:.1f}%)")

    # ------------------------------------------------------- 이미지 불러오기
    def _refresh_session_list(self) -> None:
        current_text = self.session_dir_combo.currentText()
        self.session_dir_combo.clear()
        if not DEFAULT_DATASET_ROOT.is_dir():
            self.log_message.emit(f"[{self.LOG_PREFIX}] 데이터셋 폴더 없음: {DEFAULT_DATASET_ROOT}")
            return
        sessions = sorted(
            d.name for d in DEFAULT_DATASET_ROOT.iterdir()
            if d.is_dir() and _usable_session_frames(d)
        )
        for name in sessions:
            self.session_dir_combo.addItem(name, str(DEFAULT_DATASET_ROOT / name))
        if current_text:
            self.session_dir_combo.setEditText(current_text)
        self.log_message.emit(f"[{self.LOG_PREFIX}] 세션 폴더 스캔: {len(sessions)}개 (프레임 있는 것만)")

    def _on_browse_session_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", str(DEFAULT_DATASET_ROOT))
        if path:
            self.session_dir_combo.setEditText(path)
            self._on_scan_frames()

    def _resolve_session_dir(self) -> Path | None:
        text = self.session_dir_combo.currentText().strip()
        if not text:
            return None
        idx = self.session_dir_combo.findText(text)
        if idx >= 0:
            data = self.session_dir_combo.itemData(idx)
            if data:
                return Path(data)
        return Path(text)

    def _on_scan_frames(self) -> None:
        session_dir = self._resolve_session_dir()
        if session_dir is None:
            QMessageBox.warning(self, "알림", "세션 폴더를 먼저 지정하세요.")
            return
        if not session_dir.is_dir():
            QMessageBox.warning(self, "알림", f"폴더를 찾을 수 없습니다: {session_dir}")
            return

        frames = _usable_session_frames(session_dir)
        if not frames:
            QMessageBox.information(
                self, "프레임 없음",
                f"{session_dir}\n에서 intensity/pointcloud_organized/valid_mask가 "
                "모두 있는 프레임을 찾지 못했습니다.",
            )
            return

        n_added = 0
        for frame_name in frames:
            label = f"{session_dir.name}/{frame_name}"
            if label in self._frame_entries:
                continue
            self._frame_entries[label] = _FrameEntry(
                kind="session", session_dir=session_dir, frame_name=frame_name,
            )
            self.capture_list.addItem(QListWidgetItem(label))
            n_added += 1

        self.log_message.emit(
            f"[{self.LOG_PREFIX}] {session_dir.name}: 프레임 {len(frames)}개 중 {n_added}개 목록에 추가"
        )
        if n_added > 0:
            self.capture_list.setCurrentRow(self.capture_list.count() - 1)

    # ------------------------------------------------------------- 프레임 선택
    def _on_capture_row_changed(self, row: int) -> None:
        if row < 0:
            self.btn_save_session.setEnabled(False)
            return
        label = self.capture_list.item(row).text()
        entry = self._frame_entries[label]

        if entry.kind == "capture":
            self._current_image_path = entry.image_path
            self._pcd_organized = entry.pcd_organized
            self._valid_mask = entry.valid_mask
            self._pcd_std = entry.pcd_std
            self.btn_save_session.setEnabled(True)
        else:  # "session" - 디스크에서 지금 읽는다 (스캔 시점엔 안 읽음)
            image_path = entry.session_dir / "intensity" / f"{entry.frame_name}.png"
            try:
                pcd_organized = np.load(entry.session_dir / "pointcloud_organized" / f"{entry.frame_name}.npy")
                valid_mask = np.load(entry.session_dir / "valid_mask" / f"{entry.frame_name}.npy")
            except OSError as exc:
                QMessageBox.critical(self, "불러오기 오류", f"프레임을 읽을 수 없습니다: {exc}")
                self.btn_save_session.setEnabled(False)
                return
            self._current_image_path = str(image_path)
            self._pcd_organized = pcd_organized
            self._valid_mask = valid_mask
            self._pcd_std = None
            # 이미 세션 폴더에 있는 프레임이라 다시 저장할 필요가 없음.
            self.btn_save_session.setEnabled(False)

        self._on_new_frame_acquired(label)

    # -------------------------------------------------------------- 세션 저장
    def _on_save_as_session(self) -> None:
        row = self.capture_list.currentRow()
        if row < 0:
            return
        label = self.capture_list.item(row).text()
        entry = self._frame_entries[label]
        if entry.kind != "capture":
            return  # 이미 세션 폴더에서 온 프레임 - 버튼이 비활성화돼 있어야 하지만 방어적으로 한 번 더 체크

        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = DEFAULT_DATASET_ROOT / session_name
        for sub in ("intensity", "pointcloud_organized", "valid_mask"):
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        frame_name = "frame_0000"
        image = cv2.imread(entry.image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(session_dir / "intensity" / f"{frame_name}.png"), image)
        np.save(session_dir / "pointcloud_organized" / f"{frame_name}.npy", entry.pcd_organized)
        np.save(session_dir / "valid_mask" / f"{frame_name}.npy", entry.valid_mask)

        self.log_message.emit(f"[{self.LOG_PREFIX}] 세션으로 저장 완료: {session_dir}")
        QMessageBox.information(
            self, "저장 완료",
            f"세션 폴더로 저장했습니다:\n{session_dir}\n\n"
            "'이미지 불러오기' 모드에서 이 폴더를 그대로 불러올 수 있습니다.",
        )