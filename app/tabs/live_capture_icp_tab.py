"""탭 5: ICP 정합테스트(TCP) - 실시간 촬영 기반.

세션 폴더 대신 카메라로 즉시 촬영해서 바로 검출+ICP까지 돌린다.
공통 로직(파이프라인 실행, ICP/FGR 파라미터, 결과 패널, 3D 뷰어)은
ICPWorkbenchTab에 있고, 이 파일은 "프레임을 어떻게 얻는지"(카메라 타입
선택 + 촬영 버튼)만 구현한다 - app/tabs/session_icp_tab.py와 정확히
같은 확장 지점을 쓴다.

포인트클라우드 품질 개선: src.camera.create_camera()의 다중 프레임 평균화
(configs/camera_config_helios.yaml / camera_config_femto.yaml의 `averaging`
섹션)를 그대로 재사용한다. num_frames/min_valid_ratio를 이 탭에서 조절 가능하게 노출했다.

주의 (알려진 제약):
    촬영(cam.capture())은 이 탭 안에서 동기적으로 실행된다 - 즉 촬영 중에는
    UI가 잠깐 멈춘다. num_frames를 크게 잡을수록(예: 10장 평균화) 촬영
    시간이 비례해서 늘어나므로 몇 초간 응답이 없을 수 있다. 테스트용 도구
    라서 우선 단순하게 만들었고, 실제로 불편하면 QThread로 옮기면 된다.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QMessageBox,
)

from app.core.paths import PROJECT_ROOT, DEFAULT_DATASET_ROOT
from app.tabs.icp_workbench_base import ICPWorkbenchTab

CAMERA_CONFIG_PATHS = {
    "lucid_helios": PROJECT_ROOT / "configs" / "camera_config_helios.yaml",
    "femto_bolt": PROJECT_ROOT / "configs" / "camera_config_femto.yaml",
}
DEFAULT_AVERAGING_FRAMES = 8  # 1이면 평균화 없이 기존과 동일 (품질 개선 원하면 5~10 권장)
DEFAULT_MIN_VALID_RATIO = 0.6  # config yaml 기본값과 동일 - 이 탭에서 override 가능


class LiveCaptureICPTab(ICPWorkbenchTab):
    LOG_PREFIX = "ICP(TCP) 탭"

    def __init__(self, parent=None):
        self._capture_history: dict[str, dict] = {}  # label -> {image_path, pcd, valid_mask}
        super().__init__(parent)

    def _build_acquisition_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("카메라 타입"))
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(list(CAMERA_CONFIG_PATHS.keys()))
        layout.addWidget(self.camera_type_combo)

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("평균화 프레임 수"))
        self.spin_avg_frames = QSpinBox()
        self.spin_avg_frames.setRange(1, 30)
        self.spin_avg_frames.setValue(DEFAULT_AVERAGING_FRAMES)
        avg_row.addWidget(self.spin_avg_frames)
        layout.addLayout(avg_row)

        # 2026-07 추가: min_valid_ratio도 UI에서 조절 가능하게 노출.
        # "매 프레임 랜덤 드롭아웃으로 생긴 구멍"을 다중 프레임으로 메우는
        # 핵심 파라미터라 실험적으로 자주 바꿔볼 값이라 판단했다 (num_frames
        # 옆에 나란히 둠). 낮출수록 커버리지↑, 노이즈 유입 위험도 같이↑.
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("min_valid_ratio"))
        self.spin_min_valid_ratio = QDoubleSpinBox()
        self.spin_min_valid_ratio.setRange(0.05, 1.0)
        self.spin_min_valid_ratio.setSingleStep(0.05)
        self.spin_min_valid_ratio.setDecimals(2)
        self.spin_min_valid_ratio.setValue(DEFAULT_MIN_VALID_RATIO)
        ratio_row.addWidget(self.spin_min_valid_ratio)
        layout.addLayout(ratio_row)
        ratio_hint = QLabel("픽셀이 최종 유효로 인정되려면 N프레임 중 최소 이 비율 이상에서\n"
                             "유효해야 함. 낮출수록 커버리지(빈틈 메움)는 늘지만 노이즈 낀\n"
                             "픽셀도 더 받아들이게 됨 - config yaml 기본값(0.6)을 이 탭에서만\n"
                             "실험적으로 override.")
        ratio_hint.setStyleSheet("color: #888; font-size: 10px;")
        ratio_hint.setWordWrap(True)
        layout.addWidget(ratio_hint)
        avg_hint = QLabel("1=평균화 없음(기존과 동일). 5~10부터 depth 노이즈\n"
                           "감소가 체감됨 - 커질수록 촬영 시간도 비례해서 늘어남.")
        avg_hint.setStyleSheet("color: #888; font-size: 10px;")
        avg_hint.setWordWrap(True)
        layout.addWidget(avg_hint)

        self.btn_capture = QPushButton("촬영")
        self.btn_capture.clicked.connect(self._on_capture)
        layout.addWidget(self.btn_capture)

        layout.addWidget(QLabel("촬영 이력 (이번 실행 중, 메모리에만 보관)"))
        self.capture_list = QListWidget()
        self.capture_list.currentRowChanged.connect(self._on_capture_row_changed)
        layout.addWidget(self.capture_list, stretch=1)

        self.btn_save_session = QPushButton("세션으로 저장")
        self.btn_save_session.setToolTip(
            "현재 촬영본을 data/dataset/ 밑에 표준 세션 폴더로 저장 - "
            "이후 '4. ICP 정합 테스트'나 학습 파이프라인에서 재사용 가능"
        )
        self.btn_save_session.clicked.connect(self._on_save_as_session)
        self.btn_save_session.setEnabled(False)
        layout.addWidget(self.btn_save_session)

        return panel

    # ----------------------------------------------------------------- 촬영
    def _on_capture(self) -> None:
        camera_type = self.camera_type_combo.currentText()
        config_path = CAMERA_CONFIG_PATHS[camera_type]
        if not config_path.is_file():
            QMessageBox.critical(self, "설정 파일 없음", f"{config_path} 파일을 찾을 수 없습니다.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        cam_cfg = dict(full_cfg["camera"])
        # 이 탭의 스핀박스 값으로 averaging.num_frames/min_valid_ratio만 override -
        # 나머지(노출시간, 동작거리 모드 등)는 config 파일 값 그대로 사용.
        avg_cfg = dict(cam_cfg.get("averaging") or {})
        avg_cfg["num_frames"] = self.spin_avg_frames.value()
        avg_cfg["min_valid_ratio"] = self.spin_min_valid_ratio.value()
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

        self._capture_history[label] = {
            "image_path": str(tmp_path),
            "pcd_organized": frame.points_organized,
            "valid_mask": frame.valid_mask,
            "pcd_std": frame.points_std,  # num_frames=1이면 None (자동 폴백 대상)
        }
        self.capture_list.addItem(QListWidgetItem(label))
        self.capture_list.setCurrentRow(self.capture_list.count() - 1)

        valid_ratio = 100.0 * frame.valid_mask.sum() / frame.valid_mask.size
        self.log_message.emit(f"[{self.LOG_PREFIX}] 촬영 완료: {label} (유효 픽셀 {valid_ratio:.1f}%)")

    def _on_capture_row_changed(self, row: int) -> None:
        if row < 0:
            self.btn_save_session.setEnabled(False)
            return
        label = self.capture_list.item(row).text()
        entry = self._capture_history[label]

        self._current_image_path = entry["image_path"]
        self._pcd_organized = entry["pcd_organized"]
        self._valid_mask = entry["valid_mask"]
        self._pcd_std = entry.get("pcd_std")
        self.btn_save_session.setEnabled(True)

        self._on_new_frame_acquired(label)

    # -------------------------------------------------------------- 세션 저장
    def _on_save_as_session(self) -> None:
        row = self.capture_list.currentRow()
        if row < 0:
            return
        label = self.capture_list.item(row).text()
        entry = self._capture_history[label]

        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = DEFAULT_DATASET_ROOT / session_name
        for sub in ("intensity", "pointcloud_organized", "valid_mask"):
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        frame_name = "frame_0000"
        image = cv2.imread(entry["image_path"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(session_dir / "intensity" / f"{frame_name}.png"), image)
        np.save(session_dir / "pointcloud_organized" / f"{frame_name}.npy", entry["pcd_organized"])
        np.save(session_dir / "valid_mask" / f"{frame_name}.npy", entry["valid_mask"])

        self.log_message.emit(f"[{self.LOG_PREFIX}] 세션으로 저장 완료: {session_dir}")
        QMessageBox.information(
            self, "저장 완료",
            f"세션 폴더로 저장했습니다:\n{session_dir}\n\n"
            "'4. ICP 정합 테스트'나 학습 파이프라인에서 이 폴더를 그대로 불러올 수 있습니다."
        )