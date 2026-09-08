"""탭 8: 수동 라벨링 (RotHead rotation_matrix + PVNet keypoints_2d 동시 생성).

ICP 자동 정합이 애매한 자세(겹침·기울어짐이 심함)에서 실패할 때, 사람이 직접
obj별 회전각(Rx/Ry/Rz)을 입력해서 라벨을 만드는 툴이다. LiveCaptureICPTab을
상속해서 촬영/검출/CAD 선택/이미지 뷰어(obj 번호 오버레이 포함)는 그대로 재사용
하고, ICP 정합 관련 UI(ICP 정합 실행 버튼, 결과 카드, 3D 뷰어)는 숨긴 뒤 그
자리에 각도 입력 패널을 넣었다.

핵심 흐름: 각도 입력 -> 저장 누르면 즉시
    1) Rx/Ry/Rz -> 회전행렬 R = Rz @ Ry @ Rx (icp_runner._Rx/_Ry/_Rz와 동일 컨벤션)
    2) 마스크 영역 포인트클라우드 중심(centroid)을 위치(t)로 사용 (ICP 없이)
    3) rotation_matrix 라벨 저장 (RotHead용, data/rotation_labels.json - 기존
       LiveLabelGenerationTab이 쓰는 것과 완전히 같은 파일/스키마, 이어서 누적됨)
    4) 원통형 대칭(cylindrical_*)이면 canonicalize_axial_rotation()으로 스핀 제거
    5) CAD 키포인트(FPS)를 그 회전+위치로 투영 -> keypoints_2d 라벨 저장
       (PVNet용, data/pvnet_labels.json)

즉 한 번의 각도 입력으로 RotHead와 PVNet 라벨을 동시에 만든다.

실시간 오버레이: obj별 Rx/Ry/Rz 스핀박스 값이 바뀔 때마다, CAD 점군(가볍게
서브샘플링한 것)을 현재 입력 각도+centroid 위치로 투영해서 이미지 위에 반투명
노란 점으로 그린다 (ImageViewer.set_pose_overlay). 이 점들이 사진 속 실제 물체
실루엣과 겹치면 입력한 각도가 잘 맞다는 뜻 - 저장하기 전에 눈으로 바로 확인
가능하다.

주의: "CAD 축보정" 값은 스핀박스를 바꿔도 실시간으로 CAD를 다시 로드하지
않는다(무거운 연산이라 매 입력마다 하면 느려짐) - 검출을 다시 실행하거나
저장을 누를 때 반영된다. 이 값은 부품마다 한 번만 맞춰두는 값이라 큰 문제가
안 된다.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from app.core import icp_runner
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points
from app.core.icp_runner import _Rx, _Ry, _Rz
from app.core.paths import PROJECT_ROOT
from app.tabs.live_capture_icp_tab import LiveCaptureICPTab
from src.detection.pvnet import canonicalize_axial_rotation, farthest_point_sampling

ROTMDET_TAB_NAME = "RTMDet"

DEFAULT_MASK_OUT_DIR = PROJECT_ROOT / "data" / "rotation_labels_masks"       # RotHead와 동일 폴더 공유
DEFAULT_IMAGE_OUT_DIR = PROJECT_ROOT / "data" / "rotation_labels_images"     # RotHead와 동일 폴더 공유
DEFAULT_ROTATION_LABELS_OUT = PROJECT_ROOT / "data" / "rotation_labels.json"  # RotHead와 완전히 같은 파일
DEFAULT_PVNET_LABELS_OUT = PROJECT_ROOT / "data" / "pvnet_labels.json"
DEFAULT_KEYPOINTS_3D_OUT = PROJECT_ROOT / "data" / "pvnet_keypoints_3d.npy"

PVNET_NUM_KEYPOINTS = 8
SYMMETRY_AXIS_CHOICES = ["none", "x", "y", "z"]
DEFAULT_SYMMETRY_AXIS = "y"  # 볼트(원통형) 기준 - 부품 바뀌면 UI에서 변경
OVERLAY_MAX_POINTS = 300  # 실시간 오버레이용 CAD 서브샘플 점 개수 (속도용)

# 이 탭은 ICP 정합을 아예 안 하므로, ICPParams 전체(fitness threshold, outlier,
# FGR 파라미터 등)가 필요 없다. 실제로 쓰는 건 CAD 로딩(축보정)과 마스크 침식뿐이라
# 그 두 가지만 이 탭 자체의 작은 입력칸으로 따로 둔다 - 아래 기본값은
# icp_runner.ICPParams()의 기본값과 동일하게 맞췄다.
DEFAULT_CAD_AXIS_RX_DEG = -90.0
DEFAULT_CAD_AXIS_RY_DEG = 90.0
DEFAULT_CAD_AXIS_RZ_DEG = 90.0
DEFAULT_MASK_ERODE_PX = 1

# obj별 각도 입력칸 기본값/범위 - ICPParams의 init_*/*_limit_deg에 더는 의존하지
# 않고 이 탭 자체 상수로 관리한다 (ICP 파라미터 박스를 통째로 숨겼으므로).
DEFAULT_ANGLE_RX_DEG = 180.0
DEFAULT_ANGLE_RY_DEG = 180.0
DEFAULT_ANGLE_RZ_DEG = 90.0

AXIS_KEYS = ("Rx", "Ry", "Rz")


class ManualLabelingTab(LiveCaptureICPTab):
    LOG_PREFIX = "수동 라벨링 탭"

    def __init__(self, parent=None):
        self._rotation_labels: list[dict] = []
        self._pvnet_labels: list[dict] = []
        self._n_saved_session = 0
        self._angle_widgets: dict[int, dict] = {}
        self._include_checkboxes: dict[int, QCheckBox] = {}
        self._instance_centroids_mm: dict[int, np.ndarray] = {}
        self._keypoints_3d: np.ndarray | None = None
        self._cad_overlay_points_m: np.ndarray | None = None
        self._cad_center_m: np.ndarray | None = None
        self._keypoints_3d_cad_path: str | None = None
        self._intrinsics: tuple | None = None
        super().__init__(parent)
        self._restrict_to_rtmdet()
        self._hide_icp_only_widgets()
        self._build_manual_panel()
        self._load_existing_labels()

    # ------------------------------------------------------- UI 정리
    def _restrict_to_rtmdet(self) -> None:
        """RotHead pose는 여기서 필요 없음 (사람이 직접 각도를 입력하므로) -
        RTMDet 서브탭만 남겨서 순수 검출(마스크/bbox)만 받는다."""
        remove_indices = [
            i for i in range(self.pipeline_tabs.count())
            if self.pipeline_tabs.tabText(i) != ROTMDET_TAB_NAME
        ]
        for i in reversed(remove_indices):
            widget = self.pipeline_tabs.widget(i)
            self.pipeline_tabs.removeTab(i)
            widget.deleteLater()
        self.pipeline_tabs.tabBar().setVisible(False)

    def _hide_icp_only_widgets(self) -> None:
        """ICP 정합은 여기서 안 쓰므로 ICP 파라미터 박스 전체를 숨긴다.

        fitness threshold, outlier 제거, FGR 파라미터 등은 이 탭에서 전혀
        안 쓰인다 - 실제로 필요한 건 CAD 축보정과 마스크 침식뿐이라, 그
        둘만 _build_manual_panel()에서 이 탭 전용의 작은 입력칸으로 따로 둔다.
        """
        self.btn_run_icp.hide()
        self.params_scroll.hide()
        self.result_title_label.hide()
        self.result_scroll.hide()
        self.btn_open_viewer.hide()

    def _build_manual_panel(self) -> None:
        right_layout = self.result_scroll.parentWidget().layout()
        insert_idx = right_layout.indexOf(self.result_scroll)

        # --- CAD 로딩에 실제로 쓰이는 값만 (ICP 파라미터 박스 대체) ---
        cad_axis_label = QLabel("CAD 축보정 (deg)")
        cad_axis_label.setStyleSheet("font-weight: 600;")
        right_layout.insertWidget(insert_idx, cad_axis_label)

        self.cad_axis_spins: dict[str, QDoubleSpinBox] = {}
        cad_axis_defaults = {
            "Rx": DEFAULT_CAD_AXIS_RX_DEG,
            "Ry": DEFAULT_CAD_AXIS_RY_DEG,
            "Rz": DEFAULT_CAD_AXIS_RZ_DEG,
        }
        cad_axis_row = QHBoxLayout()
        for key in AXIS_KEYS:
            cad_axis_row.addWidget(QLabel(key))
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setValue(cad_axis_defaults[key])
            spin.setSingleStep(1.0)
            spin.setMaximumWidth(70)
            cad_axis_row.addWidget(spin)
            self.cad_axis_spins[key] = spin
        right_layout.insertLayout(insert_idx + 1, cad_axis_row)

        erode_row = QHBoxLayout()
        erode_row.addWidget(QLabel("마스크 침식 px"))
        self.mask_erode_spin = QDoubleSpinBox()
        self.mask_erode_spin.setDecimals(0)
        self.mask_erode_spin.setRange(0, 10)
        self.mask_erode_spin.setValue(DEFAULT_MASK_ERODE_PX)
        erode_row.addWidget(self.mask_erode_spin)
        erode_row.addStretch(1)
        right_layout.insertLayout(insert_idx + 2, erode_row)

        sym_row = QHBoxLayout()
        sym_row.addWidget(QLabel("대칭축"))
        self.symmetry_axis_combo = QComboBox()
        self.symmetry_axis_combo.addItems(SYMMETRY_AXIS_CHOICES)
        self.symmetry_axis_combo.setCurrentText(DEFAULT_SYMMETRY_AXIS)
        self.symmetry_axis_combo.setToolTip(
            "CAD 로컬 좌표계에서 원통 대칭축 (scripts/check_cad_symmetry_axis.py로 확인).\n"
            "none이면 정규화 없이 입력한 각도 그대로 keypoints_2d를 계산합니다."
        )
        sym_row.addWidget(self.symmetry_axis_combo)
        right_layout.insertLayout(insert_idx + 3, sym_row)

        title = QLabel("obj별 각도 입력 (Rx/Ry/Rz, deg)")
        right_layout.insertWidget(insert_idx + 4, title)
        hint = QLabel("점 오버레이가 사진 속 물체와 겹치면 각도가 맞는 것")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        right_layout.insertWidget(insert_idx + 5, hint)

        self.manual_scroll = QScrollArea()
        self.manual_scroll.setWidgetResizable(True)
        self.manual_container = QWidget()
        self.manual_layout = QVBoxLayout(self.manual_container)
        self.manual_layout.addStretch(1)
        self.manual_scroll.setWidget(self.manual_container)
        right_layout.insertWidget(insert_idx + 6, self.manual_scroll, 1)

        self.btn_save_manual = QPushButton("라벨 저장 (현재 프레임)")
        self.btn_save_manual.clicked.connect(self._on_save_manual_labels)
        right_layout.insertWidget(insert_idx + 7, self.btn_save_manual)

        self.manual_count_label = QLabel(f"누적 저장: {self._n_saved_session}건")
        right_layout.insertWidget(insert_idx + 8, self.manual_count_label)

    # ------------------------------------------------------- 검출 훅
    def _on_run_detection(self) -> None:
        super()._on_run_detection()
        self._ensure_cad_and_intrinsics_ready()
        self._render_manual_panel()

    def _on_new_frame_acquired(self, frame_label: str) -> None:
        super()._on_new_frame_acquired(frame_label)
        self._clear_manual_panel()  # 새 프레임이면 이전 각도 입력칸은 의미 없어짐
        self._instance_centroids_mm.clear()
        self.image_viewer.clear_pose_overlays()

    def _clear_manual_panel(self) -> None:
        self._angle_widgets.clear()
        self._include_checkboxes.clear()
        while self.manual_layout.count() > 1:  # 마지막 addStretch(1)은 남김
            item = self.manual_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # ------------------------------------------------------- CAD/intrinsic 준비
    def _ensure_cad_and_intrinsics_ready(self) -> None:
        """검출 직후 한 번 호출 - CAD 로드 + 오버레이용 서브샘플 점 + 키포인트
        + intrinsic을 전부 준비해둔다 (각도 조정 시 즉시 오버레이를 그릴 수 있게).
        """
        cad_path = self.cad_combo.itemData(self.cad_combo.currentIndex())
        if not cad_path:
            return

        cad_axis_params = icp_runner.ICPParams(
            cad_axis_roll_deg=self.cad_axis_spins["Rx"].value(),
            cad_axis_pitch_deg=self.cad_axis_spins["Ry"].value(),
            cad_axis_yaw_deg=self.cad_axis_spins["Rz"].value(),
        )
        axis_tuple = cad_axis_params.cad_axis_correction_deg
        if self._cad_pcd is None or self._cad_path_loaded != cad_path or self._cad_axis_loaded != axis_tuple:
            self.log_message.emit(f"[{self.LOG_PREFIX}] CAD 로드 중: {cad_path}")
            self._cad_pcd = icp_runner.load_cad_as_pcd(cad_path, cad_axis_params)
            self._cad_path_loaded = cad_path
            self._cad_axis_loaded = axis_tuple

            cad_points = np.asarray(self._cad_pcd.points)
            self._cad_center_m = cad_points.mean(axis=0)  # CAD 로컬 원점이 아니라
            # 실제 기하학적 중심 - 회전은 반드시 이 점을 축으로 해야 함 (원점이
            # 물체 모서리 등에 있으면 원점 기준 회전은 물체가 궤도를 도는 것처럼
            # 보이는 오차가 생김)

            self._keypoints_3d = farthest_point_sampling(
                cad_points, num_keypoints=PVNET_NUM_KEYPOINTS, include_centroid=True
            )
            self._keypoints_3d_cad_path = cad_path
            np.save(DEFAULT_KEYPOINTS_3D_OUT, self._keypoints_3d)

            if len(cad_points) > OVERLAY_MAX_POINTS:
                idx = np.random.default_rng(0).choice(len(cad_points), OVERLAY_MAX_POINTS, replace=False)
                self._cad_overlay_points_m = cad_points[idx]
            else:
                self._cad_overlay_points_m = cad_points

        if self._intrinsics is None and self._pcd_organized is not None and self._valid_mask is not None:
            self._intrinsics = estimate_intrinsics_from_organized_pcd(self._pcd_organized, self._valid_mask)
            fx, fy, cx, cy = self._intrinsics
            self.log_message.emit(
                f"[{self.LOG_PREFIX}] intrinsic 추정: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
            )

    # ------------------------------------------------------- 각도 입력 패널
    def _render_manual_panel(self) -> None:
        """검출된 인스턴스마다 Rx/Ry/Rz 입력 카드를 만든다.

        기본값은 이 탭 상단의 상수(DEFAULT_ANGLE_*_DEG)를 쓰고, 범위는 -180~180
        전체를 허용한다. 카드가 만들어지자마자 기본 각도로 오버레이를 한 번
        그려서, 만들어진 직후부터 바로 비교해볼 수 있게 한다.

        맨 처음에 이미지 뷰어의 오버레이를 전부 지운다 - 안 그러면 이전 검출
        (예: conf 0.5로 6개 검출)에서 그려진 obj3~5 오버레이가, 이번 검출
        (conf 0.8로 3개만 검출)에서 obj0~2만 다시 그려도 안 지워지고 그대로
        남는 갱신 버그가 있었다 (인덱스별로만 덮어쓰고, 이번엔 아예 안 만들어진
        인덱스는 그대로 방치됐었음).
        """
        self.image_viewer.clear_pose_overlays()
        self._clear_manual_panel()
        defaults = {"Rx": DEFAULT_ANGLE_RX_DEG, "Ry": DEFAULT_ANGLE_RY_DEG, "Rz": DEFAULT_ANGLE_RZ_DEG}

        for i, det in enumerate(self._last_detections):
            self._instance_centroids_mm[i] = self._compute_centroid_mm(det)

            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setStyleSheet(
                "QFrame { border: 1px solid #ddd; border-radius: 6px; padding: 4px; margin-bottom: 4px; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)

            title_row = QHBoxLayout()
            title = QLabel(f"obj{i}")
            title.setStyleSheet("font-weight: 600;")
            title_row.addWidget(title)
            title_row.addStretch(1)
            include_cb = QCheckBox("라벨링 포함")
            include_cb.setChecked(True)
            include_cb.toggled.connect(lambda checked, idx=i: self._on_include_toggled(idx, checked))
            title_row.addWidget(include_cb)
            card_layout.addLayout(title_row)
            self._include_checkboxes[i] = include_cb

            spins: dict[str, QDoubleSpinBox] = {}
            for key in AXIS_KEYS:
                row = QHBoxLayout()
                row.addWidget(QLabel(key))
                spin = QDoubleSpinBox()
                spin.setRange(-180.0, 180.0)
                spin.setValue(defaults[key])
                spin.setSingleStep(1.0)
                spin.valueChanged.connect(lambda _v, idx=i: self._update_pose_overlay(idx))
                row.addWidget(spin)
                card_layout.addLayout(row)
                spins[key] = spin
            self._angle_widgets[i] = spins

            self.manual_layout.insertWidget(self.manual_layout.count() - 1, card)
            self._update_pose_overlay(i)  # 기본 각도로 즉시 한 번 그려둠

    def _compute_centroid_mm(self, det) -> np.ndarray | None:
        if self._pcd_organized is None or self._valid_mask is None:
            return None
        pts_mm = icp_runner.extract_instance_points_mm(
            det.mask, self._pcd_organized, self._valid_mask, erode_px=int(self.mask_erode_spin.value())
        )
        if pts_mm is None or len(pts_mm) < 3:
            return None
        return pts_mm.mean(axis=0)

    def _on_include_toggled(self, obj_index: int, checked: bool) -> None:
        """체크 해제 -> 각도 입력칸 비활성화 + 오버레이 즉시 제거 (시각화에 반영).
        다시 체크 -> 입력칸 재활성화 + 현재 각도로 오버레이 다시 그림."""
        spins = self._angle_widgets.get(obj_index)
        if spins:
            for spin in spins.values():
                spin.setEnabled(checked)

        if checked:
            self._update_pose_overlay(obj_index)
        else:
            self.image_viewer.set_pose_overlay(obj_index, None)

    def _update_pose_overlay(self, obj_index: int) -> None:
        """obj_index의 현재 Rx/Ry/Rz 스핀박스 값으로 CAD 서브샘플 점을 투영해서
        이미지 뷰어에 반투명 점으로 그린다. 필요한 것(centroid/CAD/intrinsic) 중
        하나라도 없거나, 이 오브젝트가 '라벨링 포함' 체크 해제 상태면 조용히
        아무것도 안 그린다."""
        include_cb = self._include_checkboxes.get(obj_index)
        if include_cb is not None and not include_cb.isChecked():
            self.image_viewer.set_pose_overlay(obj_index, None)
            return

        spins = self._angle_widgets.get(obj_index)
        centroid_mm = self._instance_centroids_mm.get(obj_index)
        if (spins is None or centroid_mm is None or self._cad_overlay_points_m is None
                or self._cad_center_m is None or self._intrinsics is None):
            self.image_viewer.set_pose_overlay(obj_index, None)
            return

        Rx, Ry, Rz = spins["Rx"].value(), spins["Ry"].value(), spins["Rz"].value()
        R = _Rz(Rz) @ _Ry(Ry) @ _Rx(Rx)

        points_centered_m = self._cad_overlay_points_m - self._cad_center_m  # CAD 중심을 원점으로
        points_cam_mm = (R @ points_centered_m.T).T * 1000.0 + centroid_mm
        points_2d = project_points(points_cam_mm, self._intrinsics)

        # 클러터가 심하면 마스크 경계가 옆 인스턴스로 새서 centroid가 잘못 잡히거나,
        # 다른 원인으로 투영이 어긋나 다른(검출 안 된) 볼트 위까지 오버레이가 번질
        # 수 있다. 방어적으로 "이 인스턴스 자신의 검출 bbox 주변(10% 여유)"을 벗어난
        # 점은 그리지 않는다 - 자기 자리가 아니면 절대 안 보이게 강제하는 안전장치.
        # (여유를 50%로 뒀을 때 볼트가 다닥다닥 붙은 클러터 장면에서 바로 옆 볼트
        # 영역까지 포함돼버려서 10%로 대폭 줄임 - 필요하면 더 좁혀도 됨.)
        det = self._last_detections[obj_index] if obj_index < len(self._last_detections) else None
        if det is not None:
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1
            margin_x, margin_y = bw * 0.1, bh * 0.1
            in_bounds = (
                (points_2d[:, 0] >= x1 - margin_x) & (points_2d[:, 0] <= x2 + margin_x)
                & (points_2d[:, 1] >= y1 - margin_y) & (points_2d[:, 1] <= y2 + margin_y)
            )
            n_dropped = int((~in_bounds).sum())
            if n_dropped > 0:
                bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                overlay_center = points_2d.mean(axis=0)
                dist = np.linalg.norm(overlay_center - bbox_center)
                self.log_message.emit(
                    f"[{self.LOG_PREFIX}] obj{obj_index}: 오버레이 점 {n_dropped}/{len(points_2d)}개가 "
                    f"자기 bbox 밖으로 나가서 제외됨 (bbox중심={bbox_center.round(1)}, "
                    f"투영중심={overlay_center.round(1)}, 거리={dist:.1f}px, centroid_mm={centroid_mm.round(1)})"
                )
            points_2d = points_2d[in_bounds]

        self.image_viewer.set_pose_overlay(obj_index, points_2d)

    # ------------------------------------------------------- 저장
    def _on_save_manual_labels(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 2D 검출을 실행하세요.")
            return
        if not self._current_image_path or not os.path.isfile(self._current_image_path):
            QMessageBox.warning(self, "알림", "촬영 이미지를 찾을 수 없습니다. 다시 촬영하세요.")
            return

        self._ensure_cad_and_intrinsics_ready()
        if self._cad_pcd is None:
            QMessageBox.warning(self, "알림", "CAD 모델을 선택하세요.")
            return
        if self._intrinsics is None:
            QMessageBox.warning(self, "알림", "카메라 intrinsic을 추정하지 못했습니다 (유효 픽셀 부족).")
            return

        symmetry_axis = self.symmetry_axis_combo.currentText()

        DEFAULT_MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_tag = self._current_frame or "frame"
        image_ext = os.path.splitext(self._current_image_path)[1] or ".png"
        permanent_image_path = DEFAULT_IMAGE_OUT_DIR / f"{stamp}_{frame_tag}{image_ext}"
        shutil.copy2(self._current_image_path, permanent_image_path)

        n_saved = 0
        for i, det in enumerate(self._last_detections):
            include_cb = self._include_checkboxes.get(i)
            if include_cb is not None and not include_cb.isChecked():
                self.log_message.emit(f"[{self.LOG_PREFIX}] obj{i}: '라벨링 포함' 체크 해제됨 - 건너뜀")
                continue

            spins = self._angle_widgets.get(i)
            centroid_mm = self._instance_centroids_mm.get(i)
            if spins is None or centroid_mm is None:
                self.log_message.emit(f"[{self.LOG_PREFIX}] obj{i}: 유효 포인트 부족 - 건너뜀")
                continue

            Rx, Ry, Rz = spins["Rx"].value(), spins["Ry"].value(), spins["Rz"].value()
            R = _Rz(Rz) @ _Ry(Ry) @ _Rx(Rx)

            mask_filename = f"{stamp}_{frame_tag}_obj{i}.npy"
            mask_path = DEFAULT_MASK_OUT_DIR / mask_filename
            np.save(mask_path, det.mask.astype(bool))

            # --- RotHead 라벨 (rotation_matrix, 정규화 안 함 - 실제 측정값 그대로) ---
            self._rotation_labels.append({
                "image": str(permanent_image_path),
                "mask": str(mask_path),
                "bbox": [float(v) for v in det.bbox],
                "rotation_matrix": R.tolist(),
            })

            # --- PVNet 라벨 (keypoints_2d, 대칭축 있으면 정규화 후 투영) ---
            R_for_projection = canonicalize_axial_rotation(R, symmetry_axis) if symmetry_axis != "none" else R
            keypoints_centered_m = self._keypoints_3d - self._cad_center_m  # CAD 중심을 원점으로
            keypoints_cam_mm = (R_for_projection @ keypoints_centered_m.T).T * 1000.0 + centroid_mm
            keypoints_2d = project_points(keypoints_cam_mm, self._intrinsics)

            self._pvnet_labels.append({
                "image": str(permanent_image_path),
                "mask": str(mask_path),
                "bbox": [float(v) for v in det.bbox],
                "keypoints_2d": keypoints_2d.tolist(),
            })

            n_saved += 1
            self._n_saved_session += 1

        if n_saved == 0:
            QMessageBox.information(self, "저장 안 됨", "저장된 인스턴스가 없습니다 (유효 포인트 부족 등).")
            return

        DEFAULT_ROTATION_LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_ROTATION_LABELS_OUT, "w", encoding="utf-8") as f:
            json.dump(self._rotation_labels, f, ensure_ascii=False, indent=2)
        with open(DEFAULT_PVNET_LABELS_OUT, "w", encoding="utf-8") as f:
            json.dump(self._pvnet_labels, f, ensure_ascii=False, indent=2)

        self.manual_count_label.setText(f"누적 저장: {self._n_saved_session}건")
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] {n_saved}건 저장 (누적 {self._n_saved_session}건) -> "
            f"{DEFAULT_ROTATION_LABELS_OUT.name} + {DEFAULT_PVNET_LABELS_OUT.name}"
        )

    def _load_existing_labels(self) -> None:
        """탭을 다시 켰을 때 기존 라벨 파일이 있으면 이어서 누적한다."""
        if DEFAULT_ROTATION_LABELS_OUT.is_file():
            try:
                with open(DEFAULT_ROTATION_LABELS_OUT, "r", encoding="utf-8") as f:
                    self._rotation_labels = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                self.log_message.emit(f"[{self.LOG_PREFIX}] rotation_labels.json 로드 실패 (무시): {exc}")

        if DEFAULT_PVNET_LABELS_OUT.is_file():
            try:
                with open(DEFAULT_PVNET_LABELS_OUT, "r", encoding="utf-8") as f:
                    self._pvnet_labels = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                self.log_message.emit(f"[{self.LOG_PREFIX}] pvnet_labels.json 로드 실패 (무시): {exc}")

        self._n_saved_session = len(self._pvnet_labels)
        if self._n_saved_session:
            self.manual_count_label.setText(f"누적 저장: {self._n_saved_session}건 (기존 파일에 이어서 씀)")