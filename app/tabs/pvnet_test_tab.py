"""탭: PVNet 테스트 (학습된 모델로 실시간 추론).

"5. ICP 정합테스트(TCP)"(LiveCaptureICPTab)를 그대로 상속해서 [촬영] ->
[2D 검출 실행]까지 100% 동일한 화면(카메라 캡처, RTMDet 검출, 이미지
뷰어)을 그대로 쓰고, ICP 정합 대신 "학습된 PVNet 모델로 추론"을 추가한다.

    [촬영]              -- LiveCaptureICPTab 그대로
      -> [2D 검출 실행]  -- ICPWorkbenchTab 그대로 (RTMDet bbox/mask)
      -> [모델 로드]     -- 이 파일에서 추가. scripts/train_pvnet.py 출력
                            폴더(train_config.json + 체크포인트 + keypoints_3d.npy)를
                            로드한다.
      -> [PVNet 추론 실행] -- 이 파일에서 추가. RTMDet 마스크로 크롭 -> PVNet
                            forward -> voting -> PnP. ICP처럼 point cloud/
                            CAD 가시면/초기 회전값 준비가 전혀 필요 없다 -
                            크롭 이미지 한 장과 로드된 모델만 있으면 된다.

결과 시각화는 새 UI를 안 만들고 기존 "라벨 미리보기" 오버레이
(image_viewer.set_label_preview_overlay, "PVNet 라벨 생성" 탭에서 만든 것과
동일)를 그대로 재사용한다 - 마스크(초록) + 키포인트(센트로이드=노랑,
나머지=주황)가 화면에 얹히고, "라벨 미리보기" 슬라이더로 투명도를 조절할
수 있다. ICP를 아예 안 쓰므로 우측 "ICP 결과" 패널/CAD 오버레이/3D 뷰어는
이 탭에서는 비어있다(CAD 경로/체크포인트 설정 UI는 ICPWorkbenchTab에서
그대로 상속돼 화면에 보이지만 이 탭 로직에서는 사용하지 않는다).

카메라 intrinsic(fx,fy,cx,cy)은 여전히 촬영된 포인트클라우드에서 추정한다
(estimate_intrinsics_from_organized_pcd) - PVNet 추론 자체엔 3D 정보가
필요 없지만, 키포인트를 화면(2D)에 재투영해서 보여주려면 필요하다.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QWidget,
)

from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd
from app.tabs.live_capture_icp_tab import LiveCaptureICPTab
from src.detection.rtmdet_inferencer_pvnet import (
    PVNetBundle, DEFAULT_CHECKPOINT_NAME, load_pvnet_bundle, run_pvnet_on_detections,
)


class PVNetTestTab(LiveCaptureICPTab):
    LOG_PREFIX = "PVNet 테스트 탭"

    def __init__(self, parent=None):
        self._pvnet_bundle: PVNetBundle | None = None
        super().__init__(parent)

    # ----------------------------------------------------- UI 확장 (좌측 패널)
    def _build_acquisition_panel(self) -> QWidget:
        panel = super()._build_acquisition_panel()
        layout = panel.layout()

        layout.addWidget(QLabel("PVNet 모델 (학습 출력 폴더)"))

        dir_row = QHBoxLayout()
        self.pvnet_output_dir_edit = QLineEdit()
        self.pvnet_output_dir_edit.setPlaceholderText("예: data/pvnet_output")
        dir_row.addWidget(self.pvnet_output_dir_edit, stretch=1)
        btn_browse_dir = QPushButton("선택")
        btn_browse_dir.clicked.connect(self._on_browse_pvnet_output_dir)
        dir_row.addWidget(btn_browse_dir)
        layout.addLayout(dir_row)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("체크포인트"))
        self.pvnet_checkpoint_combo = QComboBox()
        self.pvnet_checkpoint_combo.addItem(DEFAULT_CHECKPOINT_NAME)
        ckpt_row.addWidget(self.pvnet_checkpoint_combo, stretch=1)
        layout.addLayout(ckpt_row)

        self.btn_load_pvnet_model = QPushButton("모델 로드")
        self.btn_load_pvnet_model.setToolTip(
            "train_config.json + 체크포인트(.pth) + keypoints_3d.npy가\n"
            "같은 폴더에 있어야 합니다 (scripts/train_pvnet.py 출력 그대로)."
        )
        self.btn_load_pvnet_model.clicked.connect(self._on_load_pvnet_model)
        layout.addWidget(self.btn_load_pvnet_model)

        self.pvnet_model_status_label = QLabel("모델 아직 로드 안 됨")
        self.pvnet_model_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.pvnet_model_status_label.setWordWrap(True)
        layout.addWidget(self.pvnet_model_status_label)

        self.btn_run_pvnet = QPushButton("PVNet 추론 실행")
        self.btn_run_pvnet.setToolTip("먼저 '2D 검출 실행'과 '모델 로드'를 마쳐야 활성화됩니다.")
        self.btn_run_pvnet.clicked.connect(self._on_run_pvnet)
        self.btn_run_pvnet.setEnabled(False)
        layout.addWidget(self.btn_run_pvnet)

        self.pvnet_result_label = QLabel("")
        self.pvnet_result_label.setStyleSheet("color: #666; font-size: 11px;")
        self.pvnet_result_label.setWordWrap(True)
        layout.addWidget(self.pvnet_result_label)

        return panel

    def _on_browse_pvnet_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "PVNet 학습 출력 폴더 선택", self.pvnet_output_dir_edit.text(),
        )
        if not path:
            return
        self.pvnet_output_dir_edit.setText(path)
        self._refresh_pvnet_checkpoint_choices(path)

    def _refresh_pvnet_checkpoint_choices(self, output_dir: str) -> None:
        """폴더 안의 *.pth를 스캔해 콤보박스를 채운다. best.pth가 있으면 맨 위로."""
        self.pvnet_checkpoint_combo.clear()
        d = Path(output_dir)
        if not d.is_dir():
            return
        names = sorted((p.name for p in d.glob("*.pth")), key=lambda n: (n != DEFAULT_CHECKPOINT_NAME, n))
        if not names:
            self.pvnet_checkpoint_combo.addItem(DEFAULT_CHECKPOINT_NAME)
            return
        for name in names:
            self.pvnet_checkpoint_combo.addItem(name)

    # ----------------------------------------------------- 프레임/검출 훅
    def _on_new_frame_acquired(self, frame_label: str) -> None:
        super()._on_new_frame_acquired(frame_label)
        self.btn_run_pvnet.setEnabled(False)
        self.image_viewer.clear_label_preview_overlays()
        self.pvnet_result_label.setText("")

    def _on_run_detection(self) -> None:
        super()._on_run_detection()
        self.image_viewer.clear_label_preview_overlays()
        self.pvnet_result_label.setText("")
        self.btn_run_pvnet.setEnabled(bool(self._last_detections) and self._pvnet_bundle is not None)

    # ----------------------------------------------------- 모델 로드
    def _on_load_pvnet_model(self) -> None:
        """버튼 클릭 슬롯 - 예외 처리 원칙은 라벨 생성/저장 탭과 동일
        (PyQt 슬롯 안 예외는 조용히 죽으므로 항상 팝업+로그로 드러낸다)."""
        try:
            self._on_load_pvnet_model_impl()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"[{self.LOG_PREFIX}] PVNet 모델 로드 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "모델 로드 실패",
                f"예상치 못한 오류로 모델 로드에 실패했습니다:\n\n{exc}\n\n"
                "터미널의 상세 트레이스백을 확인하세요.",
            )
            raise

    def _on_load_pvnet_model_impl(self) -> None:
        output_dir = self.pvnet_output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "알림", "먼저 PVNet 학습 출력 폴더를 지정하세요.")
            return
        if not Path(output_dir).is_dir():
            QMessageBox.warning(self, "알림", f"폴더를 찾을 수 없습니다: {output_dir}")
            return

        checkpoint_name = self.pvnet_checkpoint_combo.currentText().strip() or DEFAULT_CHECKPOINT_NAME
        self._pvnet_bundle = load_pvnet_bundle(output_dir, checkpoint_name=checkpoint_name, device="cpu")

        cfg = self._pvnet_bundle.config
        self.pvnet_model_status_label.setText(
            f"로드됨: {checkpoint_name}\n"
            f"키포인트 {self._pvnet_bundle.keypoints_3d.shape[0]}개(센트로이드 포함), "
            f"crop_size={self._pvnet_bundle.crop_size}, "
            f"epoch={cfg.get('epochs', '?')}"
        )
        self.log_message.emit(f"[{self.LOG_PREFIX}] PVNet 모델 로드: {self._pvnet_bundle.checkpoint_path}")
        self.btn_run_pvnet.setEnabled(bool(self._last_detections))

    # ----------------------------------------------------- PVNet 추론
    def _on_run_pvnet(self) -> None:
        try:
            self._on_run_pvnet_impl()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"[{self.LOG_PREFIX}] PVNet 추론 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "PVNet 추론 실패",
                f"예상치 못한 오류로 추론에 실패했습니다:\n\n{exc}\n\n"
                "터미널의 상세 트레이스백을 확인하세요.",
            )
            raise

    def _on_run_pvnet_impl(self) -> None:
        if not self._last_detections:
            QMessageBox.warning(self, "알림", "먼저 2D 검출을 실행하세요.")
            return
        if self._pvnet_bundle is None:
            QMessageBox.warning(self, "알림", "먼저 PVNet 모델을 로드하세요.")
            return
        if self._pcd_organized is None or self._valid_mask is None:
            QMessageBox.warning(
                self, "알림",
                "포인트클라우드가 없습니다 (카메라 intrinsic 추정에 필요). 다시 촬영하세요.",
            )
            return
        if not self._current_image_path or not os.path.isfile(self._current_image_path):
            QMessageBox.warning(
                self, "알림",
                f"촬영 이미지 파일을 찾을 수 없습니다: {self._current_image_path}\n다시 촬영하세요.",
            )
            return

        try:
            fx, fy, cx, cy = estimate_intrinsics_from_organized_pcd(self._pcd_organized, self._valid_mask)
        except ValueError:
            QMessageBox.warning(self, "알림", "카메라 intrinsic을 추정하지 못했습니다 (유효 픽셀 부족).")
            return
        intrinsics = (fx, fy, cx, cy)

        gray = cv2.imread(self._current_image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            QMessageBox.warning(self, "알림", f"이미지를 읽지 못했습니다: {self._current_image_path}")
            return
        image_bgr = np.stack([gray, gray, gray], axis=-1)

        self.image_viewer.clear_label_preview_overlays()
        results = run_pvnet_on_detections(self._pvnet_bundle, image_bgr, self._last_detections, intrinsics)

        lines: list[str] = []
        n_ok = 0
        for r in results:
            if r.ok:
                n_ok += 1
                det = self._last_detections[r.instance_id]
                self.image_viewer.set_label_preview_overlay(
                    r.instance_id, det.mask.astype(bool), r.keypoints_2d,
                )
                lines.append(f"obj{r.instance_id}: 재투영 오차 {r.reprojection_error:.2f}px")
            else:
                lines.append(f"obj{r.instance_id}: 실패 - {r.error}")

        self.pvnet_result_label.setText(
            f"PVNet 추론 완료: {n_ok}/{len(results)}건 성공\n" + "\n".join(lines)
        )
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] PVNet 추론: {n_ok}/{len(results)}건 성공 "
            f"(모델: {self._pvnet_bundle.checkpoint_path})"
        )
        if n_ok == 0:
            QMessageBox.information(
                self, "추론 성공 없음",
                "모든 인스턴스에서 PVNet 추론이 실패했습니다.\n"
                "위 요약에서 각 인스턴스의 실패 사유를 확인하세요.\n"
                "(흔한 원인: 세그멘테이션이 전경을 거의 못 찾음 - 학습이 더 필요하거나\n"
                "실제 촬영 조건이 학습 데이터와 많이 다를 수 있습니다.)",
            )