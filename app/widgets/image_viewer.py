"""이미지 위에 검출 박스 + 마스크 오버레이 + (선택) pose 미리보기 오버레이를 보여주는 위젯."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt6.QtWidgets import QLabel, QSizePolicy

from app.core.detector import Detection

MASK_ALPHA = 100  # 0~255, 마스크 반투명도 (낮을수록 더 투명)
POSE_OVERLAY_ALPHA = 100  # 0~255, pose 미리보기 윤곽선 반투명도 (낮을수록 더 투명)
POSE_OVERLAY_RADIUS = 2.5  # px
POSE_OVERLAY_LINE_WIDTH = 1.2  # px, 윤곽선 두께 (채워진 원이 아니라 속이 빈 원)
POSE_OVERLAY_COLOR = QColor(0, 220, 255, POSE_OVERLAY_ALPHA)  # 하늘색 - 노란 볼트 실물과 안 겹치게


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f2f1ec; border-radius: 8px;")
        self.setMinimumHeight(280)
        self._base_pixmap: QPixmap | None = None
        self._detections: list[Detection] = []
        self._pose_overlays: dict[int, np.ndarray] = {}  # obj index -> (N,2) 투영된 2D 점
        self.setText("이미지를 불러오세요")

    def load_image(self, path: str) -> None:
        self._base_pixmap = QPixmap(path)
        self._detections = []
        self._pose_overlays = {}
        self._refresh()

    def set_detections(self, detections: list[Detection]) -> None:
        self._detections = detections
        self._refresh()

    def set_pose_overlay(self, obj_index: int, points_2d: np.ndarray | None) -> None:
        """obj_index 인스턴스의 pose 미리보기 점(CAD를 현재 입력 각도로 투영한 것)을
        설정/갱신한다. points_2d가 None이면 그 인스턴스의 오버레이를 지운다.

        수동 라벨링 탭에서 각도 스핀박스를 조정할 때마다 호출되어, "지금 입력한
        각도가 실제 사진 속 물체와 얼마나 맞는지"를 즉시 눈으로 확인할 수 있게 한다.
        """
        if points_2d is None:
            self._pose_overlays.pop(obj_index, None)
        else:
            self._pose_overlays[obj_index] = points_2d
        self._refresh()

    def clear_pose_overlays(self) -> None:
        self._pose_overlays = {}
        self._refresh()

    def _refresh(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.setText("이미지를 불러오세요")
            return

        canvas = QPixmap(self._base_pixmap)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont()
        font.setPointSize(11)
        painter.setFont(font)

        colors = [QColor("#1D9E75"), QColor("#D85A30"), QColor("#378ADD"), QColor("#D4537E")]

        # 1단계: 마스크 영역을 반투명 색으로 먼저 채운다 (bbox/라벨보다 아래에 깔림)
        for i, det in enumerate(self._detections):
            if det.mask is None:
                continue
            color = colors[i % len(colors)]
            mask_image = self._mask_to_qimage(det.mask, color)
            if mask_image is not None:
                painter.drawImage(0, 0, mask_image)

        # 2단계: bbox + 라벨
        for i, det in enumerate(self._detections):
            color = colors[i % len(colors)]
            pen = QPen(color, 3)
            painter.setPen(pen)
            x1, y1, x2, y2 = det.bbox
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            label_text = f"obj{i}: {det.label} {det.confidence:.2f}"
            painter.fillRect(QRectF(x1, y1 - 20, 8 * len(label_text), 20), color)
            painter.setPen(QPen(QColor("white")))
            painter.drawText(int(x1) + 4, int(y1) - 5, label_text)
            painter.setPen(pen)

        # 3단계: pose 미리보기 오버레이 (반투명 노란 "윤곽선" 원 - CAD를 현재
        # 입력 각도로 투영한 것. 속이 빈 원이라 실제 물체 사진이 안 가려지고,
        # 실루엣과 겹치면 입력한 각도가 잘 맞다는 뜻).
        overlay_pen = QPen(POSE_OVERLAY_COLOR, POSE_OVERLAY_LINE_WIDTH)
        painter.setPen(overlay_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for points_2d in self._pose_overlays.values():
            for x, y in points_2d:
                painter.drawEllipse(QRectF(x - POSE_OVERLAY_RADIUS, y - POSE_OVERLAY_RADIUS,
                                            POSE_OVERLAY_RADIUS * 2, POSE_OVERLAY_RADIUS * 2))

        painter.end()

        scaled = canvas.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    @staticmethod
    def _mask_to_qimage(mask: np.ndarray, color: QColor) -> QImage | None:
        """(H, W) bool 마스크를 반투명 RGBA QImage로 변환한다 (원본 이미지 크기와 동일해야 함)."""
        if mask is None or mask.ndim != 2:
            return None

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.where(mask, MASK_ALPHA, 0).astype(np.uint8)

        data = rgba.tobytes()
        qimage = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return qimage.copy()  # 자체 버퍼를 소유하도록 깊은 복사 (data가 GC돼도 안전)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()