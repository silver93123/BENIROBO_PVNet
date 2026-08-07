"""이미지 위에 검출 박스 + 마스크 오버레이 + (선택) pose 미리보기 오버레이를 보여주는 위젯."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage, QPolygonF
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
        self._show_axis_gizmo = True
        self.setText("이미지를 불러오세요")

    def set_axis_gizmo_visible(self, visible: bool) -> None:
        self._show_axis_gizmo = visible
        self._refresh()

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

        if self._show_axis_gizmo:
            gizmo_painter = QPainter(scaled)
            gizmo_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_axis_gizmo(gizmo_painter, scaled.width(), scaled.height())
            gizmo_painter.end()

        self.setPixmap(scaled)

    def _draw_axis_gizmo(self, painter: QPainter, canvas_width: int, canvas_height: int) -> None:
        """카메라(씬) 좌표축과 roll/pitch/yaw 대응관계를 이미지 우측 상단에
        항상 작게 표시한다 - "Δroll을 올리면 정확히 어느 방향으로 도는지"가
        항상 헷갈린다는 피드백에 대응. 스케일링이 끝난 최종 픽스맵 위에
        그려서, 이미지 해상도나 확대/축소와 무관하게 화면에서 항상 같은
        크기로 보인다 (이미지 좌표계에 그리면 이미지가 클수록 아이콘도
        커져버림).

        이 앱의 회전 컨벤션(icp_runner._Rx/_Ry/_Rz, R = Rz(yaw)@Ry(pitch)@Rx(roll),
        카메라 좌표계는 X=오른쪽·Y=아래·Z=깊이) 기준:
            Roll  = X축 회전 (오른쪽 방향 화살표)
            Pitch = Y축 회전 (아래쪽 방향 화살표)
            Yaw   = Z축 회전 (카메라를 정면으로 바라보는 축이라 유일하게
                    2D 이미지 평면 위 회전으로 정확히 나타낼 수 있음)
        Yaw만 정확한 회전방향 원호(시계방향)를 같이 그렸다 - _Rz(+d)가
        +X를 +Y(아래) 쪽으로 돌리므로 이미지에서 양의 yaw는 시계방향이라는
        걸 직접 계산해서 검증한 뒤 반영한 값이다. Roll/Pitch는 화면 안쪽으로
        파고드는 회전이라 2D 평면 위에 왜곡 없이 그릴 수 없어 축 방향
        화살표까지만 표시한다.
        """
        box_w, box_h = 128, 116
        margin = 10
        x0 = canvas_width - box_w - margin
        y0 = margin
        if x0 < margin:
            return  # 위젯이 너무 좁으면 그리지 않음 (겹침 방지)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 225))
        painter.drawRoundedRect(QRectF(x0, y0, box_w, box_h), 8, 8)

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        origin_x = x0 + 26
        label_x = x0 + 48
        row_h = 30

        red = QColor("#D8303A")
        green = QColor("#1D9E75")
        blue = QColor("#378ADD")

        # Roll = X축 (오른쪽)
        y_row = y0 + 22
        self._draw_gizmo_arrow(painter, origin_x, y_row, origin_x + 22, y_row, red)
        painter.setPen(red)
        painter.drawText(label_x, y_row + 4, "Roll (X)")

        # Pitch = Y축 (아래쪽)
        y_row = y0 + 22 + row_h
        self._draw_gizmo_arrow(painter, origin_x, y_row - 11, origin_x, y_row + 11, green)
        painter.setPen(green)
        painter.drawText(label_x, y_row + 4, "Pitch (Y)")

        # Yaw = Z축 (화면 안쪽, ⊗ 심볼) + 정확한 +방향 원호(시계방향)
        y_row = y0 + 22 + row_h * 2
        cx, cy, r = float(origin_x), float(y_row), 7.0
        painter.setPen(QPen(blue, 2.0))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        d = r * 0.65
        painter.drawLine(int(cx - d), int(cy - d), int(cx + d), int(cy + d))
        painter.drawLine(int(cx - d), int(cy + d), int(cx + d), int(cy - d))

        arc_r = 15.0
        arc_rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)
        start_deg, span_deg = 40.0, -280.0  # Qt drawArc: 음수 span = 화면상 시계방향
        painter.drawArc(arc_rect, int(start_deg * 16), int(span_deg * 16))
        end_rad = math.radians(start_deg + span_deg)
        ex = cx + arc_r * math.cos(end_rad)
        ey = cy - arc_r * math.sin(end_rad)  # Qt 각도는 y-up 수학 convention
        tangent = end_rad - math.pi / 2
        painter.setBrush(blue)
        head = 6.0
        painter.drawPolygon(QPolygonF([
            QPointF(ex, ey),
            QPointF(ex - head * math.cos(tangent - 0.4), ey + head * math.sin(tangent - 0.4)),
            QPointF(ex - head * math.cos(tangent + 0.4), ey + head * math.sin(tangent + 0.4)),
        ]))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(blue)
        painter.drawText(label_x, y_row + 4, "Yaw (Z)")

        caption_font = QFont()
        caption_font.setPointSize(7)
        painter.setFont(caption_font)
        painter.setPen(QColor("#888"))
        painter.drawText(QRectF(x0 + 6, y0 + box_h - 16, box_w - 12, 14), Qt.AlignmentFlag.AlignLeft, "카메라 시점 기준")

        painter.restore()

    @staticmethod
    def _draw_gizmo_arrow(painter: QPainter, x1: float, y1: float, x2: float, y2: float, color: QColor) -> None:
        painter.setPen(QPen(color, 2.4))
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 7.0
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([
            QPointF(x2, y2),
            QPointF(x2 - head * math.cos(angle - 0.4), y2 - head * math.sin(angle - 0.4)),
            QPointF(x2 - head * math.cos(angle + 0.4), y2 - head * math.sin(angle + 0.4)),
        ]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

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