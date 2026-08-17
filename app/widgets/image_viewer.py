"""이미지 위에 검출 박스 + 마스크 오버레이 + (선택) pose 미리보기 오버레이를 보여주는 위젯.

확대/축소(줌)와 ROI(검출 대상 영역) 지정 기능도 여기서 담당한다. 두 기능 다
"원본 이미지 픽셀 좌표"를 기준으로 상태를 저장한다 - 화면에 보이는 크기(줌
배율)가 바뀌어도 좌표 자체는 안 흔들리게 하기 위함이다.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage, QPolygonF, QCursor
from PyQt6.QtWidgets import QLabel, QSizePolicy

from app.core.detector import Detection

DEFAULT_MASK_ALPHA = 100  # 0~255, 마스크 반투명도 기본값 (낮을수록 더 투명)
DEFAULT_LABEL_ALPHA = 255  # 0~255, bbox 선/라벨 배경 반투명도 기본값 (기존과 동일하게 완전 불투명)
DEFAULT_POSE_OVERLAY_ALPHA = 100  # 0~255, pose/CAD 오버레이 반투명도 기본값
POSE_OVERLAY_RADIUS = 2.5  # px
POSE_OVERLAY_LINE_WIDTH = 1.2  # px, 윤곽선 두께 (채워진 원이 아니라 속이 빈 원)
POSE_OVERLAY_BASE_COLOR = (0, 220, 255)  # 하늘색 - 노란 볼트 실물과 안 겹치게
ROI_COLOR = (255, 179, 0)  # 앰버 - 마스크(초록/주황 계열)와 안 겹치는 색

MIN_ZOOM = 0.1   # 10%
MAX_ZOOM = 8.0   # 800%
ZOOM_STEP = 1.15  # 휠 한 칸당 배율


class ImageViewer(QLabel):
    #: ROI가 새로 지정되거나(값=(x1,y1,x2,y2), 원본 이미지 픽셀 좌표) 지워지면(값=None) emit.
    roi_changed = pyqtSignal(object)
    #: ROI 그리기 모드가 켜지거나(외부 토글 버튼) 드래그 완료로 자동으로 꺼질 때 emit -
    #: 토글 버튼의 체크 상태를 실제 모드와 맞추기 위함.
    roi_draw_mode_changed = pyqtSignal(bool)
    #: 줌 배율이 바뀔 때마다(버튼/휠 어느 경로든) emit, 인자는 zoom_percent() 값.
    #: eventFilter로 wheelEvent를 가로채 갱신하는 방식은 실패한다 - Qt는 이벤트
    #: 필터를 위젯 자신의 이벤트 핸들러보다 먼저 호출하므로, 그 시점엔 아직
    #: 줌 값이 갱신되기 전이다. 시그널을 실제 값이 바뀐 "이후"에 emit하는 게 맞다.
    zoom_changed = pyqtSignal(int)

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
        self._mask_alpha = DEFAULT_MASK_ALPHA
        self._label_alpha = DEFAULT_LABEL_ALPHA
        self._pose_overlay_alpha = DEFAULT_POSE_OVERLAY_ALPHA

        # 줌: fit_scale(뷰포트에 꽉 맞추는 배율) * zoom(사용자 배율 1.0=100% of fit).
        # fit_scale은 뷰포트 크기가 바뀔 때마다(리사이즈, 처음 로드) 다시 계산한다.
        self._zoom = 1.0
        self._fit_scale = 1.0
        self._render_scale = 1.0  # 마지막으로 실제 그릴 때 쓴 최종 배율(원본 픽셀 -> 화면 픽셀) - 마우스 좌표 역산용

        # ROI: 원본 이미지 픽셀 좌표 (x1, y1, x2, y2), 없으면 None.
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_draw_mode = False
        self._roi_drag_start_img: tuple[float, float] | None = None
        self._roi_drag_current_img: tuple[float, float] | None = None

        self.setMouseTracking(True)
        self.setText("이미지를 불러오세요")

    # ----------------------------------------------------------- 표시 옵션
    def set_axis_gizmo_visible(self, visible: bool) -> None:
        self._show_axis_gizmo = visible
        self._refresh()

    def set_mask_alpha(self, alpha: int) -> None:
        """2D 검출 마스크 채우기 반투명도 (0=완전 투명, 255=완전 불투명)."""
        self._mask_alpha = alpha
        self._refresh()

    def set_label_alpha(self, alpha: int) -> None:
        """bbox 테두리 + 라벨 배경 반투명도."""
        self._label_alpha = alpha
        self._refresh()

    def set_pose_overlay_alpha(self, alpha: int) -> None:
        """pose/CAD 정합 결과 오버레이(윤곽선 점) 반투명도."""
        self._pose_overlay_alpha = alpha
        self._refresh()

    # ----------------------------------------------------------------- 줌
    def zoom_percent(self) -> int:
        return round(self._zoom * 100)

    def set_zoom(self, zoom: float, anchor_viewport_pos: QPoint | None = None) -> None:
        """zoom=1.0이 '뷰포트에 꽉 맞춤'(기존 기본 동작)이다.

        anchor_viewport_pos를 주면(보통 마우스 위치) 그 지점이 화면에서
        가리키는 원본 이미지 지점이 줌 전후로 같은 화면 위치에 남도록
        스크롤 위치를 맞춰준다(휠 줌의 표준 동작 - 커서 아래 지점을
        기준으로 확대/축소).
        """
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if new_zoom == self._zoom:
            return

        img_pt = None
        if anchor_viewport_pos is not None:
            img_pt = self._viewport_pos_to_image(anchor_viewport_pos)

        self._zoom = new_zoom
        self._refresh()
        self.zoom_changed.emit(self.zoom_percent())

        if img_pt is not None and anchor_viewport_pos is not None:
            self._scroll_to_show(img_pt, anchor_viewport_pos)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / ZOOM_STEP)

    def zoom_fit(self) -> None:
        """줌을 100%(뷰포트에 꽉 맞춤)로 리셋."""
        self.set_zoom(1.0)

    # ----------------------------------------------------------------- ROI
    def set_roi_draw_mode(self, enabled: bool) -> None:
        """True면 다음 드래그로 ROI를 새로 그린다. 드래그 완료 시 자동으로 꺼진다."""
        if enabled == self._roi_draw_mode:
            return
        self._roi_draw_mode = enabled
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor) if enabled else QCursor(Qt.CursorShape.ArrowCursor))
        self.roi_draw_mode_changed.emit(enabled)

    def get_roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self._refresh()
        self.roi_changed.emit(roi)

    def clear_roi(self) -> None:
        self.set_roi(None)

    def load_image(self, path: str) -> None:
        self._base_pixmap = QPixmap(path)
        self._detections = []
        self._pose_overlays = {}
        self._zoom = 1.0  # 새 프레임은 항상 '맞춤'으로 시작 (이전 줌/스크롤 위치가 새 이미지에서 엉뚱한 곳을 보여줄 수 있음)
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

        # 0단계: ROI 바깥을 살짝 어둡게 덮어서 "이 안쪽만 검출 대상"임을 시각적으로 강조.
        # 드래그로 그리는 중이면 확정된 self._roi 대신 지금 드래그 중인 사각형을 미리 보여준다.
        roi_to_draw = self._roi
        if self._roi_draw_mode and self._roi_drag_start_img is not None and self._roi_drag_current_img is not None:
            roi_to_draw = self._normalized_roi(self._roi_drag_start_img, self._roi_drag_current_img)
        if roi_to_draw is not None:
            rx1, ry1, rx2, ry2 = roi_to_draw
            img_w, img_h = canvas.width(), canvas.height()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 90))
            for dim_rect in (
                QRectF(0, 0, img_w, ry1),
                QRectF(0, ry2, img_w, img_h - ry2),
                QRectF(0, ry1, rx1, ry2 - ry1),
                QRectF(rx2, ry1, img_w - rx2, ry2 - ry1),
            ):
                painter.drawRect(dim_rect)
            roi_color = QColor(*ROI_COLOR)
            pen = QPen(roi_color, 2.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

        # 1단계: 마스크 영역을 반투명 색으로 먼저 채운다 (bbox/라벨보다 아래에 깔림)
        for i, det in enumerate(self._detections):
            if det.mask is None:
                continue
            color = colors[i % len(colors)]
            mask_image = self._mask_to_qimage(det.mask, color, self._mask_alpha)
            if mask_image is not None:
                painter.drawImage(0, 0, mask_image)

        # 2단계: bbox + 라벨
        for i, det in enumerate(self._detections):
            color = colors[i % len(colors)]
            box_color = QColor(color)
            box_color.setAlpha(self._label_alpha)
            pen = QPen(box_color, 3)
            painter.setPen(pen)
            x1, y1, x2, y2 = det.bbox
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            label_text = f"obj{i}: {det.label} {det.confidence:.2f}"
            painter.fillRect(QRectF(x1, y1 - 20, 8 * len(label_text), 20), box_color)
            label_text_color = QColor("white")
            label_text_color.setAlpha(self._label_alpha)
            painter.setPen(QPen(label_text_color))
            painter.drawText(int(x1) + 4, int(y1) - 5, label_text)
            painter.setPen(pen)

        # 3단계: pose 미리보기 오버레이 (반투명 노란 "윤곽선" 원 - CAD를 현재
        # 입력 각도로 투영한 것. 속이 빈 원이라 실제 물체 사진이 안 가려지고,
        # 실루엣과 겹치면 입력한 각도가 잘 맞다는 뜻).
        overlay_color = QColor(*POSE_OVERLAY_BASE_COLOR)
        overlay_color.setAlpha(self._pose_overlay_alpha)
        overlay_pen = QPen(overlay_color, POSE_OVERLAY_LINE_WIDTH)
        painter.setPen(overlay_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for points_2d in self._pose_overlays.values():
            for x, y in points_2d:
                painter.drawEllipse(QRectF(x - POSE_OVERLAY_RADIUS, y - POSE_OVERLAY_RADIUS,
                                            POSE_OVERLAY_RADIUS * 2, POSE_OVERLAY_RADIUS * 2))

        painter.end()

        img_w, img_h = canvas.width(), canvas.height()
        viewport = self._viewport_size()
        if img_w > 0 and img_h > 0 and viewport.width() > 0 and viewport.height() > 0:
            self._fit_scale = min(viewport.width() / img_w, viewport.height() / img_h)
        else:
            self._fit_scale = 1.0
        self._render_scale = max(self._fit_scale * self._zoom, 0.01)

        target_w = max(1, round(img_w * self._render_scale))
        target_h = max(1, round(img_h * self._render_scale))
        scaled = canvas.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # 실제 렌더 배율을 스케일된 결과 기준으로 다시 맞춘다 - KeepAspectRatio가
        # target_w/target_h를 정확히 못 맞추고 한쪽을 살짝 줄일 수 있어서(반올림
        # 오차), 마우스 좌표 역산이 어긋나지 않도록 실측값으로 보정.
        if scaled.width() > 0:
            self._render_scale = scaled.width() / img_w

        if self._show_axis_gizmo:
            gizmo_painter = QPainter(scaled)
            gizmo_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_axis_gizmo(gizmo_painter, scaled.width(), scaled.height())
            gizmo_painter.end()

        self.setPixmap(scaled)
        self.resize(scaled.size())

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
    def _mask_to_qimage(mask: np.ndarray, color: QColor, alpha: int) -> QImage | None:
        """(H, W) bool 마스크를 반투명 RGBA QImage로 변환한다 (원본 이미지 크기와 동일해야 함)."""
        if mask is None or mask.ndim != 2:
            return None

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)

        data = rgba.tobytes()
        qimage = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return qimage.copy()  # 자체 버퍼를 소유하도록 깊은 복사 (data가 GC돼도 안전)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    # ------------------------------------------------------- 뷰포트/좌표 변환
    def _viewport_size(self):
        """줌 기준이 되는 '뷰포트' 크기. QScrollArea.setWidget(self)로 담겨있으면
        parentWidget()이 바로 그 스크롤 영역의 뷰포트라 이 크기를 그대로 쓰면
        되고, 스크롤 영역 없이 일반 레이아웃 안에 있으면 self.size()로 폴백한다."""
        parent = self.parentWidget()
        if parent is not None and parent.width() > 0 and parent.height() > 0:
            return parent.size()
        return self.size()

    def _viewport_pos_to_image(self, viewport_pos: QPoint) -> tuple[float, float] | None:
        """QScrollArea 뷰포트 기준 마우스 좌표 -> 원본 이미지 픽셀 좌표.
        self는 QScrollArea 안에서 스크롤에 따라 이동하므로, mapFromParent로
        먼저 self(=렌더된 픽스맵) 기준 좌표로 바꾼 뒤 렌더 배율로 나눈다."""
        if self._base_pixmap is None or self._render_scale <= 0:
            return None
        local = self.mapFromParent(viewport_pos)
        return (local.x() / self._render_scale, local.y() / self._render_scale)

    def _widget_pos_to_image(self, widget_pos: QPoint) -> tuple[float, float] | None:
        """self(=렌더된 픽스맵 QLabel) 기준 마우스 좌표 -> 원본 이미지 픽셀 좌표."""
        if self._base_pixmap is None or self._render_scale <= 0:
            return None
        return (widget_pos.x() / self._render_scale, widget_pos.y() / self._render_scale)

    def _scroll_to_show(self, img_pt: tuple[float, float], viewport_pos: QPoint) -> None:
        """줌 이후 img_pt(원본 이미지 좌표)가 viewport_pos(뷰포트 기준 화면 좌표)에
        그대로 보이도록 스크롤 위치를 맞춘다 - 마우스 아래 지점을 고정한 채 확대."""
        from PyQt6.QtWidgets import QScrollArea
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        if area is None:
            return
        target_x = img_pt[0] * self._render_scale - viewport_pos.x()
        target_y = img_pt[1] * self._render_scale - viewport_pos.y()
        area.horizontalScrollBar().setValue(int(target_x))
        area.verticalScrollBar().setValue(int(target_y))

    @staticmethod
    def _normalized_roi(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[int, int, int, int]:
        x1, y1 = p1
        x2, y2 = p2
        return (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))

    # ----------------------------------------------------------- 휠 = 줌
    def wheelEvent(self, event) -> None:
        if self._base_pixmap is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else (1.0 / ZOOM_STEP)
        # event.position()은 self(QLabel) 기준 좌표 - set_zoom의 anchor는 부모(뷰포트)
        # 기준을 기대하므로 mapToParent로 변환.
        anchor = self.mapToParent(event.position().toPoint())
        self.set_zoom(self._zoom * factor, anchor_viewport_pos=anchor)
        event.accept()

    # ----------------------------------------------------------- ROI 드래그
    def mousePressEvent(self, event) -> None:
        if self._roi_draw_mode and event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_pos_to_image(event.pos())
            if img_pt is not None:
                self._roi_drag_start_img = img_pt
                self._roi_drag_current_img = img_pt
                self._refresh()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_draw_mode and self._roi_drag_start_img is not None:
            img_pt = self._widget_pos_to_image(event.pos())
            if img_pt is not None:
                self._roi_drag_current_img = img_pt
                self._refresh()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._roi_draw_mode and event.button() == Qt.MouseButton.LeftButton and self._roi_drag_start_img is not None:
            img_pt = self._widget_pos_to_image(event.pos())
            if img_pt is not None and self._base_pixmap is not None:
                img_w, img_h = self._base_pixmap.width(), self._base_pixmap.height()
                x1, y1, x2, y2 = self._normalized_roi(self._roi_drag_start_img, img_pt)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                # 너무 작게(거의 클릭만) 그리면 실수로 그린 것으로 보고 무시.
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    self.set_roi((x1, y1, x2, y2))
            self._roi_drag_start_img = None
            self._roi_drag_current_img = None
            self.set_roi_draw_mode(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)