"""기존 RTMDet-Ins 파이프라인.

register()는 오버라이드하지 않는다 - ICPPipelineTab의 기본 구현을 그대로
쓴다 (이 파이프라인은 Detection.initial_pose가 항상 None이라 매번
build_icp_init() fallback을 타지만, 그건 기본 구현이 이미 처리한다).
"""
from __future__ import annotations

from typing import List

from PyQt6.QtWidgets import QLabel, QVBoxLayout

from app.core.detector import Detection, Detector
from app.core.pipeline_context import FrameContext
from app.tabs.icp_pipelines.base import ICPPipelineTab

BACKEND_NAME = "rtmdet_ins"


class RTMDetPipelineTab(ICPPipelineTab):
    pipeline_name = "RTMDet"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "기존 검출 방식. 회전 헤드가 없으므로 ICP 초기 pose는 항상\n"
            "상단 'ICP 파라미터' 박스의 '정합 알고리즘'(FGR/기본) + "
            "'초기 roll/pitch/yaw'로 결정됩니다."
        )
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

    def detect(self, ctx: FrameContext) -> List[Detection]:
        detector = Detector(
            checkpoint_path=ctx.checkpoint_path,
            config_path=ctx.config_path,
            score_threshold=ctx.score_threshold,
            backend=BACKEND_NAME,
        )
        detections = detector.predict(ctx.image_path, conf_threshold=ctx.score_threshold)
        return [d for d in detections if d.mask is not None]