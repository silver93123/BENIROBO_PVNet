"""ICP 정합 테스트 섹션의 파이프라인 탭 공통 베이스.

공유 헤더(app/tabs/icp_test_tab.py)는 이 인터페이스만 알면 된다:
    detections = active_tab.detect(ctx)
    results = active_tab.register(detections, ctx, params)

detect()는 파이프라인마다 반드시 다르다 (어떤 Detector backend를 쓰는지가
파이프라인의 정체성 그 자체이므로) - 서브클래스가 필수로 구현해야 한다.

register()는 기본 구현을 여기 둔다. 실제로 확인해보면 RTMDet 파이프라인과
RotHead 파이프라인의 ICP 실행 로직은 완전히 동일하다:
    - Detection.initial_pose가 있으면 그걸 T_init으로 그대로 쓰고
    - 없으면 icp_runner.build_icp_init() fallback
RTMDet 파이프라인은 initial_pose가 애초에 항상 None이라 매번 fallback을
타는 것뿐이고, 코드 경로 자체는 하나다. 그래서 register()는 오버라이드
없이 이 기본 구현을 그대로 상속받아 쓰면 된다 - 파이프라인을 가르는 건
오직 detect()뿐이다. 다른 정합 방식이 필요한 새 파이프라인(예: 별도
confidence-weighted ICP)만 이걸 override하면 된다.
"""
from __future__ import annotations

from typing import List

from PyQt6.QtWidgets import QWidget

from app.core import icp_runner
from app.core.detector import Detection
from app.core.icp_runner import ICPParams, ICPResult
from app.core.pipeline_context import FrameContext


class ICPPipelineTab(QWidget):
    #: 내부 QTabWidget에 표시될 탭 이름. 서브클래스에서 반드시 지정.
    pipeline_name: str = "unnamed"

    def detect(self, ctx: FrameContext) -> List[Detection]:
        """이 파이프라인의 2D 검출을 수행한다. 서브클래스 필수 구현.

        mask가 없는 Detection은 여기서 걸러서 반환하는 게 원칙
        (register()는 mask가 있다고 가정하고 동작함).
        """
        raise NotImplementedError

    def register(
        self, detections: List[Detection], ctx: FrameContext, params: ICPParams
    ) -> List[ICPResult]:
        """기본 구현 - 대부분의 파이프라인은 오버라이드 안 해도 된다."""
        results: List[ICPResult] = []
        for i, det in enumerate(detections):
            if params.pc_upsample_method == "probabilistic" and params.pc_upsample_factor > 1:
                pts_mm = icp_runner.extract_instance_points_probabilistic(
                    det.mask, ctx.pcd_organized_mm, ctx.pcd_std_mm, ctx.valid_mask,
                    erode_px=params.mask_erode_px, samples_per_point=params.pc_upsample_factor,
                )
            else:
                pts_mm = icp_runner.extract_instance_points_mm(
                    det.mask, ctx.pcd_organized_mm, ctx.valid_mask, erode_px=params.mask_erode_px,
                    upsample_factor=params.pc_upsample_factor, upsample_method=params.pc_upsample_method,
                )
            result = icp_runner.run_icp_for_instance(
                i, pts_mm, ctx.cad_pcd, ctx.cad_visible_normal, ctx.cad_visible_flipped,
                params=params, T_init_override=getattr(det, "initial_pose", None),
            )
            results.append(result)
        return results