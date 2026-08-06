"""ICP 정합 테스트 섹션의 파이프라인 탭 레지스트리.

app/core/registration, src/detection과 동일한 팩토리 패턴이지만, 여기서는
인스턴스가 QWidget(UI를 가짐)이라 "타입 문자열 -> 클래스" 목록 형태로 둔다.
공유 헤더(icp_test_tab.py)가 이 목록으로 내부 QTabWidget을 채운다.

새 파이프라인 추가 방법:
    1. ICPPipelineTab을 상속하는 새 파일을 이 폴더에 작성 (detect()만
       필수 구현, register()는 대부분 오버라이드 불필요 - base.py 참고).
    2. 아래 AVAILABLE_ICP_PIPELINES에 (이름, 클래스) 한 줄 추가.
그 외에는 icp_test_tab.py를 포함해 아무 것도 손댈 필요 없다.
"""
from __future__ import annotations

from app.tabs.icp_pipelines.base import ICPPipelineTab
from app.tabs.icp_pipelines.foundationpose_pipeline_tab import FoundationPosePipelineTab
from app.tabs.icp_pipelines.rothead_pipeline_tab import RotHeadPipelineTab
from app.tabs.icp_pipelines.rtmdet_pipeline_tab import RTMDetPipelineTab

AVAILABLE_ICP_PIPELINES = [
    ("RTMDet", RTMDetPipelineTab),
    ("RotHead", RotHeadPipelineTab),
    ("FoundationPose", FoundationPosePipelineTab),
]

__all__ = [
    "ICPPipelineTab", "RTMDetPipelineTab", "RotHeadPipelineTab",
    "FoundationPosePipelineTab", "AVAILABLE_ICP_PIPELINES",
]