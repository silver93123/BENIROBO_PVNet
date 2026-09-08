"""다중 프레임 평균화 카메라 데코레이터.

정적 씬(카메라와 대상물 모두 안 움직이는 상태 - 빈피킹 촬영은 대부분 이 조건을
만족한다)을 가정하고, capture() 1회 호출마다 내부 카메라에서 N프레임을
연속으로 찍어 픽셀별로 합쳐서 반환한다. ToF/구조광 depth 노이즈는 랜덤
성분이 커서, N프레임을 평균/중앙값으로 합치면 이론상 노이즈가 sqrt(N)배
줄어든다 (Helios2 정밀도 최적화 논의에서 나온 방법을 실제 캡처 계층에 반영).

CameraBase를 그대로 구현하는 "래퍼"라서, 이 데코레이터를 씌워도 호출부
(collect_dataset.py, 7_Test_binpicking_offline.py, inference_test_tab.py 등)는
capture()가 FrameData 하나를 돌려준다는 점 외에는 아무것도 몰라도 된다 -
내부적으로 몇 프레임을 찍고 있는지는 완전히 숨겨진다.

사용 예:
    from src.camera import create_camera
    cfg = {
        "type": "lucid_helios", ...,
        "averaging": {"num_frames": 8, "method": "median"},
    }
    with create_camera(cfg) as cam:      # AveragingCamera로 자동으로 감싸짐
        frame = cam.capture()            # 내부적으로 8프레임 찍어서 합친 결과
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .base import CameraBase, FrameData

logger = logging.getLogger(__name__)

_VALID_METHODS = ("median", "mean")


class AveragingCamera(CameraBase):
    """다중 프레임 평균화 데코레이터.

    내부 카메라(inner)의 open()/close()는 그대로 위임하고, capture()만
    가로채서 num_frames번 연속 캡처 후 픽셀 단위로 합친다.

    Args:
        inner: 실제 하드웨어에 접근하는 CameraBase 구현체
            (LucidHeliosCamera, FemtoBoltCamera 등 무엇이든 상관없음).
        num_frames: 합칠 프레임 수. 1이면 평균화 없이 inner.capture()를
            그대로 반환한다 (오버헤드 없음 - 옵션을 꺼둔 것과 동일).
        method: "median"(기본, flying pixel 등 이상치에 강건) 또는
            "mean"(순수 랜덤 노이즈만 있다는 가정 하에 이론적으로 더 매끈함).
            깨끗한 실내 환경 + 안정된 씬이면 mean도 괜찮지만, 엣지 근처
            flying pixel이 섞이기 쉬운 조건에서는 median을 권장.
        min_valid_ratio: 한 픽셀이 최종적으로 "유효"로 인정되려면 N프레임 중
            최소 몇 비율에서 유효해야 하는지 (0~1). 예: num_frames=8,
            min_valid_ratio=0.6 -> 8프레임 중 5프레임 이상에서 유효해야
            최종 valid_mask에 포함됨. 너무 낮으면 노이즈 픽셀이 섞이고,
            너무 높으면(1.0) 한 프레임이라도 흔들리면 그 픽셀이 통째로 빠짐.
    """

    def __init__(self, inner: CameraBase, num_frames: int = 5,
                 method: str = "median", min_valid_ratio: float = 0.6):
        if num_frames < 1:
            raise ValueError(f"num_frames는 1 이상이어야 합니다: {num_frames}")
        if method not in _VALID_METHODS:
            raise ValueError(f"method는 {_VALID_METHODS} 중 하나여야 합니다: {method}")
        if not (0.0 < min_valid_ratio <= 1.0):
            raise ValueError(f"min_valid_ratio는 (0, 1] 범위여야 합니다: {min_valid_ratio}")

        self._inner = inner
        self.num_frames = num_frames
        self.method = method
        self.min_valid_ratio = min_valid_ratio

    # -- CameraBase 위임 --------------------------------------------------
    def open(self) -> None:
        self._inner.open()

    def close(self) -> None:
        self._inner.close()

    # -- 핵심: N프레임 캡처 후 픽셀 단위 집계 ------------------------------
    def capture(self) -> FrameData:
        if self.num_frames == 1:
            return self._inner.capture()

        frames = [self._inner.capture() for _ in range(self.num_frames)]
        return self._aggregate(frames)

    def _aggregate(self, frames: list[FrameData]) -> FrameData:
        pts_stack = np.stack([f.points_organized for f in frames], axis=0)   # (N,H,W,3)
        valid_stack = np.stack([f.valid_mask for f in frames], axis=0)       # (N,H,W)

        # 무효 픽셀은 NaN으로 마스킹해서 nanmedian/nanmean이 유효한 프레임만
        # 집계에 반영하게 한다 (한두 프레임에서만 튄 flying pixel도 이렇게
        # 자연스럽게 걸러진다).
        pts_masked = np.where(valid_stack[..., None], pts_stack, np.nan)
        valid_count = valid_stack.sum(axis=0)  # (H,W) - 이 픽셀이 유효했던 프레임 수

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # 모든 프레임에서 무효였던 픽셀은 전부 NaN이라 nanmedian/nanmean/nanstd가
            # "All-NaN slice" 경고를 던진다 - 결과가 NaN인 게 맞는 동작이라
            # (final_valid에서 어차피 걸러짐) 경고만 조용히 무시한다.
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            if self.method == "median":
                agg = np.nanmedian(pts_masked, axis=0)
            else:
                agg = np.nanmean(pts_masked, axis=0)
            # 2026-07 추가: 픽셀별 표준편차. Monte Carlo 확률적 샘플링
            # (icp_runner.extract_instance_points_probabilistic)에서 이
            # 값을 Normal(mean, std)의 std로 그대로 쓴다. ddof=0(모집단
            # 표준편차) - 표본이 N=num_frames로 작을 때(예: 5~8)
            # ddof=1(표본표준편차)이 이론적으로는 더 정확하지만, 여기선
            # "관측된 산포"를 그대로 반영하는 목적이라 큰 차이는 없다.
            pixel_std = np.nanstd(pts_masked, axis=0)

        min_count = max(1, int(np.ceil(self.num_frames * self.min_valid_ratio)))
        final_valid = valid_count >= min_count

        agg = np.where(final_valid[..., None], agg, np.nan).astype(np.float32)
        points = agg[final_valid].astype(np.float32)
        pixel_std = np.where(final_valid[..., None], pixel_std, np.nan).astype(np.float32)

        # intensity/confidence는 depth만큼 노이즈에 민감하지 않지만,
        # 어차피 같은 N프레임을 찍은 김에 평균내서 약간의 노이즈 감소 효과를 더한다.
        intensity = np.mean(
            np.stack([f.intensity.astype(np.float32) for f in frames], axis=0), axis=0
        ).astype(frames[0].intensity.dtype)

        confidence = None
        if frames[0].confidence is not None:
            confidence = np.mean(
                np.stack([f.confidence.astype(np.float32) for f in frames], axis=0), axis=0
            ).astype(frames[0].confidence.dtype)

        # color_rgb는 정적 씬 가정 하에 굳이 평균낼 이유가 없어 마지막 프레임 값을 그대로 사용.
        color_rgb = frames[-1].color_rgb

        n_valid_px = int(final_valid.sum())
        logger.debug(
            "AveragingCamera: %d프레임(%s) 집계 완료, 유효 픽셀 %d개 "
            "(min_valid_ratio=%.2f -> 최소 %d/%d프레임 필요)",
            self.num_frames, self.method, n_valid_px,
            self.min_valid_ratio, min_count, self.num_frames,
        )

        return FrameData(
            intensity=intensity,
            points=points,
            points_organized=agg,
            valid_mask=final_valid,
            confidence=confidence,
            color_rgb=color_rgb,
            points_std=pixel_std,
        )