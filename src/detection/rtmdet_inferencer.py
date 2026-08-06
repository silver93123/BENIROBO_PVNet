"""RTMDet-Ins MMDetection 추론 래퍼.

MMDetection 3.x API를 사용해 RTMDet-Ins 모델을 로드하고, intensity 이미지에서
인스턴스 세그멘테이션 결과를 얻는 가벼운 인터페이스 제공.

목적:
    1) Stage 4 통합 시 단순한 인터페이스 제공
    2) MMDetection 내부 API 변경에 대한 우리 코드 보호
    3) FrameData.crop_by_mask와의 자연스러운 연결

참고:
    - MMDetection 3.x DetInferencer API:
      https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html
    - RTMDet-Ins:
      https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.detection.base import DetectionResult, DetectorBase

logger = logging.getLogger(__name__)

# 하위 호환: 기존에 `from src.detection.rtmdet_inferencer import DetectionResult`로
# import하던 코드가 있을 수 있으므로 base.py의 것을 그대로 재노출.
__all__ = ["DetectionResult", "RTMDetInferencer"]


class RTMDetInferencer(DetectorBase):
    """RTMDet-Ins 모델 wrapper.

    사용 예:
        inferencer = RTMDetInferencer(
            config="configs/rtmdet-ins_tiny_8xb32-300e_coco.py",
            checkpoint="models/rtmdet-ins_tiny_*.pth",
            device="cuda:0",
        )
        results = inferencer.infer(image)  # numpy (H, W, 3) or (H, W)
        for r in results:
            print(r.class_name, r.score, r.mask.shape)
    """

    def __init__(
        self,
        config: Union[str, Path],
        checkpoint: Union[str, Path],
        device: str = "cuda:0",
        score_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            config: RTMDet-Ins config .py 파일 경로.
            checkpoint: .pth 가중치 경로.
            device: 'cuda:0' 또는 'cpu'.
            score_threshold: 이 이하 신뢰도는 결과에서 제외.
        """
        self.config = str(config)
        self.checkpoint = str(checkpoint)
        self.device = device
        self.score_threshold = score_threshold

        if not Path(self.config).exists():
            raise FileNotFoundError(f"Config not found: {self.config}")
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        logger.info("Loading RTMDet-Ins model...")
        logger.info("  Config:     %s", self.config)
        logger.info("  Checkpoint: %s", self.checkpoint)
        logger.info("  Device:     %s", self.device)

        # MMDetection 3.x 권장 방식: init_detector + inference_detector
        # (DetInferencer는 시각화까지 통합되어 있어 우리 용도엔 무거움)
        from mmdet.apis import init_detector
        self._model = init_detector(
            self.config, self.checkpoint, device=self.device
        )

        # COCO 클래스 이름 (모델 metadata에서 추출)
        self._class_names = self._model.dataset_meta.get("classes", None)
        if self._class_names is None:
            # 폴백: COCO 80 클래스 기본 이름
            self._class_names = tuple(f"class_{i}" for i in range(80))

        logger.info("Model loaded. %d classes.", len(self._class_names))

    @property
    def class_names(self) -> tuple:
        return tuple(self._class_names)

    def infer(self, image: np.ndarray) -> List[DetectionResult]:
        """단일 이미지에 대해 인스턴스 세그멘테이션 수행.

        Args:
            image: (H, W, 3) BGR uint8 또는 (H, W) mono uint8.
                Mono인 경우 자동으로 3채널 BGR로 복제됩니다.

        Returns:
            DetectionResult 리스트. score_threshold 이상만 포함.
        """
        from mmdet.apis import inference_detector

        # Mono → BGR 3채널 복제 (LUCID Helios intensity는 mono)
        if image.ndim == 2:
            image_bgr = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            image_bgr = image
        else:
            raise ValueError(
                f"image는 (H,W) 또는 (H,W,3) 형식이어야 합니다. 받음: {image.shape}"
            )

        # uint8 보장
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)

        # 추론 실행
        result = inference_detector(self._model, image_bgr)

        # mmdet 3.x: result.pred_instances 에 모든 검출 정보
        # - bboxes:  (N, 4) tensor
        # - scores:  (N,)
        # - labels:  (N,) class ids
        # - masks:   (N, H, W) bool tensor
        pred = result.pred_instances

        scores_np = pred.scores.cpu().numpy()
        keep = scores_np >= self.score_threshold

        if not keep.any():
            return []

        bboxes_np = pred.bboxes.cpu().numpy()[keep]
        labels_np = pred.labels.cpu().numpy()[keep]
        scores_np = scores_np[keep]

        # masks가 없는 경우는 일반 detection 모델일 때 (RTMDet-Ins는 항상 있어야 함)
        if not hasattr(pred, "masks") or pred.masks is None:
            raise RuntimeError(
                "결과에 mask가 없습니다. RTMDet-Ins(인스턴스 세그) 모델이 맞는지 확인."
            )

        masks_np = pred.masks.cpu().numpy()[keep]  # (N, H, W) bool

        results = []
        for i in range(len(scores_np)):
            cls_id = int(labels_np[i])
            cls_name = (self._class_names[cls_id]
                        if cls_id < len(self._class_names)
                        else f"class_{cls_id}")
            results.append(DetectionResult(
                mask=masks_np[i].astype(bool),
                bbox=bboxes_np[i].astype(np.float32),
                score=float(scores_np[i]),
                class_id=cls_id,
                class_name=cls_name,
            ))

        # 점수 내림차순 정렬
        results.sort(key=lambda r: r.score, reverse=True)
        return results