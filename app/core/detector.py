"""오프라인 2D 객체 검출 추론 (RTMDet-Ins).

실제 파이프라인 스크립트(3_Detect_and_PickPoint.py)의 [A] Detection 단계만
그대로 가져왔다. PCD 분리 / ICP 정합 / 픽포인트 계산은 포함하지 않는다
(2D 검출까지만 -> ICP는 나중 단계에서 필요할 때 추가).

의존성: <프로젝트 루트>/src/detection.py 의 RTMDetInferencer 클래스가 필요하다.
이 파일은 실제 추론 엔진 구현이 담긴 파일로, 업로드되지 않아서 이 앱 zip에는
포함되어 있지 않다. 프로젝트 루트에 src/detection.py를 두면 그대로 동작하고,
없으면 import 시점에 명확한 에러를 낸다 (더 이상 더미/랜덤 결과를 만들지 않는다).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# pcd_organized_mm/valid_mask를 실제로 쓰는 backend 집합.
# 새 3D-aware backend를 추가할 때는 predict()를 고칠 필요 없이 여기 한 줄만
# 추가하면 된다 - RotHead 추가 당시 이 분기를 문자열 하나로 하드코딩해 둔
# 탓에 FoundationPose 추가 때 조용히 depth가 안 넘어가는 버그가 있었다
# (2026-08 발견/수정).
_BACKENDS_REQUIRING_3D = {"rtmdet_ins_rothead", "rtmdet_ins_foundationpose"}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (픽셀 좌표)
    mask: np.ndarray | None = None  # (H, W) bool - 있으면 마스크 오버레이에 쓸 수 있음
    # rtmdet_ins_rothead 백엔드일 때만 채워짐 (4x4, m 단위). 없으면 None
    # -> icp_runner.py의 기존 build_icp_init() fallback을 그대로 쓰면 됨.
    initial_pose: np.ndarray | None = None


class Detector:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
        backend: str = "rtmdet_ins",
        **backend_kwargs,
    ):
        """
        Args:
            backend: "rtmdet_ins"(기존, 회전 없음) 또는
                "rtmdet_ins_rothead"(회전 헤드 포함).
                src.detection.AVAILABLE_DETECTOR_TYPES 참고.
            **backend_kwargs: 특정 backend에만 필요한 추가 생성자 인자를
                그대로 통과시킨다. 예: backend="rtmdet_ins_rothead"일 때
                rotation_checkpoint=..., rotation_backbone=..., 
                rotation_crop_size=... (RTMDetInferencerRotHead.__init__
                참고). rtmdet_ins에는 해당사항 없음 - 그냥 안 넘기면 됨.
                이렇게 열어두면 새 backend가 자기만의 파라미터를 추가해도
                이 클래스를 다시 고칠 필요가 없다.
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.score_threshold = score_threshold
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self._inferencer = None

    def load_model(self) -> None:
        """src.detection.create_detector() 팩토리로 백엔드 인스턴스를 생성한다.

        registration/camera 모듈과 동일한 팩토리 패턴 - 백엔드 종류는
        문자열 하나로 결정되고, 이 함수는 그 종류를 몰라도 된다.
        """
        if not self.checkpoint_path:
            raise ValueError("체크포인트 경로가 지정되지 않았습니다.")
        if not self.config_path:
            raise ValueError(
                "config 경로가 지정되지 않았습니다. "
                "실제 파이프라인에서는 체크포인트와 같은 work_dir 안의 "
                "rtmdet-ins_bracket.py(학습 시 저장된 사본)를 사용합니다."
            )

        try:
            from src.detection import create_detector
        except ImportError as exc:
            raise ImportError(
                "src/detection의 create_detector를 import할 수 없습니다. "
                "src/detection/ 패키지(base.py, __init__.py, "
                "rtmdet_inferencer.py 등)가 프로젝트 루트에 있는지 확인하세요."
            ) from exc

        params = {
            "config": self.config_path,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "score_threshold": self.score_threshold,
        }
        params.update(self.backend_kwargs)

        self._inferencer = create_detector({"type": self.backend, "params": params})

    def predict(
        self,
        image_path: str,
        conf_threshold: float | None = None,
        pcd_organized_mm: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> list[Detection]:
        """이미지 한 장에 대해 검출을 수행한다.

        pcd_organized_mm/valid_mask는 backend가 _BACKENDS_REQUIRING_3D에
        속할 때만 의미가 있다 (initial_pose 계산에 필요, icp_test_tab.py가
        세션에서 로드한 것과 동일한 배열을 그대로 넘기면 됨). 그 외
        backend(rtmdet_ins 등)에서는 무시된다.
        """
        if self._inferencer is None:
            self.load_model()

        threshold = self.score_threshold if conf_threshold is None else conf_threshold

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        bgr = np.stack([gray, gray, gray], axis=-1)

        if self.backend in _BACKENDS_REQUIRING_3D:
            results = self._inferencer.infer(
                bgr, pcd_organized_mm=pcd_organized_mm, valid_mask=valid_mask
            )
        else:
            results = self._inferencer.infer(bgr)

        detections: list[Detection] = []
        for r in results:
            score = float(r.score)
            if score < threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in r.bbox]
            detections.append(
                Detection(
                    label=str(r.class_name),
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    mask=getattr(r, "mask", None),
                    initial_pose=getattr(r, "initial_pose", None),
                )
            )
        return detections