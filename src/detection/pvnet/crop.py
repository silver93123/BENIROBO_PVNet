"""bbox -> 정사각형 패딩 크롭 -> 고정 크기 리사이즈 -> 정규화.

train_pvnet.py(학습)와 앞으로 만들 추론 wrapper(RTMDetInferencerPVNet류)가
반드시 "동일한" 전처리를 써야 한다 - 학습 때와 추론 때 크롭 방식이 다르면
domain shift가 생겨 voting/PnP 정확도가 크게 떨어진다. model.py, pipeline.py
여러 곳의 docstring이 "crop_and_preprocess() 출력을 그대로 재사용할 수 있다"고
가정하고 있었는데(원래 rotation_head_model.py에 있던 것), 이 프로젝트(PVNet
라벨링 전용 축소판)로 옮기며 그 파일 자체가 빠져 있었다. 이 모듈이 그 역할을
대신한다 - 이름은 기존 문서와 맞추기 위해 그대로 유지했다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_CROP_SIZE = 256
DEFAULT_BBOX_PADDING = 0.2  # bbox 각 변 기준 여유 비율. 표면 키포인트가 bbox 경계에
                            # 걸쳐 있거나 마스크 침식 등으로 bbox가 살짝 타이트해도
                            # 크롭 안에 들어오게 여유를 둔다.


@dataclass
class CropTransform:
    """원본 이미지 픽셀 좌표 -> 크롭 좌표 아핀 변환 (x/y 독립 스케일).

    crop_x = (full_x - x1) * scale_x
    crop_y = (full_y - y1) * scale_y

    x/y 스케일을 독립적으로 두는 이유: 원본 bbox가 이미 정사각형이 되도록
    compute_square_crop_box()에서 맞춰주므로 실제로는 scale_x == scale_y가
    거의 항상 성립하지만, 이미지 경계에 걸려 정사각형이 clip된 극단적인
    경우까지 안전하게 다루기 위해 일반형으로 둔다.
    """
    x1: float
    y1: float
    scale_x: float
    scale_y: float

    def to_crop(self, points_xy: np.ndarray) -> np.ndarray:
        """(..., 2) 원본 좌표 배열 -> 같은 shape의 크롭 좌표 배열."""
        points_xy = np.asarray(points_xy, dtype=np.float64)
        out = np.empty_like(points_xy)
        out[..., 0] = (points_xy[..., 0] - self.x1) * self.scale_x
        out[..., 1] = (points_xy[..., 1] - self.y1) * self.scale_y
        return out


def compute_square_crop_box(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    padding_ratio: float = DEFAULT_BBOX_PADDING,
) -> tuple[int, int, int, int]:
    """bbox(x1,y1,x2,y2)를 정사각형으로 확장(+패딩)하고 이미지 경계로 clip.

    정사각형으로 맞추는 이유: 비-정사각형 크롭을 정사각형(crop_size,
    crop_size)으로 리사이즈하면 x/y 스케일이 달라져 버리고, 그러면 벡터장이
    가리키는 "방향"이 실제 기하와 미묘하게 어긋난다 - 처음부터 정사각형으로
    잘라 리사이즈 시 x/y 스케일을 동일하게 유지하는 편이 안전하다.

    Returns:
        (x1, y1, x2, y2) 정수 픽셀 좌표, 이미지 경계 안으로 clip됨.
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox가 비정상입니다(x2<=x1 또는 y2<=y1): {bbox}")

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding_ratio)

    x1n = max(0, int(round(cx - side / 2.0)))
    y1n = max(0, int(round(cy - side / 2.0)))
    x2n = min(w, int(round(cx + side / 2.0)))
    y2n = min(h, int(round(cy + side / 2.0)))
    return x1n, y1n, x2n, y2n


def crop_and_preprocess(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    padding_ratio: float = DEFAULT_BBOX_PADDING,
    mask_background: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, CropTransform]:
    """원본 이미지 + bbox(+선택적 마스크) -> (crop_size, crop_size) 정규화 크롭.

    Args:
        image_bgr: (H, W, 3) uint8. 그레이스케일 센서 이미지를 3채널로 복제한
            것이어도 그대로 넘기면 된다(이 레포 관례).
        bbox: (x1, y1, x2, y2), 원본 이미지 픽셀 좌표.
        mask: (H, W) bool. mask_background=True면 필수.
        crop_size: 출력 정사각형 한 변(px).
        padding_ratio: compute_square_crop_box() 참고.
        mask_background: True면 물체 마스크 밖 픽셀을 0으로 지운다
            (model.py의 PVNetHead docstring이 권장하는 방식 - 배경이 지워져
            있으면 세그멘테이션 학습이 쉬워짐).

    Returns:
        crop_float: (crop_size, crop_size, 3) float32, [0, 1] 정규화됨.
        mask_crop: (crop_size, crop_size) bool, mask 인자가 None이면 None.
        transform: CropTransform. 원본 좌표계의 점(keypoints_2d 등)을
            transform.to_crop(points)로 이 크롭 좌표계에 맞게 옮길 때 쓴다.
    """
    if mask_background and mask is None:
        raise ValueError("mask_background=True인데 mask가 주어지지 않았습니다.")

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = compute_square_crop_box(bbox, (h, w), padding_ratio)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"크롭 영역이 이미지 경계 clip 이후 비정상입니다: bbox={bbox}, "
            f"image_shape={(h, w)}, clipped=({x1},{y1},{x2},{y2})"
        )

    crop = image_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    scale_x = crop_size / float(x2 - x1)
    scale_y = crop_size / float(y2 - y1)
    transform = CropTransform(x1=float(x1), y1=float(y1), scale_x=scale_x, scale_y=scale_y)

    mask_crop = None
    if mask is not None:
        mask_u8 = mask[y1:y2, x1:x2].astype(np.uint8)
        mask_crop = cv2.resize(mask_u8, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST) > 0

    crop_float = crop_resized.astype(np.float32) / 255.0
    if mask_background:
        crop_float = crop_float.copy()
        crop_float[~mask_crop] = 0.0

    return crop_float, mask_crop, transform