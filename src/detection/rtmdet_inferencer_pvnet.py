"""RTMDet(2D 검출) 크롭 -> PVNet(키포인트 투표) -> PnP 추론 wrapper.

icp_runner.py의 ICP 기반 pose 추정 경로와 나란히 두는 "학습된 PVNet
모델 기반" 경로다. scripts/train_pvnet.py의 산출물(best.pth/epoch_N.pth/
last.pth + train_config.json + keypoints_3d.npy)을 로드해서, RTMDet이 이미
뽑아둔 인스턴스별 bbox/mask를 그대로 재사용해 크롭 -> PVNetHead forward ->
voting -> uncertainty PnP까지 한 번에 처리한다.

ICP 경로와의 핵심 차이: point cloud/HPR/초기 회전값 같은 준비가 전혀
필요 없다. PVNet은 2D 크롭 이미지 한 장만으로 2D-3D 대응 자체를
네트워크가 예측해서 곧장 pose를 낸다 - 그래서 이 모듈은 point cloud를
입력으로 받지 않는다(카메라 intrinsic 추정에만 여전히 필요해서 호출부에서
따로 준비해 넘겨준다).

결과(PVNetPoseResult.T)는 icp_runner.ICPResult.T와 같은 규약(4x4,
CAD 로컬 좌표계 -> 카메라 좌표계, translation 단위 m)을 따른다 - CAD
키포인트가 애초에 같은 좌표계(self._cad_pcd.points, m)에서 뽑혔고
(pvnet_label_generation_tab._ensure_keypoints_3d 참고), cv2.solvePnP는
입력 3D 점의 단위를 그대로 translation에 반영하기 때문이다(스케일
공변). 덕분에 화면 표시(keypoints_2d 투영)에 icp_runner와 동일한 공식
(R @ kpts_3d.T * 1000 + t * 1000 -> project_points)을 그대로 재사용할 수
있다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.core.camera_intrinsics import project_points
from app.core.detector import Detection
from src.detection.pvnet.crop import CropTransform, crop_and_preprocess
from src.detection.pvnet.model import PVNetHead
from src.detection.pvnet.pipeline import estimate_pose_from_crop

DEFAULT_CHECKPOINT_NAME = "best.pth"


@dataclass
class PVNetBundle:
    """학습 산출물(체크포인트 + 설정 + 3D 키포인트) 한 벌을 담는다 - 매
    추론마다 다시 로드하지 않도록 캐싱 용도(load_pvnet_bundle()이 만든다)."""
    model: PVNetHead
    keypoints_3d: np.ndarray   # (K, 3) m, CAD 로컬 좌표계 - pvnet_label_generation_tab의
                               # self._keypoints_3d와 동일 규약(센트로이드 포함 시 [0]=센트로이드)
    crop_size: int
    bbox_padding: float
    mask_background: bool
    device: str
    checkpoint_path: str
    config: dict


@dataclass
class PVNetPoseResult:
    instance_id: int
    ok: bool
    T: np.ndarray | None = None                # (4, 4) CAD 로컬 -> 카메라, m
    keypoints_2d: np.ndarray | None = None      # (K, 2) 원본 이미지 픽셀 좌표
    reprojection_error: float | None = None     # px, PVNet 크롭 좌표계 기준 (voting.py 결과)
    error: str | None = None


def load_pvnet_bundle(
    output_dir: str | Path,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    device: str = "cpu",
) -> PVNetBundle:
    """scripts/train_pvnet.py가 만든 출력 폴더에서 추론에 필요한 걸 전부 로드한다.

    폴더 구조(사용자가 "PVNet 학습" 실행 후 실제로 받는 것과 동일):
        <output_dir>/train_config.json   -- num_keypoints/crop_size/... 하이퍼파라미터
        <output_dir>/best.pth 등          -- {"model_state_dict": ..., "epoch": ...}
        <output_dir>/keypoints_3d.npy    -- train_pvnet.py가 학습 시점에 찍어둔
                                             스냅샷 (최우선으로 씀 - 학습 이후 절대
                                             안 바뀜). 이게 없을 때만(--keypoints-3d
                                             없이 학습했거나 폴더만 옮겨온 경우)
                                             train_config.json의 원본 경로로 폴백한다.

    Raises:
        FileNotFoundError: config/체크포인트/키포인트 파일 중 하나라도 없으면.
        ValueError: keypoints_3d 개수가 train_config.json의 num_keypoints와 다르면
            (학습 때와 다른 키포인트 파일을 잘못 골랐다는 뜻).
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"train_config.json을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ckpt_path = output_dir / checkpoint_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    num_keypoints = config["num_keypoints"]
    model = PVNetHead(
        num_keypoints=num_keypoints, backbone=config.get("backbone", "resnet18"), pretrained=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    keypoints_3d_path = config.get("keypoints_3d_path")
    keypoints_3d = None

    # 2026-09 버그 수정: 처음엔 config에 기록된 원본 경로(keypoints_3d_path)를
    # 먼저 읽고, 없을 때만 학습 폴더 안 스냅샷으로 폴백했다. 그런데
    # keypoints_3d_path는 "PVNet 라벨 생성" 탭이 CAD 파일명 기준 "고정
    # 경로"(data/pvnet_keypoints_{cad_stem}.npy)에 저장한 걸 가리키는데, 이
    # 경로는 학습 이후에도 그 탭에서 같은 CAD로 키포인트 설정(개수 등)을
    # 바꿔 다시 계산하면 내용이 덮어써진다 - 파일은 그대로 존재하니 원본
    # 우선 로직이 "이제는 다른 내용으로 바뀐" 파일을 조용히 읽어버렸다.
    # 반면 <output_dir>/keypoints_3d.npy는 train_pvnet.py가 학습 "그 순간"
    # np.save()로 한 번 찍어둔 스냅샷이라 이후 절대 안 바뀐다 - 그래서
    # 이제는 이 스냅샷을 최우선으로 쓰고, 없을 때만(--keypoints-3d 없이
    # 학습했거나 폴더를 옮겨온 경우) 원본 경로로 폴백한다.
    snapshot_path = output_dir / "keypoints_3d.npy"
    if snapshot_path.is_file():
        keypoints_3d = np.load(snapshot_path).astype(np.float64)
    elif keypoints_3d_path and Path(keypoints_3d_path).is_file():
        keypoints_3d = np.load(keypoints_3d_path).astype(np.float64)
    if keypoints_3d is None:
        raise FileNotFoundError(
            f"keypoints_3d.npy를 찾을 수 없습니다 (학습 폴더 스냅샷: {snapshot_path}, "
            f"config 원본 경로: {keypoints_3d_path!r})"
        )

    if keypoints_3d.shape[0] != num_keypoints:
        raise ValueError(
            f"keypoints_3d 개수({keypoints_3d.shape[0]})가 train_config.json의 "
            f"num_keypoints({num_keypoints})와 다릅니다 - 학습 때와 다른 키포인트 파일입니다."
        )

    return PVNetBundle(
        model=model,
        keypoints_3d=keypoints_3d,
        crop_size=config.get("crop_size", 256),
        bbox_padding=config.get("bbox_padding", 0.2),
        mask_background=config.get("mask_background", True),
        device=device,
        checkpoint_path=str(ckpt_path),
        config=config,
    )


def _adjust_intrinsics_for_crop(
    intrinsics: tuple[float, float, float, float], transform: CropTransform,
) -> np.ndarray:
    """원본 이미지 전체 기준 (fx,fy,cx,cy) -> crop_and_preprocess()가 만든
    크롭(잘리고 리사이즈된) 좌표계 기준 3x3 camera_matrix.

    crop_u = (full_u - x1) * scale_x 이고 full_u = fx*(X/Z) + cx 이므로:
        crop_u = fx*scale_x*(X/Z) + (cx - x1)*scale_x
    y도 동일하게 scale_y로 유도된다. voting.py가 낸 키포인트 위치는 이
    크롭 좌표계 기준이므로, PnP에는 이 조정된 camera_matrix를 넘겨야 한다
    (crop.py 모듈 docstring의 "조정된 intrinsic을 넘겨야 한다" 요구사항).
    """
    fx, fy, cx, cy = intrinsics
    fx_c = fx * transform.scale_x
    fy_c = fy * transform.scale_y
    cx_c = (cx - transform.x1) * transform.scale_x
    cy_c = (cy - transform.y1) * transform.scale_y
    return np.array([[fx_c, 0.0, cx_c], [0.0, fy_c, cy_c], [0.0, 0.0, 1.0]], dtype=np.float64)


def _tight_bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """마스크 전경 픽셀의 최소/최대 좌표로 타이트한 bbox를 만든다.

    scripts/train_pvnet.py의 PVNetLabelDataset.__getitem__()이 정확히 이
    방식으로 bbox를 계산한다 ("마스크가 bbox보다 더 정확한 정보"라는 이유로,
    RTMDet이 낸 bbox를 안 쓰고 항상 마스크에서 직접 계산). 추론 쪽이
    RTMDet의 bbox를 그대로 쓰면, 크롭 안에서 물체가 차지하는 위치/스케일이
    학습 때와 달라지는 학습-추론 불일치(train/test skew)가 생겨 vertex
    field 예측이 흔들린다 - 이 함수로 학습과 완전히 동일한 크롭 프레이밍을
    보장한다.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("마스크에 전경 픽셀이 없습니다.")
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def run_pvnet_on_detections(
    bundle: PVNetBundle,
    image_bgr: np.ndarray,
    detections: list[Detection],
    intrinsics: tuple[float, float, float, float],
) -> list[PVNetPoseResult]:
    """RTMDet이 이미 뽑아둔 인스턴스별 mask로 크롭 -> PVNet -> PnP.

    icp_runner.run_icp_for_instance()와 나란한 역할이지만, 인스턴스 하나당
    필요한 건 크롭 이미지 한 장 뿐이다 - point cloud/CAD 가시면/초기
    회전값 준비가 전혀 필요 없다.

    bbox는 det.bbox(RTMDet 출력)를 쓰지 않고 항상 마스크에서 다시 계산한다
    (_tight_bbox_from_mask) - train_pvnet.py의 학습 데이터셋과 정확히 같은
    크롭 프레이밍을 만들기 위함. 이게 어긋나면 학습 때 배운 벡터장과 추론
    시 크롭의 물체 위치/스케일이 미묘하게 달라져 pose 정확도가 크게
    떨어질 수 있다.

    인스턴스 하나가 실패(마스크 없음, 전경 픽셀 부족, PnP 실패 등)해도
    나머지 인스턴스 처리를 막지 않는다 - ok=False + error 메시지로 개별
    보고한다.
    """
    results: list[PVNetPoseResult] = []

    for i, det in enumerate(detections):
        if det.mask is None:
            results.append(PVNetPoseResult(instance_id=i, ok=False, error="마스크 없음"))
            continue

        try:
            bbox = _tight_bbox_from_mask(det.mask)
            crop_float, _mask_crop, transform = crop_and_preprocess(
                image_bgr, bbox, mask=det.mask,
                crop_size=bundle.crop_size, padding_ratio=bundle.bbox_padding,
                mask_background=bundle.mask_background,
            )
            crop_tensor = torch.from_numpy(crop_float).permute(2, 0, 1).float()
            crop_camera_matrix = _adjust_intrinsics_for_crop(intrinsics, transform)

            pnp_result = estimate_pose_from_crop(
                bundle.model, crop_tensor, bundle.keypoints_3d, crop_camera_matrix,
                device=bundle.device,
            )
        except Exception as exc:  # noqa: BLE001 - 인스턴스 하나 실패가 전체를 막으면 안 됨
            results.append(PVNetPoseResult(instance_id=i, ok=False, error=str(exc)))
            continue

        T = pnp_result.pose
        R, t_m = T[:3, :3], T[:3, 3]
        # icp_runner/pvnet_label_generation_tab과 동일한 공식 - 원본 이미지
        # 전체 기준 intrinsics로 재투영해야 화면(원본 해상도) 위에 정확히
        # 얹힌다. PnP 자체에 쓰인 crop_camera_matrix와 헷갈리지 말 것.
        kpts_cam_mm = (R @ bundle.keypoints_3d.T).T * 1000.0 + t_m * 1000.0
        keypoints_2d = project_points(kpts_cam_mm, intrinsics)

        results.append(PVNetPoseResult(
            instance_id=i, ok=True, T=T, keypoints_2d=keypoints_2d,
            reprojection_error=pnp_result.reprojection_error,
        ))

    return results