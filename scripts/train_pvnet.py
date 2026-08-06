"""PVNetHead 학습.

data/pvnet_labels.json(라벨) + (필요시) 키포인트 3D .npy를 읽어서, 크롭
시점에 벡터장/세그멘테이션 GT를 그때그때 계산하며 학습한다. generate_pvnet_labels.py
docstring에서 이미 예고된 설계 그대로: "vertex field는 train_pvnet.py가 크롭
시점에 계산한다" - 라벨 파일에는 가벼운 표현(keypoints_2d 또는 pose+intrinsics)만
저장돼 있고, 크롭 방식(crop_size, padding, mask_background)이 바뀌어도 라벨을
다시 만들 필요가 없다.

라벨 소스 3곳의 스키마가 서로 다르다는 점에 주의 - 이 스크립트는 셋 다 지원한다:
    1) manual_labeling_tab.py  (탭1) -> keypoints_2d만 있음, pose/intrinsics 없음
    2) pvnet_label_generation_tab.py (탭2, ICP 기반) -> keypoints_2d + pose +
       camera_intrinsics + fitness 전부 있음
    3) generate_pvnet_labels.py (오프라인 배치 스크립트) -> pose + camera_intrinsics만
       있고 keypoints_2d는 없음 (--keypoints-3d로 직접 투영해서 만들어야 함)
항목별로 "keypoints_2d가 이미 있으면 그대로 쓰고, 없으면 pose+camera_intrinsics+
keypoints_3d로 투영해서 만든다"는 우선순위로 처리한다 - 세 스키마를 하나로
묶어서 처리하기 위한 최소 공통분모다.

실행 (프로젝트 루트에서):
    python scripts/train_pvnet.py \\
        --out-dir checkpoints/pvnet_bracket \\
        --keypoints-3d data/pvnet_keypoints_bracket.npy

라벨에 keypoints_2d가 없는 항목이 하나도 없다면(=탭1/탭2에서만 라벨을 만들었다면)
--keypoints-3d는 생략해도 된다.

체크포인트/설정은 --out-dir에 저장된다:
    train_config.json  - 이 실행에 쓰인 모든 하이퍼파라미터 (재현/추론 wrapper용)
    keypoints_3d.npy   - --keypoints-3d를 넘겼다면 그 사본 (추론 시 이 파일과
                          체크포인트를 항상 같이 챙기면 됨)
    best.pth            - val loss 기준 최적 체크포인트
    last.pth            - 마지막 epoch 체크포인트 (--resume으로 이어서 학습 가능)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from app.core.camera_intrinsics import project_points  # noqa: E402
from app.core.paths import PROJECT_ROOT  # noqa: E402
from src.detection.pvnet.crop import (  # noqa: E402
    DEFAULT_BBOX_PADDING, DEFAULT_CROP_SIZE, crop_and_preprocess,
)
from src.detection.pvnet.model import (  # noqa: E402
    PVNetHead, segmentation_loss, vertex_smooth_l1_loss,
)

DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "pvnet_labels.json"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 0


# =============================================================================
# 라벨 로딩 + keypoints_2d 확보 (스키마 3종 통합)
# =============================================================================
@dataclass
class LabelItem:
    image: str
    mask: str
    keypoints_2d: np.ndarray  # (K, 2), 원본 이미지 픽셀 좌표


def _project_keypoints(item: dict, keypoints_3d: np.ndarray) -> np.ndarray:
    """pose(4x4, m 단위) + camera_intrinsics로 keypoints_3d를 2D에 투영.

    pvnet_label_generation_tab.py / generate_pvnet_labels.py가 저장하는 pose는
    CAD 로컬 좌표계(keypoints_3d와 동일 좌표계) -> 카메라(씬) 좌표계 변환이므로,
    그대로 R,t를 적용하면 된다 (별도 CAD 중심 보정 불필요 - 두 저장 코드의
    주석과 동일한 결론).
    """
    pose = np.asarray(item["pose"], dtype=np.float64)
    R, t = pose[:3, :3], pose[:3, 3]
    intr = item["camera_intrinsics"]
    intrinsics = (intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    kpts_cam_mm = (R @ keypoints_3d.T).T * 1000.0 + t * 1000.0
    return project_points(kpts_cam_mm, intrinsics)


def load_label_items(
    labels_path: Path,
    keypoints_3d: np.ndarray | None,
    min_fitness: float | None,
) -> list[LabelItem]:
    """pvnet_labels.json -> 검증된 LabelItem 리스트.

    항목별로 다음을 확인하고, 실패하면 이유를 출력한 뒤 건너뛴다(학습이
    조용히 절반의 데이터로 돌아가는 것을 막기 위해 스킵 사유를 전부 로그로 남김):
        - image/mask 파일이 실제로 존재하는가
        - keypoints_2d를 확보할 수 있는가 (직접 있거나, pose+camera_intrinsics+
          keypoints_3d로 투영 가능한가)
        - min_fitness가 지정됐고 항목에 fitness가 있다면 그 기준을 만족하는가
          (fitness 필드가 아예 없는 항목 - 탭1 수동 라벨 - 은 항상 통과시킨다.
          사람이 이미 눈으로 검증한 라벨이라 fitness 개념 자체가 없다)
    """
    if not labels_path.is_file():
        raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items: list[LabelItem] = []
    n_skipped_fitness = 0
    n_skipped_missing_file = 0
    n_skipped_no_keypoints = 0
    expected_k: int | None = None if keypoints_3d is None else keypoints_3d.shape[0]

    for i, raw in enumerate(raw_items):
        if min_fitness is not None and "fitness" in raw and raw["fitness"] < min_fitness:
            n_skipped_fitness += 1
            continue

        if not Path(raw["image"]).is_file() or not Path(raw["mask"]).is_file():
            n_skipped_missing_file += 1
            print(f"[pvnet-train]   ⚠ 항목 {i}: image/mask 파일 없음, 건너뜀 (image={raw['image']})")
            continue

        if "keypoints_2d" in raw:
            keypoints_2d = np.asarray(raw["keypoints_2d"], dtype=np.float64)
        elif "pose" in raw and "camera_intrinsics" in raw:
            if keypoints_3d is None:
                n_skipped_no_keypoints += 1
                print(
                    f"[pvnet-train]   ⚠ 항목 {i}: keypoints_2d가 없고 pose/intrinsics만 있는데 "
                    "--keypoints-3d가 지정되지 않음, 건너뜀"
                )
                continue
            keypoints_2d = _project_keypoints(raw, keypoints_3d)
        else:
            n_skipped_no_keypoints += 1
            print(f"[pvnet-train]   ⚠ 항목 {i}: keypoints_2d도 pose/intrinsics도 없음, 건너뜀")
            continue

        if expected_k is None:
            expected_k = keypoints_2d.shape[0]
        elif keypoints_2d.shape[0] != expected_k:
            raise ValueError(
                f"항목 {i}의 키포인트 개수({keypoints_2d.shape[0]})가 이전 항목들"
                f"({expected_k})과 다릅니다 - 서로 다른 CAD/설정으로 만든 라벨이 "
                "섞여 있을 가능성이 있습니다. labels 파일을 확인하세요."
            )

        items.append(LabelItem(image=raw["image"], mask=raw["mask"], keypoints_2d=keypoints_2d))

    print(
        f"[pvnet-train] 라벨 {len(raw_items)}건 중 {len(items)}건 사용 "
        f"(건너뜀: fitness={n_skipped_fitness}, 파일없음={n_skipped_missing_file}, "
        f"키포인트없음={n_skipped_no_keypoints})"
    )
    if not items:
        raise RuntimeError("사용 가능한 라벨이 0건입니다. 위 스킵 사유를 확인하세요.")
    return items


# =============================================================================
# Dataset: 크롭 + 벡터장/세그멘테이션 GT를 그때그때 계산
# =============================================================================
class PVNetLabelDataset(Dataset):
    def __init__(
        self,
        items: list[LabelItem],
        crop_size: int = DEFAULT_CROP_SIZE,
        padding_ratio: float = DEFAULT_BBOX_PADDING,
        mask_background: bool = True,
    ):
        self.items = items
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.mask_background = mask_background

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]

        gray = cv2.imread(item.image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {item.image}")
        image_bgr = np.stack([gray, gray, gray], axis=-1)
        mask = np.load(item.mask)

        # bbox는 라벨 JSON에 저장돼 있지 않을 수 있으므로(마스크만 저장된 스키마도
        # 있음) 항상 마스크에서 직접 계산한다 - 마스크 자체가 bbox보다 더 정확한
        # 정보라 이렇게 하는 편이 오히려 낫다(라벨 저장 시점의 bbox가 이후
        # 마스크 침식 등으로 살짝 어긋나 있어도 항상 최신 마스크 기준으로 크롭됨).
        ys, xs = np.where(mask)
        if ys.size == 0:
            raise RuntimeError(f"마스크에 전경 픽셀이 없습니다: {item.mask}")
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

        crop_float, mask_crop, transform = crop_and_preprocess(
            image_bgr, bbox, mask,
            crop_size=self.crop_size, padding_ratio=self.padding_ratio,
            mask_background=self.mask_background,
        )
        keypoints_2d_crop = transform.to_crop(item.keypoints_2d)  # (K, 2)

        vertex_gt = _build_vertex_field(mask_crop, keypoints_2d_crop)  # (K*2, H, W) float32

        crop_tensor = torch.from_numpy(crop_float.transpose(2, 0, 1)).contiguous()
        seg_gt = torch.from_numpy(mask_crop.astype(np.int64))
        vertex_gt_tensor = torch.from_numpy(vertex_gt)
        return crop_tensor, seg_gt, vertex_gt_tensor


def _build_vertex_field(mask_crop: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray:
    """전경 마스크 + 크롭 좌표계 키포인트 -> 픽셀별 단위 방향벡터장 GT.

    논문 Eq.1: xo(p) = (x - p) / ||x - p||  (각 전경 픽셀 p에서 키포인트 x로
    향하는 단위벡터). 배경 픽셀은 0으로 채운다 - model.vertex_smooth_l1_loss가
    seg_gt로 마스킹해서 배경 기여를 어차피 제거하지만, 그전까지 텐서 자체는
    유효한 값(0)이어야 하므로 명시적으로 0을 채워둔다.

    Args:
        mask_crop: (H, W) bool.
        keypoints_2d: (K, 2) 크롭 좌표계 (x, y).

    Returns:
        (K*2, H, W) float32. 채널 순서 [k0_dx, k0_dy, k1_dx, k1_dy, ...] -
        model.py의 vertex 출력 채널 순서와 반드시 일치해야 한다.
    """
    h, w = mask_crop.shape
    k = keypoints_2d.shape[0]

    ys, xs = np.mgrid[0:h, 0:w]
    pix = np.stack([xs, ys], axis=-1).astype(np.float64)  # (H, W, 2)

    diff = keypoints_2d[:, None, None, :] - pix[None, :, :, :]  # (K, H, W, 2)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    unit = diff / np.clip(norm, 1e-8, None)

    unit = unit * mask_crop[None, :, :, None]  # 배경 픽셀 0으로
    field = np.moveaxis(unit, -1, 1)  # (K, 2, H, W) - [dx, dy] 순서 유지
    return field.reshape(k * 2, h, w).astype(np.float32)


# =============================================================================
# 학습 루프
# =============================================================================
def _split_train_val(
    items: list[LabelItem], val_ratio: float, seed: int
) -> tuple[list[LabelItem], list[LabelItem]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        # 데이터가 너무 적어 val을 떼면 train이 비는 경우 - val 없이 전부 train으로.
        train_idx, val_idx = idx, np.array([], dtype=int)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items


def _run_epoch(
    model: PVNetHead,
    loader: DataLoader,
    device: str,
    vertex_loss_weight: float,
    optimizer: optim.Optimizer | None,
    log_prefix: str,
    log_every: int,
) -> dict[str, float]:
    """optimizer가 주어지면 학습(backward 포함), None이면 평가(no_grad)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_seg, total_vertex, total_fg_iou, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (crops, seg_gt, vertex_gt) in enumerate(loader):
            crops = crops.to(device)
            seg_gt = seg_gt.to(device)
            vertex_gt = vertex_gt.to(device)

            seg_logits, vertex_pred = model(crops)
            loss_seg = segmentation_loss(seg_logits, seg_gt)
            loss_vertex = vertex_smooth_l1_loss(vertex_pred, vertex_gt, seg_gt)
            loss = loss_seg + vertex_loss_weight * loss_vertex

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                seg_pred = seg_logits.argmax(dim=1)
                intersection = ((seg_pred == 1) & (seg_gt == 1)).sum().item()
                union = ((seg_pred == 1) | (seg_gt == 1)).sum().item()
                fg_iou = intersection / union if union > 0 else 1.0

            total_seg += float(loss_seg.item())
            total_vertex += float(loss_vertex.item())
            total_fg_iou += fg_iou
            n_batches += 1

            if is_train and log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"[pvnet-train]   {log_prefix} batch {batch_idx + 1}/{len(loader)} "
                    f"seg_loss={loss_seg.item():.4f} vertex_loss={loss_vertex.item():.4f} "
                    f"fg_iou={fg_iou:.3f}"
                )

    n_batches = max(n_batches, 1)
    return {
        "seg_loss": total_seg / n_batches,
        "vertex_loss": total_vertex / n_batches,
        "total_loss": (total_seg + vertex_loss_weight * total_vertex) / n_batches,
        "fg_iou": total_fg_iou / n_batches,
    }


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[pvnet-train] ⚠ {device} 요청했지만 CUDA를 쓸 수 없어 cpu로 대체합니다.")
        device = "cpu"

    keypoints_3d = None
    if args.keypoints_3d:
        keypoints_3d = np.load(args.keypoints_3d).astype(np.float64)
        np.save(out_dir / "keypoints_3d.npy", keypoints_3d)
        print(f"[pvnet-train] 키포인트 3D 로드: {args.keypoints_3d} (K={keypoints_3d.shape[0]})")

    items = load_label_items(Path(args.labels), keypoints_3d, args.min_fitness)
    num_keypoints = items[0].keypoints_2d.shape[0]
    if keypoints_3d is not None and keypoints_3d.shape[0] != num_keypoints:
        raise ValueError(
            f"--keypoints-3d의 키포인트 개수({keypoints_3d.shape[0]})가 라벨의 "
            f"keypoints_2d 개수({num_keypoints})와 다릅니다."
        )

    train_items, val_items = _split_train_val(items, args.val_ratio, args.seed)
    print(f"[pvnet-train] train={len(train_items)}건, val={len(val_items)}건, num_keypoints={num_keypoints}")

    train_ds = PVNetLabelDataset(train_items, args.crop_size, args.bbox_padding, args.mask_background)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = None
    if val_items:
        val_ds = PVNetLabelDataset(val_items, args.crop_size, args.bbox_padding, args.mask_background)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PVNetHead(
        num_keypoints=num_keypoints, backbone="resnet18", pretrained=args.pretrained_backbone,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[pvnet-train] 체크포인트에서 재개: {args.resume} (epoch {start_epoch}부터)")

    config = {
        "num_keypoints": num_keypoints,
        "crop_size": args.crop_size,
        "bbox_padding": args.bbox_padding,
        "mask_background": args.mask_background,
        "backbone": "resnet18",
        "labels_path": str(args.labels),
        "keypoints_3d_path": str(args.keypoints_3d) if args.keypoints_3d else None,
        "vertex_loss_weight": args.vertex_loss_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = _run_epoch(
            model, train_loader, device, args.vertex_loss_weight, optimizer,
            log_prefix=f"epoch {epoch + 1}/{args.epochs} [train]", log_every=args.log_every,
        )
        scheduler.step()

        msg = (
            f"[pvnet-train] epoch {epoch + 1}/{args.epochs} "
            f"train: seg={train_metrics['seg_loss']:.4f} vertex={train_metrics['vertex_loss']:.4f} "
            f"total={train_metrics['total_loss']:.4f} fg_iou={train_metrics['fg_iou']:.3f}"
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = _run_epoch(
                model, val_loader, device, args.vertex_loss_weight, optimizer=None,
                log_prefix=f"epoch {epoch + 1}/{args.epochs} [val]", log_every=0,
            )
            msg += (
                f" | val: seg={val_metrics['seg_loss']:.4f} vertex={val_metrics['vertex_loss']:.4f} "
                f"total={val_metrics['total_loss']:.4f} fg_iou={val_metrics['fg_iou']:.3f}"
            )
        print(msg)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        torch.save(checkpoint, out_dir / "last.pth")

        monitored_loss = val_metrics["total_loss"] if val_metrics is not None else train_metrics["total_loss"]
        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, out_dir / "best.pth")
            print(f"[pvnet-train]   -> best 갱신 (loss={best_val_loss:.4f}), best.pth 저장")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, out_dir / f"epoch_{epoch + 1}.pth")

    print(f"[pvnet-train] 완료. 체크포인트: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="pvnet_labels.json 경로")
    parser.add_argument(
        "--keypoints-3d", default=None,
        help="CAD 키포인트 3D .npy 경로. 라벨에 keypoints_2d가 없는 항목(예: "
             "generate_pvnet_labels.py로 만든 라벨)이 있으면 필수.",
    )
    parser.add_argument("--out-dir", required=True, help="체크포인트/설정 저장 폴더")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--bbox-padding", type=float, default=DEFAULT_BBOX_PADDING)
    parser.add_argument(
        "--mask-background", action=argparse.BooleanOptionalAction, default=True,
        help="크롭 내 마스크 밖 픽셀을 0으로 지울지 여부 (기본 True, 권장값)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--vertex-loss-weight", type=float, default=1.0,
        help="total_loss = seg_loss + vertex_loss_weight * vertex_loss",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--min-fitness", type=float, default=None,
                         help="fitness 필드가 있는 항목(ICP 기반 라벨)에만 적용되는 추가 필터. "
                              "기본은 필터 없음(라벨 생성 시 이미 걸러졌다고 가정).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True,
        help="resnet18 ImageNet 사전학습 가중치 사용 여부",
    )
    parser.add_argument("--resume", default=None, help="이어서 학습할 체크포인트(.pth) 경로")
    parser.add_argument("--save-every", type=int, default=10, help="N epoch마다 별도 스냅샷 저장 (0이면 끔)")
    parser.add_argument("--log-every", type=int, default=20, help="N batch마다 학습 중 로그 출력")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()