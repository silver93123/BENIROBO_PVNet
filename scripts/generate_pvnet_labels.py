"""PVNet 학습용 pseudo-label 자동 생성.

generate_rotation_labels.py(RotHead용)와 원리는 완전히 동일하다: RTMDet-Ins
baseline 검출 + ICP 정합을 세션의 모든 프레임에 돌리고, fitness가 높은
인스턴스만 골라 그 ICP 결과를 "정답"처럼 채택한다. fitness가 낮으면 ICP
자체가 불확실한 것이므로 버린다.

RotHead용 라벨과의 차이 - PVNet 학습에는 이게 더 필요하다:
    1) 회전(3x3)뿐 아니라 이동(translation)까지 포함한 전체 pose(4x4).
       PVNet은 3D 키포인트를 2D로 투영해서 vertex field GT(방향벡터장)를
       만들어야 하는데, 투영에는 회전+이동이 모두 필요하다(RotHead는
       회전만 회귀하고 이동은 별도로 포인트클라우드 centroid에서 구하므로
       라벨도 회전만 있으면 충분했다).
    2) 프레임별 카메라 intrinsic(fx, fy, cx, cy). 3D->2D 투영에 필요.
       app/core/camera_intrinsics.py의 기존 최소자승 역산 유틸을 그대로
       재사용한다(manual_labeling_tab.py, 이 스크립트가 동일 유틸을 씀).
    3) CAD 표면에서 뽑은 키포인트 3D 좌표(keypoints_3d). 인스턴스마다
       달라지는 게 아니라 "이 부품(CAD) 하나당 한 번" 계산하면 되는
       값이라, 인스턴스별 JSON이 아니라 별도 .npy 파일 하나로 저장한다 -
       train_pvnet.py(다음 단계)가 이 파일을 그대로 로드해서 모든 프레임의
       vertex field GT를 계산할 때 재사용한다.

의도적으로 안 하는 것: 픽셀별 vertex field(방향벡터장) 자체를 여기서 미리
계산해서 저장하지 않는다. (H, W, K*2) 크기라 인스턴스당 용량이 크고, 크롭
방식(out_size, mask_background 등)이 학습 스크립트 쪽에서 바뀔 때마다
라벨을 통째로 재생성해야 하는 문제가 생긴다. 대신 여기서는 "pose + 카메라
intrinsic + 키포인트 3D"라는 가벼운 표현만 저장하고, vertex field는
train_pvnet.py가 크롭 시점에 그때그때 계산한다 - crop_and_preprocess()가
원본 좌표계를 크롭 좌표계로 바꿀 때 intrinsic도 같이 조정해야 한다는
pipeline.py의 기존 경고와도 맥락이 같다.

실행 (프로젝트 루트에서):
    python scripts/generate_pvnet_labels.py \\
        --dataset 20260521_114500 \\
        --cad data/cad/bracket.stl \\
        --checkpoint checkpoints/rtmdet_ins_bracket/best.pth \\
        --config configs/rtmdet-ins_bracket_used.py

여러 세션을 한 번에 넘기려면 콤마로 구분:
    --dataset 20260521_114500,20260522_090000

생성물:
    <mask-out-dir>/<session>_<frame>_obj<i>.npy     - 채택된 인스턴스의 마스크
    <keypoints-out>                                  - CAD 키포인트 3D 좌표 (K+1, 3) .npy,
                                                        이 CAD(부품) 전체에 공통
    <labels-out>                                     - train_pvnet.py가 읽을 라벨 JSON
    <preview-dir>/<session>_<frame>_obj<i>.jpg       - (--preview-dir 지정 시) 마스크
                                                        오버레이 + 투영된 키포인트를 그린
                                                        검토용 이미지. 키포인트가 실제
                                                        물체 표면 위에 정확히 찍히는지
                                                        눈으로 바로 검증할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import icp_runner  # noqa: E402
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points  # noqa: E402
from app.core.detector import Detector  # noqa: E402
from app.core.icp_runner import ICPParams  # noqa: E402
from src.detection.pvnet.keypoints import DEFAULT_NUM_SURFACE_KEYPOINTS, farthest_point_sampling  # noqa: E402

# --- 하드코딩 상수 (CLI로 override 가능) ---
DEFAULT_DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_MASK_OUT_DIR = ROOT / "data" / "pvnet_labels_masks"
DEFAULT_LABELS_OUT = ROOT / "data" / "pvnet_labels.json"
DEFAULT_KEYPOINTS_OUT_TEMPLATE = str(ROOT / "data" / "pvnet_keypoints_{cad_stem}.npy")
DEFAULT_FITNESS_MIN = 0.85  # generate_rotation_labels.py와 동일 기준 - 학습 라벨은
                            # ICP 탭 기본 fitness_threshold보다 엄격하게 잡음.
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_DEVICE = "cuda:0"
DEFAULT_PREVIEW_DIR = ROOT / "data" / "pvnet_labels_preview"
DEFAULT_PREVIEW_MAX_WIDTH = 480


def _load_frame(session_dir: Path, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
    pcd_organized = np.load(session_dir / "pointcloud_organized" / f"{frame_name}.npy")
    valid_mask = np.load(session_dir / "valid_mask" / f"{frame_name}.npy")
    return pcd_organized, valid_mask


def _usable_frames(session_dir: Path) -> list[str]:
    intensity_dir = session_dir / "intensity"
    organized_dir = session_dir / "pointcloud_organized"
    mask_dir = session_dir / "valid_mask"
    stems = sorted(f.stem for f in intensity_dir.glob("*.png"))
    return [
        s for s in stems
        if (organized_dir / f"{s}.npy").is_file() and (mask_dir / f"{s}.npy").is_file()
    ]


def _save_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    fitness: float,
    keypoints_2d: np.ndarray,
    out_path: Path,
    max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
) -> None:
    """마스크 오버레이 + bbox + fitness + 투영된 키포인트를 그려 검토용으로 저장.

    generate_rotation_labels.py의 _save_preview()와 거의 동일하되, 키포인트
    투영점을 추가로 찍는다 - "라벨이 맞게 생성됐는지"를 눈으로 바로 확인하는
    게 이 함수의 핵심 목적이다(라벨 자체는 이 이미지와 무관하게 이미
    JSON/.npy에 저장 완료된 상태).
    """
    overlay = image_bgr.copy()
    color = np.array([60, 220, 60], dtype=np.uint8)
    overlay[mask] = (
        overlay[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
    ).astype(np.uint8)

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        overlay, f"fitness={fitness:.3f}", (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
    )

    # 키포인트: 센트로이드(0번)는 노란색 큰 원, 나머지 표면점은 하늘색 작은 원.
    for idx, (u, v) in enumerate(keypoints_2d):
        u_i, v_i = int(round(u)), int(round(v))
        if idx == 0:
            cv2.circle(overlay, (u_i, v_i), 6, (0, 255, 255), -1)
        else:
            cv2.circle(overlay, (u_i, v_i), 4, (255, 200, 0), -1)
            cv2.putText(
                overlay, str(idx), (u_i + 4, v_i - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA,
            )

    h, w = overlay.shape[:2]
    if w > max_width:
        scale = max_width / w
        overlay = cv2.resize(overlay, (max_width, int(h * scale)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def generate_labels(
    dataset_names: list[str],
    cad_path: str,
    checkpoint_path: str,
    config_path: str,
    fitness_min: float,
    mask_out_dir: Path,
    labels_out: Path,
    keypoints_out: Path,
    num_keypoints: int = DEFAULT_NUM_SURFACE_KEYPOINTS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    preview_dir: Path | None = None,
) -> None:
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    keypoints_out.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pvnet-label] 미리보기 이미지 저장: {preview_dir}")

    params = ICPParams()
    print(f"[pvnet-label] CAD 로드 중: {cad_path}")
    cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)
    cad_visible_normal, cad_visible_flipped = icp_runner.build_visible_cad_pair(cad_pcd, params)

    # 키포인트는 CAD 전체 표면(가시면 필터링 전, 원본 미변환 좌표계)에서
    # 뽑는다 - build_visible_cad_pair()가 만드는 "가시면만 남긴" 서브셋은
    # 카메라 시점에 따라 달라지므로 키포인트 기준으로 쓰면 안 된다. 반드시
    # cad_pcd(load_cad_as_pcd 직후, 원본 전체) 기준이어야 프레임이 달라져도
    # 키포인트 3D 좌표가 고정된다.
    cad_points = np.asarray(cad_pcd.points)
    keypoints_3d = farthest_point_sampling(cad_points, num_keypoints=num_keypoints, include_centroid=True)
    np.save(keypoints_out, keypoints_3d)
    print(f"[pvnet-label] 키포인트 {keypoints_3d.shape[0]}개(센트로이드 포함) 저장: {keypoints_out}")

    detector = Detector(
        checkpoint_path=checkpoint_path, config_path=config_path,
        device=device, score_threshold=score_threshold, backend="rtmdet_ins",
    )

    items: list[dict] = []
    n_total_instances = 0
    n_accepted = 0

    for dataset_name in dataset_names:
        session_dir = DEFAULT_DATASET_ROOT / dataset_name
        if not session_dir.is_dir():
            print(f"[pvnet-label] ⚠ 세션 폴더 없음, 건너뜀: {session_dir}")
            continue

        frames = _usable_frames(session_dir)
        print(f"[pvnet-label] 세션 {dataset_name}: 프레임 {len(frames)}개")

        for frame_name in frames:
            pcd_organized, valid_mask = _load_frame(session_dir, frame_name)
            image_path = session_dir / "intensity" / f"{frame_name}.png"

            # 프레임마다 유효 픽셀 분포가 조금씩 달라 intrinsic 추정값도 미세하게
            # 흔들릴 수 있으므로, camera_intrinsics.py 관례대로 프레임 단위로
            # 다시 추정한다(manual_labeling_tab.py와 동일 방식).
            try:
                fx, fy, cx, cy = estimate_intrinsics_from_organized_pcd(pcd_organized, valid_mask)
            except ValueError:
                print(f"[pvnet-label]   ⚠ {frame_name}: intrinsic 추정 불가(유효 픽셀 없음), 건너뜀")
                continue

            detections = detector.predict(str(image_path), conf_threshold=score_threshold)
            detections = [d for d in detections if d.mask is not None]

            frame_bgr = None
            if preview_dir is not None and detections:
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    frame_bgr = np.stack([gray, gray, gray], axis=-1)

            for i, det in enumerate(detections):
                n_total_instances += 1
                pts_mm = icp_runner.extract_instance_points_mm(
                    det.mask, pcd_organized, valid_mask, erode_px=params.mask_erode_px
                )
                result = icp_runner.run_icp_for_instance(
                    i, pts_mm, cad_pcd, cad_visible_normal, cad_visible_flipped, params=params,
                )

                if not result.ok or result.fitness is None:
                    reason = (result.error or "ICP 실패").replace(" ", "_")
                    print(
                        f"[pvnet-label]   RESULT frame={frame_name} obj={i} "
                        f"status=REJECT fitness=none reason={reason}"
                    )
                    continue

                if result.fitness < fitness_min:
                    print(
                        f"[pvnet-label]   RESULT frame={frame_name} obj={i} "
                        f"status=REJECT fitness={result.fitness:.3f} reason=fitness<{fitness_min:.3f}"
                    )
                    continue

                mask_filename = f"{dataset_name}_{frame_name}_obj{i}.npy"
                mask_path = mask_out_dir / mask_filename
                np.save(mask_path, det.mask.astype(bool))

                items.append({
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "bbox": [float(v) for v in det.bbox],
                    # RotHead 라벨(rotation_matrix만)과 달리 전체 4x4를 저장 -
                    # PVNet 학습 시 키포인트 3D->2D 투영에 이동까지 필요하다.
                    "pose": result.T.tolist(),
                    "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                })
                n_accepted += 1

                if preview_dir is not None and frame_bgr is not None:
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    kpts_cam = (result.T[:3, :3] @ keypoints_3d.T).T + result.T[:3, 3]
                    kpts_2d = project_points(kpts_cam * 1000.0, (fx, fy, cx, cy))  # project_points는 mm 단위 관례
                    preview_path = preview_dir / mask_filename.replace(".npy", ".jpg")
                    _save_preview(frame_bgr, det.mask, det.bbox, result.fitness, kpts_2d, preview_path)

                print(
                    f"[pvnet-label]   RESULT frame={frame_name} obj={i} "
                    f"status=ACCEPT fitness={result.fitness:.3f}"
                )

            if detections:
                print(
                    f"[pvnet-label]   {frame_name}: 인스턴스 {len(detections)}개 검사 "
                    f"-> 누적 채택 {n_accepted}건"
                )

    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(
        f"[pvnet-label] 완료: 전체 인스턴스 {n_total_instances}개 중 "
        f"{n_accepted}개 채택 (fitness >= {fitness_min}) -> {labels_out}"
    )
    if n_accepted == 0:
        print(
            "[pvnet-label] ⚠ 채택된 라벨이 0건입니다. --fitness-min을 낮추거나, "
            "기존 ICP 파이프라인(icp_test_tab)에서 이 세션의 fitness가 실제로 "
            "얼마나 나오는지 먼저 확인하세요."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="세션 폴더명, 콤마로 여러 개 나열 가능")
    parser.add_argument("--cad", required=True, help="CAD 파일 경로 (.stl/.ply/.obj)")
    parser.add_argument("--checkpoint", required=True, help="RTMDet-Ins 체크포인트 (.pth)")
    parser.add_argument("--config", required=True, help="RTMDet-Ins config (.py)")
    parser.add_argument("--fitness-min", type=float, default=DEFAULT_FITNESS_MIN)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--num-keypoints", type=int, default=DEFAULT_NUM_SURFACE_KEYPOINTS,
                         help="센트로이드 제외 표면 키포인트 개수 (PVNetHead.num_keypoints와 반드시 일치해야 함)")
    parser.add_argument("--mask-out-dir", default=str(DEFAULT_MASK_OUT_DIR))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS_OUT))
    parser.add_argument("--keypoints-out", default=None,
                         help="미지정 시 CAD 파일명 기준으로 data/pvnet_keypoints_<cad_stem>.npy에 저장")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--preview-dir", default=None,
                         help="지정하면 채택된 인스턴스마다 마스크+투영 키포인트 오버레이 이미지를 저장 (선택)")
    args = parser.parse_args()

    cad_stem = Path(args.cad).stem
    keypoints_out = (
        Path(args.keypoints_out) if args.keypoints_out
        else Path(DEFAULT_KEYPOINTS_OUT_TEMPLATE.format(cad_stem=cad_stem))
    )

    dataset_names = [s.strip() for s in args.dataset.split(",") if s.strip()]
    generate_labels(
        dataset_names=dataset_names,
        cad_path=args.cad,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        fitness_min=args.fitness_min,
        mask_out_dir=Path(args.mask_out_dir),
        labels_out=Path(args.labels_out),
        keypoints_out=keypoints_out,
        num_keypoints=args.num_keypoints,
        score_threshold=args.score_threshold,
        device=args.device,
        preview_dir=Path(args.preview_dir) if args.preview_dir else None,
    )


if __name__ == "__main__":
    main()
