"""
Intensity 이미지 픽셀 <-> 포인트클라우드 매칭 테스트 (LUCID Helios)

참고로 주신 o3r_2d3d_registration.py는 RGB 케이스인데, 그건 2D(RGB) 헤드와
3D(ToF) 헤드가 물리적으로 다른 렌즈/좌표계라서 아래 절차가 전부 필요했다:
    1. 3D intrinsic으로 3D 광학 좌표계 포인트클라우드 생성
    2. 3D extrinsic으로 User 좌표계 변환
    3. 2D extrinsic 역변환으로 2D 광학 좌표계로 이동
    4. 2D inverse-intrinsic으로 2D 픽셀 좌표에 투영
    5. 그 픽셀에서 RGB 색상을 읽어옴

Helios는 이보다 훨씬 단순하다 - O3R처럼 "같은 3D 센서가 같은 격자에서 뽑아낸
서로 다른 두 값이라 registration이 필요 없는" 정도가 아니라, 애초에 픽셀
포맷(Coord3D_ABCY16) 자체가 한 픽셀당 X,Y,Z,Intensity를 "하나의 버퍼"로
묶어서 통째로 준다(src/camera/lucid_helios.py 상단 주석 참고). 그러니
points_organized[row, col]와 intensity[row, col]가 같은 원본 버퍼에서 나온
값이라 애초에 어긋날 수가 없다 - RGB처럼 별도 재투영/좌표변환은 원천적으로
필요 없다.

이 스크립트는 프로젝트의 LucidHeliosCamera 래퍼를 그대로 재사용해 intensity
값을 그레이스케일 색상으로 입힌 포인트클라우드를 만들고 PLY로 저장 + (옵션)
뷰어로 확인하는 순수 테스트용이다. 앱에는 통합하지 않은 독립 실행 스크립트다.

사용:
    python -m scripts.test_intensity_pointcloud
    python -m scripts.test_intensity_pointcloud --serial <시리얼> --no-view
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera.lucid_helios import LucidHeliosCamera  # noqa: E402

DEFAULT_SERIAL = None  # None이면 첫 번째 발견되는 카메라
DEFAULT_EXPOSURE = "Exp250Us"
DEFAULT_OPERATING_MODE = "Distance1500mm"
DEFAULT_OUT_PLY = "intensity_pointcloud.ply"
DEFAULT_POINT_SIZE = 2.0


def colorize_by_intensity(intensity_at_valid_pixels: np.ndarray) -> np.ndarray:
    """0~255 그레이스케일 intensity를 그대로 회색조 RGB(0~1)로 입힌다.

    인자는 frame.intensity를 frame.valid_mask로 걸러낸 (N,) 배열이어야
    한다 - frame.points와 순서가 1:1로 맞으려면 반드시 같은 valid_mask로
    걸러야 한다 (다른 방식으로 걸렀다간 포인트와 색이 어긋나 버린다).
    """
    gray = intensity_at_valid_pixels.astype(np.float64) / 255.0
    return np.stack([gray, gray, gray], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Helios intensity-포인트클라우드 픽셀 매칭 테스트")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="카메라 시리얼 번호 (미지정 시 첫 번째 발견 카메라)")
    parser.add_argument("--exposure", default=DEFAULT_EXPOSURE, help="예: Exp62_5Us / Exp250Us / Exp1000Us")
    parser.add_argument("--operating-mode", default=DEFAULT_OPERATING_MODE, help="예: Distance1500mm / Distance3000mm")
    parser.add_argument("--out", default=DEFAULT_OUT_PLY, help=f"저장할 PLY 경로 (기본값: {DEFAULT_OUT_PLY})")
    parser.add_argument("--no-view", action="store_true", help="저장만 하고 뷰어는 띄우지 않음")
    args = parser.parse_args()

    print(f"[test] Helios 연결 중 (serial={args.serial or '(자동탐색)'}) ...")
    with LucidHeliosCamera(
        serial=args.serial,
        exposure_time_selector=args.exposure,
        operating_mode=args.operating_mode,
    ) as cam:
        frame = cam.capture()

    n_total = frame.valid_mask.size
    n_valid = int(frame.valid_mask.sum())
    print(f"전체 픽셀: {n_total}, 유효 픽셀: {n_valid} ({n_valid / n_total * 100:.1f}%)")
    if n_valid == 0:
        print("ERROR: 유효 포인트가 하나도 없습니다 (거리 범위/노출 설정을 확인하세요).")
        return 1

    # frame.points는 이미 valid_mask로 걸러진 (N,3) mm 좌표, frame.intensity는
    # (H,W) 원본 격자 - 같은 valid_mask로 걸러야 순서가 1:1로 맞는다.
    intensity_at_valid = frame.intensity[frame.valid_mask]
    colors = colorize_by_intensity(intensity_at_valid)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame.points.astype(np.float64) / 1000.0)  # mm -> m (open3d 관례)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(args.out, pcd, write_ascii=False)
    print(f"PLY 저장 완료: {args.out} (포인트 {len(pcd.points)}개)")

    if args.no_view:
        return 0

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Helios Intensity-Colored Point Cloud (pixel-matched)")
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    vis.get_render_option().point_size = DEFAULT_POINT_SIZE
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())