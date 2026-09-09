"""ICP 결과 3D 뷰어 (별도 프로세스로 실행됨).

open3d의 시각화 창(GLFW)은 자체 이벤트 루프를 돌기 때문에 PyQt6 메인
이벤트 루프와 한 프로세스 안에서 같이 쓰면 불안정하다. 그래서 ICP 탭에서는
결과를 매니페스트(JSON) + 레이어별 PLY로 저장해두고, 이 스크립트를 QProcess로
별도 실행해서 보여준다.

2026-07 개편: 레이어(배경/CAD/마스크 등)를 하나로 합친 PLY 한 장이 아니라,
이름별로 분리된 PLY 여러 장 + 매니페스트로 받는다. open3d의 신규
`o3d.visualization.draw()` API(0.14+)는 geometry를 {"name": ..., "geometry": ...}
딕셔너리 리스트로 받으면 show_ui=True일 때 "Geometries" 패널에 이름별
체크박스를 자동으로 만들어준다 - 커스텀 GUI 코드를 직접 짤 필요가 없다.

매니페스트 포맷 (dict):
    {
        "layers": [
            {"name": "Background (Height Colormap)", "file": "bg.ply", "visible": true},
            {"name": "CAD Registration Result", "file": "cad.ply", "visible": true},
            ...
        ]
    }

2026-09 추가: 기본 카메라를 "근사 직교(pseudo-orthographic) 투영"으로 바꿨다.
Open3D의 진짜 직교 투영(Camera.Projection.Ortho)은 버전에 따라 "Camera
preconditions not met. Using default projection." 경고와 함께 조용히
무시되고 원근 투영으로 남는 경우가 보고돼 있어(Open3D 공식 이슈 트래커
#4862, #5984) 신뢰할 수 없다. 대신 시야각(FOV)을 아주 좁게 잡고 카메라를
그만큼 멀리 물러세우는 표준 기법을 쓴다 - FOV가 0에 가까워질수록 원근
왜곡이 사실상 사라지고 직교 투영에 근접한다(회전시켜도 비스듬한 왜곡
없이 XYZ 축 그대로 회전하는 것처럼 보임).

사용:
    python -m app.core.icp_viewer <manifest.json> [--title TITLE] [--fov-deg 3.0]

하위호환: .ply 파일 경로를 직접 줘도 동작한다 (레이어 하나짜리 매니페스트로
취급 - 예전 방식으로 호출하는 코드가 남아있어도 안 깨지게).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui


def _load_manifest(path: str) -> dict:
    """매니페스트 JSON 또는 (하위호환) 단일 .ply 경로를 받아 layers 리스트로 정규화."""
    if path.lower().endswith(".ply"):
        return {"layers": [{"name": "Result", "file": path, "visible": True}]}

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "layers" not in manifest or not manifest["layers"]:
        raise ValueError(f"매니페스트에 layers가 없습니다: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="ICP 결과 3D 뷰어 (레이어별 체크박스 지원)")
    parser.add_argument("manifest_path", help="매니페스트 JSON 또는 (하위호환) 단일 .ply 경로")
    parser.add_argument("--title", default="ICP 정합 결과")
    parser.add_argument(
        "--fov-deg", type=float, default=3.0,
        help="가상 카메라 시야각(도). 작을수록 원근 왜곡이 줄어 직교 투영에 가까워진다 (기본 3도). "
             "너무 작으면(<0.5도) 카메라가 지나치게 멀어져 렌더링이 이상해질 수 있어 최소 0.5도로 제한됨.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest_path)
    manifest_dir = Path(args.manifest_path).parent

    geometries = []
    any_points = False
    for layer in manifest["layers"]:
        ply_path = layer["file"]
        # 매니페스트 안 file은 상대경로일 수 있음 (같은 폴더에 저장하는 관례) - 절대경로면 그대로 사용.
        full_path = ply_path if Path(ply_path).is_absolute() else str(manifest_dir / ply_path)

        pcd = o3d.io.read_point_cloud(full_path)
        if len(pcd.points) == 0:
            print(f"[icp_viewer] 경고: '{layer['name']}' 레이어가 비어있어 건너뜀 ({full_path})", flush=True)
            continue
        any_points = True
        geometries.append({
            "name": layer["name"],
            "geometry": pcd,
            "is_visible": layer.get("visible", True),
        })

    if not any_points:
        print("ERROR: 표시할 레이어가 하나도 없습니다 (전부 빈 포인트클라우드).", flush=True)
        return 1

    # 좌표축을 항상 world origin(카메라 원점, [0,0,0])에 그리면 실제 포인트는
    # 카메라에서 대개 0.5~1m쯤 떨어져 있어 화면에서 축과 점군이 멀리 따로
    # 떨어져 보인다 (원본 리포트: 축이 점군과 동떨어진 구석에 작게 뜸).
    # 방향(XYZ)은 그대로 두고 원점만 로드된 점군들의 포인트-가중 무게중심으로
    # 옮긴다 - create_coordinate_frame(origin=...)는 축의 방향은 안 바꾸고
    # 원점 위치만 평행이동한다. 같은 중심을 카메라 lookat에도 그대로 쓴다.
    scene_center = np.zeros(3)
    max_radius = 0.05  # 최소값 - 점이 거의 없는 극단적인 경우 카메라가 물체에 바짝 붙는 것 방지
    weighted_sum = np.zeros(3)
    total_points = 0
    for g in geometries:
        geo = g["geometry"]
        if isinstance(geo, o3d.geometry.PointCloud) and len(geo.points) > 0:
            n = len(geo.points)
            weighted_sum += np.asarray(geo.get_center()) * n
            total_points += n
    if total_points > 0:
        scene_center = weighted_sum / total_points
        for g in geometries:
            geo = g["geometry"]
            if isinstance(geo, o3d.geometry.PointCloud) and len(geo.points) > 0:
                pts = np.asarray(geo.points)
                r = float(np.linalg.norm(pts - scene_center, axis=1).max())
                max_radius = max(max_radius, r)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=scene_center.tolist())
    geometries.append({"name": "Axis", "geometry": axis, "is_visible": True})

    width, height = 1100, 780

    gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer(args.title, width, height)
    # show_settings=True -> 우측에 "Geometries" 패널이 생기고, 각 레이어(name)별
    # 체크박스로 켜고 끌 수 있다 (예전 draw(show_ui=True)와 동일한 효과).
    vis.show_settings = True
    for g in geometries:
        vis.add_geometry(g["name"], g["geometry"], is_visible=g["is_visible"])

    # 근사 직교 카메라: FOV를 좁게 잡고, 물체 반경(max_radius)이 그 좁은 화각
    # 안에 다 들어오도록 카메라를 충분히 멀리 물러세운다. 시선 방향은 원본
    # 센서가 보던 그대로(+Z, 카메라 좌표계 관례) 유지 - 회전(마우스 드래그)은
    # 이 뷰를 기준으로 자유롭게 계속 가능하고, 회전 중에도 좁은 FOV 덕에
    # 원근 왜곡이 거의 안 느껴진다.
    half_fov_rad = np.radians(max(args.fov_deg, 0.5) / 2.0)
    distance = (max_radius * 1.4) / np.tan(half_fov_rad)
    focal_px = (height / 2.0) / np.tan(half_fov_rad)
    intrinsic = np.array([
        [focal_px, 0.0, width / 2.0],
        [0.0, focal_px, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    eye = scene_center + np.array([0.0, 0.0, -distance])
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = -eye  # world-to-camera (identity 회전, 평행이동만)
    vis.setup_camera(intrinsic, extrinsic, width, height)

    print(
        f"[icp_viewer] geometry {len(geometries)}개 로드 완료, "
        f"근사 직교 뷰(FOV {args.fov_deg:.1f}도)로 표시 (창이 뜰 때까지 몇 초 걸릴 수 있음)",
        flush=True,
    )
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()
    print("[icp_viewer] 창이 닫힘", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())