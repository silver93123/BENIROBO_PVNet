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

사용:
    python -m app.core.icp_viewer <manifest.json> [--title TITLE]

하위호환: .ply 파일 경로를 직접 줘도 동작한다 (레이어 하나짜리 매니페스트로
취급 - 예전 방식으로 호출하는 코드가 남아있어도 안 깨지게).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open3d as o3d


def _load_manifest(path: str) -> dict:
    """매니페스트 JSON 또는 (하위호환) 단일 .ply 경로를 받아 layers 리스트로 정규화."""
    if path.lower().endswith(".ply"):
        return {"layers": [{"name": "결과", "file": path, "visible": True}]}

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "layers" not in manifest or not manifest["layers"]:
        raise ValueError(f"매니페스트에 layers가 없습니다: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="ICP 결과 3D 뷰어 (레이어별 체크박스 지원)")
    parser.add_argument("manifest_path", help="매니페스트 JSON 또는 (하위호환) 단일 .ply 경로")
    parser.add_argument("--title", default="ICP 정합 결과")
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

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append({"name": "Axis", "geometry": axis, "is_visible": True})

    # show_ui=True -> 우측에 "Geometries" 패널이 자동으로 생기고, 각 레이어(name)별
    # 체크박스로 켜고 끌 수 있다 (open3d 0.14+ 신규 시각화 API 내장 기능).
    o3d.visualization.draw(
        geometries, title=args.title, width=1100, height=780, show_ui=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())