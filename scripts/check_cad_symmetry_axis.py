"""CAD 부품의 원통형 대칭축(x/y/z) 확인 스크립트.

train_rotation_head.py의 cylindrical_x/y/z 대칭군 중 어느 걸 써야 하는지는
"raw CAD 파일 좌표계" 기준이 아니라, icp_runner.load_cad_as_pcd()가 적용하는
축 보정(cad_axis_roll/pitch/yaw_deg)까지 반영된 뒤의 좌표계 기준이어야 한다 -
rotation_labels.json에 저장된 rotation_matrix(T[:3,:3])가 바로 그 축보정된
cad_pcd를 타깃으로 ICP 정합한 결과이기 때문이다. 그래서 STL 파일을 그냥
열어서 바운딩박스만 보면 답이 틀릴 수 있다 (축보정 값이 0이 아니면).

이 스크립트는 icp_runner.load_cad_as_pcd()를 축보정 파라미터까지 동일하게
넘겨서 그대로 호출한 뒤, 실제 라벨과 같은 좌표계의 점군을 얻어서 두 가지
방법으로 긴 축을 찾는다:
    [방법 1] 바운딩박스 크기 비교 - CAD가 이미 대략 축에 맞춰져 있으면 충분
    [방법 2] PCA 주성분 분석 - CAD가 축에 깔끔하게 안 맞아도 형상 자체의
             주축 방향을 벡터로 잡아준다 (더 신뢰할 만함)

결과 확인은 이미지 저장이 아니라 open3d 인터랙티브 3D 창으로 띄운다 -
"5. ICP 정합테스트(TCP)" 탭의 "3D 뷰어 열기"와 동일한 라이브러리(open3d)를
쓰는 것이라 별도 설치가 필요 없다. CAD 점군(회색) + 감지된 긴 축(빨간 선) +
좌표축(원점 기준 xyz 프레임)을 마우스로 돌려보면서 확인하면 된다.

사용법: 아래 "설정값" 구역만 고쳐서 실행.
    python scripts/check_cad_symmetry_axis.py

CAD_PATH, AXIS_ROLL_DEG, AXIS_PITCH_DEG, AXIS_YAW_DEG는 "6. 회전 라벨 생성"
탭(ICP 파라미터 박스)에서 이 CAD에 실제로 쓰고 있는 "CAD 축보정 roll/pitch/yaw"
값을 그대로 넣을 것 - 다르면 실제 라벨 좌표계와 결과가 안 맞는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import icp_runner  # noqa: E402
from app.core.icp_runner import ICPParams  # noqa: E402

# =============================================================================
# 설정값 - 여기만 고쳐서 실행하면 됨
# =============================================================================
CAD_PATH = "/home/silver/binpicking_vision/BENIROBO_RTMDetTrain/data/cad/bolt_M10_80-Mesh.stl"
AXIS_ROLL_DEG = 0.0    # "6. 회전 라벨 생성" 탭의 "CAD 축보정 roll"과 동일한 값
AXIS_PITCH_DEG = 0.0   # "CAD 축보정 pitch"와 동일한 값
AXIS_YAW_DEG = 0.0     # "CAD 축보정 yaw"와 동일한 값
# =============================================================================

AXIS_NAMES = "xyz"


POINT_KEEP_RATIO = 0.15  # 점군을 이 비율만큼만 남겨서 듬성듬성하게 - 축 선이 안 가려지게
POINT_SIZE = 2.0


def _show_interactive(cad_pcd, principal_dir: np.ndarray, extent: np.ndarray,
                       aabb_axis: str, pca_axis: str) -> None:
    """CAD 점군(듬성듬성) + 감지된 긴 축을 open3d 인터랙티브 창으로 띄운다.

    드래그: 회전 / 스크롤: 확대-축소 / 창을 닫으면 스크립트가 이어서 종료된다.
    빨간 선이 PCA로 찾은 긴 축, 좌표 프레임(RGB 화살표)이 x(빨강)/y(초록)/z(파랑) 참고용.

    주의: 신형 렌더러(o3d.visualization.draw + MaterialRecord)로 진짜 알파
    반투명을 시도했으나, "unlitLine" 셰이더가 일부 OpenGL 드라이버 조합에서
    "uniform named srgbColor not found" 크래시를 낸다(Open3D 0.19의 알려진
    호환성 문제). 그래서 안정적인 구버전 draw_geometries()로 되돌리고, 대신
    점 개수를 줄이고(POINT_KEEP_RATIO) 점 크기를 작게(POINT_SIZE) 그려서
    빨간 선이 점들 사이로 잘 보이게 우회한다. 구버전 렌더러는 진짜 알파블렌딩은
    지원 안 하지만 이 방법이 크래시 없이 확실하게 동작한다.
    """
    import open3d as o3d

    center = cad_pcd.get_center()
    axis_len = extent.max() * 0.6
    p0 = center - principal_dir * axis_len
    p1 = center + principal_dir * axis_len
    axis_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0, p1]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    axis_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 빨강 = 감지된 긴 축

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=extent.max() * 0.3, origin=center
    )  # x=빨강/y=초록/z=파랑 참고용 (open3d 기본 색상)

    n = len(cad_pcd.points)
    keep_idx = np.random.default_rng(0).choice(
        n, size=max(1, int(n * POINT_KEEP_RATIO)), replace=False
    )
    sparse_pcd = cad_pcd.select_by_index(keep_idx)
    sparse_pcd.paint_uniform_color([0.6, 0.6, 0.6])

    print(
        "\n3D 창을 띄웁니다 (드래그=회전, 스크롤=확대·축소). 창을 닫으면 스크립트가 종료됩니다.\n"
        f"  회색 점군(전체의 {POINT_KEEP_RATIO*100:.0f}%만 표시) = CAD, "
        f"빨간 선 = 감지된 긴 축(PCA -> {pca_axis}), 좌표 프레임 = x(빨강)/y(초록)/z(파랑)"
    )
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"CAD 대칭축 확인 - AABB={aabb_axis} / PCA={pca_axis}",
                       width=1000, height=800)
    vis.add_geometry(sparse_pcd)
    vis.add_geometry(axis_line)
    vis.add_geometry(frame)
    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.line_width = 8.0  # 드라이버에 따라 무시될 수 있음 (OpenGL 코어 프로파일 제약, 알려진 한계)
    vis.run()
    vis.destroy_window()


def check_axis(cad_path: str, axis_roll: float, axis_pitch: float, axis_yaw: float) -> None:
    params = ICPParams(
        cad_axis_roll_deg=axis_roll,
        cad_axis_pitch_deg=axis_pitch,
        cad_axis_yaw_deg=axis_yaw,
    )
    print(f"CAD 로드 중: {cad_path}")
    print(f"  축보정 적용: roll={axis_roll} pitch={axis_pitch} yaw={axis_yaw} (deg)")
    cad_pcd = icp_runner.load_cad_as_pcd(cad_path, params)

    points = np.asarray(cad_pcd.points)
    centered = points - points.mean(axis=0)

    # --- 방법 1: 바운딩박스 크기 ---
    extent = centered.max(axis=0) - centered.min(axis=0)
    aabb_axis = AXIS_NAMES[int(np.argmax(extent))]
    print(f"\n[방법 1] 바운딩박스 크기 (x, y, z) = {extent.round(4)} m")
    print(f"  -> 가장 긴 축: {aabb_axis} ({extent.max()*1000:.1f} mm)")

    # --- 방법 2: PCA 주성분 ---
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순 정렬됨
    principal_dir = eigvecs[:, -1]  # 분산이 가장 큰(=가장 긴) 축 방향벡터
    dominant_component = int(np.argmax(np.abs(principal_dir)))
    pca_axis = AXIS_NAMES[dominant_component]
    alignment = abs(principal_dir[dominant_component])  # 1.0이면 축과 완전히 일치
    print(f"\n[방법 2] PCA 주성분(가장 긴 축) 방향벡터: {principal_dir.round(4)}")
    print(f"  -> 가장 가까운 축: {pca_axis} (정렬도 {alignment:.3f}, 1.0에 가까울수록 축과 정확히 일치)")
    if alignment < 0.95:
        print(
            "  ⚠ 정렬도가 낮습니다 - CAD가 x/y/z 어느 축과도 깔끔하게 안 맞을 수 있습니다. "
            "cylindrical_x/y/z 중 가장 가까운 것을 쓰되, 대칭 근사가 완벽하지 않을 수 있습니다."
        )

    print()
    if aabb_axis == pca_axis:
        print(f"결론: --symmetry-group cylindrical_{aabb_axis} 를 쓰세요.")
    else:
        print(
            f"⚠ 두 방법 결과가 다릅니다 (바운딩박스={aabb_axis}, PCA={pca_axis}).\n"
            f"  PCA가 형상 자체의 분산을 보는 방식이라 더 신뢰할 만합니다 -> "
            f"cylindrical_{pca_axis} 권장."
        )

    _show_interactive(cad_pcd, principal_dir, extent, aabb_axis, pca_axis)


if __name__ == "__main__":
    check_axis(CAD_PATH, AXIS_ROLL_DEG, AXIS_PITCH_DEG, AXIS_YAW_DEG)