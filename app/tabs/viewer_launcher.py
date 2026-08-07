"""3D 뷰어(app.core.icp_viewer)를 QProcess로 띄우는 공용 로직.

ICPWorkbenchTab(ICP 결과 뷰어)과 CADModelSettingsTab(CAD 가시면/초기자세
미리보기)이 둘 다 이 절차(레이어별 PLY 저장 -> 매니페스트 작성 -> 별도
프로세스로 뷰어 실행 -> 실패 시 에러 표시)를 그대로 쓴다 - 완전히 동일한
로직이라 믹스인으로 뽑았다.

이 믹스인을 쓰는 클래스는 다음을 가지고 있어야 한다:
    - self.log_message: pyqtSignal(str)
    - self.LOG_PREFIX: str
    - self._viewer_process: QProcess | None (보통 __init__에서 None으로 초기화)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QMessageBox


class Viewer3DMixin:
    def _launch_viewer(
        self, components: dict, title: str, dir_tag: str, default_hidden: set | None = None,
    ) -> None:
        """components(레이어 이름 -> open3d PointCloud)를 PLY로 저장하고
        매니페스트를 만들어 app.core.icp_viewer를 서브프로세스로 띄운다."""
        import open3d as o3d

        view_dir = Path(tempfile.gettempdir()) / f"icp_view_{dir_tag}"
        view_dir.mkdir(parents=True, exist_ok=True)

        default_hidden = default_hidden or set()
        layers = []
        for i, (name, pcd) in enumerate(components.items()):
            filename = f"layer_{i}.ply"
            o3d.io.write_point_cloud(str(view_dir / filename), pcd, write_ascii=False)
            layers.append({"name": name, "file": filename, "visible": name not in default_hidden})

        manifest_path = view_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"layers": layers}, f, ensure_ascii=False, indent=2)

        if getattr(self, "_viewer_process", None) is not None \
                and self._viewer_process.state() != QProcess.ProcessState.NotRunning:
            self._viewer_process.kill()

        self._viewer_process = QProcess(self)
        # 2026-08 추가: 이전엔 완전히 fire-and-forget이라 뷰어 프로세스가
        # 조용히 죽어도(open3d 버전 문제, GPU/디스플레이 문제 등) 사용자가
        # 알 방법이 없었다 - "3D 뷰어가 안 열리는데 이유를 모르겠다" 문의의
        # 흔한 원인. 이제 stdout/stderr를 모아뒀다가 비정상 종료 시 그대로
        # 보여준다.
        self._viewer_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._viewer_process.errorOccurred.connect(self._on_viewer_process_error)
        self._viewer_process.finished.connect(self._on_viewer_process_finished)
        self._viewer_process.start(
            sys.executable,
            ["-m", "app.core.icp_viewer", str(manifest_path), "--title", title],
        )
        self.log_message.emit(f"[{self.LOG_PREFIX}] 3D 뷰어 실행: {manifest_path} ({len(layers)}개 레이어)")

    def _on_viewer_process_error(self, error) -> None:
        QMessageBox.critical(
            self, "3D 뷰어 실행 실패",
            f"뷰어 프로세스를 시작하지 못했습니다: {error}\n\n"
            f"'{sys.executable}' 실행 파일 경로/권한을 확인하세요.",
        )

    def _on_viewer_process_finished(self, exit_code: int, exit_status) -> None:
        if exit_code == 0:
            return  # 정상 종료(사용자가 창을 닫음) - 조용히 넘어감
        output = ""
        if self._viewer_process is not None:
            output = bytes(self._viewer_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        tail = output.strip()[-1500:] if output.strip() else "(출력 없음)"
        self.log_message.emit(f"[{self.LOG_PREFIX}] 3D 뷰어가 비정상 종료됨 (exit code {exit_code})")
        QMessageBox.critical(
            self, "3D 뷰어 오류",
            f"뷰어 프로세스가 오류로 종료됐습니다 (exit code {exit_code}).\n\n{tail}",
        )