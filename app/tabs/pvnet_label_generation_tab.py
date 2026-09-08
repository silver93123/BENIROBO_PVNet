"""탭 10: PVNet 라벨 생성 (실시간 촬영 기반).

live_label_generation_tab.py(탭6, RotHead용)와 완전히 동일한 설계 원칙을
따른다: "5. ICP 정합테스트(TCP)"(LiveCaptureICPTab)를 그대로 상속해서
[촬영] -> [2D 검출 실행] -> [ICP 정합 실행]까지 100% 동일한 화면(오버레이
이미지 뷰어 + 인스턴스별 결과 카드)을 그대로 쓰고, 마지막에 "라벨 생성
(미리보기)" -> "라벨로 저장" 두 단계를 추가한다.

    [촬영]              -- LiveCaptureICPTab 그대로 (카메라 타입/평균화)
      -> [2D 검출 실행]  -- ICPWorkbenchTab 그대로 (좌측 패널: 체크포인트/
                            config/CAD 경로 전부 여기 있음)
      -> [ICP 정합 실행] -- ICPWorkbenchTab 그대로 (우측 결과 카드, fitness)
      -> [라벨 생성 (미리보기)] -- 이 파일에서 추가. fitness가 기준 이상인
                            인스턴스만, ICP가 낸 전체 pose(R,t)로 CAD
                            키포인트를 2D에 투영해서 keypoints_2d를 계산하고
                            image_viewer의 "라벨 미리보기" 오버레이로 화면에
                            바로 보여준다 - 디스크에는 아직 아무 것도 안 씀.
      -> [라벨로 저장]   -- 위 "생성" 결과(self._pending_label_data)를 그대로
                            디스크에 씀(마스크 .npy, 이미지 복사, 미리보기
                            jpg, labels.json). "생성" 없이는 눌러도 막힌다.

2026-09 추가: 예전엔 [라벨로 저장] 버튼 하나가 계산+저장을 한 번에 다
했는데, "라벨이 잘못 나왔는데 이미 디스크에 저장돼버렸다"는 상황을 막기
위해 계산(생성/미리보기)과 저장(파일 쓰기)을 분리했다. 생성 결과는 새
프레임을 찍거나 ICP를 다시 돌리면 자동으로 무효화된다(_invalidate_pending
_labels) - "저장" 버튼은 그 프레임의 "생성"을 다시 실행해야만 다시
눌러진다. fitness 임계값(spin_label_fitness_min)만 생성 이후에 바꾸는 건
허용한다 - "저장" 시점에 그 값으로 다시 필터링하므로 재생성이 필요 없다
(keypoints_2d 계산 자체는 임계값과 무관하기 때문).

배치/subprocess 방식(generate_pvnet_labels.py, "기존 데이터셋 폴더를
지정해서 오프라인으로 라벨링")에서 이 방식(실시간 촬영 기반)으로 전환한
이유: 검출/ICP가 왜 실패하는지 화면에서 바로 보면서 디버깅할 수 있고,
탭6/탭8과 화면 조작 방식이 통일된다. generate_pvnet_labels.py 자체는
지우지 않았다 - 이미 수집해둔 세션 폴더를 일괄 처리하고 싶을 때는 여전히
유효하다 (스크립트를 직접 실행하면 됨).

탭8(ManualLabelingTab)과의 차이: 탭8은 사람이 Rx/Ry/Rz를 직접 입력하고
포인트클라우드 centroid를 위치로 쓰는 반면(ICP를 아예 안 씀), 이 탭은
ICP가 낸 전체 pose(R,t)를 그대로 쓴다 - CAD 로컬 좌표계 원점 기준으로 이미
정합된 값이라, 키포인트 투영 시 별도의 "CAD 중심으로 옮기기" 보정이
필요 없다(탭8은 회전만 있고 이동은 별도 측정값이라 이 보정이 필요했음).

좌측 패널 구성(경로/설정 관련은 전부 여기): 카메라 타입/평균화/촬영 이력
(LiveCaptureICPTab 상속) -> 체크포인트/config/CAD 경로(ICPWorkbenchTab
상속) -> PVNet 키포인트 설정(개수, 3D 저장 경로) -> 라벨 채택 기준(fitness)
-> 라벨 생성 (미리보기) 버튼 -> 라벨로 저장 버튼.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QWidget,
)

from app.core import settings_manager
from app.core.camera_intrinsics import estimate_intrinsics_from_organized_pcd, project_points
from app.core.paths import PROJECT_ROOT
from app.tabs.live_capture_icp_tab import LiveCaptureICPTab
from src.detection.pvnet import farthest_point_sampling

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "pvnet_data"
DEFAULT_MASK_OUT_DIR = DEFAULT_DATA_ROOT / "labels_masks"
DEFAULT_IMAGE_OUT_DIR = DEFAULT_DATA_ROOT / "labels_images"
DEFAULT_LABELS_OUT = DEFAULT_DATA_ROOT / "labels.json"
DEFAULT_PREVIEW_DIR = DEFAULT_DATA_ROOT / "labels_preview"
DEFAULT_KEYPOINTS_OUT_TEMPLATE = str(DEFAULT_DATA_ROOT / "keypoints_{cad_stem}.npy")
DEFAULT_LABEL_FITNESS_MIN = 0.85  # ICP 파라미터 박스의 fitness threshold(보통 0.6~0.7)보다
                                   # 엄격하게 - "정합 성공"과 "학습 라벨로 쓸 만큼 확실함"은 다른 기준.
DEFAULT_NUM_KEYPOINTS = 8
DEFAULT_PREVIEW_MAX_WIDTH = 480


def atomic_write_json(path: Path, data) -> None:
    """labels.json을 안전하게 덮어쓴다.

    기존엔 open(path, "w") + json.dump()로 파일을 직접 덮어썼는데, 쓰는
    도중(수백 건 누적된 상태에서) 프로세스가 죽거나 디스크가 꽉 차면 파일이
    반쯤 써진 채로 남아 - 마스크(.npy)는 인스턴스별 개별 파일이라 하나가
    깨져도 그 하나만 잃지만, 이 json은 전체 라벨이 한 파일에 묶여있어서
    한 번의 쓰기 실패가 지금까지 쌓아온 라벨 전체를 못 읽게 만들 수 있다.

    같은 디렉터리에 임시 파일로 먼저 다 쓴 뒤 os.replace()로 원자적으로
    이름을 바꿔치기한다 - 중간에 실패해도 원본은 항상 직전의 정상 상태를
    유지한다 (같은 디렉터리를 써야 os.replace가 원자적임이 보장됨 -
    파일시스템이 다르면 rename이 copy+delete로 풀려 이 보장이 깨짐).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)  # 같은 파일시스템 내 rename은 원자적
    except BaseException:
        # 쓰다가 실패했으면 임시 파일만 지우고 원본은 그대로 둔다.
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


class PVNetLabelGenerationTab(LiveCaptureICPTab):
    LOG_PREFIX = "PVNet 라벨 생성 탭"

    def __init__(self, parent=None):
        self._saved_labels: list[dict] = []
        self._n_saved_session = 0
        self._keypoints_3d: np.ndarray | None = None
        self._keypoints_3d_cad_path: str | None = None  # 이 경로가 바뀌면 캐시 무효화
        self._pending_label_data: dict | None = None  # "라벨 생성" 결과 캐시 (저장 전 미리보기)
        super().__init__(parent)
        self._load_existing_labels()

    # ----------------------------------------------------- UI 확장 (좌측 패널)
    def _build_acquisition_panel(self) -> QWidget:
        """LiveCaptureICPTab의 촬영 패널을 그대로 받아서, 그 밑에 PVNet 라벨
        저장 관련 컨트롤(경로/설정)을 이어붙인다.

        체크포인트/config/CAD 경로는 ICPWorkbenchTab이 이미 좌측 패널에
        만들어두므로(이 탭도 그걸 상속받음), 여기서는 "PVNet 라벨에만
        필요한" 나머지 경로/설정만 추가한다 - 전부 좌측 패널에 몰아둔다는
        원칙을 그대로 지킨다.
        """
        panel = super()._build_acquisition_panel()
        layout = panel.layout()

        layout.addWidget(QLabel("PVNet 키포인트 설정"))

        kpts_num_row = QHBoxLayout()
        kpts_num_row.addWidget(QLabel("키포인트 개수(센트로이드 제외)"))
        self.num_keypoints_spin = QSpinBox()
        self.num_keypoints_spin.setRange(4, 32)
        self.num_keypoints_spin.setValue(DEFAULT_NUM_KEYPOINTS)
        self.num_keypoints_spin.setToolTip(
            "PVNetHead.num_keypoints와 반드시 일치해야 합니다.\n"
            "CAD를 바꾸거나 이 값을 바꾸면 키포인트가 다시 계산됩니다."
        )
        kpts_num_row.addWidget(self.num_keypoints_spin)
        layout.addLayout(kpts_num_row)

        kpts_out_row = QHBoxLayout()
        kpts_out_row.addWidget(QLabel("키포인트 3D 저장경로"))
        self.keypoints_out_edit = QLineEdit()
        self.keypoints_out_edit.setPlaceholderText("비우면 CAD 파일명 기준 자동 결정")
        kpts_out_row.addWidget(self.keypoints_out_edit, stretch=1)
        btn_browse_kpts_out = QPushButton("선택")
        btn_browse_kpts_out.clicked.connect(self._on_browse_keypoints_out)
        kpts_out_row.addWidget(btn_browse_kpts_out)
        layout.addLayout(kpts_out_row)

        layout.addWidget(QLabel("라벨 채택 기준"))
        fitness_row = QHBoxLayout()
        fitness_row.addWidget(QLabel("fitness ≥"))
        self.spin_label_fitness_min = QDoubleSpinBox()
        self.spin_label_fitness_min.setRange(0.0, 1.0)
        self.spin_label_fitness_min.setSingleStep(0.01)
        self.spin_label_fitness_min.setValue(DEFAULT_LABEL_FITNESS_MIN)
        self.spin_label_fitness_min.setToolTip(
            "위 'ICP 파라미터' 박스의 fitness threshold(정합 성공/실패 판정)와는\n"
            "별개입니다. 여기 지정한 값 이상인 인스턴스만 학습 라벨로 저장됩니다."
        )
        fitness_row.addWidget(self.spin_label_fitness_min)
        layout.addLayout(fitness_row)

        self.btn_generate_labels = QPushButton("라벨 생성 (미리보기)")
        self.btn_generate_labels.setToolTip(
            "현재 프레임의 ICP 결과로 keypoints_2d를 계산해서 화면에 '라벨 미리보기'\n"
            "오버레이로 보여줍니다. 아직 디스크에는 아무 것도 저장하지 않습니다 -\n"
            "먼저 2D 검출 실행 -> ICP 정합 실행을 마쳐야 활성화됩니다."
        )
        self.btn_generate_labels.clicked.connect(self._on_generate_labels)
        self.btn_generate_labels.setEnabled(False)
        layout.addWidget(self.btn_generate_labels)

        self.generate_status_label = QLabel("아직 생성 안 됨")
        self.generate_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.generate_status_label.setWordWrap(True)
        layout.addWidget(self.generate_status_label)

        self.btn_save_labels = QPushButton("라벨로 저장 (현재 프레임)")
        self.btn_save_labels.setToolTip("먼저 '라벨 생성 (미리보기)'로 결과를 확인한 뒤 저장하세요.")
        self.btn_save_labels.clicked.connect(self._on_save_labels)
        self.btn_save_labels.setEnabled(False)
        layout.addWidget(self.btn_save_labels)

        self.label_count_label = QLabel(f"누적 저장: {self._n_saved_session}건")
        layout.addWidget(self.label_count_label)
        self.keypoints_status_label = QLabel("키포인트: 아직 계산 안 됨")
        self.keypoints_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.keypoints_status_label.setWordWrap(True)
        layout.addWidget(self.keypoints_status_label)

        return panel

    def _on_browse_keypoints_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "키포인트 3D 저장 위치", self.keypoints_out_edit.text(), "NumPy (*.npy)"
        )
        if path:
            self.keypoints_out_edit.setText(path)
            self._keypoints_3d_cad_path = None  # 경로 바뀌면 다음 저장 때 재계산/재저장

    # ----------------------------------------------------- 프레임/ICP 훅
    def _on_new_frame_acquired(self, frame_label: str) -> None:
        super()._on_new_frame_acquired(frame_label)
        self.btn_generate_labels.setEnabled(False)  # 새 프레임이면 검출/ICP 다시 해야 함
        self._invalidate_pending_labels()

    def _on_run_icp(self) -> None:
        super()._on_run_icp()
        self.btn_generate_labels.setEnabled(
            bool(self._last_icp_results) and any(r.ok for r in self._last_icp_results)
        )
        self._invalidate_pending_labels()

    def _invalidate_pending_labels(self) -> None:
        """새 프레임을 찍거나 ICP를 다시 돌리면 지난 "라벨 생성" 결과는 더
        이상 지금 화면의 검출/ICP 결과와 안 맞으므로 무효화한다. "저장"
        버튼은 이 프레임에 대해 "생성"을 다시 실행해야만 눌러진다."""
        self._pending_label_data = None
        self.btn_save_labels.setEnabled(False)
        self.generate_status_label.setText("아직 생성 안 됨")
        self.image_viewer.clear_label_preview_overlays()

    # ----------------------------------------------------- 키포인트 3D (CAD 1회 계산)
    def _keypoints_out_path(self, cad_path: str) -> Path:
        custom = self.keypoints_out_edit.text().strip()
        if custom:
            return Path(custom)
        cad_stem = Path(cad_path).stem
        return Path(DEFAULT_KEYPOINTS_OUT_TEMPLATE.format(cad_stem=cad_stem))

    def _ensure_keypoints_3d(self, cad_path: str) -> bool:
        """self._cad_pcd(ICPWorkbenchTab이 이미 로드해둔 것)에서 FPS로
        키포인트를 뽑아 캐시한다. CAD 경로/키포인트 개수가 바뀌지 않았으면
        재계산하지 않는다(탭8의 _keypoints_3d_cad_path 캐시 무효화 패턴과 동일).

        Returns:
            성공하면 True, CAD가 아직 안 로드됐으면 False.
        """
        cache_key = f"{cad_path}::{self.num_keypoints_spin.value()}"
        if self._keypoints_3d is not None and self._keypoints_3d_cad_path == cache_key:
            return True
        if self._cad_pcd is None:
            return False

        cad_points = np.asarray(self._cad_pcd.points)
        self._keypoints_3d = farthest_point_sampling(
            cad_points, num_keypoints=self.num_keypoints_spin.value(), include_centroid=True
        )
        self._keypoints_3d_cad_path = cache_key

        out_path = self._keypoints_out_path(cad_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self._keypoints_3d)
        self.keypoints_status_label.setText(
            f"키포인트 {self._keypoints_3d.shape[0]}개(센트로이드 포함) 저장됨: {out_path}"
        )
        self.log_message.emit(f"[{self.LOG_PREFIX}] 키포인트 재계산/저장: {out_path}")
        return True

    # ----------------------------------------------------- 라벨 생성 (미리보기)
    def _on_generate_labels(self) -> None:
        """버튼 클릭 슬롯 - 실제 로직은 _on_generate_labels_impl()에 있다.
        예외 처리 원칙은 _on_save_labels()와 동일 (조용히 죽지 않게)."""
        try:
            self._on_generate_labels_impl()
        except Exception as exc:  # noqa: BLE001 - 사용자에게 반드시 보여줘야 함
            self.log_message.emit(f"[{self.LOG_PREFIX}] 라벨 생성 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "라벨 생성 실패",
                f"예상치 못한 오류로 생성에 실패했습니다:\n\n{exc}\n\n"
                "터미널의 상세 트레이스백을 확인하세요.",
            )
            raise

    def _on_generate_labels_impl(self) -> None:
        """디스크에 아무 것도 쓰지 않고, 현재 프레임의 ICP 결과로 keypoints_2d만
        계산해서 image_viewer의 "라벨 미리보기" 오버레이 + 텍스트 요약으로
        보여준다. 여기서 통과(fitness ≥ 임계값)한 인스턴스 목록을
        self._pending_label_data에 캐시해두고, [라벨로 저장]이 이걸 그대로
        디스크에 쓴다.

        fitness 판정과 keypoints_2d 계산을 분리한 이유: keypoints_2d 자체는
        임계값과 무관하므로, 여기서는 ICP가 성공한 모든 인스턴스에 대해
        미리 계산해서 캐시해두고 "통과 여부"만 표시한다. 저장 시점에
        임계값 스핀박스를 다시 읽어서 필터링하므로, 생성 이후 임계값만
        바꾼 경우엔 재생성 없이 바로 저장할 수 있다.
        """
        if not self._last_icp_results:
            QMessageBox.warning(self, "알림", "먼저 2D 검출과 ICP 정합을 실행하세요.")
            return

        cad_path = settings_manager.load_settings().get("cad_path", "")
        if not cad_path:
            QMessageBox.warning(
                self, "알림",
                "CAD 모델이 지정되지 않았습니다. 'CAD 모델 설정' 탭에서 지정하고 저장하세요.",
            )
            return

        if not self._ensure_keypoints_3d(cad_path):
            QMessageBox.warning(self, "알림", "CAD가 아직 로드되지 않았습니다. 먼저 ICP 정합을 한 번 실행하세요.")
            return

        if self._pcd_organized is None or self._valid_mask is None:
            QMessageBox.warning(self, "알림", "포인트클라우드가 없습니다. 다시 촬영하세요.")
            return
        try:
            fx, fy, cx, cy = estimate_intrinsics_from_organized_pcd(self._pcd_organized, self._valid_mask)
        except ValueError:
            QMessageBox.warning(self, "알림", "카메라 intrinsic을 추정하지 못했습니다 (유효 픽셀 부족).")
            return
        intrinsics = (fx, fy, cx, cy)

        self.image_viewer.clear_label_preview_overlays()

        threshold = self.spin_label_fitness_min.value()
        items: list[dict] = []
        detail_lines: list[str] = []
        n_pass = 0

        for i, (det, result) in enumerate(zip(self._last_detections, self._last_icp_results)):
            if not result.ok or result.fitness is None:
                detail_lines.append(f"  obj{i}: ICP 실패 - 제외")
                continue

            # ICP가 낸 전체 pose(R,t)는 이미 CAD 로컬 좌표계 -> 씬(카메라) 좌표계로
            # 정합된 값이므로, 키포인트(같은 CAD 로컬 좌표계에서 뽑음)에 그대로
            # 적용하면 된다 - 탭8(수동 각도)과 달리 별도 중심 보정이 필요 없다.
            R, t_m = result.T[:3, :3], result.T[:3, 3]
            kpts_cam_mm = (R @ self._keypoints_3d.T).T * 1000.0 + t_m * 1000.0
            keypoints_2d = project_points(kpts_cam_mm, intrinsics)

            passed = result.fitness >= threshold
            items.append({"i": i, "keypoints_2d": keypoints_2d, "fitness": float(result.fitness)})
            if passed:
                n_pass += 1
                self.image_viewer.set_label_preview_overlay(i, det.mask.astype(bool), keypoints_2d)
                detail_lines.append(f"  obj{i}: fitness {result.fitness:.3f} - 채택")
            else:
                detail_lines.append(f"  obj{i}: fitness {result.fitness:.3f} < {threshold:.2f} - 제외")

        self._pending_label_data = {"cad_path": cad_path, "intrinsics": intrinsics, "items": items}

        n_total = len(items)
        summary = f"라벨 생성됨: {n_pass}/{n_total}개 채택 (fitness ≥ {threshold:.2f})\n" + "\n".join(detail_lines)
        self.generate_status_label.setText(summary)
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 라벨 생성 (미리보기): 채택 {n_pass}건 / 전체 {n_total}건"
        )

        self.btn_save_labels.setEnabled(n_pass > 0)
        if n_pass == 0:
            QMessageBox.information(
                self, "채택된 라벨 없음",
                f"fitness ≥ {threshold:.2f}인 인스턴스가 없습니다.\n"
                "위 요약이나 우측 결과 카드에서 각 인스턴스의 fitness 값을 확인하거나,\n"
                "'라벨 채택 기준'을 낮춰서 다시 생성해보세요.",
            )

    # ----------------------------------------------------- 라벨 저장
    def _on_save_labels(self) -> None:
        """버튼 클릭 슬롯 - 실제 로직은 _on_save_labels_impl()에 있다.

        PyQt 슬롯 안에서 예외가 나면 트레이스백이 stderr로만 나가고 화면에는
        아무 반응도 없이 조용히 끝난다(cad_combo AttributeError로 실제 겪었던
        문제). 그래서 여기서 감싸 항상 팝업 + 로그로 실패를 눈에 보이게
        만든다 - icp_workbench_base.py의 KeyError 침묵 크래시 수정과 동일한
        원칙.
        """
        try:
            self._on_save_labels_impl()
        except Exception as exc:  # noqa: BLE001 - 사용자에게 반드시 보여줘야 함
            self.log_message.emit(f"[{self.LOG_PREFIX}] 라벨 저장 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "라벨 저장 실패",
                f"예상치 못한 오류로 저장에 실패했습니다:\n\n{exc}\n\n"
                "터미널의 상세 트레이스백을 확인하세요.",
            )
            raise  # 터미널에도 전체 트레이스백이 남도록 다시 던짐

    def _on_save_labels_impl(self) -> None:
        """self._pending_label_data("라벨 생성"의 결과)를 그대로 디스크에
        쓴다 - keypoints_2d를 여기서 다시 계산하지 않는다(생성 시점 값을
        그대로 신뢰). fitness 임계값만 지금 스핀박스 값으로 다시 확인해서
        필터링한다 - 생성 이후 임계값을 바꿨을 수 있기 때문."""
        if not self._pending_label_data:
            QMessageBox.warning(
                self, "알림",
                "먼저 '라벨 생성 (미리보기)'로 결과를 확인하세요.\n"
                "확인 없이 바로 저장하는 걸 막기 위한 절차입니다.",
            )
            return

        cad_path = self._pending_label_data["cad_path"]
        fx, fy, cx, cy = self._pending_label_data["intrinsics"]

        DEFAULT_MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

        # LiveCaptureICPTab의 촬영본(self._current_image_path)은 임시 경로라
        # data/ 아래 영구 위치로 복사해둔다 (live_label_generation_tab.py와
        # 동일 이유 - 학습 라벨은 몇 주 뒤에도 유효해야 함).
        if not self._current_image_path or not os.path.isfile(self._current_image_path):
            QMessageBox.warning(
                self, "알림",
                f"촬영 이미지 파일을 찾을 수 없습니다: {self._current_image_path}\n"
                "다시 촬영 후 검출/ICP/생성을 새로 실행하세요.",
            )
            return

        threshold = self.spin_label_fitness_min.value()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_tag = self._current_frame or "frame"

        image_ext = os.path.splitext(self._current_image_path)[1] or ".png"
        permanent_image_path = DEFAULT_IMAGE_OUT_DIR / f"{stamp}_{frame_tag}{image_ext}"
        shutil.copy2(self._current_image_path, permanent_image_path)

        gray = cv2.imread(str(permanent_image_path), cv2.IMREAD_GRAYSCALE)
        frame_bgr = np.stack([gray, gray, gray], axis=-1) if gray is not None else None

        n_saved_this_frame = 0
        n_skipped = 0

        for item in self._pending_label_data["items"]:
            fitness = item["fitness"]
            if fitness < threshold:
                n_skipped += 1
                continue

            i = item["i"]
            keypoints_2d = item["keypoints_2d"]
            det = self._last_detections[i]
            result = self._last_icp_results[i]

            mask_filename = f"{stamp}_{frame_tag}_obj{i}.npy"
            mask_path = DEFAULT_MASK_OUT_DIR / mask_filename
            np.save(mask_path, det.mask.astype(bool))

            self._saved_labels.append({
                "image": str(permanent_image_path),
                "mask": str(mask_path),
                "bbox": [float(v) for v in det.bbox],
                "pose": result.T.tolist(),
                "keypoints_2d": keypoints_2d.tolist(),
                "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "fitness": float(fitness),
            })
            n_saved_this_frame += 1
            self._n_saved_session += 1

            if frame_bgr is not None:
                preview_path = DEFAULT_PREVIEW_DIR / mask_filename.replace(".npy", ".jpg")
                self._save_preview_image(
                    frame_bgr, det.mask, det.bbox, fitness, keypoints_2d, preview_path
                )

        if n_saved_this_frame == 0:
            QMessageBox.information(
                self, "저장할 라벨 없음",
                f"fitness ≥ {threshold:.2f}인 채택 인스턴스가 없습니다.\n"
                "'라벨 생성 (미리보기)' 요약이나 우측 결과 카드에서 fitness 값을 확인하세요.",
            )
            return

        atomic_write_json(DEFAULT_LABELS_OUT, self._saved_labels)

        self.label_count_label.setText(f"누적 저장: {self._n_saved_session}건")
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 라벨 {n_saved_this_frame}건 저장 "
            f"(건너뜀 {n_skipped}건, 누적 {self._n_saved_session}건) -> {DEFAULT_LABELS_OUT}"
        )

        # 저장 완료 - 이 프레임을 또 저장하려면 "생성"부터 다시 눌러야 한다
        # (중복 저장 방지 - 같은 인스턴스를 실수로 두 번 저장하는 걸 막음).
        self._pending_label_data = None
        self.btn_save_labels.setEnabled(False)
        self.generate_status_label.setText(
            f"저장 완료 ({n_saved_this_frame}건). 다시 저장하려면 '라벨 생성'을 다시 누르세요."
        )

    def _load_existing_labels(self) -> None:
        """탭을 다시 켰을 때 기존에 저장해둔 라벨 파일이 있으면 이어서 누적한다."""
        if not DEFAULT_LABELS_OUT.is_file():
            return
        try:
            with open(DEFAULT_LABELS_OUT, "r", encoding="utf-8") as f:
                self._saved_labels = json.load(f)
            self._n_saved_session = len(self._saved_labels)
            self.label_count_label.setText(f"누적 저장: {self._n_saved_session}건 (기존 파일에 이어서 씀)")
        except (json.JSONDecodeError, OSError) as exc:
            self.log_message.emit(f"[{self.LOG_PREFIX}] 기존 라벨 파일 로드 실패 (무시하고 새로 시작): {exc}")

    # ----------------------------------------------------- 미리보기 이미지
    @staticmethod
    def _save_preview_image(
        image_bgr, mask, bbox, fitness, keypoints_2d, out_path, max_width=DEFAULT_PREVIEW_MAX_WIDTH
    ):
        """live_label_generation_tab.py의 _save_preview_image()에 투영된
        키포인트 표시를 추가한 버전 (generate_pvnet_labels.py의 _save_preview와
        동일한 스타일: 센트로이드=노랑 큰 원, 표면점=하늘색 번호)."""
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

        for idx, (u, v) in enumerate(keypoints_2d):
            u_i, v_i = int(round(u)), int(round(v))
            if idx == 0:
                cv2.circle(overlay, (u_i, v_i), 6, (0, 255, 255), -1)
            else:
                cv2.circle(overlay, (u_i, v_i), 4, (255, 200, 0), -1)

        h, w = overlay.shape[:2]
        if w > max_width:
            scale = max_width / w
            overlay = cv2.resize(overlay, (max_width, int(h * scale)))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)