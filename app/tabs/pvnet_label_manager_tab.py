"""탭: PVNet 라벨 파일 관리.

data/pvnet_data/labels.json에 누적된 라벨을 미리보기 이미지(pvnet_labels_preview/
*.jpg) + 키포인트 개수와 함께 목록으로 보여주고, 잘못 저장된 항목을 골라서
지울 수 있게 한다.

이 탭이 필요해진 배경: "PVNet 라벨 생성" 탭(pvnet_label_generation_tab.py)은
CAD 파일명 기준 "고정 경로"(data/pvnet_data/keypoints_{cad_stem}.npy)에 키포인트
3D를 저장한다. 같은 CAD로 키포인트 개수 등 설정을 바꿔서 다시 계산하면
그 파일 내용이 그대로 덮어써진다 - 그 사이에 이미 저장해둔 라벨들은
keypoints_2d 개수가 서로 달라질 수 있는데, 지금까지는 이걸 알아챌 방법이
없었다("PVNet 테스트" 탭에서 keypoints_3d 개수 불일치로 처음 발견됨). 이
탭은 라벨별 키포인트 개수를 한눈에 보여주고, 다수(대부분)와 다른 개수인
라벨을 자동으로 골라 선택하는 기능으로 이 문제를 미리 찾아낼 수 있게 한다.

삭제 시 함께 정리하는 것 (라벨 하나 = 인스턴스 하나 기준):
    - labels.json에서 해당 항목 제거
    - 그 항목의 마스크 .npy (data/pvnet_data/labels_masks/)
    - 그 항목의 미리보기 .jpg (data/pvnet_data/labels_preview/)
    - 원본 이미지 복사본(data/pvnet_data/labels_images/)은 "다른 라벨이 더 이상
      참조하지 않을 때만" 같이 지운다 - 같은 프레임에서 검출된 여러
      인스턴스가 같은 이미지 파일을 공유하기 때문.

"고아 파일 정리": labels.json에는 더 이상 없는데 디스크에 남아있는
마스크/미리보기/이미지 파일(과거 크래시나 이 탭 도입 이전의 수동 편집으로
생겼을 수 있는 것)을 찾아서 지운다.

경로 상수는 pvnet_label_generation_tab.py에서 그대로 가져다 쓴다(단일
출처 유지 - 두 파일이 서로 다른 경로를 보게 되는 사고를 방지).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from app.tabs.pvnet_label_generation_tab import (
    DEFAULT_DATA_ROOT, DEFAULT_IMAGE_OUT_DIR, DEFAULT_LABELS_OUT, DEFAULT_MASK_OUT_DIR,
    DEFAULT_PREVIEW_DIR, atomic_write_json,
)

LABEL_LIST_WIDTH = 160
DETAIL_PANE_WIDTH = 230


class PVNetLabelManagerTab(QWidget):
    log_message = pyqtSignal(str)
    LOG_PREFIX = "PVNet 라벨 관리 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list[dict] = []
        self._majority_kpts_count: int | None = None
        self._build_ui()
        self._reload()

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("PVNet 라벨 파일 관리")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("새로고침")
        btn_refresh.setToolTip("디스크의 labels.json을 다시 읽어옵니다.")
        btn_refresh.clicked.connect(self._reload)
        btn_row.addWidget(btn_refresh)

        btn_select_all = QPushButton("전체 선택")
        btn_select_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_row.addWidget(btn_select_all)

        btn_select_none = QPushButton("선택 해제")
        btn_select_none.clicked.connect(lambda: self._set_all_checked(False))
        btn_row.addWidget(btn_select_none)

        btn_select_mismatched = QPushButton("개수 다른 항목만 선택")
        btn_select_mismatched.setToolTip(
            "전체 라벨 중 가장 흔한 키포인트 개수와 다른 라벨만 자동으로 체크합니다.\n"
            "CAD를 바꾸거나 키포인트 설정을 바꾼 뒤 실수로 섞인 라벨을 찾을 때 씁니다."
        )
        btn_select_mismatched.clicked.connect(self._select_mismatched)
        btn_row.addWidget(btn_select_mismatched)
        btn_row.addStretch(1)

        self.btn_delete_selected = QPushButton("선택 항목 삭제")
        self.btn_delete_selected.setStyleSheet("color: #c0392b; font-weight: 600;")
        self.btn_delete_selected.setToolTip("체크된 라벨 + 마스크/미리보기 파일을 삭제합니다 (되돌릴 수 없음).")
        self.btn_delete_selected.clicked.connect(self._on_delete_selected)
        btn_row.addWidget(self.btn_delete_selected)

        layout.addLayout(btn_row)

        layout.addWidget(QLabel("키포인트 3D 파일 (data/pvnet_data/keypoints_*.npy)"))
        self.keypoints_files_label = QLabel("")
        self.keypoints_files_label.setWordWrap(True)
        self.keypoints_files_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.keypoints_files_label)

        btn_orphan = QPushButton("고아 파일 정리 (라벨이 참조하지 않는 마스크/미리보기/이미지 삭제)")
        btn_orphan.clicked.connect(self._on_cleanup_orphans)
        layout.addWidget(btn_orphan)

        # ------------------------------------------------------- 3분할 본문
        # 좌: 라벨 목록 리스트(체크박스로 삭제 선택) - 중: 선택된 라벨의
        # 키포인트/수집 시간 등 정보 - 우: 미리보기 이미지(크게). 목록에서
        # 항목을 클릭(현재 항목 변경)하면 중앙/우측이 그 라벨 내용으로 갱신된다.
        panes = QHBoxLayout()

        self.label_list = QListWidget()
        self.label_list.setFixedWidth(LABEL_LIST_WIDTH)
        self.label_list.setToolTip(
            "클릭하면 오른쪽에 상세 정보/미리보기가 뜹니다.\n"
            "체크박스는 삭제 대상 선택용입니다."
        )
        self.label_list.currentItemChanged.connect(self._on_current_item_changed)
        panes.addWidget(self.label_list)

        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFixedWidth(DETAIL_PANE_WIDTH)
        self.detail_label = QLabel("항목을 선택하세요")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.detail_label.setStyleSheet("padding: 8px;")
        detail_scroll.setWidget(self.detail_label)
        panes.addWidget(detail_scroll)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_label = QLabel("미리보기 없음")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #f2f1ec; color: #999;")
        self.preview_scroll.setWidget(self.preview_label)
        panes.addWidget(self.preview_scroll, stretch=1)

        layout.addLayout(panes, stretch=1)

    # ------------------------------------------------------------- 상세 표시
    def _on_current_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self.detail_label.setText("항목을 선택하세요")
            self.detail_label.setStyleSheet("padding: 8px;")
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("미리보기 없음")
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        self._show_detail(idx, self._labels[idx])

    def _show_detail(self, idx: int, entry: dict) -> None:
        n_kpts = len(entry.get("keypoints_2d") or [])
        mismatch = self._majority_kpts_count is not None and n_kpts != self._majority_kpts_count
        fitness = entry.get("fitness")
        fitness_text = f"{fitness:.3f}" if isinstance(fitness, (int, float)) else "?"
        bbox = entry.get("bbox")
        bbox_text = ", ".join(f"{v:.0f}" for v in bbox) if bbox else "?"

        lines = [
            f"인덱스: #{idx}",
            f"이미지: {Path(entry.get('image', '')).name}",
            f"마스크: {Path(entry.get('mask', '')).name}",
            "",
            f"키포인트: {n_kpts}개" + ("  ⚠ 다수 개수와 다름" if mismatch else ""),
            f"fitness: {fitness_text}",
            f"bbox: [{bbox_text}]",
            f"수집 시각: {self._collected_time_text(entry)}",
        ]
        self.detail_label.setText("\n".join(lines))
        self.detail_label.setStyleSheet("padding: 8px; color: #c0392b;" if mismatch else "padding: 8px;")

        preview_path = self._preview_path_for(entry)
        pix = QPixmap(str(preview_path)) if preview_path and preview_path.is_file() else None
        if pix is not None and not pix.isNull():
            self.preview_label.setPixmap(pix)
            self.preview_label.setText("")
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("미리보기 없음")

    @staticmethod
    def _collected_time_text(entry: dict) -> str:
        """마스크(없으면 이미지) 파일명 앞부분의 YYYYMMDD_HHMMSS 스탬프를
        읽을 수 있는 형식으로 바꾼다 (pvnet_label_generation_tab.py의
        `stamp = datetime.now().strftime("%Y%m%d_%H%M%S")` 관례에 맞춤)."""
        stem = Path(entry.get("mask") or entry.get("image") or "").stem
        m = re.match(r"(\d{8})_(\d{6})", stem)
        if not m:
            return "알 수 없음"
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return "알 수 없음"

    # ------------------------------------------------------------- 경로 헬퍼
    @staticmethod
    def _preview_path_for(entry: dict) -> Path | None:
        mask_path = entry.get("mask")
        if not mask_path:
            return None
        name = Path(mask_path).name.replace(".npy", ".jpg")
        return DEFAULT_PREVIEW_DIR / name

    # ----------------------------------------------------------- 로드/표시
    def _reload(self) -> None:
        if DEFAULT_LABELS_OUT.is_file():
            try:
                with open(DEFAULT_LABELS_OUT, "r", encoding="utf-8") as f:
                    self._labels = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                QMessageBox.critical(self, "로드 실패", f"labels.json을 읽지 못했습니다:\n{exc}")
                self._labels = []
        else:
            self._labels = []

        counts = [len(e.get("keypoints_2d") or []) for e in self._labels]
        hist = Counter(counts)
        self._majority_kpts_count = hist.most_common(1)[0][0] if hist else None

        hist_text = ", ".join(f"{k}개: {v}건" for k, v in sorted(hist.items())) or "없음"
        warn = ""
        if len(hist) > 1:
            warn = "  ⚠ 키포인트 개수가 서로 다른 라벨이 섞여 있습니다 - 이대로 학습하면 문제가 될 수 있습니다."
        self.summary_label.setText(f"총 {len(self._labels)}건  |  키포인트 개수별: {hist_text}{warn}")

        self._refresh_keypoints_files_info()
        self._rebuild_list()

    def _refresh_keypoints_files_info(self) -> None:
        files = sorted(DEFAULT_DATA_ROOT.glob("keypoints_*.npy")) if DEFAULT_DATA_ROOT.is_dir() else []
        if not files:
            self.keypoints_files_label.setText("(없음)")
            return
        lines = []
        for f in files:
            try:
                arr = np.load(f)
                note = ""
                if self._majority_kpts_count is not None and arr.shape[0] != self._majority_kpts_count:
                    note = f"  ⚠ 라벨 다수 개수({self._majority_kpts_count})와 다름"
                lines.append(f"{f.name}: {arr.shape[0]}개{note}")
            except Exception as exc:  # noqa: BLE001 - 손상된 npy 파일이어도 목록 표시는 계속
                lines.append(f"{f.name}: 읽기 실패 ({exc})")
        self.keypoints_files_label.setText("\n".join(lines))

    def _rebuild_list(self) -> None:
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for idx, entry in enumerate(self._labels):
            n_kpts = len(entry.get("keypoints_2d") or [])
            mismatch = self._majority_kpts_count is not None and n_kpts != self._majority_kpts_count
            image_stem = Path(entry.get("image", "")).stem
            item = QListWidgetItem(f"#{idx} {image_stem}" + (" ⚠" if mismatch else ""))
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            if mismatch:
                item.setForeground(QColor("#c0392b"))
            self.label_list.addItem(item)
        self.label_list.blockSignals(False)

        self.detail_label.setText("항목을 선택하세요")
        self.detail_label.setStyleSheet("padding: 8px;")
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("미리보기 없음")

    # ------------------------------------------------------------- 선택
    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.label_list.count()):
            self.label_list.item(i).setCheckState(state)

    def _select_mismatched(self) -> None:
        if self._majority_kpts_count is None:
            return
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            n = len(self._labels[idx].get("keypoints_2d") or [])
            item.setCheckState(Qt.CheckState.Checked if n != self._majority_kpts_count else Qt.CheckState.Unchecked)

    # ------------------------------------------------------------- 삭제
    def _on_delete_selected(self) -> None:
        """버튼 클릭 슬롯 - 예외 처리 원칙은 다른 PVNet 탭들과 동일
        (PyQt 슬롯 안 예외는 조용히 죽으므로 항상 팝업+로그로 드러낸다)."""
        try:
            self._on_delete_selected_impl()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"[{self.LOG_PREFIX}] 삭제 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "삭제 실패",
                f"예상치 못한 오류로 삭제에 실패했습니다:\n\n{exc}\n\n터미널의 상세 트레이스백을 확인하세요.",
            )
            raise

    def _on_delete_selected_impl(self) -> None:
        selected_idx = [
            self.label_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.label_list.count())
            if self.label_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        if not selected_idx:
            QMessageBox.information(self, "알림", "삭제할 항목을 먼저 선택하세요.")
            return

        reply = QMessageBox.question(
            self, "삭제 확인",
            f"{len(selected_idx)}건을 삭제합니다 (라벨 + 마스크 + 미리보기 파일).\n"
            "원본 이미지는 다른 라벨이 더 이상 참조하지 않을 때만 같이 지웁니다.\n\n"
            "되돌릴 수 없습니다. 계속할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        selected_set = set(selected_idx)
        to_delete = [self._labels[i] for i in selected_idx]
        remaining = [e for i, e in enumerate(self._labels) if i not in selected_set]
        remaining_images = {e.get("image") for e in remaining if e.get("image")}

        n_files_deleted = 0
        for entry in to_delete:
            mask_path = entry.get("mask")
            if mask_path and os.path.isfile(mask_path):
                os.remove(mask_path)
                n_files_deleted += 1

            preview_path = self._preview_path_for(entry)
            if preview_path is not None and preview_path.is_file():
                preview_path.unlink()
                n_files_deleted += 1

            image_path = entry.get("image")
            if image_path and image_path not in remaining_images and os.path.isfile(image_path):
                os.remove(image_path)
                n_files_deleted += 1

        self._labels = remaining
        atomic_write_json(DEFAULT_LABELS_OUT, self._labels)

        self.log_message.emit(
            f"[{self.LOG_PREFIX}] {len(selected_idx)}건 삭제 (연관 파일 {n_files_deleted}개 포함), "
            f"남은 라벨 {len(self._labels)}건"
        )
        self._reload()

    # --------------------------------------------------------- 고아 파일 정리
    def _on_cleanup_orphans(self) -> None:
        try:
            self._on_cleanup_orphans_impl()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"[{self.LOG_PREFIX}] 고아 파일 정리 실패 (예외): {exc!r}")
            QMessageBox.critical(
                self, "정리 실패",
                f"예상치 못한 오류로 정리에 실패했습니다:\n\n{exc}\n\n터미널의 상세 트레이스백을 확인하세요.",
            )
            raise

    def _on_cleanup_orphans_impl(self) -> None:
        """labels.json에 더 이상 없는데 디스크에 남아있는 마스크/
        미리보기/이미지 파일을 찾아서 지운다. 경로 비교는 절대경로 기준 -
        라벨에 상대경로/절대경로가 섞여 저장돼 있어도 안전하게 비교하기 위함."""
        referenced_masks = {os.path.abspath(e["mask"]) for e in self._labels if e.get("mask")}
        referenced_images = {os.path.abspath(e["image"]) for e in self._labels if e.get("image")}
        referenced_previews = set()
        for e in self._labels:
            p = self._preview_path_for(e)
            if p is not None:
                referenced_previews.add(os.path.abspath(str(p)))

        orphan_masks = [
            p for p in DEFAULT_MASK_OUT_DIR.glob("*.npy") if os.path.abspath(str(p)) not in referenced_masks
        ] if DEFAULT_MASK_OUT_DIR.is_dir() else []
        orphan_previews = [
            p for p in DEFAULT_PREVIEW_DIR.glob("*.jpg") if os.path.abspath(str(p)) not in referenced_previews
        ] if DEFAULT_PREVIEW_DIR.is_dir() else []
        orphan_images = [
            p for p in DEFAULT_IMAGE_OUT_DIR.glob("*")
            if p.is_file() and os.path.abspath(str(p)) not in referenced_images
        ] if DEFAULT_IMAGE_OUT_DIR.is_dir() else []

        total = len(orphan_masks) + len(orphan_previews) + len(orphan_images)
        if total == 0:
            QMessageBox.information(self, "알림", "고아 파일이 없습니다.")
            return

        reply = QMessageBox.question(
            self, "고아 파일 정리",
            f"라벨에서 참조하지 않는 파일 {total}개를 지웁니다\n"
            f"(마스크 {len(orphan_masks)}건, 미리보기 {len(orphan_previews)}건, 이미지 {len(orphan_images)}건).\n\n"
            "되돌릴 수 없습니다. 계속할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for p in (*orphan_masks, *orphan_previews, *orphan_images):
            p.unlink()

        self.log_message.emit(f"[{self.LOG_PREFIX}] 고아 파일 {total}개 정리 완료")
        self._reload()