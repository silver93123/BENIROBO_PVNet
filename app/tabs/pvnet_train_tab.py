"""탭 3: PVNet 학습.

scripts/train_pvnet.py를 QProcess로 그대로 실행하는 GUI 래퍼다. 학습 코드
자체(모델/loss/데이터로딩)는 전혀 재구현하지 않는다 - 이 탭은 오직
    1) CLI 인자를 폼으로 채워서 명령줄을 조립하고
    2) 자식 프로세스로 스크립트를 띄운 뒤
    3) stdout을 실시간으로 읽어 로그 창에 스트리밍하고, "epoch N/M" 패턴을
       정규식으로 파싱해 진행률 바를 갱신하는 것
만 한다. icp_workbench_base.py가 3D 뷰어를 QProcess로 띄우는 것과 동일한
패턴이다 - 다만 여기서는 fire-and-forget이 아니라 출력을 계속 읽어야 해서
readyReadStandardOutput을 붙였다.

학습은 보통 수 시간 걸리는 작업이라 QProcess(비동기, 별도 OS 프로세스)를
썼다 - QThread로 같은 인터프리터 안에서 돌리면 GIL 경쟤 + torch DataLoader
worker 프로세스 fork 문제가 얽혀서 복잡해진다. 프로세스 하나 띄우고 stdout만
파이프로 읽는 게 훨씬 단순하고, train_pvnet.py를 터미널에서 직접 돌릴 때와
100% 동일한 코드 경로를 타므로 "GUI에서는 되는데 터미널에서는 안 된다"류의
불일치가 생길 여지도 없다.

주의: train_pvnet.py가 학습 도중 GUI와 같은 GPU를 점유한다. RTMDet-Ins 검출/
ICP를 동시에 돌리는 다른 탭과 병행 사용하면 VRAM이 모자랄 수 있다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from app.core.paths import PROJECT_ROOT

DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "pvnet_labels.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "checkpoints" / "pvnet_run"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_pvnet.py"

# "[pvnet-train] epoch 3/100 train: ..." 형태 라인에서 진행률을 뽑는다.
# train_pvnet.py의 print 포맷과 반드시 맞아야 한다 - 그쪽 포맷을 바꾸면 여기도
# 같이 고쳐야 함.
_EPOCH_RE = re.compile(r"\[pvnet-train\]\s+epoch\s+(\d+)/(\d+)\s")
_BEST_RE = re.compile(r"best 갱신 \(loss=([\d.]+)\)")


class PVNetTrainTab(QWidget):
    #: 다른 탭과 동일한 관례 - main_window.py가 이 시그널을 공용 로그
    #: 콘솔에 연결한다. 학습 전체 stdout은 이 탭 안의 log_view로 따로
    #: 스트리밍하고, 여기로는 시작/종료 같은 굵직한 이벤트만 보낸다.
    log_message = pyqtSignal(str)
    LOG_PREFIX = "PVNet 학습 탭"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._total_epochs: int = 0
        self._best_loss: float | None = None
        self._build_ui()

    # ----------------------------------------------------------- UI 조립
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ------------------------------------------------------- 좌측: 설정 폼
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        form_widget = QWidget()
        scroll.setWidget(form_widget)
        left = QVBoxLayout(form_widget)

        left.addWidget(self._build_paths_group())
        left.addWidget(self._build_crop_group())
        left.addWidget(self._build_optim_group())
        left.addWidget(self._build_misc_group())

        run_row = QHBoxLayout()
        self.btn_start = QPushButton("학습 시작")
        self.btn_start.clicked.connect(self._on_start_clicked)
        run_row.addWidget(self.btn_start)
        self.btn_stop = QPushButton("중지")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        run_row.addWidget(self.btn_stop)
        left.addLayout(run_row)

        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("color: #666;")
        left.addWidget(self.status_label)
        left.addStretch(1)

        root.addWidget(scroll)

        # ------------------------------------------------------- 우측: 진행률 + 로그
        right = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("epoch %v/%m")
        right.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; background-color: #101014; color: #d8d8d8;"
        )
        right.addWidget(self.log_view, stretch=1)

        btn_clear_log = QPushButton("로그 지우기")
        btn_clear_log.clicked.connect(self.log_view.clear)
        right.addWidget(btn_clear_log)

        root.addLayout(right, stretch=1)

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("경로")
        form = QFormLayout(group)

        self.labels_edit = QLineEdit(str(DEFAULT_LABELS_PATH))
        form.addRow("라벨 JSON", self._with_browse_button(self.labels_edit, self._on_browse_labels))

        self.keypoints3d_edit = QLineEdit()
        self.keypoints3d_edit.setPlaceholderText("keypoints_2d가 없는 라벨이 있을 때만 필요")
        form.addRow("키포인트 3D (.npy)", self._with_browse_button(self.keypoints3d_edit, self._on_browse_keypoints3d))

        self.out_dir_edit = QLineEdit(str(DEFAULT_OUT_DIR))
        form.addRow("출력 디렉토리", self._with_browse_button(self.out_dir_edit, self._on_browse_out_dir))

        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText("이어서 학습할 체크포인트(.pth), 선택")
        form.addRow("재개 체크포인트", self._with_browse_button(self.resume_edit, self._on_browse_resume))

        return group

    def _build_crop_group(self) -> QGroupBox:
        group = QGroupBox("크롭 설정")
        form = QFormLayout(group)

        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(32, 1024)
        self.crop_size_spin.setValue(256)
        form.addRow("crop size", self.crop_size_spin)

        self.bbox_padding_spin = QDoubleSpinBox()
        self.bbox_padding_spin.setRange(0.0, 2.0)
        self.bbox_padding_spin.setSingleStep(0.05)
        self.bbox_padding_spin.setValue(0.2)
        form.addRow("bbox padding", self.bbox_padding_spin)

        self.mask_background_check = QCheckBox("배경 픽셀 지우기 (권장)")
        self.mask_background_check.setChecked(True)
        form.addRow("", self.mask_background_check)

        return group

    def _build_optim_group(self) -> QGroupBox:
        group = QGroupBox("학습 하이퍼파라미터")
        form = QFormLayout(group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(8)
        form.addRow("batch size", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(100)
        form.addRow("epochs", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(1e-4)
        form.addRow("learning rate", self.lr_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(1e-5)
        self.weight_decay_spin.setValue(1e-4)
        form.addRow("weight decay", self.weight_decay_spin)

        self.vertex_loss_weight_spin = QDoubleSpinBox()
        self.vertex_loss_weight_spin.setRange(0.0, 100.0)
        self.vertex_loss_weight_spin.setSingleStep(0.1)
        self.vertex_loss_weight_spin.setValue(1.0)
        form.addRow("vertex loss weight", self.vertex_loss_weight_spin)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 0.9)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setValue(0.1)
        form.addRow("val ratio", self.val_ratio_spin)

        fitness_row = QHBoxLayout()
        self.min_fitness_check = QCheckBox("사용")
        self.min_fitness_spin = QDoubleSpinBox()
        self.min_fitness_spin.setRange(0.0, 1.0)
        self.min_fitness_spin.setSingleStep(0.01)
        self.min_fitness_spin.setValue(0.85)
        self.min_fitness_spin.setEnabled(False)
        self.min_fitness_check.toggled.connect(self.min_fitness_spin.setEnabled)
        fitness_row.addWidget(self.min_fitness_check)
        fitness_row.addWidget(self.min_fitness_spin)
        form.addRow("min fitness", fitness_row)

        return group

    def _build_misc_group(self) -> QGroupBox:
        group = QGroupBox("기타")
        form = QFormLayout(group)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        form.addRow("seed", self.seed_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 64)
        self.num_workers_spin.setValue(4)
        form.addRow("num workers", self.num_workers_spin)

        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        form.addRow("device", self.device_combo)

        self.pretrained_check = QCheckBox("ImageNet 사전학습 backbone 사용")
        self.pretrained_check.setChecked(True)
        form.addRow("", self.pretrained_check)

        self.save_every_spin = QSpinBox()
        self.save_every_spin.setRange(0, 10000)
        self.save_every_spin.setValue(10)
        form.addRow("save every (epoch)", self.save_every_spin)

        self.log_every_spin = QSpinBox()
        self.log_every_spin.setRange(0, 10000)
        self.log_every_spin.setValue(20)
        form.addRow("log every (batch)", self.log_every_spin)

        return group

    @staticmethod
    def _with_browse_button(line_edit: QLineEdit, slot) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        btn = QPushButton("선택")
        btn.setFixedWidth(48)
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return wrapper

    # ----------------------------------------------------------- 파일 선택
    def _on_browse_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "라벨 JSON 선택", self.labels_edit.text(), "JSON (*.json)")
        if path:
            self.labels_edit.setText(path)

    def _on_browse_keypoints3d(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "키포인트 3D 선택", self.keypoints3d_edit.text(), "NumPy (*.npy)"
        )
        if path:
            self.keypoints3d_edit.setText(path)

    def _on_browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 디렉토리 선택", self.out_dir_edit.text())
        if path:
            self.out_dir_edit.setText(path)

    def _on_browse_resume(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "재개 체크포인트 선택", self.resume_edit.text(), "Checkpoint (*.pth)")
        if path:
            self.resume_edit.setText(path)

    # ----------------------------------------------------------- 명령줄 조립
    def _build_command_args(self) -> list[str]:
        args = [
            str(TRAIN_SCRIPT),
            "--labels", self.labels_edit.text().strip(),
            "--out-dir", self.out_dir_edit.text().strip(),
            "--crop-size", str(self.crop_size_spin.value()),
            "--bbox-padding", str(self.bbox_padding_spin.value()),
            "--batch-size", str(self.batch_size_spin.value()),
            "--epochs", str(self.epochs_spin.value()),
            "--lr", str(self.lr_spin.value()),
            "--weight-decay", str(self.weight_decay_spin.value()),
            "--vertex-loss-weight", str(self.vertex_loss_weight_spin.value()),
            "--val-ratio", str(self.val_ratio_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--num-workers", str(self.num_workers_spin.value()),
            "--device", self.device_combo.currentText().strip(),
            "--save-every", str(self.save_every_spin.value()),
            "--log-every", str(self.log_every_spin.value()),
        ]
        args.append("--mask-background" if self.mask_background_check.isChecked() else "--no-mask-background")
        args.append("--pretrained-backbone" if self.pretrained_check.isChecked() else "--no-pretrained-backbone")

        keypoints3d = self.keypoints3d_edit.text().strip()
        if keypoints3d:
            args += ["--keypoints-3d", keypoints3d]

        resume = self.resume_edit.text().strip()
        if resume:
            args += ["--resume", resume]

        if self.min_fitness_check.isChecked():
            args += ["--min-fitness", str(self.min_fitness_spin.value())]

        return args

    def _validate_before_start(self) -> str | None:
        """문제가 있으면 사용자에게 보여줄 에러 메시지를, 없으면 None을 반환."""
        labels_path = Path(self.labels_edit.text().strip())
        if not labels_path.is_file():
            return f"라벨 JSON을 찾을 수 없습니다: {labels_path}"

        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            return "출력 디렉토리를 지정하세요."

        keypoints3d = self.keypoints3d_edit.text().strip()
        if keypoints3d and not Path(keypoints3d).is_file():
            return f"키포인트 3D 파일을 찾을 수 없습니다: {keypoints3d}"

        resume = self.resume_edit.text().strip()
        if resume and not Path(resume).is_file():
            return f"재개 체크포인트를 찾을 수 없습니다: {resume}"

        if not TRAIN_SCRIPT.is_file():
            return f"학습 스크립트를 찾을 수 없습니다: {TRAIN_SCRIPT}"

        return None

    # ----------------------------------------------------------- 실행/중지
    def _on_start_clicked(self) -> None:
        error = self._validate_before_start()
        if error:
            QMessageBox.warning(self, "확인 필요", error)
            return

        Path(self.out_dir_edit.text().strip()).mkdir(parents=True, exist_ok=True)

        self._total_epochs = self.epochs_spin.value()
        self._best_loss = None
        self.progress_bar.setRange(0, self._total_epochs)
        self.progress_bar.setValue(0)
        self.log_view.clear()

        args = self._build_command_args()

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        # -u: stdout을 완전 unbuffered로 - 이게 없으면 파이썬이 파이프로
        # 리다이렉트될 때 stdout을 블록 버퍼링해서, 학습이 다 끝난 뒤에야
        # 로그가 한꺼번에 쏟아지는 문제가 생긴다(실시간 스트리밍의 핵심).
        self._process.setProgram(sys.executable)
        self._process.setArguments(["-u"] + args)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        self._set_running_state(True)
        self._append_log(f"$ {sys.executable} -u {' '.join(args)}")
        self.status_label.setText("학습 시작 중...")
        self._process.start()

    def _on_stop_clicked(self) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            return
        reply = QMessageBox.question(
            self, "학습 중지", "정말 학습을 중지할까요? 마지막 체크포인트(last.pth)까지만 저장된 상태입니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.status_label.setText("중지 중...")
        self._process.kill()

    def _set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        for w in (
            self.labels_edit, self.keypoints3d_edit, self.out_dir_edit, self.resume_edit,
            self.crop_size_spin, self.bbox_padding_spin, self.mask_background_check,
            self.batch_size_spin, self.epochs_spin, self.lr_spin, self.weight_decay_spin,
            self.vertex_loss_weight_spin, self.val_ratio_spin, self.min_fitness_check,
            self.min_fitness_spin, self.seed_spin, self.num_workers_spin, self.device_combo,
            self.pretrained_check, self.save_every_spin, self.log_every_spin,
        ):
            w.setEnabled(not running)

    # ----------------------------------------------------------- 프로세스 이벤트
    def _on_process_output(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            if not line:
                continue
            self._append_log(line)

            m = _EPOCH_RE.search(line)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                if total != self._total_epochs:
                    self._total_epochs = total
                    self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(current)
                self.status_label.setText(f"epoch {current}/{total} 진행 중")

            m_best = _BEST_RE.search(line)
            if m_best:
                self._best_loss = float(m_best.group(1))

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        self._set_running_state(False)
        if exit_code == 0:
            best_txt = f" (best loss={self._best_loss:.4f})" if self._best_loss is not None else ""
            self.status_label.setText(f"완료{best_txt}")
            self._append_log(f"=== 학습 정상 종료 (exit code 0){best_txt} ===")
        else:
            self.status_label.setText(f"종료됨 (exit code {exit_code})")
            self._append_log(f"=== 프로세스 종료 (exit code {exit_code}) - 위 로그에서 에러 확인 ===")
        self.log_message.emit(
            f"[{self.LOG_PREFIX}] 학습 프로세스 종료 (exit code {exit_code}, out_dir={self.out_dir_edit.text().strip()})"
        )

    def _on_process_error(self, error) -> None:
        self._set_running_state(False)
        self.status_label.setText("실행 실패")
        self._append_log(f"=== 프로세스 실행 실패: {error} (파이썬 실행 파일 경로/권한을 확인하세요) ===")

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())