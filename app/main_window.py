"""메인 윈도우: 좌측 트리 내비게이션 + 우측 스택 콘텐츠 + 하단 공용 로그.

원본 BENIROBO_RTMDetTrain 프로젝트(탭 0~10)에서 PVNet 라벨링에 필요한
탭만 들고 나온 최소 버전이다:

    0. 설정           (settings_tab.py - ICP 파라미터(CAD 관련 제외) +
                        RTMDet-Ins 체크포인트/config + 카메라 설정을
                        여기서만 편집한다. data/app_settings.json에 저장돼
                        앱을 재시작해도 유지된다.)
    1. 수동 라벨링    (구 "8. 수동 라벨링", manual_labeling_tab.py)
    2. PVNet 라벨 생성 (구 "10. PVNet 라벨 생성", pvnet_label_generation_tab.py)
    3. PVNet 학습      (pvnet_train_tab.py - scripts/train_pvnet.py를
                         QProcess로 실행하는 GUI 래퍼)
    4. CAD 모델 설정   (cad_model_settings_tab.py - CAD 선택, 초기 자세/
                         축보정/HPR 파라미터, CAD 가시면 미리보기, 그리고
                         "초기 자세 vs 카메라 포인트클라우드 비교" - CAD를
                         초기 roll/pitch/yaw로 놓았을 때 실제 카메라
                         포인트클라우드와 방향이 맞는지 3D로 직접 대조)

ICP 기반 탭(1, 2 - 정확히는 이들의 공통 조상인 ICPWorkbenchTab)은 더 이상
체크포인트/config/CAD/ICP 파라미터를 직접 편집하지 않는다. 실행 시점마다
app/core/settings_manager.load_settings()로 "설정"/"CAD 모델 설정" 탭에
저장된 최신값을 다시 읽어 쓴다 - 이 탭들을 미리 열어두지 않았어도 항상
최신값이 반영된다. 각 ICP 탭의 "설정 열기"/"CAD 모델 설정 열기" 버튼은
open_settings_requested/open_cad_settings_requested 시그널을 emit하고,
이 파일이 그 시그널을 받아 nav_tree에서 해당 탭으로 전환한다.

RotHead/FoundationPose 관련 탭, 데이터 수집/세션 관리, 오프라인 검출
테스트, RTMDet 학습 파이프라인은 이 프로젝트 범위에서 제외했다 - 필요한
파트는 원본 프로젝트에서 이미 학습된 checkpoint(.pth)+config(.py)를
그대로 복사해서 재사용한다.

구조는 원본과 동일하게 QTreeWidget(좌측 내비게이션) + QStackedWidget(우측
콘텐츠) + 하단 공용 로그 콘솔을 유지했다 - 나중에 탭이 다시 늘어날 걸
대비해 원본 아키텍처(카테고리 하위 트리 지원, _add_leaf() 패턴)를 그대로
가져왔다. 새 탭을 추가할 때는 원본과 동일하게:
    1. app/tabs/에 새 파일 작성
    2. 이 파일에 import + 인스턴스 생성 + self._add_leaf(...) 한 줄
    3. 필요하면 log_message 시그널 연결 한 줄
만 하면 된다.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTreeWidget,
    QTreeWidgetItem, QStackedWidget, QLabel,
)
from PyQt6.QtCore import Qt

from app.tabs.cad_model_settings_tab import CADModelSettingsTab
from app.tabs.manual_labeling_tab import ManualLabelingTab
from app.tabs.pvnet_label_generation_tab import PVNetLabelGenerationTab
from app.tabs.pvnet_train_tab import PVNetTrainTab
from app.tabs.settings_tab import SettingsTab
from app.widgets.log_console import LogConsole

# 트리 아이템에 스택 페이지 인덱스를 저장할 때 쓰는 데이터 role.
PAGE_INDEX_ROLE = Qt.ItemDataRole.UserRole


class CurrentPageStackedWidget(QStackedWidget):
    """QStackedWidget 기본 동작 보정: sizeHint/minimumSizeHint를 '현재 보이는
    페이지' 기준으로만 계산한다.

    Qt의 QStackedWidget은 기본적으로 지금까지 추가된 *모든* 페이지 중 가장
    큰 최소 크기를 스택 전체의 최소 크기로 잡는다 - 즉 화면에 안 보이는
    탭 하나가 크면, 지금 보고 있는 다른(작은) 탭에서도 메인 윈도우 전체의
    최소 크기가 그만큼 커져버려 창 최대화가 안 되는 문제가 생긴다(원본
    프로젝트에서 2026-08에 실제로 발견/수정한 이슈 - 이 프로젝트도 탭이
    늘어날 걸 대비해 처음부터 이 보정을 포함시켰다).
    """

    def sizeHint(self):  # noqa: N802 - Qt override 관례
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):  # noqa: N802 - Qt override 관례
        current = self.currentWidget()
        return current.minimumSizeHint() if current is not None else super().minimumSizeHint()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PVNet Labeling Toolkit")
        self.resize(1200, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        top_bar = self._build_top_bar()
        outer.addWidget(top_bar)

        body = QHBoxLayout()
        outer.addLayout(body, stretch=1)

        self.nav_tree = QTreeWidget()
        self.nav_tree.setFixedWidth(200)
        self.nav_tree.setHeaderHidden(True)
        body.addWidget(self.nav_tree)

        self.stack = CurrentPageStackedWidget()
        body.addWidget(self.stack, stretch=1)

        # ---- 페이지 구성 (탭 인스턴스 생성 + 스택에 추가 + 트리 항목 연결) ----
        self.settings_tab = SettingsTab()
        self.settings_nav_item = self._add_leaf("0. 설정", self.settings_tab)

        self.manual_labeling_tab = ManualLabelingTab()
        self._add_leaf("1. 수동 라벨링", self.manual_labeling_tab)

        self.pvnet_label_gen_tab = PVNetLabelGenerationTab()
        self._add_leaf("2. PVNet 라벨 생성", self.pvnet_label_gen_tab)

        self.pvnet_train_tab = PVNetTrainTab()
        self._add_leaf("3. PVNet 학습", self.pvnet_train_tab)

        self.cad_model_settings_tab = CADModelSettingsTab()
        self.cad_settings_nav_item = self._add_leaf("4. CAD 모델 설정", self.cad_model_settings_tab)

        self.nav_tree.currentItemChanged.connect(self._on_nav_changed)

        self.log_console = LogConsole()
        outer.addWidget(self.log_console)

        # 로그 시그널 연결
        self.settings_tab.log_message.connect(self.log_console.append_log)
        self.manual_labeling_tab.log_message.connect(self.log_console.append_log)
        self.pvnet_label_gen_tab.log_message.connect(self.log_console.append_log)
        self.pvnet_train_tab.log_message.connect(self.log_console.append_log)
        self.cad_model_settings_tab.log_message.connect(self.log_console.append_log)

        # ICP 기반 탭의 "설정 열기"/"CAD 모델 설정 열기" 버튼 -> 해당 탭으로 전환
        self.manual_labeling_tab.open_settings_requested.connect(self._on_open_settings)
        self.pvnet_label_gen_tab.open_settings_requested.connect(self._on_open_settings)
        self.manual_labeling_tab.open_cad_settings_requested.connect(self._on_open_cad_settings)
        self.pvnet_label_gen_tab.open_cad_settings_requested.connect(self._on_open_cad_settings)
        # CAD 모델 설정 탭의 "카메라 설정 열기" 버튼 -> 설정 탭으로 전환
        self.cad_model_settings_tab.open_settings_requested.connect(self._on_open_settings)

        # 첫 화면: 트리의 첫 leaf 항목 선택
        first_leaf = self.nav_tree.topLevelItem(0)
        self.nav_tree.setCurrentItem(first_leaf)

    def _add_leaf(
        self, label: str, page: QWidget, parent_item: QTreeWidgetItem | None = None
    ) -> QTreeWidgetItem:
        """스택에 페이지를 추가하고, 그 인덱스를 담은 트리 leaf 항목을 만들어 반환한다.

        나중에 카테고리(하위 트리)가 필요해지면 parent_item에 QTreeWidgetItem을
        넘기면 된다 - 원본 프로젝트의 "2. 모델 학습" 카테고리 트리와 동일한
        확장 지점을 그대로 남겨뒀다.
        """
        page_index = self.stack.addWidget(page)
        item = QTreeWidgetItem([label])
        item.setData(0, PAGE_INDEX_ROLE, page_index)
        if parent_item is not None:
            parent_item.addChild(item)
        else:
            self.nav_tree.addTopLevelItem(item)
        return item

    def _on_open_settings(self) -> None:
        """ICP 기반 탭의 '설정 열기' 버튼(open_settings_requested)에 연결됨."""
        self.nav_tree.setCurrentItem(self.settings_nav_item)

    def _on_open_cad_settings(self) -> None:
        """ICP 기반 탭의 'CAD 모델 설정 열기' 버튼(open_cad_settings_requested)에 연결됨."""
        self.nav_tree.setCurrentItem(self.cad_settings_nav_item)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("PVNet Labeling Toolkit")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        layout.addStretch(1)
        return bar

    def _on_nav_changed(self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        page_index = current.data(0, PAGE_INDEX_ROLE)
        if page_index is None:
            # 카테고리(부모) 항목 - 페이지가 없으므로 펼치기/접기만 한다.
            current.setExpanded(not current.isExpanded())
            return
        self.stack.setCurrentIndex(page_index)
        # sizeHint/minimumSizeHint를 오버라이드했으므로, 탭이 바뀔 때마다
        # 레이아웃이 새 현재 페이지 기준으로 다시 계산되도록 명시적으로
        # 알려준다 (Qt가 캐시된 이전 sizeHint를 계속 쓰는 것을 방지).
        self.stack.updateGeometry()