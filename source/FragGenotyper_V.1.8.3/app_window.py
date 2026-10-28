"""Three connected workspaces with cooperative shutdown."""
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QStackedWidget, QComboBox, QApplication,
)
from app_style import get_theme_controller
from phenotype_tab import PhenotypeComparisonTab
from genotype_panel import GenotypePanel, VariantFilterPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme_controller = get_theme_controller(QApplication.instance())
        self.language = "zh"
        self.setWindowTitle("FragGenotyper 1.8.3")
        self.resize(1320, 920)
        self.setMinimumSize(1040, 720)
        root_widget = QWidget()
        root_widget.setObjectName("appRoot")
        layout = QVBoxLayout(root_widget)
        layout.setContentsMargins(18, 14, 18, 12)
        header = QHBoxLayout()
        title = QLabel("FragGenotyper")
        title.setObjectName("brand")
        header.addWidget(title)
        self.subtitle = QLabel("1.8.3 · 片段分型与表型比较")
        self.subtitle.setObjectName("subtitle")
        header.addWidget(self.subtitle)
        header.addStretch()
        self.theme_combo = QComboBox()
        for label, mode in [("跟随系统", "system"), ("浅色模式", "light"), ("深色模式", "dark")]:
            self.theme_combo.addItem(label, mode)
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(self.theme_controller.mode))
        self.theme_combo.setAccessibleName("外观主题")
        self.theme_combo.setToolTip("选择外观主题；不会改变导出图的配色")
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        self.theme_controller.theme_changed.connect(self._sync_theme_combo)
        header.addWidget(self.theme_combo)
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self.change_language)
        header.addWidget(self.language_combo)
        layout.addLayout(header)
        workspace = QHBoxLayout()
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setFixedWidth(158)
        self.navigation.addItems(["片段分型", "表型比较", "变异筛选"])
        self.stack = QStackedWidget()
        self.genotype_panel = GenotypePanel(language=self.language)
        self.phenotype_panel = PhenotypeComparisonTab(language=self.language)
        self.filter_panel = VariantFilterPanel(language=self.language)
        for panel in (self.genotype_panel, self.phenotype_panel, self.filter_panel):
            self.stack.addWidget(panel)
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        workspace.addWidget(self.navigation)
        workspace.addWidget(self.stack, 1)
        layout.addLayout(workspace, 1)
        self.setCentralWidget(root_widget)
        self.genotype_panel.genotypes_ready.connect(self.use_genotypes)
        self.phenotype_panel.message.connect(lambda text: self.statusBar().showMessage(text, 12000))
        self.statusBar().showMessage("就绪 · 旧版本保留；当前为 1.8.3")

    def use_genotypes(self, frame, origin):
        if self.phenotype_panel.is_busy():
            self.statusBar().showMessage("表型分析正在运行；请完成后重新传入分型结果。", 10000)
            return
        self.phenotype_panel.accept_genotypes(frame, origin)
        self.statusBar().showMessage("分型结果已载入“表型比较”，可继续加载表型表。", 10000)

    def change_language(self, *_):
        self.language = self.language_combo.currentData()
        english = self.language == "en"
        for index, (chinese, english_label) in enumerate([("跟随系统", "System theme"), ("浅色模式", "Light mode"), ("深色模式", "Dark mode")]):
            self.theme_combo.setItemText(index, english_label if english else chinese)
        self.theme_combo.setAccessibleName("Appearance theme" if english else "外观主题")
        self.theme_combo.setToolTip("Choose the appearance; exported figure colors stay unchanged" if english else "选择外观主题；不会改变导出图的配色")
        self.subtitle.setText("1.8.3 · Fragment genotypes and phenotypes" if english else "1.8.3 · 片段分型与表型比较")
        for i, name in enumerate(["Fragment genotyping", "Phenotype comparison", "Variant filtering"] if english else ["片段分型", "表型比较", "变异筛选"]):
            self.navigation.item(i).setText(name)
        for panel in (self.genotype_panel, self.phenotype_panel, self.filter_panel):
            panel.set_language(self.language)

    def change_theme(self, *_):
        mode = self.theme_combo.currentData()
        if mode:
            self.theme_controller.set_mode(mode)

    def _sync_theme_combo(self, *_):
        previous = self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(self.theme_controller.mode))
        self.theme_combo.blockSignals(previous)

    def closeEvent(self, event):
        panels = (self.genotype_panel, self.phenotype_panel, self.filter_panel)
        ready = [panel.begin_shutdown() for panel in panels]
        if all(ready):
            event.accept()
        else:
            self.statusBar().showMessage("正在安全停止后台任务…" if self.language == "zh" else "Stopping background tasks safely…")
            event.ignore()
            QTimer.singleShot(150, self.close)
