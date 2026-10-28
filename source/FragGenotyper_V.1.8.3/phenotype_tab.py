"""Guided phenotype comparisons, reproducible projects, and live plot options."""
from itertools import combinations
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QListWidget, QListWidgetItem, QMessageBox,
    QScrollArea, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QLineEdit, QSpinBox, QCheckBox, QProgressBar, QAbstractItemView,
    QHeaderView, QColorDialog,
)
from plot_toolbar import PlotNavigationToolbar
from workers import AnalysisWorker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

MAX_VISIBLE_PAIRS = 5000


def input_metadata(genotype, phenotype, mode, preferences):
    """Infer column defaults and match samples without touching Qt objects."""
    from haplotype_phenotype import prepare_data
    def column(columns, previous, names):
        if previous in columns:
            return previous
        lookup = {str(col).casefold(): col for col in columns}
        return next((lookup[name.casefold()] for name in names if name.casefold() in lookup), columns[0] if columns else "")
    gen_cols = list(genotype.columns) if genotype is not None else []
    pheno_cols = list(phenotype.columns) if phenotype is not None else []
    aliases = ["Sample", "sample_id", "Taxa", "IID", "Unnamed: 0"]
    sample = column(gen_cols, preferences.get("sample"), aliases)
    group = column(gen_cols, preferences.get("group"), ["Genotype", "Hap", "Haplotype", "Cluster"])
    pheno_sample = column(pheno_cols, preferences.get("phenotype_sample"), aliases)
    combined_sample = next((col for col in gen_cols if col.casefold() in {"sample", "sample_id", "taxa", "iid", "unnamed: 0"}), "Sample")
    source = genotype if mode == "combined" else phenotype
    exclude = {group, pheno_sample if mode == "separate" else combined_sample}
    choices = [col for col in (list(source.columns) if source is not None else []) if col not in exclude and col.casefold() != "fid"]
    selected = [col for col in choices if col in preferences.get("traits", [])]
    if not selected and choices:
        selected = [column(choices, None, ["Phenotype", "FibLen_BLUP", "Yolo"])]
    result = dict(sample=sample, group=group, phenotype_sample=pheno_sample,
                  traits=choices, selected_traits=selected, counts={}, available=[], report=None, error=None)
    try:
        if genotype is None or not selected:
            raise ValueError("请选择数据和至少一个性状。 / Choose data and at least one trait.")
        if mode == "separate" and phenotype is None:
            raise ValueError("请选择表型表。 / Choose a phenotype table.")
        data, result["report"] = prepare_data(genotype, phenotype if mode == "separate" else None,
            sample_col=sample if mode == "separate" else combined_sample, group_col=group,
            trait_col=selected[0], phenotype_sample_col=pheno_sample if mode == "separate" else None)
        result["counts"] = data.groupby("Genotype").size().to_dict()
        raw_groups = genotype[group].astype("string").str.strip().dropna()
        result["available"] = sorted({str(value) for value in raw_groups if value})
    except Exception as exc:
        result["error"] = str(exc)
    return result


class ComparisonCanvas(FigureCanvasQTAgg):
    """Keep the axis label above the footer as a desktop pane is resized."""
    def draw(self):
        super().draw()
        if not self.figure.axes or not self.figure.texts:
            return
        renderer = self.get_renderer()
        axis = self.figure.axes[0]
        footer_top = max(text.get_window_extent(renderer).y1 for text in self.figure.texts)
        label_bottom = axis.xaxis.label.get_window_extent(renderer).y0
        required = footer_top + 8 - label_bottom
        if required > 0:
            bottom = min(.48, self.figure.subplotpars.bottom + required / self.figure.bbox.height)
            self.figure.subplots_adjust(bottom=bottom)
            super().draw()


def populate_table(widget, frame):
    widget.clear()
    widget.setColumnCount(len(frame.columns))
    widget.setRowCount(len(frame))
    widget.setHorizontalHeaderLabels([str(col) for col in frame.columns])
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        for column_index, value in enumerate(row):
            if isinstance(value, (float, np.floating)):
                text = f"{value:.6g}" if np.isfinite(value) else "n/a"
            else:
                text = str(value)
            widget.setItem(row_index, column_index, QTableWidgetItem(text))
    widget.resizeColumnsToContents()


class PhenotypeComparisonTab(QWidget):
    message = pyqtSignal(str)

    def __init__(self, language="zh", parent=None):
        super().__init__(parent)
        self.language = language
        self.genotype_df = self.phenotype_df = None
        self.genotype_path = self.phenotype_path = None
        self.origin_path = None
        self.worker = None
        self.analysis = None
        self.canvas = self.toolbar = None
        self._updating = False
        self._had_groups = False
        self._report = None
        self._error = None
        self._pairs_checked = set()
        self._annotation_checked = set()
        self._pair_choices = {}
        self._annotation_choices = {}
        self._background_metadata = False
        self._metadata_pending = False
        self._next_group_metadata = None
        self._data_generation = 0
        self._worker_kind = None
        self.import_diagnostics = {}
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._start_metadata_refresh)
        self._texts = []
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        heading = QLabel()
        heading.setObjectName("sectionTitle")
        self.register(heading, "表型比较", "Phenotype comparison")
        root.addWidget(heading)

        project_bar = QHBoxLayout()
        self.project_hint = QLabel()
        self.register(self.project_hint, "按步骤选择数据、比较和图形设置", "Choose data, comparisons and plot settings")
        project_bar.addWidget(self.project_hint, 1)
        self.load_button = self.button("打开项目", "Open project", self.open_project)
        self.project_button = self.button("保存项目", "Save project", self.save_project_dialog)
        project_bar.addWidget(self.load_button)
        project_bar.addWidget(self.project_button)
        root.addLayout(project_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        self.controls = QTabWidget()
        self.controls.setMinimumWidth(365)
        self.controls.setMaximumWidth(560)
        splitter.addWidget(self.controls)
        self.data_page, data_layout = self.control_page("1 数据", "1 Data")
        self.comparison_page, comparison_layout = self.control_page("2 比较", "2 Compare")
        self.plot_page, plot_layout = self.control_page("3 绘图", "3 Plot")
        self.build_data_controls(data_layout)
        self.build_comparison_controls(comparison_layout)
        self.build_plot_controls(plot_layout)
        self.results_tabs = QTabWidget()
        splitter.addWidget(self.results_tabs)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.placeholder = QLabel()
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.register(self.placeholder, "完成数据和单倍型选择后，点击“运行分析”\n图形可缩放，并支持 PNG / PDF 导出", "Choose data and haplotypes, then run the analysis.\nZoom the figure or export PNG / PDF.")
        self.plot_layout.addWidget(self.placeholder, 1)
        self.results_tabs.addTab(self.plot_container, "箱线图")
        self.test_table = self.result_table()
        self.summary_table = self.result_table()
        self.results_tabs.addTab(self.test_table, "检验结果")
        self.results_tabs.addTab(self.summary_table, "描述统计")
        splitter.setSizes([400, 800])

        result_bar = QHBoxLayout()
        self.result_label = QLabel()
        self.register(self.result_label, "当前性状", "Current trait")
        result_bar.addWidget(self.result_label)
        self.result_trait_combo = QComboBox()
        self.result_trait_combo.setMinimumWidth(180)
        self.result_trait_combo.currentTextChanged.connect(self.display_result)
        result_bar.addWidget(self.result_trait_combo)
        result_bar.addStretch()
        self.result_note = QLabel()
        result_bar.addWidget(self.result_note)
        root.addLayout(result_bar)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        actions.addWidget(self.progress, 1)
        self.cancel_button = self.button("取消", "Cancel", self.cancel_task)
        self.run_button = self.button("运行分析", "Run analysis", self.run_analysis)
        self.run_button.setObjectName("primary")
        self.export_button = self.button("导出本次分析", "Export analysis", self.export_analysis)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)
        actions.addWidget(self.export_button)
        root.addLayout(actions)
        self.set_language(language)
        self.input_mode_changed()
        self.mode_changed()
        self.set_busy(False)

    def tr_text(self, zh, en):
        return en if self.language == "en" else zh

    def register(self, widget, zh, en):
        self._texts.append((widget, zh, en))
        widget.setText(self.tr_text(zh, en))
        return widget

    def button(self, zh, en, callback):
        button = QPushButton(clicked=callback)
        return self.register(button, zh, en)

    def label(self, zh, en):
        return self.register(QLabel(), zh, en)

    def control_page(self, zh, en):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(10)
        scroll.setWidget(widget)
        self.controls.addTab(scroll, self.tr_text(zh, en))
        return scroll, layout

    def result_table(self):
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return table

    def build_data_controls(self, layout):
        self.input_mode = QComboBox()
        self.input_mode.addItem("", "separate")
        self.input_mode.addItem("", "combined")
        layout.addWidget(self.input_mode)
        self.genotype_button = self.button("选择分型表 / 合并表", "Choose genotypes / combined table", self.select_genotype_file)
        self.genotype_file_label = QLabel()
        self.phenotype_button = self.button("选择表型表", "Choose phenotype table", self.select_phenotype_file)
        self.phenotype_file_label = QLabel()
        for button, label in [(self.genotype_button, self.genotype_file_label), (self.phenotype_button, self.phenotype_file_label)]:
            label.setWordWrap(True)
            layout.addWidget(button)
            layout.addWidget(label)
        form = QFormLayout()
        self.sample_combo, self.group_combo = QComboBox(), QComboBox()
        self.phenotype_sample_combo = QComboBox()
        self.sample_label = self.label("分型表样本列", "Genotype sample column")
        self.group_label = self.label("单倍型 / 分组列", "Haplotype / group column")
        self.phenotype_sample_label = self.label("表型表样本列", "Phenotype sample column")
        for label, combo in [(self.sample_label, self.sample_combo), (self.group_label, self.group_combo), (self.phenotype_sample_label, self.phenotype_sample_combo)]:
            form.addRow(label, combo)
            combo.currentIndexChanged.connect(self.columns_changed)
        layout.addLayout(form)
        layout.addWidget(self.label("勾选一个或多个性状", "Select one or more traits"))
        self.trait_list = QListWidget()
        self.trait_list.setMinimumHeight(140)
        self.trait_list.setMaximumHeight(260)
        self.trait_list.itemChanged.connect(self.traits_changed)
        layout.addWidget(self.trait_list)
        trait_actions = QHBoxLayout()
        trait_actions.addWidget(self.button("全选性状", "Select all traits", lambda: self.check_all(self.trait_list, True)))
        trait_actions.addWidget(self.button("清空", "Clear", lambda: self.check_all(self.trait_list, False)))
        layout.addLayout(trait_actions)
        self.refresh_button = self.button("刷新数据检查", "Refresh data checks", self.refresh_groups)
        layout.addWidget(self.refresh_button)
        self.data_note = QLabel()
        self.data_note.setWordWrap(True)
        self.register(self.data_note, "CSV / TSV / 空格分隔文本；样本名一对一匹配。\n单表可仅含分组和表型。非数值、缺失及无穷表型会剔除。", "CSV / TSV / space-delimited text; sample IDs match one-to-one.\nA combined table needs only group and trait columns. Invalid traits are removed.")
        layout.addWidget(self.data_note)
        layout.addStretch()
        self.input_mode.currentIndexChanged.connect(self.input_mode_changed)

    def build_comparison_controls(self, layout):
        self.group_note = self.label("勾选参与分析的单倍型；n 按首个所选性状统计", "Select haplotypes; n uses the first selected trait")
        self.group_note.setWordWrap(True)
        layout.addWidget(self.group_note)
        self.group_list = QListWidget()
        self.group_list.setMinimumHeight(180)
        self.group_list.itemChanged.connect(self.groups_changed)
        layout.addWidget(self.group_list)
        actions = QHBoxLayout()
        actions.addWidget(self.button("全选", "Select all", lambda: self.check_all(self.group_list, True)))
        actions.addWidget(self.button("清空", "Clear", lambda: self.check_all(self.group_list, False)))
        actions.addWidget(self.button("上移", "Up", lambda: self.move_group(-1)))
        actions.addWidget(self.button("下移", "Down", lambda: self.move_group(1)))
        layout.addLayout(actions)
        form = QFormLayout()
        self.comparison_combo = QComboBox()
        for mode in ("pairwise", "reference", "custom"):
            self.comparison_combo.addItem("", mode)
        self.reference_combo = QComboBox()
        self.reference_label = self.label("参考单倍型", "Reference")
        self.correction_combo = QComboBox()
        self.correction_combo.addItem("BH (FDR)", "bh")
        self.correction_combo.addItem("", "none")
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("", "per_trait")
        self.scope_combo.addItem("", "global")
        form.addRow(self.label("比较方式", "Comparisons"), self.comparison_combo)
        form.addRow(self.reference_label, self.reference_combo)
        form.addRow(self.label("P 值校正", "P correction"), self.correction_combo)
        form.addRow(self.label("校正范围", "Correction family"), self.scope_combo)
        layout.addLayout(form)
        self.pairs_label = self.label("勾选实际检验的比较", "Select comparisons to test")
        layout.addWidget(self.pairs_label)
        self.pair_list = QListWidget()
        self.pair_list.setMinimumHeight(100)
        self.pair_list.setMaximumHeight(180)
        layout.addWidget(self.pair_list)
        self.pair_list.itemChanged.connect(self.test_pairs_changed)
        self.pair_notice = QLabel()
        self.pair_notice.setWordWrap(True)
        layout.addWidget(self.pair_notice)
        note = self.label("Welch 双侧 t 检验；每组至少 2 个有效值。\n结果含均值差和 95% 置信区间。n 不足的性状列入失败清单，其余继续。", "Two-sided Welch t-test; at least 2 valid values per group.\nResults include mean differences and 95% CIs. Failed traits are listed; other traits continue.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.comparison_combo.currentIndexChanged.connect(self.mode_changed)
        self.reference_combo.currentTextChanged.connect(self.mode_changed)
        self.correction_combo.currentIndexChanged.connect(self.invalidate_analysis)
        self.scope_combo.currentIndexChanged.connect(self.invalidate_analysis)

    def build_plot_controls(self, layout):
        self.title_edit, self.y_label_edit = QLineEdit(), QLineEdit()
        self.title_edit.setPlaceholderText("自动 / Automatic")
        self.y_label_edit.setPlaceholderText("性状名称及单位 / Trait and unit")
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(11)
        self.point_spin = QSpinBox()
        self.point_spin.setRange(5, 80)
        self.point_spin.setValue(25)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["tab10", "Set2", "Dark2", "viridis"])
        self.annotation_style = QComboBox()
        for value in ("both", "stars", "p"):
            self.annotation_style.addItem("", value)
        self.annotation_filter = QComboBox()
        for value in ("all", "significant", "custom", "none"):
            self.annotation_filter.addItem("", value)
        self.points_check = QCheckBox()
        self.means_check = QCheckBox()
        self.points_check.setChecked(True)
        self.means_check.setChecked(True)
        self.register(self.points_check, "显示原始散点", "Show observations")
        self.register(self.means_check, "显示均值菱形", "Show mean diamonds")
        form = QFormLayout()
        for zh, en, control in [
            ("图标题", "Title", self.title_edit), ("纵轴及单位", "Y label / unit", self.y_label_edit),
            ("字号", "Font size", self.font_spin), ("散点大小", "Point size", self.point_spin),
            ("配色", "Palette", self.palette_combo), ("标注内容", "Annotation", self.annotation_style),
            ("标注哪些比较", "Visible comparisons", self.annotation_filter),
        ]:
            form.addRow(self.label(zh, en), control)
        layout.addLayout(form)
        layout.addWidget(self.points_check)
        layout.addWidget(self.means_check)
        self.annotations_list = QListWidget()
        self.annotations_list.setMaximumHeight(130)
        layout.addWidget(self.annotations_list)
        self.annotation_notice = QLabel()
        self.annotation_notice.setWordWrap(True)
        layout.addWidget(self.annotation_notice)
        layout.addWidget(self.label("显示名和颜色（不改变原始分组）", "Display labels and colors (original IDs preserved)"))
        self.labels_table = QTableWidget(0, 3)
        self.labels_table.setHorizontalHeaderLabels(["原始分组", "显示名", "颜色"])
        self.labels_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.labels_table.setMinimumHeight(160)
        self.labels_table.verticalHeader().hide()
        self.labels_table.itemChanged.connect(self.redraw_if_ready)
        self.labels_table.cellDoubleClicked.connect(self.choose_group_color)
        layout.addWidget(self.labels_table)
        note = self.label("显示筛选只影响图中括号，导出保留完整检验结果。\n双击颜色可选色；组的绘图顺序在“比较”页调整。", "Annotation filters only affect the plot; every test is exported.\nDouble-click a color to choose it. Reorder groups on the Compare page.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        for control in (self.title_edit, self.y_label_edit):
            control.editingFinished.connect(self.redraw_if_ready)
        for control in (self.font_spin, self.point_spin):
            control.valueChanged.connect(self.redraw_if_ready)
        self.palette_combo.currentTextChanged.connect(self.palette_changed)
        self.annotation_style.currentIndexChanged.connect(self.redraw_if_ready)
        self.annotation_filter.currentIndexChanged.connect(self.annotation_changed)
        self.annotations_list.itemChanged.connect(self.redraw_if_ready)
        self.points_check.toggled.connect(self.redraw_if_ready)
        self.means_check.toggled.connect(self.redraw_if_ready)

    def set_language(self, language):
        self.language = language
        for widget, zh, en in self._texts:
            widget.setText(self.tr_text(zh, en))
        for i, labels in enumerate([("1 数据", "1 Data"), ("2 比较", "2 Compare"), ("3 绘图", "3 Plot")]):
            self.controls.setTabText(i, self.tr_text(*labels))
        for i, labels in enumerate([("箱线图", "Boxplot"), ("检验结果", "Tests"), ("描述统计", "Summary")]):
            self.results_tabs.setTabText(i, self.tr_text(*labels))
        self.input_mode.setItemText(0, self.tr_text("分型表 + 表型表", "Genotypes + phenotypes"))
        self.input_mode.setItemText(1, self.tr_text("一张合并表", "One combined table"))
        for i, labels in enumerate([("所选组两两比较", "All selected pairs"), ("与参考单倍型比较", "Against reference"), ("自定义比较", "Custom pairs")]):
            self.comparison_combo.setItemText(i, self.tr_text(*labels))
        self.correction_combo.setItemText(1, self.tr_text("不校正（原始 P）", "None (raw P)"))
        self.scope_combo.setItemText(0, self.tr_text("每个性状分别校正", "Within each trait"))
        self.scope_combo.setItemText(1, self.tr_text("本次全部性状一起校正", "Across all traits in this run"))
        for combo, names in [(self.annotation_style, [("P 值 + 星号", "P + stars"), ("仅星号", "Stars only"), ("仅 P 值", "P only")]), (self.annotation_filter, [("全部比较", "All"), ("仅显著比较", "Significant only"), ("指定比较", "Selected comparisons"), ("不显示括号", "No brackets")])]:
            for i, labels in enumerate(names):
                combo.setItemText(i, self.tr_text(*labels))
        self.labels_table.setHorizontalHeaderLabels([self.tr_text("原始分组", "Original ID"), self.tr_text("显示名", "Display label"), self.tr_text("颜色", "Color")])
        self.update_paths()
        self.annotation_changed()
        self.update_status()

    def update_paths(self):
        for label, path, embedded in [(self.genotype_file_label, self.genotype_path, self.genotype_df is not None), (self.phenotype_file_label, self.phenotype_path, False)]:
            label.setText(Path(path).name if path else (self.tr_text("已接收当前分型结果", "Using current genotype results") if embedded else self.tr_text("未选择文件", "No file selected")))
            label.setToolTip(str(path or self.origin_path or ""))

    def choose_table(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr_text("选择数据表", "Choose table"), filter="Text tables (*.csv *.tsv *.txt);;All files (*)")
        return path or None

    def select_genotype_file(self):
        path = self.choose_table()
        if path:
            self.start_table_import(path, kind="genotype")

    def select_phenotype_file(self):
        path = self.choose_table()
        if path:
            self.start_table_import(path, kind="phenotype")

    def metadata_preferences(self):
        return dict(sample=self.sample_combo.currentText(), group=self.group_combo.currentText(),
                    phenotype_sample=self.phenotype_sample_combo.currentText(), traits=self.checked(self.trait_list))

    def start_table_import(self, path, kind="phenotype"):
        if self.worker is not None or kind not in {"genotype", "phenotype"}:
            return
        from haplotype_phenotype import read_table
        preferences, mode = self.metadata_preferences(), self.input_mode.currentData()
        genotype, phenotype = self.genotype_df, self.phenotype_df
        self._refresh_timer.stop()
        self._metadata_pending = False
        self._data_generation += 1
        generation = self._data_generation
        def read_and_prepare(progress, cancel_event):
            began = time.perf_counter()
            table = read_table(path, progress_callback=lambda value, text: progress(value * .7, text), cancel_event=cancel_event)
            read_seconds = time.perf_counter() - began
            if cancel_event.is_set():
                raise InterruptedError()
            progress(75, "正在匹配样本并检查表型… / Matching samples and checking traits…")
            prepared = input_metadata(table if kind == "genotype" else genotype,
                                      table if kind == "phenotype" else phenotype, mode, preferences)
            if cancel_event.is_set():
                raise InterruptedError()
            return dict(path=str(path), kind=kind, table=table, prepared=prepared, generation=generation,
                        read_seconds=read_seconds, prepare_seconds=time.perf_counter() - began - read_seconds)
        self.worker = AnalysisWorker(read_and_prepare, self)
        self._worker_kind = "import"
        self.worker.progress.connect(self.on_progress)
        self.worker.succeeded.connect(self._table_import_ready)
        self.worker.failed.connect(self.task_failed)
        self.worker.cancelled.connect(self.task_cancelled)
        self.worker.finished.connect(self.worker_finished)
        self.set_busy(True)
        self.progress.setValue(0)
        self.status_label.setText(self.tr_text("正在导入数据，界面保持响应…", "Importing data; the interface remains responsive…"))
        self.worker.start()

    def _table_import_ready(self, payload):
        if payload["generation"] != self._data_generation:
            return
        began = time.perf_counter()
        if payload["kind"] == "genotype":
            self.genotype_path, self.genotype_df = payload["path"], payload["table"]
            self.origin_path = None
            self._had_groups = False
        else:
            self.phenotype_path, self.phenotype_df = payload["path"], payload["table"]
        self._background_metadata = True
        self.populate_columns(prepared=payload["prepared"])
        self.import_diagnostics = dict(read_seconds=payload["read_seconds"], prepare_seconds=payload["prepare_seconds"],
            gui_commit_seconds=time.perf_counter() - began, rows=len(payload["table"]),
            pair_items=self.pair_list.count(), annotation_items=self.annotations_list.count(),
            table_import=payload["table"].attrs.get("table_import", {}))
        self.progress.setValue(100)
        self.update_status()
        details = payload["table"].attrs.get("table_import", {})
        if details.get("header") in {"none", "absent"}:
            self.status_label.setText(self.status_label.text() + "\n" + self.tr_text("已识别无表头表型表，首个样本已保留。", "Headerless phenotype table detected; the first sample was retained."))

    def _request_metadata_refresh(self):
        self.invalidate_analysis()
        self._data_generation += 1
        self._metadata_pending = True
        self.run_button.setEnabled(False)
        self._refresh_timer.start()

    def _start_metadata_refresh(self):
        if not self._metadata_pending or self.worker is not None:
            return
        generation = self._data_generation
        genotype, phenotype = self.genotype_df, self.phenotype_df
        mode, preferences = self.input_mode.currentData(), self.metadata_preferences()
        self._metadata_pending = False
        def prepare(progress, cancel_event):
            progress(10, "正在更新样本匹配与有效样本数… / Updating sample matches and counts…")
            result = input_metadata(genotype, phenotype, mode, preferences)
            if cancel_event.is_set():
                raise InterruptedError()
            return generation, result
        self.worker = AnalysisWorker(prepare, self)
        self._worker_kind = "metadata"
        self.worker.succeeded.connect(self._metadata_ready)
        self.worker.failed.connect(self.task_failed)
        self.worker.cancelled.connect(self.task_cancelled)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.worker_finished)
        self.set_busy(True)
        self.worker.start()

    def _metadata_ready(self, payload):
        generation, prepared = payload
        if generation == self._data_generation:
            self._apply_group_metadata(prepared)
            self.progress.setValue(100)

    def accept_genotypes(self, frame, origin=""):
        self.genotype_df = frame.copy()
        self.genotype_path = None
        self.origin_path = origin
        self.input_mode.setCurrentIndex(0)
        self._had_groups = False
        self.populate_columns()
        self.controls.setCurrentIndex(0)

    def fill_columns(self, combo, columns, preferred):
        previous = combo.currentText()
        combo.clear()
        combo.addItems(columns)
        if previous in columns:
            combo.setCurrentText(previous)
        else:
            lookup = {col.casefold(): col for col in columns}
            for name in preferred:
                if name.casefold() in lookup:
                    combo.setCurrentText(lookup[name.casefold()])
                    break

    def populate_columns(self, prepared=None):
        self._updating = True
        gen_cols = list(self.genotype_df.columns) if self.genotype_df is not None else []
        pheno_cols = list(self.phenotype_df.columns) if self.phenotype_df is not None else []
        samples = ["Sample", "sample_id", "Taxa", "IID", "Unnamed: 0"]
        self.fill_columns(self.sample_combo, gen_cols, samples)
        self.fill_columns(self.group_combo, gen_cols, ["Genotype", "Hap", "Haplotype", "Cluster"])
        self.fill_columns(self.phenotype_sample_combo, pheno_cols, samples)
        if prepared is not None:
            self.sample_combo.setCurrentText(prepared["sample"])
            self.group_combo.setCurrentText(prepared["group"])
            self.phenotype_sample_combo.setCurrentText(prepared["phenotype_sample"])
        self._updating = False
        self.rebuild_traits(prepared=prepared)
        self.update_paths()

    def input_mode_changed(self, *_):
        separate = self.input_mode.currentData() == "separate"
        for widget in (self.phenotype_button, self.phenotype_file_label, self.sample_label, self.sample_combo, self.phenotype_sample_label, self.phenotype_sample_combo):
            widget.setVisible(separate)
        if not self._updating:
            self._had_groups = False
            self.populate_columns()

    def columns_changed(self, *_):
        if not self._updating:
            self.rebuild_traits()

    def rebuild_traits(self, prepared=None):
        previous = set(self.checked(self.trait_list))
        source = self.genotype_df if self.input_mode.currentData() == "combined" else self.phenotype_df
        columns = list(source.columns) if source is not None else []
        exclude = {self.group_combo.currentText(), self.phenotype_sample_combo.currentText() if self.input_mode.currentData() == "separate" else self.combined_sample_column()}
        choices = [col for col in columns if col not in exclude and col.casefold() != "fid"]
        previous.intersection_update(choices)
        if prepared is not None:
            previous = set(prepared["selected_traits"])
        default = next((col for name in ["Phenotype", "FibLen_BLUP", "Yolo"] for col in choices if col.casefold() == name.casefold()), choices[0] if choices else None)
        self.trait_list.blockSignals(True)
        self.trait_list.clear()
        for col in choices:
            self.add_check_item(self.trait_list, col, col, col in previous if previous else col == default)
        self.trait_list.blockSignals(False)
        self.refresh_groups(prepared=prepared)

    def combined_sample_column(self):
        columns = self.genotype_df.columns if self.genotype_df is not None else []
        return next((col for col in columns if col.casefold() in {"sample", "sample_id", "taxa", "iid", "unnamed: 0"}), "Sample")

    def prepare_current_data(self, trait=None):
        from haplotype_phenotype import prepare_data
        traits = self.checked(self.trait_list)
        trait = trait or (traits[0] if traits else None)
        if self.genotype_df is None or not trait:
            raise ValueError(self.tr_text("请选择数据和至少一个性状。", "Choose data and at least one trait."))
        separate = self.input_mode.currentData() == "separate"
        if separate and self.phenotype_df is None:
            raise ValueError(self.tr_text("请选择表型表。", "Choose a phenotype table."))
        return prepare_data(self.genotype_df, self.phenotype_df if separate else None,
                            sample_col=self.sample_combo.currentText() if separate else self.combined_sample_column(),
                            group_col=self.group_combo.currentText(), trait_col=trait,
                            phenotype_sample_col=self.phenotype_sample_combo.currentText() if separate else None)

    @staticmethod
    def add_check_item(widget, text, value, checked=True):
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, value)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        widget.addItem(item)
        return item

    @staticmethod
    def checked(widget):
        return [widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(widget.count())
                if widget.item(i).checkState() == Qt.CheckState.Checked]

    def selected_groups(self):
        return self.checked(self.group_list)

    def check_all(self, widget, checked):
        widget.blockSignals(True)
        for i in range(widget.count()):
            widget.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        widget.blockSignals(False)
        self.traits_changed() if widget is self.trait_list else self.groups_changed()

    def traits_changed(self, *_):
        if not self._updating:
            self.refresh_groups()

    def refresh_groups(self, *_, prepared=None):
        if self._updating:
            return
        if prepared is not None:
            self._apply_group_metadata(prepared)
        elif self._background_metadata:
            self._request_metadata_refresh()
        else:
            self._apply_group_metadata(input_metadata(self.genotype_df, self.phenotype_df,
                                                     self.input_mode.currentData(), self.metadata_preferences()))

    def _apply_group_metadata(self, prepared):
        previous = set(self.selected_groups())
        order = [self.group_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.group_list.count())]
        self.invalidate_analysis()
        self._report, self._error = prepared["report"], prepared["error"]
        self.group_list.blockSignals(True)
        self.group_list.clear()
        try:
            counts, available = prepared["counts"], set(prepared["available"])
            groups = [group for group in order if group in available] + [group for group in sorted(available) if group not in order]
            for group in groups:
                self.add_check_item(self.group_list, f"{group}  ·  n={counts.get(group, 0)}", group,
                                    group in previous if self._had_groups else True)
            if not self._error:
                self._had_groups = True
        except Exception as exc:
            self._error = str(exc)
        finally:
            self.group_list.blockSignals(False)
        self.groups_changed()

    def groups_changed(self, *_):
        if self._updating:
            return
        reference = self.reference_combo.currentText()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItems(self.selected_groups())
        if reference in self.selected_groups():
            self.reference_combo.setCurrentText(reference)
        self.reference_combo.blockSignals(False)
        self.rebuild_pair_lists()
        self.rebuild_labels()
        self.invalidate_analysis()
        self.update_status()

    def move_group(self, direction):
        row = self.group_list.currentRow()
        new_row = row + direction
        if row >= 0 and 0 <= new_row < self.group_list.count():
            item = self.group_list.takeItem(row)
            self.group_list.insertItem(new_row, item)
            self.group_list.setCurrentRow(new_row)
            # Order changes presentation and the sign convention of pairwise differences.
            self.groups_changed()

    def mode_changed(self, *_):
        reference = self.comparison_combo.currentData() == "reference"
        custom = self.comparison_combo.currentData() == "custom"
        self.reference_label.setVisible(reference)
        self.reference_combo.setVisible(reference)
        self.pair_list.setVisible(custom)
        self.pairs_label.setVisible(custom)
        if not self._updating:
            self.rebuild_pair_lists()
            self.invalidate_analysis()

    def rebuild_pair_lists(self):
        groups = self.selected_groups()
        reference = self.reference_combo.currentText()
        mode = self.comparison_combo.currentData()
        pair_count = len(groups) * max(0, len(groups) - 1) // 2
        for widget, cache, needed, notice in [
            (self.pair_list, self._pair_choices, mode == "custom", self.pair_notice),
            (self.annotations_list, self._annotation_choices, self.annotation_filter.currentData() == "custom", self.annotation_notice),
        ]:
            for i in range(widget.count()):
                item = widget.item(i)
                cache[frozenset(item.data(Qt.ItemDataRole.UserRole))] = item.checkState() == Qt.CheckState.Checked
            count = pair_count
            candidates = None
            if widget is self.annotations_list and mode == "custom":
                candidates = self.checked(self.pair_list)
                count = len(candidates)
            elif mode == "reference":
                count = max(0, len(groups) - 1)
            widget.blockSignals(True)
            widget.setUpdatesEnabled(False)
            widget.clear()
            notice.clear()
            if needed and count > MAX_VISIBLE_PAIRS:
                notice.setText(self.tr_text(
                    f"共有 {count:,} 个候选比较，暂不展开列表（上限 {MAX_VISIBLE_PAIRS:,}）。请减少分组或使用参考组比较。实际检验不会被静默截断。",
                    f"{count:,} candidate pairs exceed the visible-list limit ({MAX_VISIBLE_PAIRS:,}). Select fewer groups or use a reference. Tests are never silently truncated."))
            elif needed:
                if candidates is None:
                    candidates = [(reference, group) for group in groups if group != reference] if mode == "reference" else combinations(groups, 2)
                for pair in candidates:
                    self.add_check_item(widget, f"{pair[0]}  vs  {pair[1]}", pair, cache.get(frozenset(pair), True))
            widget.blockSignals(False)
            widget.setUpdatesEnabled(True)

    def test_pairs_changed(self, *_):
        if not self._updating:
            self.rebuild_pair_lists()
            self.invalidate_analysis()

    def group_styles(self):
        labels, colors = {}, {}
        for row in range(self.labels_table.rowCount()):
            group = self.labels_table.item(row, 0).text()
            labels[group] = self.labels_table.item(row, 1).text().strip() or group
            colors[group] = self.labels_table.item(row, 2).text().strip()
        return labels, colors

    def rebuild_labels(self, reset_colors=False):
        from matplotlib import colormaps
        old_labels, old_colors = self.group_styles()
        groups = self.selected_groups()
        cmap = colormaps[self.palette_combo.currentText()]
        from matplotlib.colors import to_hex
        self.labels_table.blockSignals(True)
        self.labels_table.setRowCount(len(groups))
        for row, group in enumerate(groups):
            original = QTableWidgetItem(group)
            original.setFlags(original.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.labels_table.setItem(row, 0, original)
            self.labels_table.setItem(row, 1, QTableWidgetItem(old_labels.get(group, group)))
            color = to_hex(cmap(row % cmap.N if cmap.N <= 20 else row / max(len(groups) - 1, 1)))
            self.labels_table.setItem(row, 2, QTableWidgetItem(color if reset_colors else old_colors.get(group, color)))
        self.labels_table.blockSignals(False)

    def choose_group_color(self, row, column):
        if column == 2:
            color = QColorDialog.getColor(parent=self)
            if color.isValid():
                self.labels_table.item(row, column).setText(color.name())

    def palette_changed(self, *_):
        self.rebuild_labels(reset_colors=True)
        self.redraw_if_ready()

    def annotation_changed(self, *_):
        self.annotations_list.setVisible(self.annotation_filter.currentData() == "custom")
        if not self._updating:
            self.rebuild_pair_lists()
        self.redraw_if_ready()

    def plot_options(self):
        labels, colors = self.group_styles()
        annotation_filter = self.annotation_filter.currentData()
        if annotation_filter == "custom" and not self.checked(self.annotations_list):
            annotation_filter = "none"
        return dict(group_labels=labels, group_colors=colors, group_order=self.selected_groups(),
                    y_label=self.y_label_edit.text().strip() or None, title=self.title_edit.text().strip() or None,
                    font_size=self.font_spin.value(), point_size=self.point_spin.value(),
                    show_points=self.points_check.isChecked(), show_means=self.means_check.isChecked(),
                    annotation_style=self.annotation_style.currentData(), annotation_filter=annotation_filter,
                    annotation_pairs=self.checked(self.annotations_list))

    def clear_plot(self):
        if self.toolbar is not None:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.figure.clear()
            self.canvas.deleteLater()
            self.canvas = None
        self.placeholder.show()

    def invalidate_analysis(self, *_):
        if self._updating:
            return
        self.analysis = None
        self.clear_plot()
        self.result_trait_combo.blockSignals(True)
        self.result_trait_combo.clear()
        self.result_trait_combo.blockSignals(False)
        self.test_table.setRowCount(0)
        self.summary_table.setRowCount(0)
        self.export_button.setEnabled(False) if hasattr(self, "export_button") else None
        self.result_note.setText("") if hasattr(self, "result_note") else None

    def redraw_if_ready(self, *_):
        if not self._updating and self.analysis is not None:
            self.display_result()

    def analysis_parameters(self):
        separate = self.input_mode.currentData() == "separate"
        return dict(traits=self.checked(self.trait_list), selected_groups=self.selected_groups(),
                    sample_col=self.sample_combo.currentText() if separate else self.combined_sample_column(),
                    group_col=self.group_combo.currentText(),
                    phenotype_sample_col=self.phenotype_sample_combo.currentText() if separate else None,
                    mode=self.comparison_combo.currentData(), reference=self.reference_combo.currentText(),
                    pairs=self.checked(self.pair_list), correction=self.correction_combo.currentData(),
                    correction_scope=self.scope_combo.currentData())

    def run_analysis(self, *_):
        from haplotype_phenotype import analyze_traits
        if self.is_busy() or self._metadata_pending:
            return
        try:
            parameters = self.analysis_parameters()
            if not parameters["traits"] or len(parameters["selected_groups"]) < 2:
                raise ValueError(self.tr_text("至少选择一个性状和两个单倍型。", "Select at least one trait and two haplotypes."))
            if parameters["mode"] == "custom" and not parameters["pairs"]:
                raise ValueError(self.tr_text("请勾选自定义比较；候选过多时先减少分组或选择参考组。", "Choose custom comparisons; reduce groups or use a reference if the candidate list is too large."))
            if self.genotype_df is None or (self.input_mode.currentData() == "separate" and self.phenotype_df is None):
                raise ValueError(self.tr_text("请先导入分型和表型数据。", "Import genotype and phenotype data first."))
            # Controls are disabled while the worker reads these immutable inputs;
            # cleaning and all dataframe copies belong to the worker, not Qt.
            genotype = self.genotype_df
            phenotype = self.phenotype_df if self.input_mode.currentData() == "separate" else None
        except Exception as exc:
            self.show_error(exc)
            return
        self.invalidate_analysis()
        def compute(progress, cancel_event):
            return analyze_traits(genotype, phenotype, **parameters,
                                  progress_callback=lambda done, total, trait: progress(done * 100 / max(total, 1), str(trait)),
                                  cancel_event=cancel_event)
        self.worker = AnalysisWorker(compute, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.succeeded.connect(self.analysis_finished)
        self.worker.failed.connect(self.task_failed)
        self.worker.cancelled.connect(self.task_cancelled)
        self.worker.finished.connect(self.worker_finished)
        self.set_busy(True)
        self.progress.setValue(0)
        self.status_label.setText(self.tr_text("正在分析…", "Analyzing…"))
        self.worker.start()

    def analysis_finished(self, result):
        self.analysis = result
        self.result_trait_combo.blockSignals(True)
        self.result_trait_combo.clear()
        self.result_trait_combo.addItems(list(result["analyses"]))
        self.result_trait_combo.blockSignals(False)
        self.progress.setValue(100)
        self.display_result()
        count, failures = len(result["analyses"]), result["failures"]
        self.status_label.setText(self.tr_text(f"完成 {count} 个性状；失败 {len(failures)} 个。", f"Completed {count} traits; failed {len(failures)}.") + ("\n" + "\n".join(f"{trait}: {reason}" for trait, reason in failures.items()) if failures else ""))
        if not count:
            self.message.emit(self.status_label.text())

    def display_result(self, *_):
        if self.analysis is None:
            return
        trait = self.result_trait_combo.currentText()
        if trait not in self.analysis["analyses"]:
            return
        from haplotype_phenotype import plot_comparison
        result = self.analysis["analyses"][trait]
        try:
            figure = plot_comparison(result["data"], self.selected_groups(), result["results"],
                                     trait_label=trait, correction=self.correction_combo.currentData(), **self.plot_options())
            self.clear_plot()
            self.canvas = ComparisonCanvas(figure)
            self.toolbar = PlotNavigationToolbar(self.canvas, self)
            self.plot_layout.insertWidget(0, self.toolbar)
            self.plot_layout.addWidget(self.canvas, 1)
            self.placeholder.hide()
            self.canvas.draw()
            populate_table(self.test_table, result["results"])
            populate_table(self.summary_table, result["summary"])
            self.result_note.setText(self.tr_text(f"{len(result['results'])} 个比较 · 逐比较 95% CI（未校正）", f"{len(result['results'])} comparisons · individual 95% CI (unadjusted)"))
        except Exception as exc:
            self.status_label.setText(str(exc))
            self.message.emit(str(exc))

    def update_status(self):
        if not hasattr(self, "status_label") or self.is_busy():
            return
        if self._error:
            self.status_label.setText(self._error)
        elif self._report:
            r = self._report
            self.status_label.setText(self.tr_text(
                f"首个性状：匹配 {r['matched_rows']}，有效 {r['valid_rows']}，剔除 {r['dropped_rows']}；未匹配分型样本 {r['unmatched_genotype_samples']}。\n已选 {len(self.selected_groups())} 个单倍型、{len(self.checked(self.trait_list))} 个性状。",
                f"First trait: matched {r['matched_rows']}, valid {r['valid_rows']}, dropped {r['dropped_rows']}; unmatched genotype samples {r['unmatched_genotype_samples']}.\nSelected {len(self.selected_groups())} haplotypes and {len(self.checked(self.trait_list))} traits."))
        else:
            self.status_label.setText(self.tr_text("从“数据”页开始。", "Start on the Data page."))

    def is_busy(self):
        return self.worker is not None

    def set_busy(self, busy):
        self.controls.setEnabled(not busy)
        for control in (self.load_button, self.project_button):
            control.setEnabled(not busy)
        self.run_button.setEnabled(not busy and not self._metadata_pending)
        self.cancel_button.setEnabled(busy)
        self.result_trait_combo.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.analysis is not None and bool(self.analysis["analyses"]))

    def on_progress(self, value, message):
        self.progress.setValue(value)
        self.status_label.setText(message)

    def cancel_task(self, *_):
        self._refresh_timer.stop()
        self._metadata_pending = False
        self._data_generation += 1
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText(self.tr_text("正在取消，等待当前计算完成…", "Cancelling after the current calculation…"))

    def begin_shutdown(self):
        self._refresh_timer.stop()
        self._metadata_pending = False
        if self.is_busy():
            self.cancel_task()
            return False
        return True

    def worker_finished(self):
        kind = self._worker_kind
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        self.set_busy(False)
        self._worker_kind = None
        if self._metadata_pending:
            QTimer.singleShot(0, self._start_metadata_refresh)
        elif kind in {"import", "metadata"} and self.progress.value() == 100:
            self.update_status()
            if kind == "import" and self.import_diagnostics.get("table_import", {}).get("header") == "none":
                self.status_label.setText(self.status_label.text() + "\n" + self.tr_text("已识别无表头表型表，首个样本已保留。", "Headerless phenotype table detected; the first sample was retained."))

    def task_failed(self, reason):
        self.status_label.setText(reason)
        self.message.emit(reason)

    def task_cancelled(self):
        self.status_label.setText(self.tr_text("已取消；未导出未完成结果。", "Cancelled; unfinished results were not exported."))

    def show_error(self, exc):
        QMessageBox.warning(self, self.tr_text("检查输入", "Check input"), str(exc))

    def project_settings(self):
        return {
            "inputs": {"mode": self.input_mode.currentData(),
                       "genotype_file": self.genotype_path, "phenotype_file": self.phenotype_path,
                       "embedded_genotypes": self.genotype_df.to_dict("records")
                       if self.genotype_df is not None and self.genotype_path is None else None,
                       "origin": self.origin_path},
            "columns": {"sample": self.sample_combo.currentText(), "group": self.group_combo.currentText(),
                        "phenotype_sample": self.phenotype_sample_combo.currentText()},
            "analysis": self.analysis_parameters(),
            "group_order": [self.group_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.group_list.count())],
            "plot": dict(self.plot_options(), palette=self.palette_combo.currentText()),
            "software_version": "1.8.3",
        }

    def save_project_dialog(self, *_):
        from project_state import save_project
        path, _ = QFileDialog.getSaveFileName(self, self.tr_text("保存分析项目", "Save analysis project"),
                                            "analysis.fgproject.json", "FragGenotyper project (*.json)")
        if not path:
            return
        try:
            save_project(path, self.project_settings())
            self.project_hint.setText(Path(path).name)
            self.message.emit(self.tr_text("项目已保存", "Project saved"))
        except Exception as exc:
            self.show_error(exc)

    def open_project(self, *_):
        from project_state import load_project
        path, _ = QFileDialog.getOpenFileName(self, self.tr_text("打开分析项目", "Open analysis project"), filter="FragGenotyper project (*.json)")
        if not path:
            return
        try:
            self.apply_project(load_project(path))
            self.project_hint.setText(Path(path).name)
        except Exception as exc:
            self.show_error(exc)

    def apply_project(self, state):
        from haplotype_phenotype import read_table, prepare_data
        inputs, columns, params, plot = (state.get(key, {}) for key in ("inputs", "columns", "analysis", "plot"))
        mode = inputs.get("mode", "separate")
        if mode not in {"separate", "combined"}:
            raise ValueError("Unknown project input mode.")
        gen_path, pheno_path = inputs.get("genotype_file"), inputs.get("phenotype_file")
        genotype = read_table(gen_path) if gen_path else pd.DataFrame(inputs.get("embedded_genotypes") or [])
        phenotype = read_table(pheno_path) if mode == "separate" and pheno_path else None
        if mode == "separate" and phenotype is None:
            raise ValueError(self.tr_text("项目的表型文件不可用，请重新选择。", "The project's phenotype file is unavailable; choose it again."))
        traits = params.get("traits", [])
        if not traits:
            raise ValueError("The project does not select any trait.")
        # Validate file/column mappings before touching the currently open project.
        prepare_data(genotype, phenotype,
                     sample_col=params.get("sample_col", columns.get("sample", "Sample")),
                     group_col=columns.get("group", "Genotype"), trait_col=traits[0],
                     phenotype_sample_col=columns.get("phenotype_sample", "Sample"))
        source = genotype if mode == "combined" else phenotype
        if any(trait not in source.columns for trait in traits):
            raise ValueError("One or more saved traits no longer exist in the source table.")
        background_metadata = self._background_metadata
        self._background_metadata = False
        self._refresh_timer.stop()
        self._metadata_pending = False
        self.invalidate_analysis()
        self._updating = True
        self.genotype_path, self.phenotype_path = gen_path, pheno_path
        self.genotype_df, self.phenotype_df = genotype, phenotype
        self.origin_path = inputs.get("origin")
        self.input_mode.setCurrentIndex(self.input_mode.findData(mode))
        self._updating = False
        self.input_mode_changed()
        self._updating = True
        self.sample_combo.setCurrentText(columns.get("sample", "Sample"))
        self.group_combo.setCurrentText(columns.get("group", "Genotype"))
        self.phenotype_sample_combo.setCurrentText(columns.get("phenotype_sample", "Sample"))
        self._updating = False
        self.rebuild_traits()
        self.trait_list.blockSignals(True)
        for i in range(self.trait_list.count()):
            item = self.trait_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in traits else Qt.CheckState.Unchecked)
        self.trait_list.blockSignals(False)
        self._had_groups = False
        self.refresh_groups()
        desired_order = state.get("group_order", [])
        for target, group in enumerate(desired_order):
            for row in range(self.group_list.count()):
                if self.group_list.item(row).data(Qt.ItemDataRole.UserRole) == group:
                    self.group_list.insertItem(target, self.group_list.takeItem(row))
                    break
        selected = params.get("selected_groups", [])
        self.group_list.blockSignals(True)
        for i in range(self.group_list.count()):
            item = self.group_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in selected else Qt.CheckState.Unchecked)
        self.group_list.blockSignals(False)
        self.groups_changed()
        self._updating = True
        for combo, key, default in [(self.comparison_combo, "mode", "pairwise"), (self.correction_combo, "correction", "bh"), (self.scope_combo, "correction_scope", "per_trait")]:
            combo.setCurrentIndex(max(0, combo.findData(params.get(key, default))))
        self.reference_combo.setCurrentText(params.get("reference") or "")
        self._updating = False
        self.mode_changed()
        saved_pairs = {frozenset(pair) for pair in params.get("pairs", [])}
        self.pair_list.blockSignals(True)
        for i in range(self.pair_list.count()):
            item = self.pair_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if frozenset(item.data(Qt.ItemDataRole.UserRole)) in saved_pairs else Qt.CheckState.Unchecked)
        self.pair_list.blockSignals(False)
        self._updating = True
        self.title_edit.setText(plot.get("title") or "")
        self.y_label_edit.setText(plot.get("y_label") or "")
        self.font_spin.setValue(int(plot.get("font_size", 11)))
        self.point_spin.setValue(int(plot.get("point_size", 25)))
        self.points_check.setChecked(bool(plot.get("show_points", True)))
        self.means_check.setChecked(bool(plot.get("show_means", True)))
        self.palette_combo.setCurrentText(plot.get("palette", "tab10"))
        self.annotation_style.setCurrentIndex(max(0, self.annotation_style.findData(plot.get("annotation_style", "both"))))
        self.annotation_filter.setCurrentIndex(max(0, self.annotation_filter.findData(plot.get("annotation_filter", "all"))))
        self.rebuild_pair_lists()
        self.rebuild_labels()
        self.labels_table.blockSignals(True)
        for row in range(self.labels_table.rowCount()):
            group = self.labels_table.item(row, 0).text()
            self.labels_table.item(row, 1).setText(plot.get("group_labels", {}).get(group, group))
            self.labels_table.item(row, 2).setText(plot.get("group_colors", {}).get(group, self.labels_table.item(row, 2).text()))
        self.labels_table.blockSignals(False)
        saved_annotations = {frozenset(pair) for pair in plot.get("annotation_pairs", [])}
        self.annotations_list.blockSignals(True)
        for i in range(self.annotations_list.count()):
            item = self.annotations_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if frozenset(item.data(Qt.ItemDataRole.UserRole)) in saved_annotations else Qt.CheckState.Unchecked)
        self.annotations_list.blockSignals(False)
        self._updating = False
        self._background_metadata = background_metadata
        self.annotation_changed()
        self.update_paths()
        self.update_status()
        available = {self.group_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.group_list.count())}
        missing = set(selected) - available
        self.status_label.setText(self.status_label.text() + "\n" + (self.tr_text(f"源数据中已不存在分组：{', '.join(sorted(missing))}；请重新选择。", f"Groups missing from source data: {', '.join(sorted(missing))}; review selection.") if missing else self.tr_text("项目已恢复，请重新运行分析。", "Project restored; run the analysis again.")))

    def export_analysis(self, *_):
        if self.is_busy() or not self.analysis or not self.analysis["analyses"]:
            return
        parent = QFileDialog.getExistingDirectory(self, self.tr_text("选择导出目录", "Choose output folder"))
        if not parent:
            return
        parent = Path(parent)
        target = parent / "FragGenotyper_analysis"
        number = 2
        while target.exists():
            target = parent / f"FragGenotyper_analysis_{number}"
            number += 1
        analysis, options, state = self.analysis, self.plot_options(), self.project_settings()
        selected, correction = self.selected_groups(), self.correction_combo.currentData()
        def export(progress, cancel_event):
            import os
            import shutil
            import tempfile
            from haplotype_phenotype import plot_comparison
            staging = Path(tempfile.mkdtemp(prefix=".fraggenotyper-export-", dir=parent))
            try:
                manifest, frames = {}, []
                for index, (trait, result) in enumerate(analysis["analyses"].items(), 1):
                    if cancel_event.is_set():
                        raise InterruptedError()
                    safe_trait = re.sub(r"[^\w.-]+", "_", trait, flags=re.UNICODE).strip("._")[:60] or "trait"
                    stem = f"{index:03d}_{safe_trait}"
                    manifest[trait] = stem
                    result["results"].to_csv(staging / f"{stem}_tests.csv", index=False)
                    result["summary"].to_csv(staging / f"{stem}_summary.csv", index=False)
                    result["data"].to_csv(staging / f"{stem}_data.csv", index=False)
                    frames.append(result["results"].assign(Trait=trait))
                    figure = plot_comparison(result["data"], selected, result["results"], trait_label=trait,
                                             correction=correction, **options)
                    figure.savefig(staging / f"{stem}.png", dpi=300, bbox_inches="tight")
                    figure.savefig(staging / f"{stem}.pdf", bbox_inches="tight")
                    figure.clear()
                    progress(index * 100 / len(analysis["analyses"]), trait)
                if cancel_event.is_set():
                    raise InterruptedError()
                pd.concat(frames, ignore_index=True).to_csv(staging / "all_tests.csv", index=False)
                metadata = dict(state, trait_files=manifest, failures=analysis["failures"],
                                cleaning_reports={trait: result["report"] for trait, result in analysis["analyses"].items()},
                                ci_note="Unadjusted, per-comparison 95% Welch confidence intervals; not simultaneous intervals.",
                                test="Two-sided Welch t-test")
                (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                if target.exists():
                    raise FileExistsError(f"Output folder already exists: {target}")
                os.rename(staging, target)
                return str(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        self.worker = AnalysisWorker(export, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.succeeded.connect(self.export_finished)
        self.worker.failed.connect(self.task_failed)
        self.worker.cancelled.connect(self.task_cancelled)
        self.worker.finished.connect(self.worker_finished)
        self.set_busy(True)
        self.progress.setValue(0)
        self.status_label.setText(self.tr_text("正在导出图片和结果…", "Exporting figures and tables…"))
        self.worker.start()

    def export_finished(self, path):
        self.progress.setValue(100)
        self.status_label.setText(self.tr_text("已导出到：", "Exported to: ") + path)
        self.message.emit(self.status_label.text())
