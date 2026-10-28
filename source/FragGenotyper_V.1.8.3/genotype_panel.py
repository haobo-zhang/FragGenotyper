"""Embedded genotype/QC panels; all computation runs in cooperative workers."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QSplitter, QPlainTextEdit,
)

from genotype_core import (
    read_variants, similarity_groups, kmeans_groups, search_k, groups_frame,
    matrix_frame, newick_tree, filter_vcf, prepare_kmeans,
)
from plot_toolbar import PlotNavigationToolbar
from workers import AnalysisWorker
from genotype_plot import draw_genotype_heatmap
from genotype_export import export_genotype_plots


def _save_dialog(parent, title, suggestion, filters):
    return QFileDialog.getSaveFileName(parent, title, suggestion, filters)[0]


def _write_json(path, report):
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class _WorkerPanel(QWidget):
    """Keep QThreads alive until finished; closing only requests cancellation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._busy_controls = []

    def is_busy(self):
        return self.worker is not None

    def begin_shutdown(self):
        if self.is_busy():
            self.cancel_analysis()
            return False
        return True

    def cancel_analysis(self):
        if self.worker is not None:
            self.worker.cancel()
            self.status_label.setText(self._tr("正在请求取消，等待当前计算安全结束…", "Cancellation requested; waiting for the current calculation…"))

    def _tr(self, chinese, english):
        return chinese if self.language == "zh" else english

    def _submit(self, function, completed):
        if self.is_busy():
            return
        for control in self._busy_controls:
            control.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(function)
        self.worker.progress.connect(self._progress)
        def deliver(result):
            try:
                completed(result)
            except Exception as exc:
                self._failed(str(exc))
        self.worker.succeeded.connect(deliver)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(lambda: self.status_label.setText(self._tr("已取消；原文件未改动。", "Cancelled; input files are unchanged.")))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _failed(self, message):
        self.status_label.setText(self._tr("计算失败：", "Analysis failed: ") + message)
        QMessageBox.warning(self, self._tr("无法完成分析", "Analysis could not finish"), message)

    def _finished(self):
        previous = self.worker
        self.worker = None
        for control in self._busy_controls:
            control.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if previous is not None:
            previous.deleteLater()

    def closeEvent(self, event):
        if self.begin_shutdown():
            event.accept()
        else:
            event.ignore()


class GenotypePanel(_WorkerPanel):
    genotypes_ready = pyqtSignal(object, str)

    def __init__(self, parent=None, language="zh"):
        super().__init__(parent)
        self.language = language
        self.data = None
        self.groups = None
        self.genotype_result = None
        self.source_path = ""
        self.parameters = {}
        self.k_result = None
        self.last_plot_export = None
        root = QVBoxLayout(self)
        self.file_box = QGroupBox()
        file_layout = QHBoxLayout(self.file_box)
        self.source_edit = QLineEdit()
        self.browse_button = QPushButton()
        self.browse_button.clicked.connect(self.choose_file)
        file_layout.addWidget(self.source_edit, 1)
        file_layout.addWidget(self.browse_button)
        root.addWidget(self.file_box)

        self.options_box = QGroupBox()
        option_layout = QHBoxLayout(self.options_box)
        form = QGridLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["", "", ""])
        self.method_label, self.threshold_label = QLabel(), QLabel()
        self.common_label, self.k_label, self.max_k_label = QLabel(), QLabel(), QLabel()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.9)
        self.common_spin = QSpinBox()
        self.common_spin.setRange(1, 1_000_000_000)
        self.common_spin.setValue(1)
        self.k_spin, self.max_k_spin = QSpinBox(), QSpinBox()
        for widget, value in [(self.k_spin, 2), (self.max_k_spin, 20)]:
            widget.setRange(1, 100000)
            widget.setValue(value)
        for index, (label, widget) in enumerate([(self.method_label, self.method_combo), (self.threshold_label, self.threshold_spin),
                                               (self.common_label, self.common_spin), (self.k_label, self.k_spin), (self.max_k_label, self.max_k_spin)]):
            row, column = index % 3, index // 3 * 2
            form.addWidget(label, row, column)
            form.addWidget(widget, row, column + 1)
        form.setColumnStretch(1, 2)
        form.setColumnStretch(3, 1)
        option_layout.addLayout(form, 3)
        action_layout = QVBoxLayout()
        self.method_note = QLabel()
        self.method_note.setWordWrap(True)
        self.method_note.setMinimumWidth(250)
        self.run_button, self.auto_k_button, self.cancel_button = QPushButton(), QPushButton(), QPushButton()
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_analysis)
        self.auto_k_button.clicked.connect(self.search_k)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)
        action_layout.addWidget(self.method_note)
        action_layout.addStretch()
        row = QHBoxLayout()
        for button in [self.run_button, self.auto_k_button, self.cancel_button]:
            row.addWidget(button)
        action_layout.addLayout(row)
        option_layout.addLayout(action_layout, 2)
        root.addWidget(self.options_box)
        self.method_combo.currentIndexChanged.connect(self._method_changed)

        self.progress_bar, self.status_label = QProgressBar(), QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.progress_bar)
        root.addWidget(self.status_label)
        self.tabs = QTabWidget()
        group_page, group_layout = QWidget(), QHBoxLayout()
        group_page.setLayout(group_layout)
        group_selector = QWidget()
        group_selector.setMaximumWidth(280)
        selector_layout = QVBoxLayout(group_selector)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        self.selection_note = QLabel()
        self.selection_note.setWordWrap(True)
        selector_layout.addWidget(self.selection_note)
        selection_buttons = QHBoxLayout()
        self.select_all_button, self.select_none_button = QPushButton(), QPushButton()
        self.select_all_button.clicked.connect(lambda: self.select_plot_groups(True))
        self.select_none_button.clicked.connect(lambda: self.select_plot_groups(False))
        selection_buttons.addWidget(self.select_all_button)
        selection_buttons.addWidget(self.select_none_button)
        selector_layout.addLayout(selection_buttons)
        self.group_table = QTableWidget(0, 2)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.group_table.itemChanged.connect(self._plot_selection_changed)
        selector_layout.addWidget(self.group_table, 1)
        self.selection_summary = QLabel()
        selector_layout.addWidget(self.selection_summary)
        plot_options = QFormLayout()
        self.label_side_label, self.plot_format_label = QLabel(), QLabel()
        self.label_side_combo, self.plot_format_combo = QComboBox(), QComboBox()
        self.label_side_combo.addItem("", "right")
        self.label_side_combo.addItem("", "left")
        self.plot_format_combo.addItem("PNG · 300 dpi", "png")
        self.plot_format_combo.addItem("PDF", "pdf")
        self.label_side_combo.currentIndexChanged.connect(self._plot_selection_changed)
        plot_options.addRow(self.label_side_label, self.label_side_combo)
        plot_options.addRow(self.plot_format_label, self.plot_format_combo)
        selector_layout.addLayout(plot_options)
        self.export_selected_button = QPushButton()
        self.export_selected_button.clicked.connect(lambda: self._export_clicked(self.export_selected_images))
        selector_layout.addWidget(self.export_selected_button)
        self.heatmap_figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_figure)
        heatmap_holder = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_holder)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.addWidget(PlotNavigationToolbar(self.heatmap_canvas, self))
        heatmap_layout.addWidget(self.heatmap_canvas, 1)
        group_layout.addWidget(group_selector)
        group_layout.addWidget(heatmap_holder, 1)
        self.tabs.addTab(group_page, "")
        k_page = QWidget()
        k_layout = QHBoxLayout(k_page)
        self.k_table = QTableWidget(0, 4)
        self.k_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.k_table.setMaximumWidth(370)
        self.k_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.k_figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.k_canvas = FigureCanvasQTAgg(self.k_figure)
        k_holder = QWidget()
        k_holder_layout = QVBoxLayout(k_holder)
        k_holder_layout.setContentsMargins(0, 0, 0, 0)
        k_holder_layout.addWidget(PlotNavigationToolbar(self.k_canvas, self))
        k_holder_layout.addWidget(self.k_canvas, 1)
        k_layout.addWidget(self.k_table)
        k_layout.addWidget(k_holder, 1)
        self.tabs.addTab(k_page, "")
        self.tabs.currentChanged.connect(lambda *_: self._update_actions())
        root.addWidget(self.tabs, 1)
        export_layout = QHBoxLayout()
        self.export_group_button, self.export_matrix_button = QPushButton(), QPushButton()
        self.export_image_button, self.export_tree_button = QPushButton(), QPushButton()
        self.export_qc_button, self.export_k_button = QPushButton(), QPushButton()
        for button, method in [(self.export_group_button, self.export_groups), (self.export_matrix_button, self.export_matrix),
                               (self.export_image_button, self.export_image), (self.export_tree_button, self.export_tree),
                               (self.export_qc_button, self.export_qc), (self.export_k_button, self.export_k_table)]:
            button.clicked.connect(lambda checked=False, handler=method: self._export_clicked(handler))
            export_layout.addWidget(button)
        root.addLayout(export_layout)
        self._busy_controls = [self.source_edit, self.browse_button, self.method_combo, self.threshold_spin,
                               self.common_spin, self.k_spin, self.max_k_spin, self.run_button, self.auto_k_button,
                               self.export_group_button, self.export_matrix_button, self.export_image_button,
                               self.export_tree_button, self.export_qc_button, self.export_k_button,
                               self.group_table, self.select_all_button, self.select_none_button,
                               self.label_side_combo, self.plot_format_combo, self.export_selected_button]
        self.set_language(language)
        self.status_label.setText(self._tr("选择 VCF/VCF.GZ 或 CHROM POS REF ALT + GT 表。", "Choose VCF/VCF.GZ or a CHROM POS REF ALT + GT table."))
        self.source_edit.textChanged.connect(self._source_changed)

    def set_language(self, language):
        self.language = "zh" if language == "zh" else "en"
        entries = [
            (self.file_box.setTitle, "变异文件", "Variant input"), (self.browse_button.setText, "选择文件", "Browse"),
            (self.options_box.setTitle, "分型设置", "Genotyping settings"),
            (self.method_label.setText, "方法", "Method"), (self.threshold_label.setText, "相似度阈值", "Similarity threshold"),
            (self.common_label.setText, "最少共同有效 SNP", "Minimum shared SNPs"),
            (self.k_label.setText, "K-means 的 K", "K-means K"), (self.max_k_label.setText, "自动选 K 上限", "Search K up to"),
            (self.run_button.setText, "开始分型", "Run genotyping"), (self.auto_k_button.setText, "评估并推荐 K", "Evaluate K"),
            (self.cancel_button.setText, "取消", "Cancel"), (self.export_group_button.setText, "导出分组 CSV", "Groups CSV"),
            (self.export_matrix_button.setText, "导出矩阵 CSV", "Matrix CSV"), (self.export_image_button.setText, "导出当前图", "Export plot"),
            (self.export_tree_button.setText, "导出聚类树 .tre", "Clustering tree .tre"), (self.export_qc_button.setText, "导出 QC JSON", "QC JSON"),
            (self.export_k_button.setText, "导出 K 候选表", "K candidates CSV"),
            (self.selection_note.setText, "勾选要展示和导出的单倍型", "Select groups to display and export"),
            (self.select_all_button.setText, "全选", "Select all"),
            (self.select_none_button.setText, "全不选", "Clear selection"),
            (self.label_side_label.setText, "名称位置", "Label side"),
            (self.plot_format_label.setText, "单独图格式", "Individual format"),
            (self.export_selected_button.setText, "分别导出所选单倍型图", "Export selected groups individually"),
        ]
        for setter, chinese, english in entries:
            setter(self._tr(chinese, english))
        for index, (chinese, english) in enumerate([("矩阵相似度（首样本为中心）", "Similarity: seed grouping"),
                                                  ("逐一相似度（任一成员匹配）", "Similarity: any-member grouping"),
                                                  ("K-means", "K-means")]):
            self.method_combo.setItemText(index, self._tr(chinese, english))
        self.tabs.setTabText(0, self._tr("分组与热图", "Groups and heatmap"))
        self.tabs.setTabText(1, self._tr("K 候选与诊断", "K candidates and diagnostics"))
        self.group_table.setHorizontalHeaderLabels([self._tr("单倍型 / 分组", "Group"), self._tr("样本数", "Samples")])
        self.label_side_combo.setItemText(0, self._tr("右侧", "Right"))
        self.label_side_combo.setItemText(1, self._tr("左侧", "Left"))
        self.k_table.setHorizontalHeaderLabels(["K", "SSE", "Silhouette", self._tr("实际簇数", "Clusters")])
        self._method_changed()
        self._update_selection_summary()
        if self.data is not None and self.groups is not None and not self.is_busy():
            self._draw_heatmap()

    def _method_changed(self, *_):
        is_kmeans = self.method_combo.currentIndex() == 2
        self.threshold_spin.setEnabled(not is_kmeans and not self.is_busy())
        self.common_spin.setEnabled(not is_kmeans and not self.is_busy())
        self.k_spin.setEnabled(is_kmeans and not self.is_busy())
        self._update_actions()
        self.method_note.setText(self._tr(
            "相似度只比较双方有效 SNP；最少共同位点默认 1，可提高以减少稀疏数据误分组。两种贪心策略均受样本顺序影响。\nK-means 按 SNP 均值填补缺失，剔除全缺失 SNP；全缺失样本拒绝分型。\n结果是未定相基因型模式分组。",
            "Similarity uses shared called SNPs only. The default minimum is 1; increase it for sparse data. Both greedy strategies depend on sample order.\nK-means mean-imputes each SNP and removes all-missing SNPs; all-missing samples are rejected.\nGroups describe unphased genotype patterns."))

    def _update_actions(self):
        busy = self.is_busy()
        for button in (self.export_group_button, self.export_matrix_button, self.export_image_button, self.export_tree_button, self.export_qc_button):
            button.setEnabled(not busy and self.data is not None)
        self.export_group_button.setEnabled(not busy and self.genotype_result is not None)
        self.export_k_button.setEnabled(not busy and self.k_result is not None)
        has_groups = self.genotype_result is not None and self.groups is not None
        selected = bool(self.selected_plot_groups())
        for control in (self.group_table, self.select_all_button, self.select_none_button, self.label_side_combo):
            control.setEnabled(not busy and has_groups)
        self.plot_format_combo.setEnabled(not busy and has_groups and selected)
        self.export_selected_button.setEnabled(not busy and has_groups and selected)
        has_plot = bool(self.k_figure.axes) if self.tabs.currentIndex() == 1 else has_groups and selected
        self.export_image_button.setEnabled(not busy and has_plot)

    def _source_changed(self, *_):
        if self.is_busy():
            return
        if self.data is not None:
            self.data = self.groups = self.genotype_result = self.k_result = None
            self.source_path = ""
            self.parameters = {}
            self.group_table.setRowCount(0)
            self.last_plot_export = None
            self._update_selection_summary()
            self.k_table.setRowCount(0)
            for figure, canvas in ((self.heatmap_figure, self.heatmap_canvas), (self.k_figure, self.k_canvas)):
                figure.clear()
                canvas.draw_idle()
            self.status_label.setText(self._tr("输入已更改，请重新运行分型。", "Input changed; run genotyping again."))
        self._update_actions()

    def _finished(self):
        super()._finished()
        self._method_changed()

    def set_source(self, path):
        self.source_edit.setText(str(path))

    load_path = set_source

    def choose_file(self):
        path = QFileDialog.getOpenFileName(self, self._tr("选择变异文件", "Choose variant file"), "",
                                           "Variants (*.vcf *.vcf.gz *.hmp *.txt *.tsv);;All files (*)")[0]
        if path:
            self.set_source(path)

    def _input_path(self):
        path = self.source_edit.text().strip()
        if not path or not Path(path).is_file():
            QMessageBox.warning(self, self._tr("需要输入文件", "Input required"), self._tr("请选择存在的变异文件。", "Choose an existing variant file."))
            return None
        return path

    def run_analysis(self):
        path = self._input_path()
        if path is None or self.is_busy():
            return
        method = ["matrix", "onebyone", "kmeans"][self.method_combo.currentIndex()]
        parameters = {"method": method, "threshold": self.threshold_spin.value(),
                      "min_common_snps": self.common_spin.value(), "K": self.k_spin.value()}
        def compute(progress, cancelled):
            data = read_variants(path, lambda p, m: progress(p // 4, m), cancelled)
            callback = lambda p, m: progress(25 + p * 3 // 4, m)
            if method == "kmeans":
                groups = kmeans_groups(data, parameters["K"], callback, cancelled)
                prepared = prepare_kmeans(data)
                parameters["preprocessing"] = {"mean_imputed_calls": prepared.imputed_calls,
                                                "dropped_all_missing_loci": prepared.dropped_loci}
            else:
                groups = similarity_groups(data, parameters["threshold"], method, parameters["min_common_snps"], callback, cancelled)
            return {"data": data, "groups": groups, "parameters": parameters, "source": str(Path(path).resolve())}
        self._submit(compute, self._analysis_ready)

    def _analysis_ready(self, result):
        if not self._same_data(result["data"]):
            self.k_result = None
            self.k_table.setRowCount(0)
            self.k_figure.clear()
            self.k_canvas.draw_idle()
        self.data, self.groups = result["data"], result["groups"]
        self.parameters, self.source_path = result["parameters"], result["source"]
        self.genotype_result = groups_frame(self.data, self.groups)
        blocked = self.group_table.blockSignals(True)
        try:
            self.group_table.setRowCount(len(self.groups))
            for row, members in enumerate(self.groups):
                label = str(self.genotype_result.iloc[members[0]]["Genotype"])
                item = QTableWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, label)
                item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
                item.setCheckState(Qt.CheckState.Checked)
                self.group_table.setItem(row, 0, item)
                self.group_table.setItem(row, 1, QTableWidgetItem(str(len(members))))
        finally:
            self.group_table.blockSignals(blocked)
        self._draw_heatmap()
        self.tabs.setCurrentIndex(0)
        self.progress_bar.setValue(100)
        self.status_label.setText(self._tr(f"完成：{len(self.data.samples)} 个样本，{len(self.data.positions)} 个位点，{len(self.groups)} 个分组。灰色表示缺失。",
                                           f"Complete: {len(self.data.samples)} samples, {len(self.data.positions)} loci, {len(self.groups)} groups. Gray indicates missing calls."))
        self.genotypes_ready.emit(self.genotype_result.copy(), self.source_path)

    def search_k(self):
        path = self._input_path()
        if path is None or self.is_busy():
            return
        maximum = self.max_k_spin.value()
        def compute(progress, cancelled):
            data = read_variants(path, lambda p, m: progress(p // 4, m), cancelled)
            result = search_k(data, maximum, lambda p, m: progress(25 + p * 3 // 4, m), cancelled)
            return {"data": data, "result": result, "source": str(Path(path).resolve())}
        self._submit(compute, self._k_ready)

    def _k_ready(self, payload):
        if self.source_path != payload["source"] or not self._same_data(payload["data"]):
            self.groups, self.genotype_result = None, None
            self.parameters = {}
            self.group_table.setRowCount(0)
            self._update_selection_summary()
            self.heatmap_figure.clear()
            self.heatmap_canvas.draw_idle()
        self.data, self.source_path = payload["data"], payload["source"]
        self.k_result = payload["result"]
        table = self.k_result["table"]
        self.k_spin.setValue(self.k_result["recommended_k"])
        self.k_table.setRowCount(len(table))
        for row, (_, record) in enumerate(table.iterrows()):
            for column, key in enumerate(["K", "SSE", "silhouette", "clusters"]):
                number = record[key]
                value = "n/a" if not np.isfinite(number) else (str(int(number)) if key in {"K", "clusters"} else f"{number:.5g}")
                self.k_table.setItem(row, column, QTableWidgetItem(value))
        self.k_figure.clear()
        axes = self.k_figure.subplots(2, 1, sharex=True)
        axes[0].plot(table.K, table.SSE, "o-", color="#4C78A8")
        axes[0].set_ylabel("SSE")
        axes[1].plot(table.K, table.silhouette, "o-", color="#E45756")
        axes[1].set_ylabel("Silhouette")
        axes[1].set_xlabel("K")
        axes[1].set_xticks(table.K)
        for axis in axes:
            axis.axvline(self.k_result["recommended_k"], color="#54A24B", linestyle="--")
            axis.grid(alpha=0.2)
        self.k_canvas.draw_idle()
        self.tabs.setCurrentIndex(1)
        self.progress_bar.setValue(100)
        k = self.k_result["recommended_k"]
        self.status_label.setText(self._tr(f"推荐 K={k}：有效 silhouette 最大；无有效评分时保守推荐 1。正式分型使用相同的样本×位点矩阵及填补步骤。",
                                           f"Recommended K={k}: maximum valid silhouette; without a valid score, conservatively use 1. Search and final fitting use identical sample × locus preprocessing."))

    def _same_data(self, data):
        return (self.data is not None and self.data.samples == data.samples and self.data.positions == data.positions
                and np.array_equal(self.data.matrix, data.matrix, equal_nan=True))

    def _draw_heatmap(self):
        if self.data is not None and self.groups is not None and self.genotype_result is not None:
            draw_genotype_heatmap(self.heatmap_figure, self.data, self.groups, self.genotype_result,
                                  selected_labels=self.selected_plot_groups(), label_side=self.label_side_combo.currentData())
        else:
            self.heatmap_figure.clear()
        self._update_selection_summary()
        self.heatmap_canvas.draw_idle()
        self._update_actions()

    def selected_plot_groups(self):
        return [str(self.group_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                for row in range(self.group_table.rowCount())
                if self.group_table.item(row, 0) is not None
                and self.group_table.item(row, 0).checkState() == Qt.CheckState.Checked]

    def _update_selection_summary(self):
        selected, total = len(self.selected_plot_groups()), self.group_table.rowCount()
        self.selection_summary.setText(self._tr(f"已选择 {selected} / {total} 个单倍型", f"Selected {selected} / {total} groups"))

    def _plot_selection_changed(self, item=None):
        if isinstance(item, QTableWidgetItem) and item.column() != 0:
            return
        if not self.is_busy():
            self._draw_heatmap()

    def select_plot_groups(self, selected):
        if self.is_busy():
            return
        blocked = self.group_table.blockSignals(True)
        try:
            for row in range(self.group_table.rowCount()):
                self.group_table.item(row, 0).setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
        finally:
            self.group_table.blockSignals(blocked)
        self._draw_heatmap()

    def _require_data(self, groups=False):
        if self.data is None or groups and self.genotype_result is None:
            self.status_label.setText(self._tr("请先完成分型。", "Run genotyping first."))
            return False
        return True

    def _export_clicked(self, handler):
        try:
            handler()
        except Exception as exc:
            self._failed(str(exc))

    def export_groups(self, path=None):
        if not self._require_data(groups=True):
            return
        path = path or _save_dialog(self, "Groups CSV", "genotypes.csv", "CSV (*.csv)")
        if path:
            self.genotype_result.to_csv(path, index=False)
            return str(path)

    def export_matrix(self, path=None):
        if not self._require_data():
            return
        path = path or _save_dialog(self, "Matrix CSV", "genotype_matrix.csv", "CSV (*.csv)")
        if path:
            frame = matrix_frame(self.data)
            if self.genotype_result is not None:
                frame["Genotype"] = self.genotype_result.set_index("Sample").loc[frame.index, "Genotype"]
            frame.to_csv(path, na_rep="")
            return str(path)

    def export_image(self, path=None):
        if self.is_busy():
            return
        if self.tabs.currentIndex() == 0 and not self.selected_plot_groups():
            self.status_label.setText(self._tr("请先勾选要导出的单倍型。", "Select at least one group to export."))
            return
        figure = self.k_figure if self.tabs.currentIndex() == 1 else self.heatmap_figure
        if not figure.axes:
            self.status_label.setText(self._tr("当前没有可导出的图。", "There is no plot to export."))
            return
        path = path or _save_dialog(self, "Export plot", "genotypes.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            figure.savefig(path, dpi=300, bbox_inches="tight")
            return str(path)

    def export_selected_images(self, directory=None, file_format=None):
        if self.is_busy() or not self._require_data(groups=True):
            return
        labels = self.selected_plot_groups()
        if not labels:
            self.status_label.setText(self._tr("请先勾选要导出的单倍型。", "Select at least one group to export."))
            return
        directory = directory or QFileDialog.getExistingDirectory(self, self._tr("选择单倍型图输出目录", "Choose output folder"))
        if not directory:
            return
        data, groups, frame = self.data, self.groups, self.genotype_result
        label_side = self.label_side_combo.currentData()
        file_format = file_format or self.plot_format_combo.currentData()
        self.last_plot_export = None
        def compute(progress, cancelled):
            return export_genotype_plots(directory, data, groups, frame, labels, label_side=label_side,
                                         file_format=file_format, progress_callback=progress, cancel_event=cancelled)
        def completed(folder):
            self.last_plot_export = str(folder)
            self.progress_bar.setValue(100)
            self.status_label.setText(self._tr(f"已分别导出 {len(labels)} 个单倍型图：{folder}",
                                               f"Exported {len(labels)} individual group plots: {folder}"))
        self._submit(compute, completed)

    def export_k_table(self, path=None):
        if self.k_result is None:
            self.status_label.setText(self._tr("请先评估 K。", "Evaluate K first."))
            return
        path = path or _save_dialog(self, "K candidates CSV", "k_candidates.csv", "CSV (*.csv)")
        if path:
            self.k_result["table"].to_csv(path, index=False, na_rep="")
            return str(path)

    def export_qc(self, path=None):
        if not self._require_data():
            return
        path = path or _save_dialog(self, "QC JSON", "genotype_qc.json", "JSON (*.json)")
        if path:
            report = {"input_qc": self.data.report, "analysis_parameters": self.parameters}
            if self.k_result is not None:
                report["k_selection"] = {key: value for key, value in self.k_result.items() if key != "table"}
                report["k_selection"]["candidates"] = json.loads(self.k_result["table"].to_json(orient="records"))
            _write_json(path, report)
            return str(path)

    def export_tree(self, path=None):
        if not self._require_data() or self.is_busy():
            return
        path = path or _save_dialog(self, "Average-linkage clustering tree (not an evolutionary phylogeny)", "genotype_clustering.tre", "Newick tree (*.tre)")
        if not path:
            return
        data = self.data
        def compute(progress, cancelled):
            progress(5, "Building average-linkage clustering tree; missing SNPs are mean-imputed")
            result = newick_tree(data)
            return result
        def completed(tree):
            Path(path).write_text(tree, encoding="utf-8")
            self.status_label.setText(self._tr("已导出平均连接聚类树；它不是进化树。", "Average-linkage clustering tree exported; this is not an evolutionary phylogeny."))
        self._submit(compute, completed)


class VariantFilterPanel(_WorkerPanel):
    def __init__(self, parent=None, language="zh"):
        super().__init__(parent)
        self.language = language
        self.report = None
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.source_edit, self.output_edit, self.chrom_edit = QLineEdit(), QLineEdit(), QLineEdit()
        self.input_label, self.output_label, self.chrom_label = QLabel(), QLabel(), QLabel()
        self.start_label, self.end_label, self.samples_label = QLabel(), QLabel(), QLabel()
        self.maf_label, self.missing_label = QLabel(), QLabel()
        self.start_spin, self.end_spin = QSpinBox(), QSpinBox()
        for spin in [self.start_spin, self.end_spin]:
            spin.setRange(0, 2_147_483_647)
            spin.setSpecialValueText("—")
        self.samples_edit = QPlainTextEdit()
        self.samples_edit.setMaximumHeight(95)
        self.maf_spin, self.missing_spin = QDoubleSpinBox(), QDoubleSpinBox()
        self.maf_spin.setRange(0, 0.5)
        self.missing_spin.setRange(0, 1)
        for spin, value in [(self.maf_spin, 0.05), (self.missing_spin, 0.1)]:
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setValue(value)
        self.browse_button, self.output_button = QPushButton(), QPushButton()
        self.browse_button.clicked.connect(self.choose_file)
        self.output_button.clicked.connect(self.choose_output)
        for label, edit, button in [(self.input_label, self.source_edit, self.browse_button), (self.output_label, self.output_edit, self.output_button)]:
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(button)
            form.addRow(label, row)
        for label, widget in [(self.chrom_label, self.chrom_edit), (self.start_label, self.start_spin),
                              (self.end_label, self.end_spin), (self.samples_label, self.samples_edit),
                              (self.maf_label, self.maf_spin), (self.missing_label, self.missing_spin)]:
            form.addRow(label, widget)
        root.addLayout(form)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        root.addWidget(self.note_label)
        buttons = QHBoxLayout()
        self.run_button, self.cancel_button, self.qc_button = QPushButton(), QPushButton(), QPushButton()
        self.run_button.clicked.connect(self.run_filter)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.qc_button.clicked.connect(lambda: self.export_qc())
        self.cancel_button.setEnabled(False)
        for button in [self.run_button, self.cancel_button, self.qc_button]:
            buttons.addWidget(button)
        root.addLayout(buttons)
        self.progress_bar, self.status_label = QProgressBar(), QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.progress_bar)
        root.addWidget(self.status_label)
        self.report_edit = QPlainTextEdit()
        self.report_edit.setReadOnly(True)
        root.addWidget(self.report_edit, 1)
        self._busy_controls = [self.source_edit, self.output_edit, self.chrom_edit, self.start_spin,
                               self.end_spin, self.samples_edit, self.maf_spin, self.missing_spin,
                               self.browse_button, self.output_button, self.run_button, self.qc_button]
        self.set_language(language)

    def set_language(self, language):
        self.language = "zh" if language == "zh" else "en"
        for setter, chinese, english in [
            (self.input_label.setText, "输入 VCF", "Input VCF"), (self.output_label.setText, "输出 VCF", "Output VCF"),
            (self.chrom_label.setText, "染色体（可空）", "Chromosome (optional)"),
            (self.start_label.setText, "起点（0 不限制）", "Start (0 = unbounded)"),
            (self.end_label.setText, "终点（0 不限制）", "End (0 = unbounded)"),
            (self.samples_label.setText, "保留样本（空白=全部）", "Keep samples (blank = all)"),
            (self.maf_label.setText, "最低 MAF", "Minimum MAF"), (self.missing_label.setText, "最高样本缺失率", "Maximum sample missingness"),
            (self.browse_button.setText, "选择文件", "Browse"), (self.output_button.setText, "选择路径", "Save as"),
            (self.run_button.setText, "开始筛选", "Run filtering"), (self.cancel_button.setText, "取消", "Cancel"),
            (self.qc_button.setText, "导出筛选 QC JSON", "Export filter QC JSON"),
        ]:
            setter(self._tr(chinese, english))
        self.samples_edit.setPlaceholderText(self._tr("每行一个样本 ID，也可使用逗号或空白分隔", "One sample ID per line, or comma/whitespace separated"))
        self.note_label.setText(self._tr("坐标从 1 开始，区间包含两端。仅保留明确二等位 SNP；部分缺失、非二倍体及非法 GT 按整个样本缺失计算。MAF 和缺失率在保留样本中重新计算。输出 .gz 为普通 gzip，不支持 tabix 索引。",
                                       "Coordinates are one-based and inclusive. Only biallelic SNPs are retained. Partial, non-diploid or invalid GTs count as missing samples. MAF and missingness are recomputed within retained samples. A .gz output uses ordinary gzip and cannot be indexed by tabix."))

    def set_source(self, path):
        self.source_edit.setText(str(path))
        name = Path(path).name
        for suffix in [".gz", ".vcf"]:
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)]
        self.output_edit.setText(str(Path(path).with_name(name + ".filtered.vcf.gz")))

    load_path = set_source

    def choose_file(self):
        path = QFileDialog.getOpenFileName(self, "Input VCF", "", "VCF (*.vcf *.vcf.gz);;All files (*)")[0]
        if path:
            self.set_source(path)

    def choose_output(self):
        path = _save_dialog(self, "Output VCF", self.output_edit.text(), "Gzip VCF (*.vcf.gz);;VCF (*.vcf)")
        if path:
            self.output_edit.setText(path)

    def run_filter(self):
        source, output = self.source_edit.text().strip(), self.output_edit.text().strip()
        if not source or not Path(source).is_file() or not output:
            QMessageBox.warning(self, self._tr("需要文件路径", "Paths required"), self._tr("请选择存在的输入文件和输出路径。", "Choose an existing input file and an output path."))
            return
        parameters = {"chrom": self.chrom_edit.text().strip() or None,
                      "start": self.start_spin.value() or None, "end": self.end_spin.value() or None,
                      "samples": re.split(r"[\s,]+", self.samples_edit.toPlainText().strip()) if self.samples_edit.toPlainText().strip() else None,
                      "min_maf": self.maf_spin.value(), "max_missing": self.missing_spin.value()}
        def compute(progress, cancelled):
            return filter_vcf(source, output, **parameters, progress_callback=progress, cancel_event=cancelled)
        self._submit(compute, self._filter_ready)

    def _filter_ready(self, report):
        self.report = report
        self.report_edit.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self.progress_bar.setValue(100)
        self.status_label.setText(self._tr(f"完成：保留 {report['retained_variants']} / {report['input_variants']} 个变异。",
                                           f"Complete: retained {report['retained_variants']} / {report['input_variants']} variants."))

    def export_qc(self, path=None):
        if self.report is None:
            self.status_label.setText(self._tr("请先运行筛选。", "Run filtering first."))
            return
        path = path or _save_dialog(self, "Filter QC JSON", "variant_filter_qc.json", "JSON (*.json)")
        if path:
            _write_json(path, self.report)
            return str(path)
