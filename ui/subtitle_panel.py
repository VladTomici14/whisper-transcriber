"""
SubtitlePanel – left-hand side of the layout.

Consists of:
  • A QTableWidget listing all subtitle entries (index, start, end, text preview)
  • An inline editor (start time, end time, full text) for the selected entry
  • Add / Remove buttons
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QLineEdit, QTextEdit, QPushButton, QSplitter, QAbstractItemView,
    QHeaderView, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from core.srt_parser import SRTFile, SubtitleEntry


class TimeEdit(QLineEdit):
    """A line-edit that validates HH:MM:SS,mmm format."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("00:00:00,000")
        self.setFixedWidth(110)


class SubtitlePanel(QWidget):
    """
    Signals:
        entry_changed()   – emitted whenever an entry is modified/added/removed
                            so the main window can save the SRT and refresh mpv
    """

    entry_changed = pyqtSignal()

    def __init__(self, srt: SRTFile, parent: QWidget | None = None):
        super().__init__(parent)
        self._srt = srt
        self._loading = False       # guard against recursive signals
        self._current_row: int = -1
        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # ── Table ─────────────────────────────────────────────────────── #
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["#", "Start", "End", "Text"])
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setColumnWidth(0, 36)
        self._table.setColumnWidth(1, 100)
        self._table.setColumnWidth(2, 100)
        self._table.currentRowChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        # ── Editor ────────────────────────────────────────────────────── #
        editor_frame = QFrame()
        editor_frame.setFrameShape(QFrame.Shape.StyledPanel)
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setSpacing(6)

        # Time row
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Start"))
        self._start_edit = TimeEdit()
        self._start_edit.editingFinished.connect(self._commit_times)
        time_row.addWidget(self._start_edit)
        time_row.addSpacing(12)
        time_row.addWidget(QLabel("End"))
        self._end_edit = TimeEdit()
        self._end_edit.editingFinished.connect(self._commit_times)
        time_row.addWidget(self._end_edit)
        time_row.addStretch()
        editor_layout.addLayout(time_row)

        # Text
        editor_layout.addWidget(QLabel("Text"))
        self._text_edit = QTextEdit()
        self._text_edit.setFixedHeight(80)
        self._text_edit.setPlaceholderText("Subtitle text…")
        self._text_edit.textChanged.connect(self._commit_text)
        editor_layout.addWidget(self._text_edit)

        # Buttons
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("＋ Add")
        self._add_btn.clicked.connect(self._add_entry)
        self._del_btn = QPushButton("✕ Remove")
        self._del_btn.clicked.connect(self._remove_entry)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()
        editor_layout.addLayout(btn_row)

        splitter.addWidget(editor_frame)
        splitter.setSizes([400, 200])
        root.addWidget(splitter)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def set_srt(self, srt: SRTFile) -> None:
        self._srt = srt
        self._populate_table()

    def highlight_entry_at(self, position_ms: int) -> None:
        """Select the row corresponding to the active subtitle."""
        for i, e in enumerate(self._srt.entries):
            if e.start_ms <= position_ms <= e.end_ms:
                if self._table.currentRow() != i:
                    self._table.blockSignals(True)
                    self._table.setCurrentCell(i, 0)
                    self._table.blockSignals(False)
                return

    # ------------------------------------------------------------------ #
    # Table helpers                                                        #
    # ------------------------------------------------------------------ #
    def _populate_table(self) -> None:
        self._loading = True
        self._table.setRowCount(0)
        for e in self._srt.entries:
            self._add_table_row(e)
        self._loading = False

    def _add_table_row(self, e: SubtitleEntry) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(str(e.index)))
        self._table.setItem(row, 1, QTableWidgetItem(e.start_str))
        self._table.setItem(row, 2, QTableWidgetItem(e.end_str))
        preview = e.text.replace("\n", " ")
        self._table.setItem(row, 3, QTableWidgetItem(preview))

    def _refresh_row(self, row: int) -> None:
        e = self._srt.entries[row]
        self._table.item(row, 1).setText(e.start_str)
        self._table.item(row, 2).setText(e.end_str)
        self._table.item(row, 3).setText(e.text.replace("\n", " "))

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #
    def _on_row_selected(self, row: int) -> None:
        self._current_row = row
        if row < 0 or row >= len(self._srt.entries):
            return
        e = self._srt.entries[row]
        self._loading = True
        self._start_edit.setText(e.start_str)
        self._end_edit.setText(e.end_str)
        self._text_edit.setPlainText(e.text)
        self._loading = False

    def _commit_text(self) -> None:
        if self._loading or self._current_row < 0:
            return
        new_text = self._text_edit.toPlainText()
        self._srt.update_entry(self._current_row, text=new_text)
        self._refresh_row(self._current_row)
        self.entry_changed.emit()

    def _commit_times(self) -> None:
        if self._loading or self._current_row < 0:
            return
        try:
            s = SubtitleEntry.ms_from_srt_time(self._start_edit.text())
            e = SubtitleEntry.ms_from_srt_time(self._end_edit.text())
            self._srt.update_entry(self._current_row, start_ms=s, end_ms=e)
            self._refresh_row(self._current_row)
            self.entry_changed.emit()
        except (ValueError, KeyError):
            pass  # Invalid time string – ignore silently

    def _add_entry(self) -> None:
        # Default: 2 s window starting after the last entry
        last_end = self._srt.entries[-1].end_ms if self._srt.entries else 0
        new_entry = self._srt.add_entry(last_end + 500, last_end + 2500, "New subtitle")
        self._populate_table()
        # Select the new row
        for i, e in enumerate(self._srt.entries):
            if e is new_entry:
                self._table.setCurrentCell(i, 0)
                break
        self.entry_changed.emit()

    def _remove_entry(self) -> None:
        row = self._current_row
        if row < 0 or row >= len(self._srt.entries):
            return
        self._srt.remove_entry(row)
        self._table.removeRow(row)
        self._current_row = -1
        self.entry_changed.emit()