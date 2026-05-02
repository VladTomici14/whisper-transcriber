"""
MainWindow – assembles VideoPlayerWidget, SubtitlePanel, and PlaybackControls.

Layout:
  ┌────────────────────────────────┬──────────────────────┐
  │                                │                      │
  │       Video (mpv)              │  Subtitle list       │
  │                                │  + inline editor     │
  │                                │                      │
  ├────────────────────────────────┴──────────────────────┤
  │   ▶  seek-bar  00:00:00 / 00:00:00                    │
  └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import tempfile
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QToolBar, QStatusBar,
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QTimer

from core.srt_parser import SRTFile
from ui.video_player import VideoPlayerWidget
from ui.subtitle_panel import SubtitlePanel
from ui.playback_controls import PlaybackControls


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Editor")
        self.resize(1280, 720)

        self._srt = SRTFile()
        self._video_path: str | None = None
        # Temporary SRT written on every edit so mpv can reload it
        self._tmp_srt: Path = Path(tempfile.mktemp(suffix=".srt"))

        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._player = VideoPlayerWidget()
        splitter.addWidget(self._player)

        self._sub_panel = SubtitlePanel(self._srt)
        self._sub_panel.setMinimumWidth(340)
        splitter.addWidget(self._sub_panel)

        splitter.setSizes([860, 420])
        root.addWidget(splitter, stretch=1)

        # Playback controls pinned at the bottom
        self._controls = PlaybackControls()
        root.addWidget(self._controls)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        open_video_act = QAction("Open &Video…", self)
        open_video_act.setShortcut(QKeySequence("Ctrl+O"))
        open_video_act.triggered.connect(self._open_video)
        file_menu.addAction(open_video_act)

        open_srt_act = QAction("Open &Subtitles (.srt)…", self)
        open_srt_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_srt_act.triggered.connect(self._open_srt)
        file_menu.addAction(open_srt_act)

        file_menu.addSeparator()

        save_act = QAction("&Save Subtitles", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_srt)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Subtitles &As…", self)
        save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_act.triggered.connect(self._save_srt_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        export_act = QAction("&Export with ffmpeg…", self)
        export_act.triggered.connect(self._export_video)
        file_menu.addAction(export_act)

    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Open a video and an SRT file to begin.")

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #
    def _connect_signals(self) -> None:
        # Player → controls
        self._player.position_changed.connect(self._on_position_changed)
        self._player.duration_changed.connect(self._controls.set_duration)
        self._player.playback_ended.connect(lambda: self._controls.set_playing(False))

        # Controls → player
        self._controls.play_pause_clicked.connect(self._on_play_pause)
        self._controls.seek_requested.connect(self._player.seek)

        # Subtitle panel → save & reload
        self._sub_panel.entry_changed.connect(self._on_subtitle_edited)

    # ------------------------------------------------------------------ #
    # Menu actions                                                         #
    # ------------------------------------------------------------------ #
    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
        )
        if path:
            self._video_path = path
            self._player.load_video(path)
            if self._tmp_srt.exists():
                self._player.load_subtitle(str(self._tmp_srt))
            self._status.showMessage(f"Video: {path}")

    def _open_srt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Subtitles", "", "SRT Files (*.srt);;All Files (*)"
        )
        if not path:
            return
        try:
            self._srt.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not load SRT:\n{exc}")
            return
        self._sub_panel.set_srt(self._srt)
        self._flush_srt_to_temp()
        if self._video_path:
            self._player.load_subtitle(str(self._tmp_srt))
        self._status.showMessage(
            f"Subtitles: {path} — {len(self._srt.entries)} entries loaded."
        )

    def _save_srt(self) -> None:
        if self._srt.path:
            self._srt.save()
            self._status.showMessage(f"Saved → {self._srt.path}")
        else:
            self._save_srt_as()

    def _save_srt_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Subtitles As", "", "SRT Files (*.srt)"
        )
        if path:
            self._srt.save(path)
            self._status.showMessage(f"Saved → {path}")

    def _export_video(self) -> None:
        """Build and display the ffmpeg command; optionally run it."""
        if not self._video_path:
            QMessageBox.information(self, "Export", "No video loaded.")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "MP4 Files (*.mp4)"
        )
        if not out:
            return
        srt_path = str(self._srt.path or self._tmp_srt)
        cmd = (
            f'ffmpeg -i "{self._video_path}" '
            f'-vf subtitles="{srt_path}" '
            f'"{out}"'
        )
        reply = QMessageBox.question(
            self, "Export with ffmpeg",
            f"Run this command?\n\n{cmd}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            import subprocess
            try:
                subprocess.Popen(cmd, shell=True)
                self._status.showMessage("ffmpeg started in background…")
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))

    # ------------------------------------------------------------------ #
    # Playback helpers                                                     #
    # ------------------------------------------------------------------ #
    def _on_play_pause(self) -> None:
        self._player.toggle_play()
        self._controls.set_playing(self._player.is_playing)

    def _on_position_changed(self, ms: int) -> None:
        self._controls.update_position(ms)
        self._sub_panel.highlight_entry_at(ms)

    # ------------------------------------------------------------------ #
    # Live subtitle reload                                                 #
    # ------------------------------------------------------------------ #
    def _on_subtitle_edited(self) -> None:
        """Flush edits to temp file and ask mpv to reload subtitles."""
        self._flush_srt_to_temp()
        self._player.reload_subtitle()
        self._status.showMessage("Subtitles updated (unsaved changes)")

    def _flush_srt_to_temp(self) -> None:
        self._tmp_srt.write_text(self._srt.to_srt_text(), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Keyboard shortcuts                                                   #
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._on_play_pause()
        elif event.key() == Qt.Key.Key_Left:
            self._player.seek(max(0, self._player.position_ms - 5000))
        elif event.key() == Qt.Key.Key_Right:
            self._player.seek(self._player.position_ms + 5000)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        # Clean up temp file
        try:
            self._tmp_srt.unlink(missing_ok=True)
        except Exception:
            pass
        super().closeEvent(event)