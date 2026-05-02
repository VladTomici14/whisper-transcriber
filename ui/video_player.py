"""
VideoPlayer widget — wraps python-mpv for hardware-accelerated playback
with live subtitle injection.

Dependencies:
    pip install python-mpv PyQt6

mpv must also be installed on the system:
    macOS:   brew install mpv
    Ubuntu:  sudo apt install mpv libmpv-dev
    Windows: download mpv and add to PATH
"""

from __future__ import annotations
import ctypes
import sys

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QColor

try:
    import mpv
    MPV_AVAILABLE = True
except ImportError:
    MPV_AVAILABLE = False


class VideoPlayerWidget(QWidget):
    """
    Embeds an mpv player into a Qt widget.

    Signals:
        position_changed(int)   – current position in milliseconds
        duration_changed(int)   – total duration in milliseconds
        playback_ended()        – fired when video reaches end
    """

    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    playback_ended   = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background: #0a0a0a;")

        self._player: "mpv.MPV | None" = None
        self._current_subtitle_file: str | None = None

        # Poll position every 250 ms
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_position)

        if MPV_AVAILABLE:
            self._init_mpv()

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #
    def _init_mpv(self) -> None:
        wid = self.winId().__int__() if sys.platform != "darwin" else None

        kwargs = dict(
            vo="gpu" if sys.platform != "darwin" else "libmpv",
            hwdec="auto",
            keep_open=True,
            hr_seek="yes",
        )
        if wid:
            kwargs["wid"] = wid

        self._player = mpv.MPV(**kwargs)

        # Observe duration when it becomes known
        @self._player.property_observer("duration")
        def _on_duration(name, value):
            if value is not None:
                self.duration_changed.emit(int(value * 1000))

        @self._player.event_callback("end-file")
        def _on_end(event):
            self.playback_ended.emit()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def load_video(self, path: str) -> None:
        if not self._player:
            return
        self._player.play(path)
        self._player.pause = True
        self._poll_timer.start()

    def load_subtitle(self, path: str) -> None:
        """Load/reload an SRT file as the active subtitle track."""
        if not self._player:
            return
        self._current_subtitle_file = path
        # Remove old external tracks, then add the new one
        try:
            self._player.command("sub-remove")
        except Exception:
            pass
        self._player.command("sub-add", path, "select", "Live Subtitles", "")

    def reload_subtitle(self) -> None:
        """Re-read the current subtitle file (called after in-place edits)."""
        if self._current_subtitle_file:
            self.load_subtitle(self._current_subtitle_file)

    def play(self) -> None:
        if self._player:
            self._player.pause = False

    def pause(self) -> None:
        if self._player:
            self._player.pause = True

    def toggle_play(self) -> None:
        if self._player:
            self._player.pause = not self._player.pause

    def seek(self, position_ms: int) -> None:
        if self._player:
            self._player.seek(position_ms / 1000.0, "absolute")

    @property
    def is_playing(self) -> bool:
        return bool(self._player and not self._player.pause)

    @property
    def position_ms(self) -> int:
        if self._player:
            pos = self._player.time_pos
            return int(pos * 1000) if pos is not None else 0
        return 0

    @property
    def duration_ms(self) -> int:
        if self._player:
            dur = self._player.duration
            return int(dur * 1000) if dur is not None else 0
        return 0

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #
    def _poll_position(self) -> None:
        self.position_changed.emit(self.position_ms)

    def paintEvent(self, event):
        if not MPV_AVAILABLE:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#0a0a0a"))
            p.setPen(QColor("#555"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "python-mpv not installed.\nSee README for setup.")

    def closeEvent(self, event):
        self._poll_timer.stop()
        if self._player:
            self._player.terminate()
        super().closeEvent(event)