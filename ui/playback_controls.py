"""
PlaybackControls – seek bar + play/pause + time display.
"""

from __future__ import annotations
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSlider, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal


def _fmt(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class PlaybackControls(QWidget):
    """
    Signals:
        seek_requested(int)  – user dragged the slider (milliseconds)
        play_pause_clicked()
    """

    seek_requested   = pyqtSignal(int)
    play_pause_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = 0
        self._user_seeking = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self.play_pause_clicked)
        layout.addWidget(self._play_btn)

        self._time_label = QLabel("00:00:00 / 00:00:00")
        self._time_label.setFixedWidth(160)
        layout.addWidget(self._time_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderPressed.connect(self._slider_pressed)
        self._slider.sliderReleased.connect(self._slider_released)
        layout.addWidget(self._slider, stretch=1)

    # ------------------------------------------------------------------ #
    # Public API called by MainWindow                                      #
    # ------------------------------------------------------------------ #
    def set_duration(self, ms: int) -> None:
        self._duration_ms = ms

    def update_position(self, ms: int) -> None:
        if self._user_seeking or not self._duration_ms:
            return
        ratio = ms / self._duration_ms
        self._slider.setValue(int(ratio * 1000))
        self._time_label.setText(f"{_fmt(ms)} / {_fmt(self._duration_ms)}")

    def set_playing(self, playing: bool) -> None:
        self._play_btn.setText("⏸" if playing else "▶")

    # ------------------------------------------------------------------ #
    # Slider interaction                                                   #
    # ------------------------------------------------------------------ #
    def _slider_pressed(self) -> None:
        self._user_seeking = True

    def _slider_released(self) -> None:
        self._user_seeking = False
        ratio = self._slider.value() / 1000.0
        target_ms = int(ratio * self._duration_ms)
        self.seek_requested.emit(target_ms)