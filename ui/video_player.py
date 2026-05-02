"""
VideoPlayer widget — uses mpv's OpenGL render API via MpvRenderContext.
This approach works on all platforms including macOS Apple Silicon.

Dependencies:
    pip install python-mpv PyQt6 PyOpenGL
    brew install mpv  (macOS)
"""

from __future__ import annotations
import ctypes

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    import mpv
    MPV_AVAILABLE = True
except ImportError:
    MPV_AVAILABLE = False


class VideoPlayerWidget(QOpenGLWidget):
    """
    Renders mpv output via OpenGL — works on macOS Apple Silicon, Linux, Windows.

    Signals:
        position_changed(int)   – current position in milliseconds
        duration_changed(int)   – total duration in milliseconds
        playback_ended()        – fired when video reaches end
    """

    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    playback_ended   = pyqtSignal()
    _frame_ready     = pyqtSignal()   # internal: cross-thread repaint trigger

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)

        self._player: "mpv.MPV | None" = None
        self._ctx:    "mpv.MpvRenderContext | None" = None
        self._current_subtitle_file: str | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_position)

        # _frame_ready is emitted from mpv's thread; Qt delivers it safely
        # to the main thread via a queued connection before calling update()
        self._frame_ready.connect(self.update, Qt.ConnectionType.QueuedConnection)

    # ------------------------------------------------------------------ #
    # QOpenGLWidget lifecycle                                              #
    # ------------------------------------------------------------------ #
    def initializeGL(self) -> None:
        if not MPV_AVAILABLE:
            return

        self._player = mpv.MPV(
            vo="libmpv",          # render into our context, never open a window
            hwdec="no",           # disable hwdec — causes issues with libmpv on macOS
            keep_open=True,
            hr_seek="yes",
        )

        @self._player.property_observer("duration")
        def _on_duration(name, value):
            if value is not None:
                self.duration_changed.emit(int(value * 1000))

        @self._player.event_callback("end-file")
        def _on_end(event):
            self.playback_ended.emit()

        from PyQt6.QtGui import QOpenGLContext

        # mpv requires a true C function pointer, not a plain Python callable
        GlProcAddressT = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)

        def _get_proc_address_py(_, name: bytes) -> int:
            ctx = QOpenGLContext.currentContext()
            if ctx is None:
                return 0
            addr = ctx.getProcAddress(name)
            # getProcAddress returns a sip.voidptr on PyQt6 — cast via int()
            try:
                return int(addr) or 0
            except (TypeError, ValueError):
                return 0

        # Keep reference on self so the C callback isn't garbage-collected
        self._get_proc_address_cb = GlProcAddressT(_get_proc_address_py)

        self._ctx = mpv.MpvRenderContext(
            self._player,
            "opengl",
            opengl_init_params={"get_proc_address": self._get_proc_address_cb},
        )
        # mpv calls this from its render thread — emit a signal so Qt
        # marshals the repaint safely onto the main thread
        self._ctx.update_cb = self._frame_ready.emit

    def paintGL(self) -> None:
        if self._ctx is None:
            return
        ratio = self.devicePixelRatio()
        self._ctx.render(
            flip_y=True,
            opengl_fbo={
                "w": int(self.width() * ratio),
                "h": int(self.height() * ratio),
                "fbo": self.defaultFramebufferObject(),
            },
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.update()

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
        if not self._player:
            return
        self._current_subtitle_file = path
        try:
            self._player.command("sub-remove")
        except Exception:
            pass
        self._player.command("sub-add", path, "select", "Live Subtitles", "")

    def reload_subtitle(self) -> None:
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

    def _poll_position(self) -> None:
        self.position_changed.emit(self.position_ms)

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        if self._ctx:
            self._ctx.free()
        if self._player:
            self._player.terminate()
        super().closeEvent(event)