"""
SRT file parser and writer.
Handles reading, writing, and in-memory representation of SRT subtitle files.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubtitleEntry:
    index: int
    start_ms: int       # milliseconds
    end_ms: int         # milliseconds
    text: str           # may contain newlines

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def ms_from_srt_time(t: str) -> int:
        """Parse '00:01:23,456' → milliseconds."""
        h, m, rest = t.split(":")
        s, ms = rest.replace(",", ".").split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    @staticmethod
    def ms_to_srt_time(ms: int) -> str:
        """milliseconds → '00:01:23,456'"""
        h = ms // 3_600_000;  ms %= 3_600_000
        m = ms // 60_000;     ms %= 60_000
        s = ms // 1_000;      ms %= 1_000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @property
    def start_str(self) -> str:
        return self.ms_to_srt_time(self.start_ms)

    @property
    def end_str(self) -> str:
        return self.ms_to_srt_time(self.end_ms)

    def to_srt_block(self) -> str:
        return f"{self.index}\n{self.start_str} --> {self.end_str}\n{self.text}\n"


class SRTFile:
    """In-memory representation of an SRT file with load/save support."""

    _ARROW = " --> "
    _TIME_RE = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
        r"\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    )

    def __init__(self):
        self.entries: list[SubtitleEntry] = []
        self.path: Path | None = None

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    def load(self, path: str | Path) -> None:
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8-sig", errors="replace")
        self.entries = self._parse(text)

    def save(self, path: str | Path | None = None) -> None:
        dest = Path(path) if path else self.path
        if dest is None:
            raise ValueError("No path specified for save.")
        dest.write_text(self.to_srt_text(), encoding="utf-8")
        self.path = dest

    def to_srt_text(self) -> str:
        return "\n".join(e.to_srt_block() for e in self.entries) + "\n"

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #
    def _parse(self, raw: str) -> list[SubtitleEntry]:
        # Normalise line endings
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n{2,}", raw.strip())
        entries = []
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue
            # Find the timing line (not always line index 1)
            timing_idx = None
            for i, line in enumerate(lines):
                if self._TIME_RE.search(line):
                    timing_idx = i
                    break
            if timing_idx is None:
                continue
            m = self._TIME_RE.search(lines[timing_idx])
            start_ms = SubtitleEntry.ms_from_srt_time(m.group(1).replace(".", ","))
            end_ms   = SubtitleEntry.ms_from_srt_time(m.group(2).replace(".", ","))
            # Index is usually line before timing
            try:
                idx = int(lines[timing_idx - 1]) if timing_idx > 0 else len(entries) + 1
            except ValueError:
                idx = len(entries) + 1
            text = "\n".join(lines[timing_idx + 1:]).strip()
            entries.append(SubtitleEntry(idx, start_ms, end_ms, text))
        return entries

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #
    def update_entry(self, index: int, text: str | None = None,
                     start_ms: int | None = None, end_ms: int | None = None) -> None:
        e = self.entries[index]
        if text is not None:
            e.text = text
        if start_ms is not None:
            e.start_ms = start_ms
        if end_ms is not None:
            e.end_ms = end_ms

    def add_entry(self, start_ms: int, end_ms: int, text: str = "") -> SubtitleEntry:
        idx = max((e.index for e in self.entries), default=0) + 1
        entry = SubtitleEntry(idx, start_ms, end_ms, text)
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.start_ms)
        self._renumber()
        return entry

    def remove_entry(self, list_index: int) -> None:
        self.entries.pop(list_index)
        self._renumber()

    def _renumber(self) -> None:
        for i, e in enumerate(self.entries, 1):
            e.index = i

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #
    def entry_at(self, position_ms: int) -> SubtitleEntry | None:
        """Return the subtitle active at a given playback position."""
        for e in self.entries:
            if e.start_ms <= position_ms <= e.end_ms:
                return e
        return None